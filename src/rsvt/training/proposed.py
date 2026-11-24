"""Memory-bounded training of the proposed RSVT on existing full-FOV CT views.

Theory: volumetric patch tokens (UNETR), soft/hard token selection (DynamicViT),
and upper-tail loss optimization (Rockafellar--Uryasev). See docs/rsvt_theory.md.
Our combination is a research hypothesis, not a reproduction of those models.
"""

from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from rsvt.data.phase2 import _json
from rsvt.data.inventory import write_csv
from rsvt.data.splits import digest_json
from rsvt.evaluation.classification import classification_metrics
from rsvt.experiments.phase5 import _plain
from rsvt.losses.classification import training_class_weights
from rsvt.losses.cvar import class_conditional_cvar
from rsvt.losses.budget import ProjectedTokenDual
from rsvt.training.checkpoint import load_checkpoint, save_checkpoint
from rsvt.training.optimization import build_optimizer, build_scheduler, optimization_report
from rsvt.training.risk_trainer import configure_stage
from rsvt.utils.reproducibility import seed_everything, rng_state, restore_rng


def stratified_batches(labels, logical_size, seed):
    """One visit per case, no oversampling; spread each class over logical batches."""
    if isinstance(logical_size, bool) or not isinstance(logical_size, int) or logical_size < 4:
        raise ValueError('Logical batch size must be an integer >=4')
    if not labels or any(label not in range(4) for label in labels):
        raise ValueError('Four-class training labels required')
    rng = np.random.default_rng(seed)
    batches = [[] for _ in range(math.ceil(len(labels)/logical_size))]
    cursor = 0
    for label in range(4):
        own = [i for i, value in enumerate(labels) if value == label]
        rng.shuffle(own)
        for index in own:
            batches[cursor % len(batches)].append(index); cursor += 1
    rng.shuffle(batches)
    for batch in batches: rng.shuffle(batch)
    return batches


def _forward(model, sample, device, mode):
    image = sample['image'][None].to(device)
    valid = sample['valid_mask'][None].to(device)
    affine = sample['voxel_to_lps'][None].to(device)
    output = model(image, valid, affine, routing_mode=mode)
    target = torch.tensor([sample['target']], device=device, dtype=torch.long)
    loss = F.cross_entropy(output['logits'], target)
    if not torch.isfinite(loss): raise ValueError('Nonfinite RSVT classification loss')
    expected = output['routing'].get('expected_sequence_counts')
    expected = expected[0] if expected is not None else loss.new_tensor(output['compute']['actual_sequence_length_including_cls'])
    return output, loss, expected


def logical_objective(losses, targets, expected, weights, risk, dual, cap, *, risk_active, budget_active):
    """Differentiate a logical-batch objective with respect to per-case scalars."""
    ce = (losses * weights[targets]).mean()
    tail = class_conditional_cvar(losses, targets, alpha=risk['alpha'], fb_weight=risk['fb_weight'],
        fm_weight=risk['fm_weight'], min_class_samples=risk['min_class_samples'])
    budget = dual.terms(expected, cap) if budget_active else {'loss': losses.sum()*0., 'constraint': losses.sum()*0.}
    total = ce + (risk['cvar_weight']*tail['loss'] if risk_active else 0.) + budget['loss']
    return total, {'classification': ce, 'cvar': tail, 'budget': budget,
                   'risk_active': risk_active, 'budget_active': budget_active}


def replay_step(model, optimizer, dataset, indices, device, weights, risk, dual, *, stage, gradient_clip):
    """Exact first-order logical-batch gradient, replayed one volume at a time.

    Pass 1 measures losses/expected token counts without retaining image graphs.
    Differentiate CE+CVaR+token constraint in scalar space. Replay each case with
    its original RNG state and backpropagate those scalar coefficients. This
    avoids the meaningless batch-size-one CVaR shortcut and large graph batches.
    """
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError('Replay requires stateless normalization (RSVT uses GroupNorm/LayerNorm)')
    settings = configure_stage(model, 'B5_full', stage, localization_active=False)
    optimizer.zero_grad(set_to_none=True)
    states, losses, counts, targets = [], [], [], []
    for index in indices:
        states.append(rng_state())
        sample = dataset[index]
        with torch.no_grad(): _, loss, count = _forward(model, sample, device, settings['routing_mode'])
        losses.append(loss.detach()); counts.append(count.detach()); targets.append(sample['target'])
    after_measurement = rng_state()
    probe = torch.stack(losses).requires_grad_()
    expected = torch.stack(counts).requires_grad_()
    target_tensor = torch.tensor(targets, device=device, dtype=torch.long)
    total, details = logical_objective(probe, target_tensor, expected, weights, risk, dual, model.sequence_cap,
        risk_active=settings['risk_enabled'], budget_active=settings['dual_enabled'])
    if not torch.isfinite(total): raise ValueError('Nonfinite logical-batch objective')
    ce_coefficients, token_coefficients = torch.autograd.grad(total, (probe, expected), allow_unused=True)
    if token_coefficients is None: token_coefficients = torch.zeros_like(expected)
    try:
        for position, index in enumerate(indices):
            restore_rng(states[position])
            _, loss, count = _forward(model, dataset[index], device, settings['routing_mode'])
            if (not torch.isclose(loss.detach(), losses[position], rtol=1e-5, atol=1e-6)
                    or not torch.isclose(count.detach(), counts[position], rtol=1e-5, atol=1e-6)):
                raise ValueError('Replay differs from measurement; optimizer was not stepped')
            surrogate = ce_coefficients[position].detach()*loss + token_coefficients[position].detach()*count
            surrogate.backward()
    finally:
        restore_rng(after_measurement)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not any(p.grad is not None for p in parameters): raise ValueError('No trainable gradient')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True))
    optimizer.step()
    if settings['dual_enabled']: dual.update(details['budget']['constraint'])
    return {'loss': float(total.detach()), 'details': _plain(details), 'cases': len(indices),
            'gradient_norm_before_clip': norm, 'gradient_clipped': norm > gradient_clip,
            'optimizer_steps': 1, 'mode': settings['routing_mode'], 'training_is_dense': True}


def evaluate_rsvt(model, dataset, device, partition):
    if partition not in {'train', 'validation'} or any(r['split'] != partition for r in dataset.rows):
        raise ValueError('RSVT evaluation only accepts matching train/validation data')
    model.eval(); targets, probabilities, predictions, losses = [], [], [], []
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            out, loss, _ = _forward(model, sample, device, 'hard')
            probs = out['logits'].softmax(1)[0].cpu().tolist()
            targets.append(sample['target']); probabilities.append(probs); losses.append(float(loss))
            route = out['routing']; coarse = route['source_valid_mask'] & (route['source_geometry']['scale_ids'] == 0)
            if not bool(route['selected_source_mask'][coarse].all()): raise ValueError('Coarse coverage lost')
            predictions.append({'case_key': sample['case_key'], 'target': sample['target'], 'probabilities': probs,
                'sequence_tokens_including_cls': out['compute']['actual_sequence_length_including_cls'],
                'dense_tokens_including_cls': out['compute']['dense_sequence_length_including_cls']})
    metrics = classification_metrics(targets, probabilities, partition=partition)
    metrics['loss'] = float(np.mean(losses))
    return metrics, predictions


def fit_rsvt(model, train, validation, config, output_dir, contract, *, resume=False, stop_after_epoch=None, progress=None):
    if (any(r['split'] != 'train' for r in train.rows) or any(r['split'] != 'validation' for r in validation.rows)):
        raise ValueError('Training excludes test cases')
    if model.localizer is not None: raise ValueError('This volume-classification job does not train localization')
    t = config['training']; destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    seed_everything(t['seed'], t['threads'])
    if t['device'] == 'cuda' and not torch.cuda.is_available(): raise ValueError('A Colab GPU is required')
    device = torch.device(t['device']); model.to(device)
    optimizer = build_optimizer(model, t); scheduler = build_scheduler(optimizer, t)
    dual = ProjectedTokenDual(**config['dual'])
    weights, class_counts = training_class_weights([r['label_index'] for r in train.rows]); weights = weights.to(device)
    binding = digest_json(contract); history, best_loss, bad, start, best_epoch = [], math.inf, 0, 0, 0
    if resume:
        state = load_checkpoint(destination/'last.pt', binding, expected_format='rsvt_real_state_v1')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); dual.load_state_dict(state['dual']); restore_rng(state['rng'])
        history, best_loss, bad, start, best_epoch = state['history'], state['best_loss'], state['bad_epochs'], state['epoch'], state['best_epoch']
        if state.get('last_epoch_report') is not None:
            _json(destination/f'epoch_{start:03d}.json', state['last_epoch_report'])
    elif (destination/'last.pt').exists() or (destination/'best.pt').exists():
        raise FileExistsError('RSVT checkpoints exist; explicit resume is required')
    begun = time.perf_counter()
    stages = config['stages']
    for epoch in range(start, t['epochs']):
        if bad >= t['patience']: break
        stage = ('baseline' if epoch < stages['baseline_epochs'] else 'routing_warmup'
                 if epoch < stages['baseline_epochs']+stages['routing_warmup_epochs'] else 'risk_finetune')
        batch_indices = stratified_batches([r['label_index'] for r in train.rows], t['logical_batch_size'], t['seed']+1009*epoch)
        rows = []
        for batch_number, indices in enumerate(batch_indices, 1):
            rows.append(replay_step(model, optimizer, train, indices, device, weights, config['risk'], dual,
                stage=stage, gradient_clip=t['gradient_clip']))
            if progress: progress({'event': 'logical_batch', 'epoch': epoch+1, 'stage': stage,
                'batch': batch_number, 'batches': len(batch_indices), 'loss': rows[-1]['loss']})
        train_metrics, _ = evaluate_rsvt(model, train, device, 'train')
        validation_metrics, predictions = evaluate_rsvt(model, validation, device, 'validation')
        eligible = stage == 'risk_finetune'
        improved = eligible and validation_metrics['loss'] < best_loss-t['min_delta']
        if improved: best_loss, bad, best_epoch = validation_metrics['loss'], 0, epoch+1
        elif eligible: bad += 1
        else: bad = 0
        row = {'epoch': epoch+1, 'stage': stage, 'checkpoint_eligible': eligible,
               'learning_rate': optimizer.param_groups[0]['lr'],
               'optimizer_steps': len(rows), 'training_logical_objective': float(np.mean([r['loss'] for r in rows])),
               'training_loss': train_metrics['loss'], 'training_macro_f1': train_metrics['macro_f1'],
               'training_balanced_accuracy': train_metrics['balanced_accuracy'],
               'training_accuracy_hard': train_metrics['accuracy'], 'validation_loss': validation_metrics['loss'],
               'validation_accuracy': validation_metrics['accuracy'], 'validation_macro_f1': validation_metrics['macro_f1'],
               'validation_tamper_auroc': validation_metrics['tamper_auroc'],
               'validation_balanced_accuracy': validation_metrics['balanced_accuracy'],
               'clipped_steps': sum(r['gradient_clipped'] for r in rows), 'dual_multiplier': dual.multiplier}
        history.append(row); scheduler.step()
        epoch_report = {'steps': rows, 'training_metrics_hard': train_metrics, 'validation_metrics': validation_metrics}
        state = {'format_version': 'rsvt_real_state_v1', 'contract_sha256': binding, 'epoch': epoch+1,
            'model': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'dual': dual.state_dict(),
            'rng': rng_state(), 'history': history, 'best_loss': best_loss, 'bad_epochs': bad, 'best_epoch': best_epoch,
            'last_epoch_report': epoch_report}
        if improved:
            save_checkpoint(destination/'best.pt', state); _json(destination/'best_validation_predictions.json', predictions)
        save_checkpoint(destination/'last.pt', state)
        _json(destination/'history.json', history); _json(destination/f'epoch_{epoch+1:03d}.json',
            epoch_report)
        write_csv(destination/'history.csv', tuple(row), history)
        if progress: progress(row)
        if stop_after_epoch is not None and epoch+1 >= stop_after_epoch: break
    _json(destination/'history.json', history)
    write_csv(destination/'history.csv', tuple(history[-1]), history)
    has_risk_checkpoint = (destination/'best.pt').is_file()
    selected = destination/('best.pt' if has_risk_checkpoint else 'last.pt')
    state = load_checkpoint(selected, binding, expected_format='rsvt_real_state_v1')
    model.load_state_dict(state['model'])
    metrics, predictions = evaluate_rsvt(model, validation, device, 'validation')
    _json(destination/'selected_validation_predictions.json', predictions)
    if has_risk_checkpoint: _json(destination/'best_validation_predictions.json', predictions)
    summary = {'ok': True, 'model': 'Risk-Aware Sparse Volumetric Transformer', 'scope': 'exploratory_real_CT_volume_classification',
        'training_ready': False, 'phase_f04_complete': False, 'epochs_completed': history[-1]['epoch'],
        'training_finished': history[-1]['epoch'] >= t['epochs'] or bad >= t['patience'],
        'stopped_early': bad >= t['patience'],
        'best_epoch': state['epoch'] if has_risk_checkpoint else None,
        'reported_checkpoint_epoch': state['epoch'],
        'checkpoint_stage': 'risk_finetune' if has_risk_checkpoint else 'provisional_warmup_only',
        'train_cases': len(train), 'validation_cases': len(validation), 'test_cases_opened': 0,
        'training_class_counts': class_counts.tolist(), 'training_class_weights': weights.cpu().tolist(),
        'validation_metrics': metrics, 'history': history,
        'device': str(device), 'precision': 'float32', 'optimization': optimization_report(optimizer, t),
        'parameters': sum(p.numel() for p in model.parameters()), 'wall_seconds_this_invocation': time.perf_counter()-begun,
        'selection_rule': 'minimum_unweighted_validation_CE_hard_packed_among_risk_finetuning_epochs_only',
        'inference_sequence_cap_including_cls': model.sequence_cap, 'training_soft_routing_is_dense': True,
        'cvar_estimator': 'class_conditional_fractional_empirical_tail_per_logical_batch',
        'generator_variance_active': False, 'localization_active': False, 'annotation_points_used': False,
        'limitations': ['Source identity and leakage assurance remain unresolved.',
            '96-cube full-FOV coarsening may suppress small tampering evidence.',
            'Hyperparameters are provisional; no efficiency or superiority claim.',
            'Two-pass replay recomputes forwards; sparse inference does not reduce training token construction.']}
    _json(destination/'summary.json', summary)
    return summary
