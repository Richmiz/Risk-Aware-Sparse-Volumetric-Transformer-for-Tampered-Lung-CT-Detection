"""Exploratory candidate-target baseline training, validation, and epoch-boundary resume."""

from __future__ import annotations
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from rsvt.data.phase2 import _json
from rsvt.data.inventory import write_csv
from rsvt.data.splits import digest_json
from rsvt.evaluation.classification import LABELS, classification_metrics
from rsvt.losses.classification import balanced_cross_entropy, training_class_weights
from rsvt.training.checkpoint import load_checkpoint, save_checkpoint
from rsvt.training.diagnostics import gradient_groups
from rsvt.training.optimization import build_optimizer, build_scheduler, optimization_report
from rsvt.utils.reproducibility import rng_state, restore_rng, seed_everything


def evaluate_partition(model,dataset,device,*,partition):
    if partition not in {"train","validation"} or not len(dataset) or any(r["split"] != partition for r in dataset.rows):
        raise ValueError("Evaluation requires a nonempty matching train/validation partition")
    loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    model.eval()
    targets,probabilities,predictions,losses = [],[],[],[]
    with torch.inference_mode():
        for batch in loader:
            image,valid,target = batch["image"].to(device),batch["valid_mask"].to(device),batch["target"].to(device)
            logits = model(image,valid)["logits"]
            loss = torch.nn.functional.cross_entropy(logits,target)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError("Nonfinite evaluation outputs")
            probs = logits.softmax(1).cpu().numpy()
            targets += target.cpu().tolist()
            probabilities += probs.tolist()
            losses.append(float(loss))
            predictions.append({"case_key":batch["case_key"][0],"candidate_target":int(target[0]),
                                "probabilities":probs[0].tolist(),"predicted_class":int(probs[0].argmax())})
    metrics = classification_metrics(targets,probabilities,partition=partition)
    metrics["loss"] = float(np.mean(losses))
    return metrics,predictions


def evaluate_validation(model,dataset,device):
    return evaluate_partition(model,dataset,device,partition="validation")


def _train_fit_passed(metrics,config):
    return metrics["accuracy"] >= config["train_fit_accuracy"] and metrics["loss"] <= config["train_fit_max_loss"]


def _fit_streak(history):
    count = 0
    for row in reversed(history):
        if not row["train_fit_passed"]:
            break
        count += 1
    return count


def profile_forward(model,dataset,device,iterations=5):
    sample = dataset[0]
    image,mask = sample["image"][None].to(device),sample["valid_mask"][None].to(device)
    sync = lambda:torch.cuda.synchronize(device) if device.type == "cuda" else None
    model.eval()
    with torch.inference_mode():
        for _ in range(2):
            model(image,mask)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elapsed = []
        for _ in range(iterations):
            sync(); start=time.perf_counter()
            model(image,mask)
            sync(); elapsed.append((time.perf_counter()-start)*1000.)
    return {"scope":"device_forward_only_excludes_view_preparation_cache_io_and_transfer",
            "source":"training_sample", "case_key":sample["case_key"], "iterations":iterations,
            "latency_ms_p50":float(np.percentile(elapsed,50)),"latency_ms_p95":float(np.percentile(elapsed,95)),
            "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "flops":None,"flops_status":"not_measured","parameters":sum(p.numel() for p in model.parameters())}


def fit_baseline(model,train_dataset,validation_dataset,config,output_dir,contract,*,resume=False,
                 resume_checkpoint="last",stop_after_epoch=None,progress=None):
    if config.get("mode") != "engineering_pilot" or config.get("candidate_targets_acknowledged") is not True:
        raise ValueError("Candidate-target training requires explicit engineering_pilot configuration")
    if any(r["split"] != "train" for r in train_dataset.rows) or any(r["split"] != "validation" for r in validation_dataset.rows):
        raise ValueError("Trainer accepts training and source-validation inputs only")
    if config["batch_size"] != 1 or resume_checkpoint not in {"last","best"}:
        raise ValueError("This pilot supports batch_size=1 and explicit last/best resume")
    fit_stop = config["stop_after_train_fit_epochs"]
    if (isinstance(fit_stop,bool) or not isinstance(fit_stop,int) or not 0 <= fit_stop <= config["epochs"]
            or fit_stop and config["early_stopping"]):
        raise ValueError("Train-fit stopping requires a valid count and disabled validation early stopping")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    seed_everything(config["seed"],config["threads"])
    requested = config["device"]
    if requested not in {"auto","cpu","cuda"}:
        raise ValueError("Device must be auto/cpu/cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose a Colab GPU/runtime with CUDA-enabled PyTorch")
    device = torch.device("cuda" if (requested == "cuda" or requested == "auto" and torch.cuda.is_available()) else "cpu")
    model.to(device)
    weights,counts = training_class_weights([r["label_index"] for r in train_dataset.rows])
    weights = weights.to(device)
    optimizer = build_optimizer(model,config)
    scheduler = build_scheduler(optimizer,config)
    amp = bool(config["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda",enabled=amp)
    binding = digest_json(contract)
    history, best_loss, best_epoch, bad_epochs, start_epoch = [],math.inf,0,0,0
    epoch_diagnostics,initial_evaluation = [],None
    resume_origin = "new_run"
    if resume:
        path = output_dir/(resume_checkpoint+".pt")
        if not path.exists() and resume_checkpoint == "last" and (output_dir/"best.pt").exists():
            path = output_dir/"best.pt"
            resume_origin = "best_checkpoint_because_last_is_absent"
        if path.exists():
            state = load_checkpoint(path,binding)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            history,best_loss,best_epoch,bad_epochs = state["history"],state["best_loss"],state["best_epoch"],state["bad_epochs"]
            epoch_diagnostics,initial_evaluation = state["epoch_diagnostics"],state["initial_evaluation"]
            start_epoch = state["epoch"]
            restore_rng(state["rng"])
            resume_origin = resume_origin if resume_origin != "new_run" else resume_checkpoint+"_checkpoint"
        else:
            resume_origin = "no_completed_epoch_restart_from_initialization"
    elif (output_dir/"last.pt").exists() or (output_dir/"best.pt").exists():
        raise FileExistsError("Model checkpoint exists; explicit resume is required")
    accumulation = config["accumulation_steps"]
    total_started = time.perf_counter()
    if config["diagnostic_logging"] and initial_evaluation is None:
        initial_evaluation = {"train":evaluate_partition(model,train_dataset,device,partition="train")[0],
                              "validation":evaluate_validation(model,validation_dataset,device)[0]}
    if initial_evaluation is not None:
        _json(output_dir/"initial_evaluation.json",initial_evaluation)
    for epoch in range(start_epoch,config["epochs"]):
        if fit_stop and _fit_streak(history) >= fit_stop:
            break
        if config["early_stopping"] and bad_epochs >= config["patience"]:
            break
        started = time.perf_counter()
        loader = DataLoader(train_dataset,batch_size=1,shuffle=True,num_workers=0,
                            generator=torch.Generator().manual_seed(config["seed"]+1009*epoch))
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses,gradients,steps,skipped = [],[],0,0
        last_gradient_groups = {}
        lr = optimizer.param_groups[0]["lr"]
        for index,batch in enumerate(loader):
            image,mask,target = batch["image"].to(device),batch["valid_mask"].to(device),batch["target"].to(device)
            window = min(accumulation,len(loader)-(index//accumulation)*accumulation)
            with torch.amp.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                logits = model(image,mask)["logits"]
                loss = balanced_cross_entropy(logits,target,weights)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            losses.append(float(loss.detach()))
            scaler.scale(loss/window).backward()
            if (index+1)%accumulation == 0 or index+1 == len(loader):
                scaler.unscale_(optimizer)
                if config["diagnostic_logging"] and index+1 == len(loader):
                    last_gradient_groups = gradient_groups(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config["gradient_clip"],error_if_nonfinite=not amp)
                if torch.isfinite(norm):
                    gradients.append(float(norm))
                previous_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < previous_scale:
                    skipped += 1
                else:
                    steps += 1
                optimizer.zero_grad(set_to_none=True)
        train_metrics,train_predictions = evaluate_partition(model,train_dataset,device,partition="train")
        metrics,predictions = evaluate_validation(model,validation_dataset,device)
        improved = metrics["loss"] < best_loss-config["min_delta"]
        if improved:
            best_loss,best_epoch,bad_epochs = metrics["loss"],epoch+1,0
        else:
            bad_epochs += 1
        scheduler.step()
        row = {"epoch":epoch+1,"training_weighted_loss":float(np.mean(losses)),
               "training_eval_loss":train_metrics["loss"],"training_accuracy":train_metrics["accuracy"],
               "training_macro_f1":train_metrics["macro_f1"],"train_fit_passed":_train_fit_passed(train_metrics,config),
               "validation_loss":metrics["loss"],
               "validation_accuracy":metrics["accuracy"],"validation_macro_f1":metrics["macro_f1"],
               "validation_tamper_auroc":metrics["tamper_auroc"],"learning_rate":lr,
               "optimizer_steps":steps,"amp_skipped_steps":skipped,"max_gradient_norm":max(gradients) if gradients else None,
               "mean_gradient_norm":float(np.mean(gradients)) if gradients else None,
               "clipped_gradient_steps":sum(n>config["gradient_clip"] for n in gradients),
               "zero_gradient_steps":sum(n==0 for n in gradients),
               "nonfinite_gradient_steps":steps+skipped-len(gradients),
               "mean_gradient_clip_scale":float(np.mean([min(1.,config["gradient_clip"]/(n+1e-6)) for n in gradients])) if gradients else None,
               "elapsed_seconds":time.perf_counter()-started}
        for prefix,item in (("training",train_metrics),("validation",metrics)):
            row[prefix+"_balanced_accuracy"] = item["balanced_accuracy"]
            row[prefix+"_dominant_prediction_fraction"] = item["dominant_prediction_fraction"]
            for label in LABELS:
                row[prefix+"_predicted_"+label] = item["predicted_class_counts"][label]
                row[prefix+"_recall_"+label] = item["class_recall"][label]
        history.append(row)
        epoch_diagnostics.append({"epoch":epoch+1,"training_metrics":train_metrics,"validation_metrics":metrics,
            "gradient_groups_preclip_last_attempt":last_gradient_groups,
            "gradient_scope":"last_optimizer_attempt_of_epoch_after_unscale_before_clip" if config["diagnostic_logging"] else "not_sampled"})
        state = {"format_version":"f04_state_v1","contract_sha256":binding,"epoch":epoch+1,
                 "model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                 "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                 "rng":rng_state(),"history":history,"best_loss":best_loss,"best_epoch":best_epoch,"bad_epochs":bad_epochs,
                 "epoch_diagnostics":epoch_diagnostics,"initial_evaluation":initial_evaluation,
                 "torch_version":str(torch.__version__),"device":str(device),"training_ready":False}
        if improved:
            save_checkpoint(output_dir/"best.pt",state)
            _json(output_dir/"best_validation_predictions.json",predictions)
        save_checkpoint(output_dir/"last.pt",state)
        _json(output_dir/"last_training_predictions.json",train_predictions)
        _json(output_dir/"history.json",history)
        _json(output_dir/"epoch_diagnostics.json",epoch_diagnostics)
        write_csv(output_dir/"history.csv",tuple(row),history)
        if progress:
            progress({"stage":"epoch",**row})
        if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
            break
    if not (output_dir/"best.pt").is_file():
        raise ValueError("No completed valid epoch/checkpoint is available")
    if not sum(row["optimizer_steps"] for row in history):
        raise ValueError("No optimizer update completed; inspect AMP overflow or gradient failures")
    # Derived logs can be recovered from an epoch checkpoint after interrupted I/O.
    _json(output_dir/"history.json",history)
    _json(output_dir/"epoch_diagnostics.json",epoch_diagnostics)
    write_csv(output_dir/"history.csv",tuple(history[-1]),history)
    # Recover this small derived report even if a previous invocation stopped
    # after committing last.pt but before writing its predictions/history files.
    _,last_predictions = evaluate_partition(model,train_dataset,device,partition="train")
    _json(output_dir/"last_training_predictions.json",last_predictions)
    best = load_checkpoint(output_dir/"best.pt",binding)
    model.load_state_dict(best["model"])
    best_train_metrics,best_train_predictions = evaluate_partition(model,train_dataset,device,partition="train")
    _json(output_dir/"best_training_predictions.json",best_train_predictions)
    metrics,predictions = evaluate_validation(model,validation_dataset,device)
    _json(output_dir/"best_validation_predictions.json",predictions)
    profiling = profile_forward(model,train_dataset,device)
    passing = [row["epoch"] for row in history if row["train_fit_passed"]]
    diagnostic = {"criterion":"end_of_epoch_eval_mode_training_accuracy_and_unweighted_cross_entropy",
        "accuracy_target":config["train_fit_accuracy"],"maximum_loss":config["train_fit_max_loss"],
        "consecutive_passes_to_stop":fit_stop,"last_pass_streak":_fit_streak(history),
        "first_passing_epoch":passing[0] if passing else None,"ever_passed":bool(passing),
        "last_epoch":{"epoch":history[-1]["epoch"],"accuracy":history[-1]["training_accuracy"],
                      "loss":history[-1]["training_eval_loss"],"passed":history[-1]["train_fit_passed"]},
        "best_validation_checkpoint":{"epoch":best["epoch"],"accuracy":best_train_metrics["accuracy"],
                      "loss":best_train_metrics["loss"],"passed":_train_fit_passed(best_train_metrics,config)},
        "scope":"training_subset_fitting_only_not_generalization_or_target_validation",
        "controls":{"dropout_disabled":all(not isinstance(layer,torch.nn.Dropout) or layer.p == 0 for layer in model.modules()),
                    "weight_decay":config["weight_decay"],"early_stopping":config["early_stopping"]},
        "optimizer_updates_completed":sum(row["optimizer_steps"] for row in history)}
    _json(output_dir/"training_diagnostic.json",diagnostic)
    stopped_on_fit = bool(fit_stop and _fit_streak(history)>=fit_stop)
    stopped_on_validation = bool(config["early_stopping"] and bad_epochs>=config["patience"])
    stop_reason = ("train_fit_control_satisfied" if stopped_on_fit else "source_validation_patience" if stopped_on_validation
                   else "epoch_budget_exhausted" if history[-1]["epoch"]>=config["epochs"] else "test_epoch_boundary_stop")
    summary = {"ok":True,"mode":"engineering_pilot","candidate_targets_used":True,"training_ready":False,
        "contract_sha256":binding,"device":str(device),"torch_version":str(torch.__version__),"mixed_precision_active":amp,
        "epochs_completed":history[-1]["epoch"] if history else 0,"best_epoch":best["epoch"],"best_validation_loss":metrics["loss"],
        "stopped_early":stopped_on_validation,"stopped_after_train_fit":stopped_on_fit,"stop_reason":stop_reason,"resume_origin":resume_origin,
        "train_cases":len(train_dataset),"validation_cases":len(validation_dataset),"test_cases_opened":0,
        "training_class_counts":counts.tolist(),"training_class_weights":weights.cpu().tolist(),
        "training_metrics_at_best_validation":best_train_metrics,"train_fit_diagnostic":diagnostic,
        "initial_evaluation":initial_evaluation,"epoch_diagnostics_file":"epoch_diagnostics.json",
        "validation_metrics":metrics,"history":history,"wall_seconds_this_invocation":time.perf_counter()-total_started,
        "profile":profiling,"checkpoint_scope":"completed_epoch_boundary",
        "wall_scope":"initial_eval_training_train_eval_validation_checkpoints_logs_best_reload_profile_excludes_model_setup_and_view_preparation",
        "selection_rule":"minimum_unweighted_source_validation_cross_entropy",
        "optimization":optimization_report(optimizer,config)}
    _json(output_dir/"summary.json",summary)
    return summary


def risk_training_step(*args,**kwargs):
    """Lazy F07 entry point; the existing F04 baseline/resume path is unchanged."""
    from rsvt.training.risk_trainer import risk_training_step as implementation
    return implementation(*args,**kwargs)
