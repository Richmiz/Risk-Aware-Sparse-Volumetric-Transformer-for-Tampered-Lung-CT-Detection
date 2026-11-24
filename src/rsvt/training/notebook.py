"""Human-readable Colab interface; medical pixels remain on Colab/Drive."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from rsvt.data.acquisition import sha256_file
from rsvt.data.dataset import CTVolumeDataset
from rsvt.data.phase2 import _json
from rsvt.data.splits import CLASS_INDEX, digest_json
from rsvt.models.proposed import RSVT
from rsvt.training.proposed_run import load_proposed_config, run_proposed, implementation_fingerprint

F03_RELATIVE = 'outputs/partitions/f03_20261110T013617_031412Z'
LABELS = tuple(CLASS_INDEX)


def table_html(headers, rows):
    head = ''.join('<th style="padding:8px;text-align:left">'+escape(str(h))+'</th>' for h in headers)
    body = ''.join('<tr>'+''.join('<td style="padding:7px 10px;border-bottom:1px solid #ddd">'+
        escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    return '<table style="border-collapse:collapse"><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table>'


def _toml_value(value):
    if isinstance(value, bool): return str(value).lower()
    if isinstance(value, str): return json.dumps(value)
    if isinstance(value, list): return '['+', '.join(_toml_value(v) for v in value)+']'
    return str(value)


def write_profile(path, config):
    text = '\n\n'.join('['+table+']\n'+'\n'.join(k+' = '+_toml_value(v) for k,v in values.items())
                         for table,values in config.items())+'\n'
    path.write_text(text, encoding='utf-8')
    if path.read_text(encoding='utf-8') != text: raise OSError('Configuration readback failed')


class NotebookSession:
    def __init__(self, project, code_root, *, epochs=30, learning_rate=1e-4, patience=10,
                 seed=13, preview_images=4, resume_session='', configuration=None,
                 f03_path=None, enforce_drive=True, display_enabled=True):
        self.project, self.code_root = Path(project).resolve(), Path(code_root).resolve()
        if enforce_drive and (platform.system() != 'Linux' or
                not self.project.is_relative_to(Path('/content/drive/MyDrive').resolve())):
            raise RuntimeError('Use the Colab kernel and mount project Drive; medical images stay there.')
        if not self.project.is_dir(): raise FileNotFoundError('Project Drive is not mounted')
        if isinstance(preview_images,bool) or preview_images not in (2,3,4): raise ValueError('Choose 2, 3, or 4 preview images')
        self.display_enabled, self.preview_images = display_enabled, preview_images
        self.fixture_mode = not enforce_drive
        self.config = deepcopy(configuration) if configuration is not None else load_proposed_config(
            self.code_root/'configs/training/rsvt_real_ct.toml')
        self.config['training'].update(epochs=epochs,learning_rate=learning_rate,patience=patience,seed=seed)
        self.f03 = Path(f03_path) if f03_path is not None else self.project/F03_RELATIVE
        self.manifest = self.f03/'manifest.draft.json'
        document = json.loads(self.manifest.read_text(encoding='utf-8'))
        evidence = json.loads((self.f03/'summary.json').read_text(encoding='utf-8'))
        self.manifest_sha = sha256_file(self.manifest)
        if (evidence.get('ok') is not True or evidence['manifest_sha256'] != self.manifest_sha
                or document['assignments_sha256'] != evidence['assignments_sha256']
                or document['assignments_sha256'] != self.config['data']['expected_assignment_sha256']
                or document['class_index'] != CLASS_INDEX):
            raise ValueError('Dataset manifest does not match its pinned receipt')
        self.records = document['records']
        self.counts = {split:sum(r['split']==split for r in self.records) for split in ('train','validation','test')}
        parent = self.project/'outputs/rsvt_training'
        self.root = (Path(resume_session).resolve() if resume_session else parent/(
            'rsvt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')))
        if not self.root.is_relative_to(parent) or self.root == parent: raise ValueError('Keep sessions inside project Drive outputs/rsvt_training')
        binding = {'config_sha256':digest_json(self.config),'manifest_sha256':self.manifest_sha,
                   'implementation_sha256':implementation_fingerprint()}
        if resume_session:
            stored=json.loads((self.root/'notebook_session.json').read_text(encoding='utf-8'))
            if stored['binding'] != binding: raise ValueError('Resume requires the same configuration, code and data; use the original release for older runs')
        else:
            self.root.mkdir(parents=True,exist_ok=False)
        self.training_dir=self.root/'training'; self.assets=self.root/'figures'; self.assets.mkdir(exist_ok=True)
        self.profile=self.root/'training_config.toml'; write_profile(self.profile,self.config)
        load_proposed_config(self.profile)
        self.artifacts={}; self._status_handle=None; self._table_handle=None; self._curve_handle=None
        self._history=[]; self._view_count=0
        self._save_json('notebook_session.json',{'binding':binding,'dataset_counts':self.counts,
            'training_directory':str(self.training_dir),
            'created_utc':stored['created_utc'] if resume_session else datetime.now(timezone.utc).isoformat(),
            'test_status':'reserved_for_final_evaluation','preview_partition':'train'})
        if not (self.root/'status.json').exists(): self._status('prepared','Dataset connected; training has not started.')

    def _display_html(self, text):
        if self.display_enabled:
            from IPython.display import display, HTML
            display(HTML(text))

    def _save_json(self, name, value):
        path=self.root/name; _json(path,value)
        if json.loads(path.read_text(encoding='utf-8')) != value: raise OSError('Drive JSON readback failed: '+name)
        if hasattr(self,'artifacts'): self.artifacts[name]={'sha256':sha256_file(path),'bytes':path.stat().st_size}

    def _status(self, status, message):
        self._save_json('status.json',{'status':status,'message':message,
            'updated_utc':datetime.now(timezone.utc).isoformat(),'training_directory':str(self.training_dir)})

    def _save_figure(self, figure, name):
        buffer=BytesIO(); figure.savefig(buffer,format='png',dpi=150,bbox_inches='tight')
        data=buffer.getvalue(); path=self.assets/name; path.write_bytes(data)
        if path.read_bytes() != data: raise OSError('Drive figure readback failed: '+name)
        self.artifacts['figures/'+name]={'sha256':sha256_file(path),'bytes':len(data)}
        return path

    @staticmethod
    def _figure_output(figure):
        # Explicit PNG output works in VS Code without relying on the active
        # Matplotlib frontend/formatter to recognize a Figure object.
        from IPython.display import Image
        buffer=BytesIO(); figure.savefig(buffer,format='png',dpi=120,bbox_inches='tight')
        return Image(data=buffer.getvalue())

    def show_runtime(self):
        available=torch.cuda.is_available()
        info={'Python':platform.python_version(),'PyTorch':str(torch.__version__),
              'CUDA build':str(torch.version.cuda),'GPU detected':available,
              'GPU name':torch.cuda.get_device_name(0) if available else 'No GPU detected',
              'Python executable':sys.executable,'Output in Drive':str(self.root)}
        self._display_html(table_html(['Runtime check','Value'],info.items()))
        if not available: self._display_html('<p><b>Dataset previews can run now. Training requires selecting a Colab GPU kernel.</b></p>')
        self._save_json('runtime.json',info)
        return info

    def show_dataset(self):
        rows=[]
        for split in ('train','validation','test'):
            counts=Counter(r['label_candidate'] for r in self.records if r['split']==split)
            rows.append([split,self.counts[split],*[counts.get(c,0) for c in LABELS],
                         'Reserved; images not opened' if split=='test' else 'Available'])
        self._display_html('<h3>Connected dataset: CT-GAN</h3>'+table_html(
            ['Partition','Volumes',*LABELS,'Use'],rows))
        self._display_html('<p>TB/TM: authentic-labelled scans; FB: nodule removal; FM: nodule insertion. '
            'Counts come from the existing manifest. Preview examples below are training cases.</p>')
        self._save_json('dataset_connection.json',{'manifest':str(self.manifest),'sha256':self.manifest_sha,
            'assignments_sha256':self.config['data']['expected_assignment_sha256'],
            'counts':self.counts,'test_images_opened':0})
        return self.counts

    def show_previews(self):
        import matplotlib.pyplot as plt
        dataset=CTVolumeDataset(self.manifest,'train',purpose='engineering_pilot',allow_candidate_targets=True,
            max_decode_bytes=self.config['data']['max_decode_bytes'])
        selected=[]
        for label in LABELS[:self.preview_images]:
            selected.append(next(i for i,r in enumerate(dataset.rows) if r['label_candidate']==label))
        figure,axes=plt.subplots(1,len(selected),figsize=(4.1*len(selected),4.4),squeeze=False)
        previews=[]
        for axis,index in zip(axes[0],selected):
            sample=dataset[index]; valid=sample['valid_mask'][0]
            support=np.flatnonzero(valid.any(axis=(1,2))); z=int(support[len(support)//2])
            image=sample['image'][0,z]; label=dataset.rows[index]['label_candidate']
            axis.imshow(image,cmap='gray',vmin=0,vmax=1,interpolation='nearest'); axis.axis('off')
            axis.set_title(label+' | '+sample['case_key']+'\ntrain | slice z='+str(z),fontsize=10)
            previews.append({'case_key':sample['case_key'],'label':label,'split':'train','slice_z':z,
                'source_normalized_sha256':dataset.rows[index]['normalized_sha256']})
            del sample,valid,image
        title=('Synthetic fixture — preview rendering test only' if self.fixture_mode else
               'Connected CT-GAN training scans — representative slices, not annotation-selected')
        figure.suptitle(title,fontsize=13)
        figure.tight_layout(); path=self._save_figure(figure,'dataset_preview.png')
        if self.display_enabled:
            from IPython.display import display
            display(self._figure_output(figure))
        plt.close(figure)
        self._save_json('dataset_previews.json',{'examples':previews,'figure':str(path),'test_images_opened':0})
        return previews

    def show_configuration(self):
        t,m,r=self.config['training'],self.config['model'],self.config['risk']; stages=self.config['stages']
        rows=[('Training cases',self.counts['train']),('Validation cases',self.counts['validation']),
              ('Test cases (reserved)',self.counts['test']),('Maximum epochs',t['epochs']),
              ('Optimizer','AdamW'),('Peak learning rate',t['learning_rate']),
              ('Schedule',f"{t['warmup_epochs']} warmup epochs, then cosine decay"),
              ('Initial LR',t['learning_rate']*t.get('warmup_start_ratio',.1)),
              ('Minimum LR',t['learning_rate']*t.get('min_lr_ratio',0.)),('Weight decay',t['weight_decay']),
              ('Early stopping',f"{t['patience']} risk-finetuning epochs without validation CE improvement > {t['min_delta']}"),
              ('Microbatch / logical target',f"1 CT / {t['logical_batch_size']} CTs"),
              ('Precision','FP32'),('Gradient clipping',t['gradient_clip']),('Seed',t['seed']),
              ('Input full-FOV grid',str(self.config['data']['shape_zyx'])),
              ('Transformer dimension / layers / heads',f"{m['dimension']} / {m['layers']} / {m['heads']}"),
              ('Dropout',m['dropout']),('3D patch sizes',str(m['patch_sizes'])),
              ('Inference token cap including CLS',m['sequence_cap']),
              ('CVaR alpha / coefficient',f"{r['alpha']} / {r['cvar_weight']}"),
              ('FB / FM risk weights',f"{r['fb_weight']} / {r['fm_weight']}"),
              ('Training stages',f"{stages['baseline_epochs']} backbone + {stages['routing_warmup_epochs']} router warmup + remaining risk finetuning"),
              ('Checkpoint selection','Lowest validation CE among risk-finetuning epochs'),
              ('Spatial localization / generator variance','Disabled for current data')]
        self._display_html(table_html(['Setting','Value'],rows)); self._save_json('displayed_configuration.json',{'settings':[list(x) for x in rows]})
        return rows

    def show_model(self):
        model=RSVT(**self.config['model']); count=sum(p.numel() for p in model.parameters())
        self._display_html('<pre>CT volume → global coarse encoder + multiscale local tokens\n'
            '→ physical-position embeddings → adaptive router → Transformer → TB / TM / FB / FM</pre>'+
            table_html(['Model','Value'],[('Name','Risk-Aware Sparse Volumetric Transformer'),
                                        ('Parameters',f'{count:,}'),('Train routing','Soft gates, dense sequence'),
                                        ('Prediction routing','Hard top-k, packed tokens')]))
        self._save_json('model_information.json',{'name':'RSVT','parameters':count,'model_config':self.config['model']})
        del model
        return count

    def history(self):
        path=self.training_dir/'history.json'
        rows=json.loads(path.read_text(encoding='utf-8')) if path.exists() else deepcopy(self._history)
        for row in rows:
            if 'training_loss' not in row:
                report=self.training_dir/f"epoch_{row['epoch']:03d}.json"
                if report.exists(): row['training_loss']=json.loads(report.read_text())['training_metrics_hard']['loss']
        return rows

    @staticmethod
    def _number(value, percent=False):
        return '—' if value is None else (f'{100*value:.1f}%' if percent else f'{value:.4f}')

    def _history_html(self, history):
        return table_html(['Epoch','Stage','LR','Train loss','Train accuracy','Val loss','Val accuracy','Val macro-F1','Tamper AUROC'],
            [[r['epoch'],r['stage'],f"{r['learning_rate']:.2g}",self._number(r.get('training_loss')),
              self._number(r.get('training_accuracy_hard'),True),self._number(r.get('validation_loss')),
              self._number(r.get('validation_accuracy'),True),self._number(r.get('validation_macro_f1')),
              self._number(r.get('validation_tamper_auroc'))] for r in history])

    def _curves(self, history):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        fig,axes=plt.subplots(1,2,figsize=(11,3.7)); epochs=[r['epoch'] for r in history]
        for key,label in [('training_loss','Train'),('validation_loss','Validation')]:
            axes[0].plot(epochs,[r.get(key,np.nan) for r in history],marker='o',markersize=3,label=label)
        for key,label in [('training_accuracy_hard','Train'),('validation_accuracy','Validation')]:
            axes[1].plot(epochs,[100*r.get(key,np.nan) for r in history],marker='o',markersize=3,label=label)
        for ax,title,y in zip(axes,['Classification loss','Classification accuracy'],['Unweighted cross-entropy','Accuracy (%)']):
            ax.set(title=title,xlabel='Epoch',ylabel=y); ax.grid(alpha=.2); ax.legend()
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].set_ylim(0,100); fig.tight_layout()
        return fig

    def _progress(self, event):
        if event.get('stage')=='coarse_view':
            self._view_count+=1; message=f"Preparing model inputs: {self._view_count}/{self.counts['train']+self.counts['validation']} volumes"
        elif event.get('event')=='logical_batch':
            message=f"Epoch {event['epoch']}/{self.config['training']['epochs']} — {event['stage']} — batch {event['batch']}/{event['batches']}"
        elif 'validation_loss' in event:
            self._history=[r for r in self._history if r['epoch']!=event['epoch']]+[event]
            message=f"Epoch {event['epoch']} finished: train accuracy {100*event['training_accuracy_hard']:.1f}%, validation accuracy {100*event['validation_accuracy']:.1f}%"
            if self.display_enabled:
                from IPython.display import HTML
                self._table_handle.update(HTML(self._history_html(self._history)))
                import matplotlib.pyplot as plt
                fig=self._curves(self._history); self._curve_handle.update(self._figure_output(fig)); plt.close(fig)
        else: message=str(event)
        if self.display_enabled:
            from IPython.display import HTML
            self._status_handle.update(HTML('<p><b>'+escape(message)+'</b></p>'))

    def train(self):
        if self.config['training']['device']=='cuda' and not torch.cuda.is_available():
            self._status('blocked_no_gpu','Dataset is connected, but this kernel cannot see a CUDA GPU.')
            raise RuntimeError('GPU not detected. Select the GPU kernel, rerun setup with this session path, then train. Dataset previews are already saved.')
        self._history=self.history(); self._view_count=0
        if self.display_enabled:
            from IPython.display import display, HTML
            self._status_handle=display(HTML('<p>Preparing training…</p>'),display_id=True)
            self._table_handle=display(HTML(self._history_html(self._history)),display_id=True)
            self._curve_handle=display(HTML('<p>Learning curves appear after the first epoch.</p>'),display_id=True)
        existing=self.training_dir/'summary.json'
        if existing.exists():
            summary=json.loads(existing.read_text(encoding='utf-8'))
            if summary.get('ok') and summary.get('training_finished'):
                self._status('completed','Using this session’s completed training run; no retraining.')
                self._display_html('<p><b>This session already finished. Loading its saved results.</b></p>')
                self.verify_saved(); return summary
        self._status('running','Training started; epoch logs and checkpoints save to Drive.')
        try:
            summary=run_proposed(self.f03,self.profile,self.training_dir,
                resume=(self.training_dir/'run_contract.json').exists(),progress=self._progress)
            self._status('completed' if summary.get('training_finished') else 'partial',
                         'Training finished.' if summary.get('training_finished') else 'A partial training run was saved.')
            self.show_history(); self.verify_saved()
            return summary
        except Exception as exc:
            try: self._status('failed',str(exc))
            except OSError: pass
            raise

    def show_history(self):
        history=self.history()
        if not history:
            self._display_html('<p>No epoch has completed yet. Metrics will appear after training.</p>'); return []
        self._display_html(self._history_html(history))
        fig=self._curves(history); self._save_figure(fig,'learning_curves.png')
        if self.display_enabled:
            from IPython.display import display
            display(self._figure_output(fig))
        import matplotlib.pyplot as plt
        plt.close(fig); return history

    def show_results(self):
        path=self.training_dir/'summary.json'
        if not path.exists():
            self._display_html('<p>Training results are not available yet. No scores are invented.</p>'); return None
        summary=json.loads(path.read_text(encoding='utf-8')); epoch=summary.get('best_epoch')
        report=self.training_dir/f'epoch_{epoch:03d}.json' if epoch is not None else None
        train=json.loads(report.read_text())['training_metrics_hard'] if report and report.exists() else {}
        val=summary['validation_metrics']; rows=[]
        for name,metrics,count in [('Train',train,self.counts['train']),('Validation',val,self.counts['validation'])]:
            rows.append([name,count,epoch if epoch is not None else 'provisional',self._number(metrics.get('loss')),
                self._number(metrics.get('accuracy'),True),self._number(metrics.get('macro_f1')),self._number(metrics.get('tamper_auroc'))])
        rows.append(['Test',self.counts['test'],'Reserved','Not evaluated','Not evaluated','Not evaluated','Not evaluated'])
        self._display_html(table_html(['Partition','Cases','Checkpoint epoch','Loss','Accuracy','Macro-F1','Tamper AUROC'],rows))
        self._display_html('<p>Train and validation scores above refer to the same selected checkpoint. '
            'Test is reserved for a separate final evaluation after model selection.</p>')
        import matplotlib.pyplot as plt
        matrix=np.asarray(val['confusion_matrix']); fig,ax=plt.subplots(figsize=(5,4))
        ax.imshow(matrix,cmap='Blues'); ax.set(xticks=range(4),yticks=range(4),xticklabels=LABELS,yticklabels=LABELS,
            xlabel='Predicted class',ylabel='Dataset label',title='Validation confusion matrix')
        for i in range(4):
            for j in range(4): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
        fig.tight_layout(); self._save_figure(fig,'validation_confusion_matrix.png')
        if self.display_enabled:
            from IPython.display import display
            display(self._figure_output(fig))
        plt.close(fig)
        self._display_html(table_html(['Class','Validation cases','Recall','F1','Predicted cases'],
            [[c,val['class_support'][c],self._number(val['class_recall'][c],True),self._number(val['class_f1'][c]),
              val['predicted_class_counts'][c]] for c in LABELS]))
        self._save_json('results_overview.json',{'checkpoint_epoch':epoch,'train':train,'validation':val,
            'test':{'cases':self.counts['test'],'status':'not_evaluated_reserved','images_opened':0}})
        if val.get('constant_argmax'): self._display_html('<p><b>Warning: validation predictions collapsed to one class. Inspect the confusion matrix before claiming useful detection.</b></p>')
        return summary

    def show_test_status(self):
        self._display_html(table_html(['Test set','Status'],[('Held-out cases',self.counts['test']),
            ('Images opened by this notebook',0),('Accuracy / loss','Not evaluated'),
            ('Reason','Keep test independent of training, checkpoint selection and hyperparameter tuning')]))

    def verify_saved(self):
        summary_path=self.training_dir/'summary.json'
        summary=json.loads(summary_path.read_text()) if summary_path.exists() else None
        text='# RSVT training session\n\n'+f'Output: `{self.root}`\n\n'+\
            f"Dataset: {self.counts['train']} train / {self.counts['validation']} validation / {self.counts['test']} reserved test.\n\n"
        if summary is not None:
            metrics=summary['validation_metrics']
            text+=f"Completed epochs: {summary['epochs_completed']}; selected risk-finetuning epoch: {summary.get('best_epoch')}.\n\n"
            text+=f"Validation accuracy: {metrics['accuracy']:.4f}; macro-F1: {metrics['macro_f1']}; tamper AUROC: {metrics['tamper_auroc']}.\n\n"
        else: text+='Training has not produced a completed epoch yet.\n\n'
        text+='Test metrics: not evaluated; test images opened: 0.\n\n'
        for name in ('dataset_preview.png','learning_curves.png','validation_confusion_matrix.png'):
            if (self.assets/name).exists(): text+=f'![{name}](figures/{name})\n\n'
        report=self.root/'run_overview.md'; report.write_text(text,encoding='utf-8')
        if report.read_text(encoding='utf-8')!=text: raise OSError('Run report readback failed')
        if (self.training_dir/'summary.json').exists():
            # Existing writer receipts bind checkpoints to their run contract.
            for path in self.training_dir.glob('*.pt'):
                receipt=json.loads(path.with_name(path.name+'.json').read_text())
                if sha256_file(path)!=receipt['sha256'] or path.stat().st_size!=receipt['bytes']:
                    raise OSError('Checkpoint readback failed: '+path.name)
        records={}
        for path in self.root.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.toml','.csv','.md','.png') and path.name!='notebook_receipt.json':
                records[path.relative_to(self.root).as_posix()]={'sha256':sha256_file(path),'bytes':path.stat().st_size}
        receipt={'schema_version':'human_readable_notebook_v1','files':records,'test_images_opened':0}
        self._save_json('notebook_receipt.json',receipt)
        for name,info in records.items():
            if sha256_file(self.root/name)!=info['sha256']: raise OSError('Drive readback mismatch: '+name)
        self._display_html('<p><b>Saved and verified in Google Drive:</b><br>'+escape(str(self.root))+'</p>')
        return receipt


def create_session(project, code_root, **settings):
    return NotebookSession(project,code_root,**settings)
