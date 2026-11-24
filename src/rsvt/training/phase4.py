"""Orchestrate F04 engineering baselines on a pinned F03 draft without promoting its QA flags."""

from __future__ import annotations
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import torch

from rsvt.data.acquisition import sha256_file
from rsvt.data.baseline_views import CachedViews, prepare_views, select_pilot_rows
from rsvt.data.inventory import write_csv
from rsvt.data.phase2 import _json
from rsvt.data.splits import CLASS_INDEX, digest_json
from rsvt.models.baselines import build_baseline
from rsvt.training.trainer import fit_baseline
from rsvt.training.diagnostics import audit_model_inputs
from rsvt.training.optimization import validate_optimization
from rsvt.utils.config import load_config
from rsvt.utils.reproducibility import seed_everything


def load_phase4_config(path):
    config = load_config(path)
    t,r,s,m = (config[key] for key in ("training","representation","selection","models"))
    validate_optimization(t)
    if t["mode"] != "engineering_pilot" or t["candidate_targets_acknowledged"] is not True:
        raise ValueError("F04 preset must explicitly acknowledge exploratory candidate targets")
    for key,minimum in (("seed",0),("epochs",1),("batch_size",1),("accumulation_steps",1),("patience",1),("threads",1)):
        if isinstance(t[key],bool) or not isinstance(t[key],int) or t[key] < minimum:
            raise ValueError("Invalid training integer: "+key)
    for key,allow_zero in (("learning_rate",False),("weight_decay",True),("min_delta",True),("gradient_clip",False)):
        if (isinstance(t[key],bool) or not isinstance(t[key],(float,int)) or not math.isfinite(t[key])
                or t[key] < 0 or not allow_zero and t[key] == 0):
            raise ValueError("Invalid training scalar: "+key)
    if t["batch_size"] != 1 or t["device"] not in {"auto","cpu","cuda"} or not isinstance(t["mixed_precision"],bool):
        raise ValueError("Invalid batch/device/mixed precision policy")
    if (not isinstance(t["early_stopping"],bool) or isinstance(t["train_fit_accuracy"],bool)
            or not isinstance(t["train_fit_accuracy"],(int,float)) or not math.isfinite(t["train_fit_accuracy"])
            or not 0 < t["train_fit_accuracy"] <= 1 or isinstance(t["train_fit_max_loss"],bool)
            or not isinstance(t["train_fit_max_loss"],(int,float)) or not math.isfinite(t["train_fit_max_loss"])
            or t["train_fit_max_loss"] <= 0):
        raise ValueError("Invalid training-fit diagnostic criterion or stopping policy")
    fit_stop = t["stop_after_train_fit_epochs"]
    if (isinstance(fit_stop,bool) or not isinstance(fit_stop,int) or not 0 <= fit_stop <= t["epochs"]
            or fit_stop and t["early_stopping"] or not isinstance(t["diagnostic_logging"],bool)):
        raise ValueError("Invalid fitting-control stop count/logging or conflicting early stopping")
    if (r["policy"] != "full_fov_linear_image_nearest_validity" or len(r["shape_zyx"]) != 3
            or any(isinstance(n,bool) or not isinstance(n,int) or n < 32 or n > 128 for n in r["shape_zyx"])
            or not isinstance(r["max_decode_bytes"],int) or isinstance(r["max_decode_bytes"],bool) or r["max_decode_bytes"] <= 0):
        raise ValueError("Unsupported pilot representation shape or decode limit")
    if not re.fullmatch(r"[0-9a-f]{64}",s["expected_assignment_sha256"]):
        raise ValueError("Pin the reviewed F03 assignment fingerprint")
    for key in ("train_per_class","validation_per_class"):
        if isinstance(s[key],bool) or not isinstance(s[key],int) or s[key] < 0:
            raise ValueError("Invalid subset limit")
    if (not m["names"] or len(set(m["names"])) != len(m["names"])
            or not set(m["names"]).issubset({"resnet50_3d","vit3d"})):
        raise ValueError("Unknown/duplicated baseline names")
    for key,minimum in (("cnn_width",2),("patch_size",1),("dimension",4),("layers",1),("heads",1)):
        if isinstance(m[key],bool) or not isinstance(m[key],int) or m[key] < minimum:
            raise ValueError("Invalid model integer: "+key)
    if (any(n % m["patch_size"] for n in r["shape_zyx"]) or m["dimension"] % m["heads"]
            or math.prod(n//m["patch_size"] for n in r["shape_zyx"]) > 4096
            or isinstance(m["dropout"],bool) or not math.isfinite(m["dropout"]) or not 0 <= m["dropout"] < 1):
        raise ValueError("Invalid fixed ViT grid or attention configuration")
    return config


def implementation_fingerprint():
    modules = ("rsvt.training.phase4","rsvt.training.trainer","rsvt.training.optimization","rsvt.training.checkpoint","rsvt.training.diagnostics","rsvt.data.baseline_views",
               "rsvt.data.dataset","rsvt.models.baselines","rsvt.losses.classification","rsvt.evaluation.classification",
               "rsvt.utils.reproducibility")
    return digest_json({name:sha256_file(Path(importlib.import_module(name).__file__)) for name in modules})


def run_phase4(f03_run,config_path,output_dir,*,resume=False,resume_checkpoint="last",progress=None):
    config = load_phase4_config(config_path)
    f03_run,output_dir = Path(f03_run).resolve(),Path(output_dir).resolve()
    if output_dir.is_relative_to(f03_run) or f03_run.is_relative_to(output_dir):
        raise ValueError("Training outputs must be separate from F03 evidence")
    if output_dir.exists() and not resume:
        raise FileExistsError("F04 output exists; use a new run ID or explicit resume")
    if resume and not output_dir.is_dir():
        raise FileNotFoundError("Resume requires an existing F04 run directory")
    summary3 = json.loads((f03_run/"summary.json").read_text(encoding="utf-8"))
    manifest_path = f03_run/"manifest.draft.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignment = document["assignments_sha256"]
    if (summary3.get("ok") is not True or summary3["manifest_sha256"] != sha256_file(manifest_path)
            or assignment != summary3["assignments_sha256"] or assignment != config["selection"]["expected_assignment_sha256"]
            or document["class_index"] != CLASS_INDEX):
        raise ValueError("F03 evidence differs from the complete, pinned draft")
    selected = select_pilot_rows(document["records"],config["training"]["seed"],
                                config["selection"]["train_per_class"],config["selection"]["validation_per_class"])
    for row in document["records"]:
        original = Path(row["volume_path"]).parent.resolve()
        if output_dir.is_relative_to(original) or original.is_relative_to(output_dir):
            raise ValueError("F04 outputs must be separate from original processed volumes")
    contract = {"implementation_sha256":implementation_fingerprint(),"config_sha256":digest_json(config),
                "f03_manifest_sha256":sha256_file(manifest_path),"assignments_sha256":assignment,
                "selected_cases_sha256":digest_json([{key:r[key] for key in ("case_key","split","label_candidate","normalized_sha256","valid_mask_sha256")} for r in selected]),
                "class_index":CLASS_INDEX,"mode":"engineering_pilot"}
    if resume:
        previous = json.loads((output_dir/"run_contract.json").read_text(encoding="utf-8"))
        if previous != contract:
            raise ValueError("Resume requires identical F03 input, subset, resolved configuration, and implementation")
    else:
        output_dir.parent.mkdir(parents=True,exist_ok=True)
        required = len(selected)*(math.prod(config["representation"]["shape_zyx"])*5+65536)+1024**3
        if shutil.disk_usage(output_dir.parent).free < required:
            raise ValueError("Insufficient filesystem storage for coarse views and checkpoint reserve")
        output_dir.mkdir()
        _json(output_dir/"run_contract.json",contract)
        _json(output_dir/"resolved_config.json",config)
        _json(output_dir/"selected_cases.json",[{key:r[key] for key in ("case_key","split","label_candidate","source_group_candidate")} for r in selected])
    summary = {"created_utc":datetime.now(timezone.utc).isoformat(),"f03_run":str(f03_run),"mode":"engineering_pilot",
        "training_ready":False,"phase_f04_complete":False,"candidate_targets_used":True,"test_cases_opened":0,
        "ok":False,"run_contract":contract,"shape_zyx":config["representation"]["shape_zyx"],"models":{}}
    _json(output_dir/"summary.json",summary)
    started = time.perf_counter()
    try:
        views = prepare_views(manifest_path,selected,output_dir/"views",config["representation"]["shape_zyx"],
                              max_decode_bytes=config["representation"]["max_decode_bytes"],progress=progress)
        summary["view_preparation_seconds"] = time.perf_counter()-started
        if config["training"]["diagnostic_logging"]:
            audit = audit_model_inputs(views)
            _json(output_dir/"input_audit.json",audit)
            write_csv(output_dir/"input_audit.csv",tuple(audit["records"][0]),audit["records"])
            summary["input_audit"] = {key:value for key,value in audit.items() if key!="records"}
            if not audit["ok"]:
                raise ValueError("Model-input audit found conflicting targets, train/validation duplicates, or out-of-range voxels; inspect input_audit.json")
        train,val = CachedViews(views,"train"),CachedViews(views,"validation")
        summary.update(train_cases=len(train),validation_cases=len(val),selected_source_groups=len({r["source_group_candidate"] for r in selected}))
        for name in config["models"]["names"]:
            seed_everything(config["training"]["seed"],config["training"]["threads"])
            model = build_baseline(name,config["models"],config["representation"]["shape_zyx"])
            model_contract = {**contract,"architecture":name,"view_index_sha256":sha256_file(output_dir/"views/index.json")}
            result = fit_baseline(model,train,val,config["training"],output_dir/name,model_contract,
                                  resume=resume,resume_checkpoint=resume_checkpoint,
                                  progress=(lambda event,n=name:progress({"model":n,**event})) if progress else None)
            summary["models"][name] = result
            _json(output_dir/"summary.json",summary)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        table = []
        for name,item in summary["models"].items():
            metrics,profile = item["validation_metrics"],item["profile"]
            table.append({"model":name,"best_epoch":item["best_epoch"],"validation_loss":metrics["loss"],
                "training_accuracy_at_best_validation":item["training_metrics_at_best_validation"]["accuracy"],
                "last_training_accuracy":item["train_fit_diagnostic"]["last_epoch"]["accuracy"],
                "last_train_fit_passed":item["train_fit_diagnostic"]["last_epoch"]["passed"],
                "optimizer_updates_completed":item["train_fit_diagnostic"]["optimizer_updates_completed"],
                "validation_accuracy":metrics["accuracy"],"validation_macro_f1":metrics["macro_f1"],
                "validation_tamper_auroc":metrics["tamper_auroc"],"parameters":profile["parameters"],
                "device_forward_ms_p50":profile["latency_ms_p50"],"device_forward_ms_p95":profile["latency_ms_p95"]})
        write_csv(output_dir/"comparison.validation.csv",tuple(table[0]),table)
        summary.update(ok=all(item["ok"] for item in summary["models"].values()),
                       train_fitting_all_models_passed_at_last_epoch=all(item["train_fit_diagnostic"]["last_epoch"]["passed"] for item in summary["models"].values()),
                       status="exploratory_candidate_target_training_complete",source_manifest_unchanged=sha256_file(manifest_path)==contract["f03_manifest_sha256"],
                       pending=["canonical source identities and validated volume targets","preprocessing/coarsening signal-preservation QA",
                                "confirmatory endpoint protocol and larger authentic validation/test support","B2 cascade and measured end-to-end FLOPs"])
        summary["ok"] = summary["ok"] and summary["source_manifest_unchanged"]
        (output_dir/"report.md").write_text("# F04 engineering baselines\n\n"+
            f"Train cases: {len(train)}; validation cases: {len(val)}; test images opened: 0.\n\n"+
            "Candidate targets remain unverified. Results are exploratory source-validation measurements, not final test performance.\n"+
            "Train fitting is evaluated separately at each epoch; see each model's training_diagnostic.json. Passing that control is not generalization.\n"+
            "See comparison.validation.csv, model histories, checkpoint receipts, and profiling scope.\n",encoding="utf-8")
    except (ValueError,OSError,RuntimeError,KeyError,MemoryError,ImportError) as exc:
        summary.update(ok=False,reason=str(exc),status="failed_or_incomplete")
    finally:
        summary["wall_seconds_this_invocation"] = time.perf_counter()-started
        _json(output_dir/"summary.json",summary)
    return summary
