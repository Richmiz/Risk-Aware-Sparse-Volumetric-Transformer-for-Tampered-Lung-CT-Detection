"""Source-only F05 synthetic integration check; never reads medical volumes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch
from torch.nn import functional as F

from rsvt.losses.localization import verified_voxel_mask_loss
from rsvt.models.rsvt import CLASS_INDEX,RSVTDenseFoundation
from rsvt.utils.config import load_config
from rsvt.utils.reproducibility import seed_everything


def load_phase5_config(path):
    config=load_config(path)
    if set(config)!={"model","engineering","protocol"}:
        raise ValueError("F05 requires only model, engineering, and protocol sections")
    if any(not isinstance(config[key],dict) for key in config):
        raise ValueError("F05 sections must be tables")
    model=config["model"]
    if model.get("name")!="rsvt_f05_dense":raise ValueError("Use the explicit F05 dense model config")
    protocol=config["protocol"]
    if (set(protocol)!={"status","medical_training_ready","adaptive_routing","risk_and_generator_losses"}
            or protocol["status"]!="synthetic_engineering_only"
            or any(protocol[key] is not False for key in ("medical_training_ready","adaptive_routing","risk_and_generator_losses"))):
        raise ValueError("F05 synthetic/dense protocol flags must remain explicit")
    engineering=config["engineering"]
    if set(engineering)!={"seed","threads","device","synthetic_shapes_zyx","learning_rate","localization_loss_weight"}:
        raise ValueError("Unknown/missing F05 engineering fields")
    for key in ("seed","threads"):
        value=engineering[key]
        if (isinstance(value,bool) or not isinstance(value,int) or value<0 or value>2**32-1
                or (key=="threads" and not 1<=value<=16)):
            raise ValueError("Use a nonnegative integer seed and 1..16 threads")
    if engineering["device"] not in ("auto","cpu","cuda"):raise ValueError("Invalid device")
    shapes=engineering["synthetic_shapes_zyx"]
    if (not isinstance(shapes,list) or len(shapes)!=4 or any(not isinstance(s,list) or len(s)!=3
            or any(isinstance(n,bool) or not isinstance(n,int) or not 1<=n<=96 for n in s) for s in shapes)):
        raise ValueError("Declare four bounded synthetic Z/Y/X shapes, axes in 1..96")
    for key in ("learning_rate","localization_loss_weight"):
        value=engineering[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError("Synthetic loss/optimizer scalars must be positive finite values")
    # Constructor validates all model dimensions/bounds; reject typos rather than ignore them.
    allowed={"name","dimension","heads","layers","coarse_width","candidate_width","coarse_patch_size",
             "patch_sizes","dropout","localization_head","max_tokens","max_batch_voxels",
             "max_attention_elements","max_feature_grid_voxels"}
    if set(model)!=allowed or model["localization_head"] is not True:
        raise ValueError("Use the complete F05 model config with localization enabled for the smoke check")
    return config


def synthetic_batch(shapes,device="cpu"):
    """Procedural fixtures only. Class IDs exercise the contract, not pathology."""
    maximum=tuple(max(shape[axis] for shape in shapes) for axis in range(3))
    image=torch.zeros((4,1,*maximum),device=device)
    valid=torch.zeros_like(image,dtype=torch.bool);padding=torch.zeros_like(valid)
    targets=torch.full_like(image,float("nan"))
    affine=torch.eye(4,dtype=torch.float64,device=device).repeat(4,1,1)
    affine[0,:3,:3]=torch.diag(torch.tensor([.7,.9,1.5],dtype=torch.float64,device=device))
    affine[1,:3,:3]=torch.tensor([[0.,-.8,0.],[.6,0.,0.],[0.,0.,2.]],device=device)
    affine[2,:3,:3]=torch.diag(torch.tensor([-.75,1.,1.25],device=device))
    affine[:,:3,3]=torch.tensor([[100.,-40.,20.],[-12.,22.,60.],[42.,0.,-10.],[0.,0.,0.]],device=device)
    for index,(z,y,x) in enumerate(shapes):
        axes=[(torch.arange(n,device=device)-(n-1)/2)/max(n/2,1) for n in (z,y,x)]
        zz,yy,xx=torch.meshgrid(*axes,indexing="ij")
        support=(zz.square()+yy.square()+xx.square())<=1.
        # A tiny box must still contain a valid voxel.
        support[z//2,y//2,x//2]=True
        value=(.4+.15*torch.sin(xx*4)+.1*torch.cos(yy*3)+.05*torch.rand((z,y,x),device=device)).clamp(0,1)
        image[index,0,:z,:y,:x]=torch.where(support,value,0.)
        valid[index,0,:z,:y,:x]=support;padding[index,0,:z,:y,:x]=True
        targets[index,0,:z,:y,:x]=0.
        if index>=2:
            targets[index,0,z//3:max(z//3+1,2*z//3),y//3:max(y//3+1,2*y//3),x//3:max(x//3+1,2*x//3)]=1.
    # The second case is deliberately unverified; its NaN target must be ignored.
    targets[1]=float("nan")
    return {"image":image,"valid_mask":valid,"padding_mask":padding,"voxel_to_lps":affine,
            "targets":targets,"annotation_verified":torch.tensor([True,False,True,True],device=device),
            "labels":torch.arange(4,device=device)}


def coarse_coverage(output,valid_mask):
    checks=[]
    geometry=output["token_geometry"]
    for index,shape in enumerate(output["original_shape_zyx"].tolist()):
        coverage=torch.zeros(shape,dtype=torch.int32,device=valid_mask.device)
        selected=(geometry["scale_ids"][index]==0)&output["token_valid_mask"][index]
        for start,end in zip(geometry["starts_xyz"][index,selected].tolist(),geometry["ends_xyz"][index,selected].tolist()):
            x,y,z=start;ex,ey,ez=end;coverage[z:ez,y:ey,x:ex]+=1
        z,y,x=shape
        checks.append(bool((coverage[valid_mask[index,0,:z,:y,:x]]==1).all()))
    return checks


def _plain(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {key:_plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(item) for item in value]
    return value


def run_phase5(config_path,output_dir,*,device=None):
    config=load_phase5_config(config_path)
    engineering=config["engineering"]
    requested=device or engineering["device"]
    if requested not in ("auto","cpu","cuda"):raise ValueError("Invalid requested device")
    actual="cuda" if requested=="auto" and torch.cuda.is_available() else "cpu" if requested=="auto" else requested
    if actual=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA requested but unavailable; select auto/cpu")
    destination=Path(output_dir)
    destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):raise ValueError("F05 output directory must be empty; use a new run name")
    seed_everything(engineering["seed"],engineering["threads"])
    # CUDA kernel determinism is recorded; unsupported deterministic kernels warn
    # instead of preventing this engineering check. No CUDA reproducibility claim.
    if actual=="cuda":torch.use_deterministic_algorithms(True,warn_only=True)
    start=time.perf_counter()
    model=RSVTDenseFoundation(**{k:v for k,v in config["model"].items() if k!="name"}).to(actual)
    batch=synthetic_batch(engineering["synthetic_shapes_zyx"],actual)
    batch["image"].requires_grad_(True)
    before=model.classifier.linear.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=engineering["learning_rate"],weight_decay=0.)
    output=model(batch["image"],batch["valid_mask"],batch["voxel_to_lps"],padding_mask=batch["padding_mask"])
    classification=F.cross_entropy(output["logits"],batch["labels"])
    localization=verified_voxel_mask_loss(output["spatial_logits"],batch["targets"],batch["valid_mask"],
        batch["annotation_verified"],annotation_kind="voxel_mask")
    loss=classification+engineering["localization_loss_weight"]*localization["loss"]
    loss.backward()
    groups={}
    modules={"coarse_encoder":model.coarse_encoder,"candidate_tokenizer":model.tokenizer,
             "position":model.position,"transformer":model.transformer,"classification":model.classifier,
             "localization":model.localizer,**{f"scale_projection_{i+1}":p for i,p in enumerate(model.tokenizer.projections)}}
    for name,module in modules.items():
        gradients=[p.grad for p in module.parameters() if p.requires_grad]
        groups[name]={"all_present_and_finite":all(g is not None and bool(torch.isfinite(g).all()) for g in gradients),
                      "l2_norm":math.sqrt(sum(float(g.detach().double().square().sum()) for g in gradients if g is not None))}
    coverage=coarse_coverage(output,batch["valid_mask"])
    checks={"finite_loss":bool(torch.isfinite(loss)),"all_module_gradients_present_finite_nonzero":
        all(g["all_present_and_finite"] and g["l2_norm"]>0 for g in groups.values()),
        "coarse_covers_every_valid_voxel_once":all(coverage),
        "invalid_image_gradient_zero":bool((batch["image"].grad[~batch["valid_mask"]]==0).all()),
        "verified_localization_cases_three":localization["eligible_cases"]==3}
    # Do not mutate weights if a numerical check has failed.
    if all(checks.values()):optimizer.step()
    checks["classifier_weights_updated"]=not torch.equal(before,model.classifier.linear.weight.detach())
    if actual=="cuda":torch.cuda.synchronize()
    resolved=json.dumps(config,sort_keys=True,indent=2)+"\n"
    source=hashlib.sha256()
    source_root=Path(__file__).resolve().parents[1]
    for path in sorted(source_root.rglob("*.py")):
        source.update(path.relative_to(source_root).as_posix().encode()+b"\0"+path.read_bytes()+b"\0")
    summary={"schema_version":"rsvt_f05_smoke_v1","ok":all(checks.values()),"phase":"F05",
        "scope":"procedural_synthetic_forward_backward_one_optimizer_step",
        "medical_data_read":False,"medical_training_ready":False,"adaptive_routing":False,
        "class_index":CLASS_INDEX,"synthetic_labels_are_contract_fixtures":True,
        "config_sha256":hashlib.sha256(resolved.encode()).hexdigest(),
        "rsvt_python_source_sha256":source.hexdigest(),
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"device":actual,
            "cuda_available":torch.cuda.is_available(),"gpu_name":torch.cuda.get_device_name() if actual=="cuda" else None},
        "determinism_policy":"seeded_strict_CPU_warn_only_CUDA_no_cross_device_reproducibility_claim",
        "parameter_count":sum(p.numel() for p in model.parameters()),"wall_seconds":time.perf_counter()-start,
        "shapes":{key:list(output[key].shape) for key in ("logits","token_features","spatial_logits")},
        "original_shape_zyx":_plain(output["original_shape_zyx"]),"grid_shapes_zyx":_plain(output["grid_shapes_zyx"]),
        "active_token_counts":_plain(output["active_token_counts"]),"compute":_plain(output["compute"]),
        "loss":{"classification":float(classification.detach()),"dice":float(localization["dice"].detach()),
                "focal":float(localization["focal"].detach()),"total":float(loss.detach()),"eligible_voxel_mask_cases":localization["eligible_cases"]},
        "gradient_groups":groups,"checks":checks,"coarse_coverage_per_case":coverage,
        "pending":["F04 CNN optimization and research QA","coarse pretraining/weights",
                   "F06 differentiable routing and measured compute","F07 risk/generator losses",
                   "verified labels/annotations and medical training/evaluation"]}
    (destination/"resolved_config.json").write_text(resolved,encoding="utf-8")
    (destination/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    return summary
