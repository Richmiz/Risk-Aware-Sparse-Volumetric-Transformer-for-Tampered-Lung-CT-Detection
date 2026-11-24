"""Small working CLI for the scaffold; training commands arrive with implementation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import tomllib

from rsvt import __version__
from rsvt.data.manifest import audit_manifest
from rsvt.data.inventory import inventory_data
from rsvt.data.m3dsynth import audit_m3dsynth
from rsvt.data.ct_gan import audit_ctgan_labels
from rsvt.utils.config import load_config, validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rsvt", description="RSVT research project scaffold")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="Show implementation status")
    config_parser = commands.add_parser("check-config", help="Compose and check an experiment TOML")
    config_parser.add_argument("path", type=Path)
    manifest_parser = commands.add_parser("check-manifest", help="Audit declared provenance and partitions")
    manifest_parser.add_argument("path", type=Path)
    manifest_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    manifest_parser.add_argument("--check-paths", action="store_true")
    manifest_parser.add_argument("--held-out-generator")
    inventory_parser = commands.add_parser("inventory-data", help="Inventory files, headers, and candidate overlap")
    inventory_parser.add_argument("--config", type=Path, default=Path("configs/inventory.toml"))
    inventory_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    inventory_parser.add_argument("--output-dir", type=Path, required=True)
    m3_parser = commands.add_parser("audit-m3dsynth", help="Audit official M3Dsynth source metadata")
    m3_parser.add_argument("--metadata-dir", type=Path, default=Path("data/raw/m3dsynth/metadata"))
    m3_parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/m3dsynth"))
    m3_parser.add_argument("--output-dir", type=Path, required=True)
    ct_parser = commands.add_parser("audit-ctgan", help="Audit official CT-GAN location labels")
    ct_parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/ct_gan/pilot"))
    ct_parser.add_argument("--output-dir", type=Path, required=True)
    phase2_parser = commands.add_parser("phase2-ctgan", help="Audit CT geometry and preprocess a pilot")
    phase2_parser.add_argument("--dataset-root", type=Path, required=True)
    phase2_parser.add_argument("--inventory-run", type=Path, required=True)
    phase2_parser.add_argument("--config", type=Path, default=Path("configs/preprocessing/ct_gan_pilot.toml"))
    phase2_parser.add_argument("--output-dir", type=Path, required=True)
    phase2_parser.add_argument("--processed-root", type=Path, required=True)
    phase2_parser.add_argument("--max-cases", type=int, default=4)
    phase2_parser.add_argument("--resume-run", type=Path, action="append", default=[])
    phase2_parser.add_argument("--max-new-output-bytes", type=int)
    phase3_parser = commands.add_parser("phase3-ctgan", help="Build source-grouped draft partitions and smoke-test the loader")
    phase3_parser.add_argument("--f02-run", type=Path, required=True)
    phase3_parser.add_argument("--inventory-run", type=Path, required=True)
    phase3_parser.add_argument("--config", type=Path, default=Path("configs/splits/ct_gan_draft.toml"))
    phase3_parser.add_argument("--output-dir", type=Path, required=True)
    phase3_parser.add_argument("--torch-smoke", action="store_true")
    phase4_parser = commands.add_parser("phase4-baselines", help="Train exploratory candidate-target baselines on a pinned F03 draft")
    phase4_parser.add_argument("--f03-run", type=Path, required=True)
    phase4_parser.add_argument("--config", type=Path, default=Path("configs/training/baseline_pilot.toml"))
    phase4_parser.add_argument("--output-dir", type=Path, required=True)
    phase4_parser.add_argument("--resume", action="store_true")
    phase4_parser.add_argument("--resume-checkpoint", choices=("last","best"), default="last")
    phase5_parser = commands.add_parser("phase5-smoke", help="Check dense RSVT with synthetic volumes only")
    phase5_parser.add_argument("--config", type=Path, default=Path("configs/models/rsvt_f05.toml"))
    phase5_parser.add_argument("--output-dir", type=Path, required=True)
    phase5_parser.add_argument("--device", choices=("auto","cpu","cuda"))
    phase6_parser = commands.add_parser("phase6-routing", help="Synthetic B3 routing check and isolated profiling")
    phase6_parser.add_argument("--config", type=Path, default=Path("configs/models/rsvt_f06.toml"))
    phase6_parser.add_argument("--output-dir", type=Path, required=True)
    phase6_parser.add_argument("--device", choices=("auto","cpu","cuda"))
    phase7_parser = commands.add_parser("phase7-risk", help="Synthetic staged B4/B5 risk and token-constraint checks")
    phase7_parser.add_argument("--config", type=Path, default=Path("configs/training/risk_synthetic.toml"))
    phase7_parser.add_argument("--output-dir", type=Path, required=True)
    phase7_parser.add_argument("--device", choices=("auto","cpu","cuda"))
    for name in ("phase8-plan","phase8-run"):
        command = commands.add_parser(name, help="Synthetic orchestration; research protocol remains draft")
        command.add_argument("--config",type=Path,default=Path("configs/orchestration/phase08.toml"))
        command.add_argument("--device",choices=("auto","cpu","cuda"))
        command.add_argument("--f03-manifest",type=Path)
        if name=="phase8-plan":command.add_argument("--output",type=Path,required=True)
        else:
            command.add_argument("--output-dir",type=Path,required=True)
            command.add_argument("--resume",action="store_true")
    infer_parser=commands.add_parser("phase8-infer",help="Replay a bound synthetic-validation checkpoint")
    infer_parser.add_argument("--checkpoint",type=Path,required=True)
    infer_parser.add_argument("--output",type=Path,required=True)
    infer_parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    evaluate_parser=commands.add_parser("phase8-evaluate",help="Evaluate validation replay with its frozen threshold")
    evaluate_parser.add_argument("--predictions",type=Path,required=True)
    evaluate_parser.add_argument("--threshold",type=Path,required=True)
    evaluate_parser.add_argument("--output",type=Path,required=True)
    phase9_parser=commands.add_parser("phase9-evaluate",help="Evaluate sealed F08 synthetic predictions; no medical/final-test access")
    phase9_parser.add_argument("--f08-run",type=Path,required=True,help="F08 orchestration directory")
    phase9_parser.add_argument("--config",type=Path,default=Path("configs/evaluation/phase09.toml"))
    phase9_parser.add_argument("--output-dir",type=Path,required=True)
    phase10_parser=commands.add_parser("phase10-run",help="Paired synthetic E3/E4/E5 and whole-source bootstrap")
    phase10_parser.add_argument("--f08-run",type=Path,required=True)
    phase10_parser.add_argument("--config",type=Path,default=Path("configs/analysis/phase10.toml"))
    phase10_parser.add_argument("--output-dir",type=Path,required=True)
    phase10_parser.add_argument("--device",choices=("auto","cpu","cuda"))
    phase10_analysis=commands.add_parser("phase10-analyze",help="Recompute paired statistics from sealed F10 outputs")
    phase10_analysis.add_argument("--run-root",type=Path,required=True)
    phase10_analysis.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "info":
            payload = {
                "project": "RSVT medical deepfake detection",
                "version": __version__,
                "status": "F10 synthetic paired ablations/statistics available; medical QA/protocol unfinished",
                "available": ["config composition", "protocol config checks", "manifest audit", "file/header inventory", "M3Dsynth metadata audit", "DICOM geometry/HU", "physical-space preprocessing", "draft grouped partitions", "volume loader", "candidate-target CNN/ViT training", "checkpoint resume", "source-validation metrics"],
                "available_f05": ["dense multiscale RSVT", "physical token geometry", "verified-mask localization", "synthetic forward/backward check"],
                "available_f06": ["differentiable soft attention gates", "hard token packing", "isolated matched-weight profiling"],
                "available_f07": ["class-conditioned empirical CVaR", "verified training-generator variance", "projected token dual", "synthetic B4/B5 stages"],
                "available_f08": ["B0-B5/E1-E7 catalog", "source-bound synthetic jobs", "job-boundary resume", "frozen validation threshold/replay", "research-readiness report"],
                "available_f09": ["sealed prediction evaluation", "four-class/threshold endpoints", "fixed-bin calibration", "generator/LOGO metadata audit", "verified point/mask metrics"],
                "available_f10": ["synthetic E3/E4/E5 paired jobs", "trained synthetic checkpoint profiles", "paired whole-cluster bootstrap", "descriptive budget frontier"],
                "planned": ["medical multi-generator comparisons", "final protocol lock", "paired ablation executors", "confirmatory evaluation"],
            }
        elif args.command == "check-config":
            config = load_config(args.path)
            warnings = validate_config(config)
            payload = {
                "config": str(args.path),
                "valid_scaffold": True,
                "experiment": config.get("experiment", {}),
                "dataset": config.get("dataset", {}).get("name"),
                "model": config.get("model", {}).get("name"),
                "warnings": warnings,
            }
        elif args.command == "inventory-data":
            payload = inventory_data(args.config, args.project_root, args.output_dir)
        elif args.command == "audit-m3dsynth":
            payload = audit_m3dsynth(args.metadata_dir, args.dataset_root, args.output_dir)
        elif args.command == "audit-ctgan":
            payload = audit_ctgan_labels(args.dataset_root, args.output_dir)
        elif args.command == "phase10-run":
            from rsvt.experiments.phase10 import run_phase10
            payload=run_phase10(args.f08_run,args.config,args.output_dir,device=args.device,
                progress=lambda event:print(json.dumps(event),flush=True))
            print(json.dumps({"ok":payload["ok"],"jobs_completed":payload["jobs_completed"],"checks":payload["checks"],
                "summary":str(args.output_dir / "summary.json"),"final_test_access_allowed":False},indent=2))
            return 0 if payload["ok"] else 1
        elif args.command == "phase10-analyze":
            from rsvt.experiments.phase10 import analyze_phase10
            payload=analyze_phase10(args.run_root,args.output_dir)
            print(json.dumps({"ok":payload["ok"],"summary":str(args.output_dir / "summary.json")},indent=2))
            return 0 if payload["ok"] else 1
        elif args.command == "phase9-evaluate":
            from rsvt.experiments.phase9 import run_phase9
            payload=run_phase9(args.f08_run,args.config,args.output_dir,
                progress=lambda event:print(json.dumps(event),flush=True))
            print(json.dumps({"ok":payload["ok"],"checks":payload["checks"],"summary":str(args.output_dir / "summary.json"),
                "final_test_access_allowed":False},indent=2))
            return 0 if payload["ok"] else 1
        elif args.command == "phase8-plan":
            from rsvt.experiments.runner import atomic_json,build_plan
            payload=build_plan(args.config,device=args.device,f03_manifest=args.f03_manifest)
            atomic_json(args.output,payload)
            print(json.dumps({"ok":True,"contract_sha256":payload["contract_sha256"],"research_ready":False,"output":str(args.output)},indent=2))
            return 0
        elif args.command == "phase8-run":
            from rsvt.experiments.runner import run_phase8
            payload=run_phase8(args.config,args.output_dir,device=args.device,f03_manifest=args.f03_manifest,resume=args.resume,
                progress=lambda event:print(json.dumps(event),flush=True))
            print(json.dumps({"ok":payload["ok"],"checks":payload["checks"],"research_ready":False,"summary":str(args.output_dir / "summary.json")},indent=2))
            return 0 if payload["ok"] else 1
        elif args.command == "phase8-infer":
            from rsvt.experiments.runner import infer_checkpoint
            payload=infer_checkpoint(args.checkpoint,args.output,device=args.device)
            print(json.dumps({"ok":True,"prediction_sha256":payload["prediction_sha256"],"output":str(args.output)},indent=2))
            return 0
        elif args.command == "phase8-evaluate":
            from rsvt.experiments.runner import evaluate_inference
            payload=evaluate_inference(args.predictions,args.threshold,args.output)
            print(json.dumps({"ok":True,"output":str(args.output),"frozen_threshold":payload["frozen_threshold"]},indent=2))
            return 0
        elif args.command == "phase7-risk":
            from rsvt.experiments.phase7 import run_phase7
            payload = run_phase7(args.config,args.output_dir,device=args.device,
                                 progress=lambda event:print(json.dumps(event),flush=True))
            print(json.dumps({"ok":payload["ok"],"checks":payload["checks"],"summary":str(args.output_dir / "summary.json")},indent=2))
            return 0 if payload["ok"] else 1
        elif args.command == "phase6-routing":
            from rsvt.experiments.phase6 import run_phase6
            payload = run_phase6(args.config,args.output_dir,device=args.device,
                                 progress=lambda event:print(json.dumps(event),flush=True))
            print(json.dumps({"ok":payload["ok"],"checks":payload["checks"],"summary":str(args.output_dir / "summary.json")},indent=2))
            return 0 if payload["ok"] else 1
        elif args.command == "phase5-smoke":
            from rsvt.experiments.phase5 import run_phase5
            payload = run_phase5(args.config,args.output_dir,device=args.device)
            print(json.dumps(payload,indent=2))
            return 0 if payload["ok"] else 1
        elif args.command == "phase4-baselines":
            from rsvt.training.phase4 import run_phase4
            payload = run_phase4(args.f03_run,args.config,args.output_dir,resume=args.resume,
                                 resume_checkpoint=args.resume_checkpoint,
                                 progress=lambda event:print(json.dumps(event),flush=True))
            print(json.dumps(payload,indent=2))
            return 0 if payload["ok"] else 1
        elif args.command == "phase3-ctgan":
            from rsvt.data.phase3 import run_phase3
            payload = run_phase3(args.f02_run, args.inventory_run, args.config, args.output_dir,
                                 torch_smoke=args.torch_smoke,
                                 progress=lambda item: print(json.dumps(item), flush=True))
            print(json.dumps(payload, indent=2))
            return 0 if payload["ok"] else 1
        elif args.command == "phase2-ctgan":
            from rsvt.data.phase2 import run_phase2
            payload = run_phase2(args.dataset_root, args.inventory_run, args.config,
                                 args.output_dir, args.processed_root, max_cases=args.max_cases,
                                 resume_runs=args.resume_run, max_new_output_bytes=args.max_new_output_bytes,
                                 progress=lambda item: print(json.dumps(item), flush=True))
            print(json.dumps(payload, indent=2))
            return 0 if payload["ok"] else 1
        else:
            audit = audit_manifest(
                args.path,
                args.project_root,
                check_paths=args.check_paths,
                held_out_generator=args.held_out_generator,
            )
            payload = {"ok": audit.ok, **asdict(audit)}
            print(json.dumps(payload, indent=2))
            return 0 if audit.ok else 1
    except (OSError, UnicodeError, csv.Error, ValueError, tomllib.TOMLDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 1
    print(json.dumps(payload, indent=2))
    return 0
