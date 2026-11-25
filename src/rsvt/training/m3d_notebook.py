"""Readable Colab session for M3Dsynth acquisition, diagnostics and coarse-to-fine RSVT.

Medical images stay on Colab/Drive. Every artifact is written to project Drive
and read back against a checksum before it is reported as saved.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import errno
from html import escape
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import shutil
import sys

import numpy as np
import torch

from rsvt.data import m3dsynth_store as store_mod
from rsvt.data.m3dsynth_dataset import M3DStore, coarse_builder, crop_shape_for
from rsvt.data.m3dsynth_store import sha256_path, write_json
from rsvt.models.coarse_to_fine import CoarseToFineRSVT, compute_estimate
from rsvt.training import coarse_to_fine as trainer
from rsvt.training.notebook import table_html, write_profile

STORE_RELATIVE = "datasets/m3dsynth_store"
RUNS_RELATIVE = "outputs/rsvt_training"
METADATA_RELATIVE = "data/raw/m3dsynth/metadata"
UID_MAP_RELATIVE = "reports/supplementary/data_label_audit_20261111/metadata_review_v2/m3dsynth_source_groups.csv"


class M3DNotebookSession:
    def __init__(self, project, code_root, *, epochs=30, learning_rate=2e-4, patience=8, seed=13,
                 patients_train=200, patients_validation=100, generators=("cycle", "pix2pix", "diffusion"),
                 resume_session="", configuration=None, store_root=None, cache_root=None, work_root=None,
                 lidc_root=None, metadata_dir=None, enforce_drive=True, display_enabled=True, code_commit=None):
        self.project, self.code_root = Path(project).resolve(), Path(code_root).resolve()
        if enforce_drive and (platform.system() != "Linux"
                              or not self.project.is_relative_to(Path("/content/drive/MyDrive").resolve())):
            raise RuntimeError("Use the Colab kernel and mount project Drive; medical images stay there.")
        if not self.project.is_dir():
            raise FileNotFoundError("Project Drive is not mounted")
        self.display_enabled, self.fixture_mode = display_enabled, not enforce_drive
        config = deepcopy(configuration) if configuration is not None else trainer.load_c2f_config(
            self.code_root / "configs/training/rsvt_m3dsynth_c2f.toml")
        config["training"].update(epochs=epochs, learning_rate=learning_rate, patience=patience, seed=seed)
        config["acquisition"].update(patients_train=patients_train, patients_validation=patients_validation, seed=seed)
        config["data"]["generators"] = list(generators)
        self.config = trainer.validate_config(config)
        self.store_root = Path(store_root) if store_root else self.project / STORE_RELATIVE
        # One local folder: acquisition's uncompressed real series are reused by training.
        self.cache_root = Path(cache_root) if cache_root else Path("/content/m3dsynth_local")
        self.work_root = Path(work_root) if work_root else self.cache_root
        self.metadata_dir = Path(metadata_dir) if metadata_dir else self.code_root / METADATA_RELATIVE
        self.lidc_root = Path(lidc_root) if lidc_root else None
        parent = self.project / RUNS_RELATIVE
        self.root = (Path(resume_session).resolve() if resume_session else
                     parent / ("c2f_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")))
        if not self.root.is_relative_to(parent) or self.root == parent:
            raise ValueError("Keep sessions inside project Drive outputs/rsvt_training")
        binding = {"config_sha256": store_mod.digest_json(self.config),
                   "implementation_sha256": trainer.implementation_fingerprint()}
        if resume_session:
            stored = json.loads((self.root / "notebook_session.json").read_text(encoding="utf-8"))
            if stored["binding"] != binding:
                raise ValueError("Resuming needs the same settings and code as the run being resumed. Restore the "
                                 "settings, or set RESUME_SESSION = \"\" to start a new run.")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        self.training_dir, self.assets = self.root / "training", self.root / "figures"
        self.assets.mkdir(exist_ok=True)
        write_profile(self.root / "training_config.toml", self.config)
        self.artifacts, self._handles, self._history = {}, {}, []
        self._save_json("notebook_session.json", {"binding": binding, "store": str(self.store_root),
                                                  "code_commit": code_commit,
                                                  "created_utc": datetime.now(timezone.utc).isoformat(),
                                                  "test_status": "not_acquired_reserved_for_final_evaluation"})
        if metadata_dir is None:
            # A GitHub checkout has no author tables: fetch them at the pinned commit (checksum-verified).
            self.metadata_report = store_mod.fetch_metadata(self.metadata_dir)
        # Without the audited UID map (kept out of the public repository) each series is found on
        # TCIA by patient and series number and its headers are checked after download.
        uid_map = self.code_root / UID_MAP_RELATIVE
        self.meta = store_mod.load_metadata(self.metadata_dir, uid_map if uid_map.exists() and not metadata_dir else None)
        acquisition = self.config["acquisition"]
        self.selection = store_mod.select_subset(
            self.meta, patients_per_split={"train": acquisition["patients_train"],
                                           "validation": acquisition["patients_validation"],
                                           "test": acquisition["patients_test"]},
            generators=self.config["data"]["generators"], seed=acquisition["seed"])

    # ------------------------------------------------------------------ helpers
    def _html(self, text):
        if self.display_enabled:
            from IPython.display import HTML, display
            display(HTML(text))

    def _save_json(self, name, value):
        write_json(self.root / name, value)
        self.artifacts[name] = {"sha256": sha256_path(self.root / name)}

    def _figure(self, figure, name):
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
        data = buffer.getvalue()
        path = self.assets / name
        path.write_bytes(data)
        if path.read_bytes() != data:
            raise OSError("Drive figure readback failed: " + name)
        if self.display_enabled:
            from IPython.display import Image, display
            display(Image(data=data))
        return path

    def _progress_printer(self, prefix):
        def report(event):
            message = prefix + " " + json.dumps(event, default=str)
            if self.display_enabled and "status" in self._handles:
                from IPython.display import HTML
                self._handles["status"].update(HTML("<p><b>" + escape(message[:500]) + "</b></p>"))
            elif self.display_enabled:
                print(message, flush=True)
        return report

    # ------------------------------------------------------------------ runtime and data
    def show_runtime(self):
        available = torch.cuda.is_available()
        info = {"Python": platform.python_version(), "PyTorch": torch.__version__, "CUDA build": str(torch.version.cuda),
                "GPU detected": available, "GPU name": torch.cuda.get_device_name(0) if available else "No GPU detected",
                "Python executable": sys.executable, "Session folder in Drive": str(self.root),
                "Dataset store in Drive": str(self.store_root)}
        self._html(table_html(["Runtime check", "Value"], info.items()))
        self._save_json("runtime.json", info)
        return info

    def plan_data(self):
        estimate = store_mod.storage_estimate(self.selection)
        counts = self.selection["counts"]
        rows = [("Patients (train / validation / test)", " / ".join(str(counts["patients"].get(s, 0))
                                                                     for s in ("train", "validation", "test"))),
                ("LIDC-IDRI real series to download", estimate["real_series"]),
                ("Manipulated samples to keep", estimate["manipulated_samples"]),
                ("Generators", ", ".join(self.selection["generators"])),
                ("Estimated LIDC download", f"{estimate['estimated_lidc_download_gb']} GB (public TCIA API)"),
                ("Archive stream per generator", ", ".join(f"{g}: {s} GB" for g, s in estimate["archive_stream_gb"].items())),
                ("Estimated Drive storage", f"{estimate['estimated_drive_gb']} GB in {STORE_RELATIVE}"),
                ("Note", estimate["note"])]
        status = self.store_status()
        rows.append(("Dataset store now", "complete for this selection" if status["complete"] else status["reason"]))
        if not status["complete"]:
            space = self.check_storage(stop=False)
            rows += [(f"Free space: {name}", f"about {info['needed_gb']} GB needed; Colab does not show your Drive quota here, "
                      "so check it yourself" if info.get("quota_reported") is False else
                      f"{info['free_gb']} GB free, about {info['needed_gb']} GB needed")
                     for name, info in space["locations"].items()]
        self._html("<h3>M3Dsynth data plan</h3>" + table_html(["Item", "Value"], rows))
        self._save_json("data_plan.json", {"selection_counts": counts, "estimate": estimate, "store_status": status,
                                           "metadata_sha256": self.meta["sha256"]})
        return estimate

    def store_status(self):
        """Does the verified store cover the current selection? Reads the manifest only (no images)."""
        path = self.store_root / "manifest.json"
        if not path.exists():
            return {"exists": False, "complete": False, "reason": "no dataset store yet", "missing_series": []}
        manifest = json.loads(path.read_text(encoding="utf-8"))
        stored = manifest.get("selection", {})
        same = all(stored.get(key) == self.selection[key] for key in ("counts", "generators", "seed"))
        rule_matches = manifest.get("box_rules") == [self.box_rule()]
        # Samples absent from an author archive stay excluded for good; only missing LIDC series are worth fetching again.
        missing = sorted(e["series_key"] for e in manifest.get("excluded", []) if "volume_id" not in e)
        reason = ("the selection changed (patients, generators or seed)" if not same else
                  "the edit boxes were made under an earlier rule, so the archives must be streamed again"
                  if not rule_matches else
                  f"{len(missing)} LIDC series not acquired yet" if missing else "complete")
        return {"exists": True, "complete": same and rule_matches and not missing, "reason": reason, "missing_series": missing}

    def box_rule(self):
        acquisition = self.config["acquisition"]
        return store_mod.box_rule(acquisition["half_box_mm"], acquisition.get("cube_half_mm", 0.))

    def check_storage(self, *, stop=True, margin_gb=2.):
        """Compare free Drive and runtime-disk space with what the download still needs."""
        estimate = store_mod.storage_estimate(self.selection)
        # Only what is still missing: LIDC scans without a file and samples without a box under the current folder.
        def missing(folder, names):
            return sum(not (self.store_root / folder / (name + ".npz")).exists() for name in names)
        scans = missing("real", [s["series_key"] for s in self.selection["series"]])
        boxes = sum(missing(f"{store_mod.BOX_FOLDER}/{g}", [s["img_id"] for s in self.selection["samples"] if s["variant"] == g])
                    for g in self.selection["generators"])
        left = store_mod.storage_estimate({**self.selection, "series": [None] * scans, "samples": [None] * boxes})
        needs = {"Google Drive": (self.project, left["estimated_drive_gb"] * 1.0737 + margin_gb),
                 # uncompressed series (~80 MB each) stay on the runtime disk while the archives stream
                 "runtime disk": (self.cache_root, estimate["real_series"] * .08 + margin_gb)}
        locations, problems = {}, []
        for name, (path, needed) in needs.items():
            probe = Path(path)
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            usage = shutil.disk_usage(probe)
            locations[name] = {"free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1),
                               "needed_gb": round(needed, 1)}
            if usage.total and usage.free < needed * 1e9:  # a size of 0 means the mount does not report space
                problems.append(f"{name}: {usage.free / 1e9:.1f} GB free, about {needed:.1f} GB needed")
        # A mount that repeats the runtime disk's size is not showing the Drive quota.
        reported = locations["Google Drive"]["total_gb"] != locations["runtime disk"]["total_gb"]
        locations["Google Drive"]["quota_reported"] = reported
        report = {"locations": locations, "problems": problems}
        self._save_json("storage_check.json", report)
        if stop and problems:
            raise RuntimeError("Not enough space, so nothing was downloaded. " + "; ".join(problems)
                               + ". Free space in Google Drive or lower PATIENTS_TRAIN / PATIENTS_VALIDATION.")
        return report

    def prepare_data(self, *, download="auto", generators=None, tcia_base=store_mod.TCIA_BASE,
                     archive_sources=None, extracted_root=None):
        """Acquire (resumably) and verify the store.

        `download="auto"` downloads only when the store does not yet cover the selection;
        otherwise the saved store is read back against its checksums.
        """
        manifest_path = self.store_root / "manifest.json"
        if download == "auto":
            status = self.store_status()
            download = not status["complete"]
            self._html(f"<p><b>Dataset store:</b> {escape(status['reason'])}"
                       f"{' — downloading what is missing.' if download else ' — verifying the saved files.'}</p>")
        if not download:
            if manifest_path.exists():
                self._html("<p><b>Existing verified store found.</b> Reading it back against its checksums.</p>")
                return self.verify_dataset()
            self._html("<p><b>No M3Dsynth store yet.</b> Nothing was downloaded.</p>")
            return None
        self.check_storage()
        if self.display_enabled:
            from IPython.display import HTML, display
            self._handles["status"] = display(HTML("<p>Preparing M3Dsynth…</p>"), display_id=True)
        report = self._progress_printer("[data]")
        build = coarse_builder()
        real_meta, failures = {}, []
        for i, item in enumerate(self.selection["series"], 1):
            try:
                real_meta[item["series_key"]] = store_mod.acquire_real_series(
                    item, self.store_root, self.work_root, lidc_root=self.lidc_root, base=tcia_base, coarse_builder=build)
            except Exception as exc:  # one bad series must not abort the rest; it is reported and excluded
                self._stop_if_drive_lost(exc, f"after {i - 1} of {len(self.selection['series'])} LIDC scans")
                failures.append({"series_key": item["series_key"], "reason": str(exc)})
            report({"stage": "real_series", "done": i, "of": len(self.selection["series"]), "failed": len(failures)})
        write_json(self.store_root / "real_acquisition_failures.json", failures)
        if len(failures) > .05 * len(self.selection["series"]):  # the archives take hours and depend on these scans
            reasons = Counter(f["reason"][:120] for f in failures).most_common(3)
            raise RuntimeError(f"{len(failures)} of {len(self.selection['series'])} LIDC scans failed, so the archives were "
                               "not streamed. Most common reasons: " + "; ".join(f"{n}x {text}" for text, n in reasons))
        available = [s for s in self.selection["samples"] if s["series_key"] in real_meta]
        streams = {}
        for generator in generators or self.selection["generators"]:
            source = (archive_sources or {}).get(generator)
            try:
                streams[generator] = store_mod.stream_generator(
                    generator, available, real_meta, self.store_root, self.work_root, source=source,
                    extracted_root=extracted_root, half_mm=self.config["acquisition"]["half_box_mm"],
                    cube_half_mm=self.config["acquisition"].get("cube_half_mm", 0.), progress=report)
            except OSError as exc:
                self._stop_if_drive_lost(exc, f"while streaming {generator}")
                raise
        manifest = store_mod.build_manifest(self.store_root, self.selection, self.meta)
        self._save_json("acquisition_report.json", {"real_series_ok": len(real_meta), "real_series_failed": failures,
                                                    "streams": streams, "manifest_counts": manifest["counts"]})
        return self.verify_dataset()

    def _stop_if_drive_lost(self, exc, when):
        """A dead Drive mount fails every later read too; stop now instead of blaming the data."""
        if not isinstance(exc, OSError):
            return
        lost = exc.errno in (errno.ENOTCONN, errno.ESTALE, errno.ENODEV)
        if not lost:
            try:
                os.stat(self.project)
            except OSError:
                lost = True
        if lost:
            raise RuntimeError(f"Google Drive disconnected {when}. Nothing is wrong with the data and finished samples are kept. "
                               "Use Runtime > Disconnect and delete runtime, reconnect, and Run all.") from exc

    def verify_dataset(self):
        manifest = json.loads((self.store_root / "manifest.json").read_text(encoding="utf-8"))
        check = store_mod.verify_store(self.store_root, manifest)
        self._save_json("dataset_verification.json", {"manifest_sha256": sha256_path(self.store_root / "manifest.json"),
                                                      **check, "counts": manifest["counts"]})
        if not check["ok"]:
            raise OSError("Store readback failed for: " + ", ".join(check["failures"][:10]))
        return manifest

    def has_store(self):
        present = (self.store_root / "manifest.json").exists()
        if not present:
            self._html("<p><b>No prepared M3Dsynth store yet.</b> Section 7 prepares it. "
                       "Nothing is trained or shown on missing data.</p>")
        return present

    def _store(self):
        return M3DStore(self.store_root, self.cache_root)

    def show_dataset(self):
        store = self._store()
        rows = []
        for split in ("train", "validation", "test"):
            volumes = [v for v in store.manifest["volumes"] if v["split"] == split]
            if not volumes:
                rows.append([split, 0, 0, "—", "—", "Not acquired (reserved)" if split == "test" else "—"])
                continue
            manipulated = Counter(f"{v['generator']}/{v['operation']}" for v in volumes if v["kind"] == "manipulated" and v["eligible"])
            rows.append([split, sum(v["kind"] == "authentic" for v in volumes), sum(manipulated.values()),
                         ", ".join(f"{k}: {n}" for k, n in sorted(manipulated.items())),
                         sum(not v["eligible"] for v in volumes), "Available"])
        self._html("<h3>Connected dataset: M3Dsynth (LIDC-IDRI sources)</h3>" + table_html(
            ["Partition", "Authentic volumes", "Manipulated (eligible)", "By generator / operation", "Ineligible", "Use"], rows))
        self._html("<p>Authentic = the LIDC-IDRI series. Removal (FB) and insertion (FM) volumes are that series with a "
                   "verified local edit. Partitions are the publisher's patient-disjoint sets.</p>")
        self._save_json("dataset_overview.json", {"rows": rows, "test_volumes_opened": 0})
        return rows

    def show_previews(self, sites=2):
        import matplotlib.pyplot as plt
        store = self._store()
        chosen = []
        for operation in ("removal", "insertion"):
            chosen += [v for v in store.manifest["volumes"] if v["split"] == "train" and v["kind"] == "manipulated"
                       and v["eligible"] and v["operation"] == operation][:max(1, sites // 2)]
        if not chosen:
            self._html("<p>No eligible training volumes to preview yet.</p>")
            return []
        figure, axes = plt.subplots(len(chosen), 3, figsize=(11, 3.8 * len(chosen)), squeeze=False)
        previews = []
        for row, volume in zip(axes, chosen):
            lo, hi = volume["mask_bbox"]
            z = (lo[0] + hi[0]) // 2
            key = volume["series_key"]
            lut = store.lut(key)
            authentic = lut[np.asarray(store.real(key)[z])]
            shape = store.series[key]["shape_zyx"]
            # centre +0.5 makes the 1-slice crop start exactly at z (no half-way rounding)
            manipulated, _, mask = store.crop(volume, (z + .5, shape[1] / 2, shape[2] / 2), (1, shape[1], shape[2]))
            panels = (("Authentic (LIDC)", authentic), (f"{volume['label']} {volume['generator']}", manipulated[0]),
                      ("Difference", manipulated[0] - authentic))
            for axis, (title, image) in zip(row, panels):
                axis.imshow(image, cmap="gray" if title != "Difference" else "coolwarm",
                            vmin=0 if title != "Difference" else -.5, vmax=1 if title != "Difference" else .5)
                axis.contour(mask[0], levels=[.5], colors="lime", linewidths=.8)
                axis.set_title(f"{title}\n{volume['volume_id']} z={z}", fontsize=9)
                axis.axis("off")
            previews.append({"volume_id": volume["volume_id"], "slice_z": int(z), "split": "train"})
        figure.suptitle("Training examples: authentic slice, manipulated slice, difference (green = author mask)")
        figure.tight_layout()
        self._figure(figure, "dataset_preview.png")
        plt.close(figure)
        self._save_json("dataset_previews.json", {"examples": previews, "test_volumes_opened": 0})
        return previews

    # ------------------------------------------------------------------ diagnostics
    def run_diagnostics(self, *, pairs_train=600, pairs_validation=200, epochs=6):
        """Run the data checks once per dataset store and reuse them; return blocking problems with the results.

        Results live beside the store, keyed by its manifest, the generators, the seed and the
        diagnostics code, so a CPU data session and a later GPU training session share them.
        """
        from rsvt.experiments import m3d_diagnostics
        names = ("shortcut_check.json", "resolution_check.json")
        key = store_mod.digest_json({"manifest": sha256_path(self.store_root / "manifest.json"),
                                     "generators": self.config["data"]["generators"], "seed": self.config["training"]["seed"],
                                     "settings": [pairs_train, pairs_validation, epochs],
                                     "code": sha256_path(Path(m3d_diagnostics.__file__))})[:16]
        folder = self.store_root / "diagnostics" / key
        reused = all((folder / name).exists() for name in names)
        if reused:
            shortcut, resolution = (json.loads((folder / name).read_text(encoding="utf-8")) for name in names)
        else:
            store = self._store()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            report = self._progress_printer("[diagnostics]")
            shortcut = m3d_diagnostics.shortcut_check(store, self.config["data"]["generators"], folder)
            resolution = m3d_diagnostics.resolution_check(
                store, self.config["data"]["generators"], folder, device=device, pairs_train=pairs_train,
                pairs_validation=pairs_validation, epochs=epochs, seed=self.config["training"]["seed"], progress=report)
        problems = self.diagnostics_problems({"shortcut": shortcut})
        number = lambda v: "—" if v is None else f"{v:.3f}"
        identity = shortcut["archive_identity"]
        per_operation = [(f"{op.capitalize()} AUROC, native / degraded to 96³ voxels",
                          f"{number(arm['native'])} / {number(arm['coarse_equivalent'])}")
                         for op, arm in resolution["by_operation"].items()]
        self._html("<h3>Pre-training checks (train/validation only)</h3>" + table_html(["Check", "Result"], [
            ("Native-crop AUROC (manipulated vs matched authentic site)", number(resolution["native"]["validation_auroc"])),
            ("Same crops degraded to the 96³ voxel size", number(resolution["coarse_equivalent"]["validation_auroc"])),
            *per_operation,
            ("Resolution conclusion", resolution["interpretation"]),
            ("Eligible manipulated volumes / receipts", f"{identity['eligible']} / {identity['manipulated_receipts']}"),
            ("Changed voxels outside stored boxes (eligible)", identity["eligible_changed_voxels_outside_box"]),
            ("Edits extending beyond the author mask box", number(identity["edits_extending_beyond_author_mask_bbox"])),
            ("Whole-scan histogram classifier AUROC", number(shortcut["global_histogram_classifier"]["validation_auroc"])),
            ("Shortcut conclusion", shortcut["interpretation"]),
            ("Results", "reused from an earlier run on this dataset" if reused else "computed now and saved with the dataset"),
            ("Blocks training", "; ".join(problems) if problems else "no")]))
        for name in names:
            self.artifacts["diagnostics/" + name] = {"sha256": sha256_path(folder / name)}
        self._save_json("diagnostics.json", {"folder": str(folder), "key": key, "reused": reused, "problems": problems})
        return {"resolution": resolution, "shortcut": shortcut, "problems": problems, "folder": str(folder)}

    @staticmethod
    def diagnostics_problems(diagnostics):
        """Data-check results that make a training run pointless until they are understood."""
        problems = []
        identity = diagnostics["shortcut"]["archive_identity"]
        receipts, eligible = identity["manipulated_receipts"], identity["eligible"]
        if receipts and eligible / receipts < .5:
            problems.append(f"only {eligible} of {receipts} manipulated samples passed archive verification")
        auroc = diagnostics["shortcut"]["global_histogram_classifier"]["validation_auroc"]
        if auroc is not None and abs(auroc - .5) >= .1:
            problems.append(f"whole-scan intensity statistics alone separate the classes (AUROC {auroc:.2f})")
        return problems

    # ------------------------------------------------------------------ model and training
    def show_configuration(self):
        t, d, r, m = (self.config[k] for k in ("training", "data", "risk", "model"))
        rows = [("Training generators", ", ".join(d["generators"])), ("Epochs (max)", t["epochs"]),
                ("Peak learning rate / schedule", f"{t['learning_rate']} / {t['warmup_epochs']} warmup epochs then cosine"),
                ("Logical batch", f"{t['authentic_per_batch']} authentic + {t['manipulated_per_batch']} manipulated volumes"),
                ("Crops per training volume", f"{t['router_crops_training']} router + {d['forced_sites']} site + {d['random_crops']} random"),
                ("Crops at inference (compute budget)", m["crops_at_inference"]),
                ("Crop extent", f"{d['crop_mm']} mm on the native voxel grid"),
                ("Augmentation", f"left-right flip p={d['flip_probability']}, crop jitter ±{d['jitter_fraction']} of crop size"),
                ("Risk stage starts", f"epoch {t['risk_start_epoch'] + 1}"),
                ("Tail risk (alpha / weight)", f"{r['alpha']} / {r['cvar_weight']}: authentic {r['authentic_weight']}, FB {r['fb_weight']}, FM {r['fm_weight']}"),
                ("Generator-variance weight", r["generator_weight"]),
                ("Checkpoint selection", "lowest validation volume BCE averaged over authentic, removal and insertion "
                                         "(risk-stage epochs)"),
                ("Early stopping", f"{t['patience']} risk-stage epochs without improvement" if t["patience"] else "off"),
                ("Seed / precision", f"{t['seed']} / FP32")]
        self._html(table_html(["Setting", "Value"], rows))
        self._save_json("displayed_configuration.json", {"settings": [list(map(str, r)) for r in rows]})
        return rows

    def show_model(self):
        model = CoarseToFineRSVT(**self.config["model"])
        count = sum(p.numel() for p in model.parameters())
        store = self._store()
        example = next(v for v in store.manifest["volumes"] if v["split"] == "train")
        info = store.series[example["series_key"]]
        estimate = compute_estimate(model, (96, 96, 96), crop_shape_for(info["geometry"]["spacing_zyx"], self.config["data"]["crop_mm"]),
                                    model.crops_at_inference, info["shape_zyx"])
        self._html("<pre>96³ coarse view → router over 12³ cells → top-K native crops (1 mm-class grid)\n"
                   "→ fine CNN per crop (tamper, visible-nodule, voxel mask) → Transformer over context + crop tokens\n"
                   "→ tampered vs authentic, removal vs insertion</pre>" + table_html(["Model", "Value"], [
                       ("Parameters", f"{count:,}"), ("Example series", example["series_key"]),
                       ("Per-volume compute (GMAC)", f"{estimate['total_gmac']:.1f}"),
                       ("Same fine encoder over the whole native volume (GMAC)", f"{estimate['dense_native_fine_gmac']:.0f}"),
                       ("Dense / sparse ratio", f"{estimate['dense_over_sparse']:.1f}×"),
                       ("Native voxels read", f"{100 * estimate['native_voxels_read_fraction']:.1f}%")]))
        self._save_json("model_information.json", {"parameters": count, "model_config": self.config["model"], "compute": estimate})
        return count

    def _history_html(self, rows):
        number = lambda v, p=False: "—" if v is None else (f"{100 * v:.1f}%" if p else f"{v:.4f}")
        return table_html(["Epoch", "Stage", "LR", "Train objective", "Val selection BCE", "Tamper AUROC", "Removal AUROC",
                           "Worst-generator AUROC", "Router recall@K", "Removal recall @10% FPR"],
                          [[r["epoch"], r["stage"], f"{r['learning_rate']:.2g}", number(r["training_objective"]),
                            number(r["validation_selection_bce"]), number(r["validation_tamper_auroc"]),
                            number(r["validation_removal_auroc"]), number(r["validation_worst_generator_auroc"]),
                            number(r["validation_router_recall"], True), number(r["validation_removal_recall_fpr10"], True)]
                           for r in rows])

    def _curves(self, rows):
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["training_objective"] for r in rows], marker="o", label="Train objective")
        axes[0].plot(epochs, [r["validation_selection_bce"] for r in rows], marker="o",
                     label="Validation BCE (authentic / removal / insertion balanced)")
        for key, label in (("validation_tamper_auroc", "Tamper AUROC"), ("validation_removal_auroc", "Removal AUROC"),
                           ("validation_worst_generator_auroc", "Worst-generator AUROC")):
            axes[1].plot(epochs, [np.nan if r[key] is None else r[key] for r in rows], marker="o", label=label)
        axes[0].set(title="Loss", xlabel="Epoch")
        axes[1].set(title="Validation discrimination", xlabel="Epoch", ylim=(0, 1))
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure.tight_layout()
        return figure

    def _train_progress(self, event):
        if not self.display_enabled:
            return
        from IPython.display import HTML
        if event.get("event") == "epoch":
            self._history = [r for r in self._history if r["epoch"] != event["epoch"]] + [event]
            self._handles["table"].update(HTML(self._history_html(self._history)))
            import matplotlib.pyplot as plt
            figure = self._curves(self._history)
            buffer = BytesIO()
            figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
            plt.close(figure)
            from IPython.display import Image
            self._handles["curves"].update(Image(data=buffer.getvalue()))
            message = f"Epoch {event['epoch']} finished"
        elif event.get("event") == "logical_batch":
            message = f"Epoch {event['epoch']}/{self.config['training']['epochs']} — batch {event['batch']}/{event['batches']}"
        else:
            message = json.dumps(event, default=str)[:200]
        self._handles["status"].update(HTML("<p><b>" + escape(message) + "</b></p>"))

    def train_when_ready(self, diagnostics=None):
        """Train (or resume) on a GPU unless a data check blocks it; on a CPU runtime explain the next step."""
        if self.config["training"]["device"] == "cuda" and not torch.cuda.is_available():
            self._html("<p><b>No GPU in this runtime, so training waits.</b> The dataset and its checks are saved. "
                       "Switch to a GPU runtime (Runtime → Change runtime type → L4) and Run all; training starts, "
                       "or resumes, automatically.</p>")
            return None
        problems = (diagnostics or {}).get("problems") or []
        if problems:
            self._html("<p><b>Training was not started</b> because a data check failed: " + escape("; ".join(problems))
                       + ". Look at section 9 before training on this dataset.</p>")
            return None
        return self.train()

    def train(self):
        if self.config["training"]["device"] == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU not detected. Select a Colab GPU kernel; data checks are already saved.")
        if self.display_enabled:
            from IPython.display import HTML, display
            self._handles = {"status": display(HTML("<p>Preparing training…</p>"), display_id=True),
                             "table": display(HTML("<p>Epoch results appear here.</p>"), display_id=True),
                             "curves": display(HTML("<p>Learning curves appear after the first epoch.</p>"), display_id=True)}
        summary_path = self.training_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("training_finished"):
                self._html("<p><b>This session already finished training; showing saved results.</b></p>")
                return summary
        summary = trainer.run_training(self.store_root, self.cache_root, self.config, self.training_dir,
                                       resume=(self.training_dir / "run_contract.json").exists(),
                                       progress=self._train_progress)
        return summary

    def show_history(self):
        path = self.training_dir / "history.json"
        if not path.exists():
            self._html("<p>No epoch has completed yet.</p>")
            return []
        rows = json.loads(path.read_text(encoding="utf-8"))
        self._html(self._history_html(rows))
        import matplotlib.pyplot as plt
        figure = self._curves(rows)
        self._figure(figure, "learning_curves.png")
        plt.close(figure)
        return rows

    def show_results(self):
        path = self.training_dir / "summary.json"
        if not path.exists():
            self._html("<p>Training results are not available yet. No scores are invented.</p>")
            return None
        summary = json.loads(path.read_text(encoding="utf-8"))
        m = summary["validation_metrics"]
        number = lambda v, p=False: "—" if v is None else (f"{100 * v:.1f}%" if p else f"{v:.3f}")
        rows = [("Validation volumes (authentic / manipulated)", f"{m['authentic']} / {m['manipulated']}"),
                ("Selected epoch", summary["best_epoch"] or "provisional"), ("Tamper AUROC", number(m["tamper_auroc"])),
                ("Removal (FB) vs authentic AUROC", number(m["removal_auroc"])),
                ("Insertion (FM) vs authentic AUROC", number(m["insertion_auroc"])),
                ("Worst-generator AUROC", number(m["worst_generator_auroc"])),
                ("Removal vs insertion accuracy", number(m.get("removal_vs_insertion_accuracy"), True)),
                ("Router recall@K (site among selected crops)", number(m.get("router_recall_at_k"), True)),
                ("Mask Dice on crops containing the site", number(m.get("mask_dice_on_hit_crops")))]
        for name, point in m["operating_points"].items():
            rows.append((f"Removal / insertion recall at {name.replace('fpr_', '')} FPR (threshold from these authentic cases)",
                         f"{number(point['removal_recall'], True)} / {number(point['insertion_recall'], True)}"))
        self._html("<h3>Selected model — validation</h3>" + table_html(["Metric", "Value"], rows))
        self._html(table_html(["Generator", "AUROC vs authentic"], [(g, number(v)) for g, v in m["per_generator_auroc"].items()]))
        self._html("<p>Operating points are exploratory: the thresholds come from the same validation authentic volumes. "
                   "Test volumes are not acquired or opened.</p>")
        self._save_json("results_overview.json", {"validation": m, "best_epoch": summary["best_epoch"],
                                                  "test": {"status": "not_acquired_reserved", "volumes_opened": 0}})
        return summary

    def show_test_status(self):
        self._html(table_html(["Test set", "Status"], [("Publisher test patients", 150),
            ("Acquired / opened by this notebook", "0 / 0"), ("Metrics", "Not evaluated"),
            ("Reason", "Keep test independent of training, checkpoint selection and threshold choice")]))

    def verify_saved(self):
        summary = self.training_dir / "summary.json"
        text = f"# RSVT coarse-to-fine session\n\nOutput: `{self.root}`\n\nStore: `{self.store_root}`\n\n"
        if summary.exists():
            m = json.loads(summary.read_text(encoding="utf-8"))["validation_metrics"]
            text += f"Validation tamper AUROC {m['tamper_auroc']}; removal AUROC {m['removal_auroc']}; worst generator {m['worst_generator_auroc']}.\n\n"
        else:
            text += "Training has not completed yet.\n\n"
        text += "Test volumes acquired/opened: 0.\n"
        (self.root / "run_overview.md").write_text(text, encoding="utf-8")
        if (self.root / "run_overview.md").read_text(encoding="utf-8") != text:
            raise OSError("Run report readback failed")
        if self.training_dir.exists() and (self.training_dir / "summary.json").exists():
            trainer.persist_receipt(self.training_dir)
        records = {}
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "notebook_receipt.json" and path.suffix in (".json", ".toml", ".md", ".png", ".pt"):
                records[path.relative_to(self.root).as_posix()] = {"sha256": sha256_path(path), "bytes": path.stat().st_size}
        write_json(self.root / "notebook_receipt.json", {"schema_version": "m3d_notebook_receipt_v1", "files": records,
                                                         "test_volumes_opened": 0})
        for name, info in records.items():
            if sha256_path(self.root / name) != info["sha256"]:
                raise OSError("Drive readback mismatch: " + name)
        self._html("<p><b>Saved and verified in Google Drive:</b><br>" + escape(str(self.root)) + "</p>")
        return records


def create_session(project, code_root, **settings):
    return M3DNotebookSession(project, code_root, **settings)
