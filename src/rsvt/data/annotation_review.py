"""Resumable human review of saved previews, with immutable verified revisions.

No raw DICOM loading, dataset acquisition, model execution, or source-audit writes.
Only a committed revision can become the resumed state. One Colab writer/session.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from uuid import uuid4

from rsvt.data.annotation_qa import persist_receipt, record_coordinate_reviews, validate_coordinate_reviews

SOURCE_RUN = "qa_20261113T005049_219029Z"
SOURCE_RECEIPT_SHA256 = "03724a00b731f8c13a01fb5e819253adddbd64b51714bff00cac3023c1cde596"
SOURCE_POINTS_SHA256 = "903ca75c3b91474de1c8ae64acb5a24884327d97a9a2fada943b5d234f64a375"
SOURCE_BUNDLE_SHA256 = "e189fa58fff594d5f077f882d55505f0ae27ba6d04040e1ffc1930012fca435f"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bound(root, relative, expected=None):
    root = Path(root).resolve()
    name = PurePosixPath(relative)
    raw_path = root / Path(*name.parts)
    path = raw_path.resolve()
    if (name.is_absolute() or ".." in name.parts or "\\" in relative or not path.is_relative_to(root)
            or any(p.is_symlink() for p in (raw_path, *raw_path.parents) if p != root)):
        raise ValueError("Unsafe review artifact path")
    raw = path.read_bytes()
    if expected and (len(raw) != expected["bytes"] or digest(raw) != expected["sha256"]):
        raise ValueError("Checksum mismatch: " + relative)
    return raw


def write_json(path, document):
    raw = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
    if Path(path).read_bytes() != raw:
        raise ValueError("Review artifact failed write/readback")
    return raw


class ReviewSource:
    """A pinned, read-only source audit; test points cannot enter the reviewer."""

    def __init__(self, root, *, receipt_sha256=SOURCE_RECEIPT_SHA256,
                 points_sha256=SOURCE_POINTS_SHA256, count=123, split_counts=None,
                 source_bundle_sha256=SOURCE_BUNDLE_SHA256):
        self.root = Path(root).resolve()
        receipt_raw = read_bound(self.root, "drive_persistence_receipt.json")
        if digest(receipt_raw) != receipt_sha256:
            raise ValueError("Source receipt checksum mismatch")
        self.receipt = json.loads(receipt_raw)
        if not self.receipt.get("all_artifacts_read_back") or self.receipt.get("training_ready") is not False:
            raise ValueError("Unexpected source receipt/readiness")
        self.points_raw = read_bound(self.root, "points.json", self.receipt["files"]["points.json"])
        if digest(self.points_raw) != points_sha256:
            raise ValueError("Source points checksum mismatch")
        self.points = json.loads(self.points_raw)
        self.by_id = {p["point_id"]: p for p in self.points}
        if len(self.points) != count or len(self.by_id) != count or count < 1:
            raise ValueError("Unexpected or repeated review points")
        splits = dict(Counter(p["split"] for p in self.points))
        if splits != (split_counts or {"train":110, "validation":13}):
            raise ValueError("Only the pinned train/validation points may be reviewed")
        if any(p["split"] not in {"train", "validation"} for p in self.points):
            raise ValueError("Test points are excluded")
        self.summary = json.loads(read_bound(self.root, "summary.json", self.receipt["files"]["summary.json"]))
        if (self.summary["code_bundle_sha256"] != source_bundle_sha256
                or self.summary["preview_points"] != count or self.summary["training_ready"] is not False
                or self.summary["test_points_excluded"] != 25 or self.summary["test_pixels_decoded"] is not False):
            raise ValueError("Source summary differs from the reviewed audit")
        for p in self.points:
            if (not re.fullmatch(r"[0-9a-f]{20}", p["point_id"])
                    or p["preview"] != "previews/point_" + p["point_id"] + ".html"
                    or p["preview"] not in self.receipt["files"] or len(p["hypotheses"]) != 8
                    or len({h["hypothesis"] for h in p["hypotheses"]}) != 8):
                raise ValueError("Unexpected preview or hypothesis binding")
        self.binding = {"source_run": self.root.name, "source_receipt_sha256":receipt_sha256,
                        "source_points_sha256":points_sha256, "source_bundle_sha256":source_bundle_sha256}

    def preview(self, point_id):
        if point_id not in self.by_id:
            raise ValueError("Unknown review point")
        name = self.by_id[point_id]["preview"]
        return read_bound(self.root, name, self.receipt["files"][name]).decode("utf-8")


class ReviewSession:
    def __init__(self, source, root, sessions_root, code_sha256):
        self.source, self.root = source, Path(root).resolve()
        self.sessions_root = Path(sessions_root).resolve()
        if self.sessions_root.is_relative_to(source.root) or source.root.is_relative_to(self.sessions_root):
            raise ValueError("Review output must be separate from the historical source audit")
        if self.root == self.sessions_root or not self.root.is_relative_to(self.sessions_root):
            raise ValueError("Review session must be inside the project Drive review directory")
        self.code_sha256 = code_sha256
        self.binding = {**source.binding, "review_code_sha256":code_sha256}
        self.state = None
        self.commit_sha256 = None
        self.sequence = -1

    @classmethod
    def open(cls, source, sessions_root, *, code_sha256, resume=True):
        """Resume the newest matching session, including an interrupted initial save."""
        sessions_root = Path(sessions_root).resolve()
        binding = {**source.binding, "review_code_sha256":code_sha256}
        if resume and sessions_root.exists():
            for path in sorted(sessions_root.glob("review_*"), reverse=True):
                if not path.is_dir() or not (path/"session.json").exists():
                    continue
                header = json.loads(read_bound(path,"session.json"))
                if header.get("source_run") != binding["source_run"]:
                    continue
                obj = cls(source,path,sessions_root,code_sha256)
                obj._load_header()
                if header["binding"] != binding:
                    raise ValueError("Existing session uses different source/code; preserve it and select a new session explicitly")
                if not list((path/"commits").glob("*.json")):
                    obj._commit({},source.points[0]["point_id"])
                else:
                    obj.reload()
                return obj
        root = sessions_root/datetime.now(timezone.utc).strftime("review_%Y%m%dT%H%M%S_%fZ")
        obj = cls(source,root,sessions_root,code_sha256)
        root.mkdir(parents=True)
        (root/"commits").mkdir()
        (root/"revisions").mkdir()
        raw = write_json(root/"session.json", {"schema":"annotation_review_session_v1", "binding":binding,
                                               "source_run":source.root.name, "source_directory":str(source.root)})
        points_path = root/"points.json"
        with points_path.open("xb") as stream: stream.write(source.points_raw)
        write_json(root/"session_receipt.json", {"session_sha256":digest(raw), "points_sha256":digest(source.points_raw)})
        obj._commit({},source.points[0]["point_id"])
        return obj

    def _load_header(self):
        receipt = json.loads(read_bound(self.root,"session_receipt.json"))
        raw = read_bound(self.root,"session.json")
        if digest(raw) != receipt["session_sha256"] or digest(read_bound(self.root,"points.json")) != receipt["points_sha256"]:
            raise ValueError("Session header checksum mismatch")
        header = json.loads(raw)
        if header["binding"] != self.binding or receipt["points_sha256"] != self.binding["source_points_sha256"]:
            raise ValueError("Session source/code binding mismatch")

    def _verified_revision(self, commit):
        folder = commit["revision"]
        if not re.fullmatch(r"revisions/[0-9]{6}_[0-9a-f]{32}",folder):
            raise ValueError("Invalid review revision path")
        receipt_raw = read_bound(self.root,folder+"/drive_persistence_receipt.json")
        if digest(receipt_raw) != commit["receipt_sha256"]:
            raise ValueError("Revision receipt checksum mismatch")
        receipt = json.loads(receipt_raw)
        if receipt.get("all_artifacts_read_back") is not True or receipt.get("training_ready") is not False:
            raise ValueError("Invalid revision receipt")
        for name,item in receipt["files"].items():
            read_bound(self.root,folder+"/"+name,item)
        base = self.root/folder
        actual={p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p.name!="drive_persistence_receipt.json"}
        if actual != set(receipt["files"]):
            raise ValueError("Revision contains unreceipted or missing files")
        state=json.loads(read_bound(base,"review_state.json",receipt["files"]["review_state.json"]))
        if (state["binding"] != self.binding or state["sequence"] != commit["sequence"]
                or state["active_point_id"] not in self.source.by_id
                or digest(read_bound(base,"points.json")) != self.binding["source_points_sha256"]):
            raise ValueError("Revision state/source binding mismatch")
        validate_coordinate_reviews(self.source.by_id,list(state["decisions"].values()))
        if any(k!=v["point_id"] for k,v in state["decisions"].items()):
            raise ValueError("Decision key mismatch")
        return state

    def reload(self):
        self._load_header()
        parent, sequence, state = None, -1, None
        for path in sorted((self.root/"commits").glob("*.json")):
            raw=read_bound(self.root,"commits/"+path.name)
            commit=json.loads(raw)
            if commit["sequence"] != sequence+1 or commit["parent_commit_sha256"] != parent:
                raise ValueError("Review commit conflict or missing revision; use one Colab writer")
            state=self._verified_revision(commit)
            parent, sequence = digest(raw), commit["sequence"]
        if state is None:
            raise ValueError("No completed review revision")
        self.state, self.commit_sha256, self.sequence = state,parent,sequence
        return self

    @property
    def active_point_id(self):
        return self.state["active_point_id"]

    @property
    def decisions(self):
        return deepcopy(self.state["decisions"])

    def _exports(self, folder, state):
        rows=[]
        for point in self.source.points:
            decision=state["decisions"].get(point["point_id"],{})
            rows.append({**{k:point[k] for k in ("point_id","case_key","split","label","slice","x","y")},
                         "status":decision.get("status","pending"),
                         **{k:decision.get(k,"") for k in ("selected_hypothesis","reviewer","evidence")}})
        counts={k:sum(r["status"]==k for r in rows) for k in ("accepted","ambiguous","rejected","pending")}
        summary={"schema":"annotation_review_summary_v1", "source_run":self.source.root.name,
                 "total_points":len(rows), "counts":counts, "review_recorded_complete":counts["pending"]==0,
                 "all_points_accepted":counts["accepted"]==len(rows),
                 "unvalidated_points":counts["pending"]+counts["ambiguous"]+counts["rejected"],
                 "test_points_excluded":25, "public_source_identities_resolved":0,
                 "README_slice_discrepancy":484, "training_ready":False,
                 "training_performed":False, "new_dataset_download_bytes":0,
                 "binding":self.binding, "sequence":state["sequence"]}
        write_json(folder/"summary.json",summary)
        with (folder/"decisions.csv").open("x",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]))
            writer.writeheader();writer.writerows(rows)
        report=("# Coordinate annotation review\n\n"
                f"Source run: {self.source.root.name}\n\n"
                + "\n".join(f"- {key}: {value}" for key,value in counts.items())
                + "\n\nAll points have a recorded decision: " + str(counts["pending"]==0)
                + "\n\nAmbiguous and rejected points remain unvalidated. "
                "Source identities are unresolved (0/100); the README discrepancy is 484 slices. "
                "Training readiness remains false. Test points were excluded and no dataset was downloaded.\n")
        with (folder/"report.md").open("x",encoding="utf-8") as stream: stream.write(report)

    def _publish_commit(self, commit):
        path=self.root/"commits"/(f"{commit['sequence']:06d}_"+uuid4().hex+".json")
        temporary=path.with_suffix(".pending")
        raw=write_json(temporary,commit)
        os.replace(temporary,path)
        if read_bound(self.root,"commits/"+path.name) != raw:
            raise ValueError("Commit failed Drive readback")

    def _commit(self, decisions, active_point_id):
        sequence=self.sequence+1
        folder=self.root/"revisions"/(f"{sequence:06d}_"+uuid4().hex)
        folder.mkdir()
        with (folder/"points.json").open("xb") as stream: stream.write(self.source.points_raw)
        ordered=[decisions[p["point_id"]] for p in self.source.points if p["point_id"] in decisions]
        # Reuse the existing decision validator/writer in this isolated revision.
        record_coordinate_reviews(folder,ordered,drive_results_root=self.sessions_root)
        state={"binding":self.binding,"sequence":sequence,"active_point_id":active_point_id,
               "decisions":decisions,"training_ready":False}
        write_json(folder/"review_state.json",state)
        self._exports(folder,state)
        persist_receipt(folder,self.sessions_root)
        receipt_raw=read_bound(folder,"drive_persistence_receipt.json")
        commit={"sequence":sequence,"parent_commit_sha256":self.commit_sha256,
                "revision":folder.relative_to(self.root).as_posix(),"receipt_sha256":digest(receipt_raw)}
        self._verified_revision(commit)
        self._publish_commit(commit)
        self.reload()
        return self

    def save(self, decision):
        """Only update memory/active point after a verified revision is committed."""
        decision=deepcopy(decision)
        validate_coordinate_reviews(self.source.by_id,[decision])
        previous=self.commit_sha256
        self.reload()
        if self.commit_sha256 != previous:
            raise ValueError("Session changed in another writer; reopen before saving")
        decisions=self.decisions
        decisions[decision["point_id"]]=decision
        keys=[p["point_id"] for p in self.source.points]
        i=keys.index(decision["point_id"])
        remaining=keys[i+1:]+keys[:i+1]
        active=next((k for k in remaining if k not in decisions),decision["point_id"])
        return self._commit(decisions,active)
