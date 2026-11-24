"""Review persistence and widget callbacks use fabricated HTML/points, never CT."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from rsvt.data.annotation_review import ReviewSession, ReviewSource, digest
from rsvt.data.annotation_review_ui import AnnotationReviewForm


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source_root=self.root/"qa_fixture"
        self.source_root.mkdir();(self.source_root/"previews").mkdir()
        self.source=self.make_source()
        self.sessions=self.root/"reviews"

    def make_source(self,count=3,test_point=False):
        points=[]
        for i in range(count):
            identifier=f"{i:020x}"
            point={"point_id":identifier,"case_key":f"fixture:{i}","split":"train" if i<count-1 else "validation",
                   "label":"TM","slice":2,"x":3,"y":4,"preview":"previews/point_"+identifier+".html",
                   "hypotheses":[{"hypothesis":f"h{n}","status":"review_pending" if n<7 else "out_of_bounds"} for n in range(8)]}
            if test_point and i==0:point["split"]="test"
            points.append(point)
            (self.source_root/point["preview"]).write_text(f"<b>Procedural preview {i}</b>",encoding="utf-8")
        summary={"preview_points":count,"test_points_excluded":25,"test_pixels_decoded":False,
                 "training_ready":False,"code_bundle_sha256":"fixture-source"}
        (self.source_root/"points.json").write_text(json.dumps(points),encoding="utf-8")
        (self.source_root/"summary.json").write_text(json.dumps(summary),encoding="utf-8")
        files={p.relative_to(self.source_root).as_posix():{"bytes":p.stat().st_size,"sha256":digest(p.read_bytes())}
               for p in self.source_root.rglob("*") if p.is_file() and p.name!="drive_persistence_receipt.json"}
        receipt={"all_artifacts_read_back":True,"training_ready":False,"files":files}
        raw=json.dumps(receipt).encode();(self.source_root/"drive_persistence_receipt.json").write_bytes(raw)
        self.source_kwargs={"receipt_sha256":digest(raw),"points_sha256":files["points.json"]["sha256"],
            "count":count,"split_counts":{"train":count-1,"validation":1},"source_bundle_sha256":"fixture-source"}
        return ReviewSource(self.source_root,**self.source_kwargs)

    def session(self,**kwargs):
        return ReviewSession.open(self.source,self.sessions,code_sha256="fixture-review",**kwargs)

    def decision(self,i=0,status="accepted",**changes):
        return {"point_id":self.source.points[i]["point_id"],"status":status,
                "selected_hypothesis":"h0" if status=="accepted" else None,
                "reviewer":"Procedural reviewer","evidence":"Synthetic fixture evidence",**changes}

    def latest(self,s):
        paths=sorted((s.root/"commits").glob("*.json"))
        return s.root/json.loads(paths[-1].read_text())["revision"]

    def form(self,s):
        with patch("rsvt.data.annotation_review_ui.display"),patch("rsvt.data.annotation_review_ui.clear_output"):
            return AnnotationReviewForm(s)

    def test_initial_state_has_no_automatic_decisions_and_source_unchanged(self):
        before={p.relative_to(self.source_root).as_posix():digest(p.read_bytes()) for p in self.source_root.rglob("*") if p.is_file()}
        s=self.session()
        self.assertEqual(s.decisions,{})
        summary=json.loads((self.latest(s)/"summary.json").read_text())
        self.assertEqual(summary["counts"],{"accepted":0,"ambiguous":0,"rejected":0,"pending":3})
        self.assertFalse(summary["training_ready"])
        self.assertEqual(before,{p.relative_to(self.source_root).as_posix():digest(p.read_bytes()) for p in self.source_root.rglob("*") if p.is_file()})

    def test_receipt_and_points_checksum_rejected(self):
        with self.assertRaisesRegex(ValueError,"receipt checksum"):
            ReviewSource(self.source_root,**{**self.source_kwargs,"receipt_sha256":"bad"})
        with (self.source_root/"points.json").open("ab") as f:f.write(b" ")
        with self.assertRaisesRegex(ValueError,"Checksum mismatch"):ReviewSource(self.source_root,**self.source_kwargs)

    def test_preview_is_verified_before_display(self):
        s=self.session();form=self.form(s);first=form.active
        next_point=self.source.points[1]
        (self.source_root/next_point["preview"]).write_text("CHANGED")
        with patch("rsvt.data.annotation_review_ui.display") as display:
            form.point.value=next_point["point_id"]
            display.assert_not_called()
        self.assertEqual(form.active,first);self.assertEqual(form.point.value,first)
        self.assertIn("Checksum mismatch",form.message.value)

    def test_test_points_are_rejected(self):
        with self.assertRaises(ValueError):self.make_source(test_point=True)

    def test_output_cannot_overlap_the_historical_source_audit(self):
        with self.assertRaisesRegex(ValueError,"separate"):
            ReviewSession.open(self.source,self.source_root,code_sha256="fixture-review")

    def test_exact_123_point_default_contract_and_split_counts(self):
        source=self.make_source(count=123)
        points=source.points
        for i,p in enumerate(points):p["split"]="train" if i<110 else "validation"
        raw=json.dumps(points).encode();(self.source_root/"points.json").write_bytes(raw)
        receipt=json.loads((self.source_root/"drive_persistence_receipt.json").read_text())
        receipt["files"]["points.json"]={"bytes":len(raw),"sha256":digest(raw)}
        receipt_raw=json.dumps(receipt).encode();(self.source_root/"drive_persistence_receipt.json").write_bytes(receipt_raw)
        loaded=ReviewSource(self.source_root,receipt_sha256=digest(receipt_raw),points_sha256=digest(raw),source_bundle_sha256="fixture-source")
        self.assertEqual(len(loaded.points),123)
        self.assertEqual(sum(p["split"]=="validation" for p in loaded.points),13)

    def test_invalid_id_hypothesis_status_and_missing_reviewer_evidence(self):
        s=self.session();initial=s.commit_sha256
        for change in ({"point_id":"unknown"},{"selected_hypothesis":"h7"},{"selected_hypothesis":None},
                       {"status":"ambiguous","selected_hypothesis":"h0"},{"status":"pending"},
                       {"reviewer":""},{"reviewer":None},{"evidence":"  "},{"evidence":None}):
            with self.subTest(change=change),self.assertRaises(ValueError):s.save(self.decision(**change))
        self.assertEqual(s.commit_sha256,initial)

    def test_save_resume_advance_and_keep_prior_revision_on_edit(self):
        s=self.session();s.save(self.decision())
        first_revision=self.latest(s);old=(first_revision/"review_state.json").read_bytes()
        self.assertEqual(s.active_point_id,self.source.points[1]["point_id"])
        resumed=self.session();self.assertEqual(resumed.root,s.root);self.assertEqual(resumed.decisions,s.decisions)
        resumed.save(self.decision(status="ambiguous",evidence="Index base remains ambiguous"))
        self.assertEqual(resumed.decisions[self.source.points[0]["point_id"]]["status"],"ambiguous")
        self.assertEqual((first_revision/"review_state.json").read_bytes(),old)

    def test_all_recorded_can_include_ambiguous_and_rejected_without_readiness(self):
        s=self.session()
        for i,status in enumerate(("accepted","ambiguous","rejected")):s.save(self.decision(i,status))
        folder=self.latest(s);summary=json.loads((folder/"summary.json").read_text())
        self.assertTrue(summary["review_recorded_complete"]);self.assertFalse(summary["all_points_accepted"])
        self.assertEqual(summary["unvalidated_points"],2);self.assertFalse(summary["training_ready"])
        with (folder/"decisions.csv").open(newline="") as f:rows=list(csv.DictReader(f))
        self.assertEqual([r["status"] for r in rows],["accepted","ambiguous","rejected"])

    def test_failed_revision_does_not_change_active_or_resumed_decisions(self):
        s=self.session();active=s.active_point_id;commit=s.commit_sha256
        with patch("rsvt.data.annotation_review.persist_receipt",side_effect=OSError("Drive unavailable")):
            with self.assertRaises(OSError):s.save(self.decision())
        self.assertEqual(s.active_point_id,active);self.assertEqual(s.commit_sha256,commit)
        self.assertEqual(self.session().decisions,{})
        s.save(self.decision());self.assertEqual(len(s.decisions),1)

    def test_orphan_revision_and_partial_commit_are_ignored_on_resume(self):
        s=self.session()
        with patch.object(s,"_publish_commit",side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):s.save(self.decision())
        (s.root/"commits/999999_partial.pending").write_text("not complete")
        self.assertEqual(self.session().decisions,{})

    def test_interrupted_initial_revision_resumes_without_creating_decisions(self):
        with patch.object(ReviewSession,"_publish_commit",side_effect=OSError("initial save interrupted")):
            with self.assertRaises(OSError):self.session()
        resumed=self.session()
        self.assertEqual(resumed.sequence,0)
        self.assertEqual(resumed.decisions,{})

    def test_committed_corruption_is_rejected_instead_of_silently_resetting(self):
        s=self.session();s.save(self.decision())
        (self.latest(s)/"review_state.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"Checksum mismatch"):self.session()

    def test_stale_writer_and_source_code_change_rejected(self):
        s1=self.session();s2=self.session();s1.save(self.decision())
        with self.assertRaisesRegex(ValueError,"another writer"):s2.save(self.decision(1))
        with self.assertRaises(ValueError):ReviewSession.open(self.source,self.sessions,code_sha256="changed")

    def test_widget_save_failure_retains_current_fields_and_has_no_success_message(self):
        s=self.session();form=self.form(s);active=form.active
        form.status.value="accepted";form.hypothesis.value="h0"
        form.reviewer.value="reviewer";form.evidence.value="evidence"
        original=form._fields()
        with patch.object(s,"save",side_effect=OSError("Drive failed")):form.save_button.click()
        self.assertEqual(form.active,active);self.assertEqual(form._fields(),original)
        self.assertIn("Gagal menyimpan",form.message.value)
        self.assertNotIn("tersimpan",form.message.value)
        self.assertFalse(form.save_button.disabled)

    def test_widget_navigation_cannot_discard_unsaved_changes(self):
        form=self.form(self.session());active=form.active
        self.assertIsNone(form.status.value);self.assertIsNone(form.hypothesis.value)
        form.evidence.value="unsaved note"
        form.point.value=self.source.points[1]["point_id"]
        self.assertEqual(form.active,active);self.assertEqual(form.point.value,active)
        self.assertEqual(form.evidence.value,"unsaved note")

    def test_widget_saves_only_explicit_decision_and_goes_to_next_pending(self):
        form=self.form(self.session())
        form.status.value="ambiguous";self.assertTrue(form.hypothesis.disabled)
        form.reviewer.value="reviewer";form.evidence.value="uncertain voxel origin"
        with patch("rsvt.data.annotation_review_ui.display"),patch("rsvt.data.annotation_review_ui.clear_output"):
            form.save_button.click()
        self.assertEqual(len(form.session.decisions),1)
        self.assertEqual(form.active,self.source.points[1]["point_id"])
        self.assertIsNone(form.status.value);self.assertIsNone(form.hypothesis.value)
        self.assertIn("tersimpan",form.message.value)
        self.assertEqual(form.reviewer.value,"reviewer")


if __name__=="__main__":unittest.main()
