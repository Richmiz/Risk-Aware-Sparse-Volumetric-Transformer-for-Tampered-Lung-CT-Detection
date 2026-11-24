"""Procedural DICOMs only: guard coordinate conventions and evidence claims."""

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rsvt.data.annotation_qa import (checked_dicom, coordinate_hypotheses, grayscale_png,
    header_evidence, native_xy_to_lps, persist_receipt, record_coordinate_reviews, render_point, run_annotation_qa)


def source(root, directory, number):
    path = root / directory / (str(number) + ".dcm")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = f"1.2.3.4.{number+1}"
    d = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0"*128)
    d.SOPClassUID, d.SOPInstanceUID = CTImageStorage, meta.MediaStorageSOPInstanceUID
    d.PatientID, d.SeriesInstanceUID, d.StudyInstanceUID = "procedural", "1.2.3.99", "1.2.3.88"
    d.Rows, d.Columns = 6, 8
    d.SamplesPerPixel, d.PhotometricInterpretation = 1, "MONOCHROME2"
    d.BitsAllocated, d.BitsStored, d.HighBit, d.PixelRepresentation = 16, 16, 15, 1
    d.RescaleSlope, d.RescaleIntercept = 2, -1000
    d.ImagePositionPatient, d.ImageOrientationPatient, d.PixelSpacing = [10,20,number*3], [0,1,0,-1,0,0], [2, .5]
    d.PixelData = (np.arange(48,dtype=np.int16).reshape(6,8) + number).tobytes()
    d.save_as(path, enforce_file_format=True)
    return {"dataset":"ct_gan", "kind":"dicom", "relative_path":path.relative_to(root).as_posix(),
            "bytes":path.stat().st_size, "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


class CoordinateQA(unittest.TestCase):
    def test_axes_and_index_bases_are_independent_and_never_selected(self):
        h = coordinate_hypotheses({"slice": 4, "x": 2, "y": 5})
        self.assertEqual(len(h), 8)
        self.assertEqual(h[0], {"hypothesis":"slice_base0_xy_base0_native", "slice_filename":"4.dcm", "x":2, "y":5})
        self.assertEqual(h[-1], {"hypothesis":"slice_base1_xy_base1_swap", "slice_filename":"3.dcm", "x":4, "y":1})

    def test_lps_uses_dicom_column_spacing_and_rotated_orientation(self):
        d = Dataset()
        d.ImagePositionPatient, d.ImageOrientationPatient, d.PixelSpacing = [10,20,30],[0,1,0,-1,0,0],[2,.5]
        self.assertEqual(native_xy_to_lps(d, 4, 3), [4,22,30])
        d.ImageOrientationPatient = [1,0,0,1,0,0]
        with self.assertRaises(ValueError): native_xy_to_lps(d, 4, 3)

    def test_display_png_preserves_declared_window_and_dimensions(self):
        png = grayscale_png(np.array([[-2000.,-1000.,-300.,400.,900.]]))
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(struct.unpack(">II", png[16:24]), (5,1))
        start = png.index(b"IDAT")
        size = struct.unpack(">I", png[start-4:start])[0]
        self.assertEqual(list(zlib.decompress(png[start+4:start+4+size])), [0,0,0,128,255,255])

    def test_dicom_hash_precedes_decoding_and_preview_records_bounds(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows={str(i)+".dcm":source(root,"scan",i) for i in (0,1)}
            html,h=render_point({"case_key":"fixture", "label":"FB", "slice":1,"x":7,"y":5},rows,root)
            self.assertIn("absence of a nodule", html)
            self.assertIn("data:image/png;base64",html)
            self.assertEqual(sum(r["status"]=="out_of_bounds" for r in h), 4)
            self.assertEqual(h[0]["point_LPS_mm"], [0,23.5,3])
            rows["1.dcm"]["sha256"]="bad"
            with patch("rsvt.data.annotation_qa.pydicom.dcmread") as read:
                with self.assertRaisesRegex(ValueError,"F01"):checked_dicom(root,rows["1.dcm"],pixels=True)
                read.assert_not_called()

    def test_catalog_candidates_are_not_canonical_patients_and_conflicts_preserved(self):
        catalog=[{"PatientID":"LIDC-IDRI-0001","StudyInstanceUID":"1.2.3.8","SeriesInstanceUID":"1.2.3.9","SeriesNumber":4,"ImageCount":9}]
        d=Dataset();d.SeriesNumber=4
        self.assertEqual(header_evidence([d],catalog,9)["status"],"metadata_candidates_only")
        self.assertEqual(header_evidence([d],catalog,9)["public_patient_UID_evidence"],"")
        d.StudyInstanceUID="1.2.3.8"
        result=header_evidence([d],catalog,9)
        self.assertEqual(result["status"],"exact_catalog_study_UID_patient_evidence")
        self.assertEqual(result["exact_public_series_UIDs"],[])
        self.assertFalse(result["independent_patient_verified"])
        d.PatientID="LIDC-IDRI-0002"
        catalog.append({**catalog[0],"PatientID":"LIDC-IDRI-0002","StudyInstanceUID":"1.2.3.18","SeriesInstanceUID":"1.2.3.19"})
        self.assertTrue(header_evidence([d],catalog,9)["conflict"])
        self.assertEqual(header_evidence([d],catalog,9)["public_patient_UID_evidence"],"")

    def test_no_private_header_values_exported_and_reference_uids_found(self):
        d=Dataset();d.PatientID="private-patient";d.SeriesInstanceUID="1.2.3.456"
        child=Dataset();child.SeriesInstanceUID="1.2.3.9"
        d.ReferencedSeriesSequence=[child]
        result=header_evidence([d],[{"PatientID":"LIDC-IDRI-0001","SeriesInstanceUID":"1.2.3.9","ImageCount":1}],1)
        self.assertEqual(result["status"],"exact_catalog_series_UID_evidence")
        self.assertNotIn("private-patient",json.dumps(result))
        self.assertNotIn("1.2.3.456",json.dumps(result))

    def test_receipt_requires_drive_subdirectory_and_hashes_all_previews(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"drive"/"run";run.mkdir(parents=True)
            (run/"preview.html").write_text("fixture")
            r=persist_receipt(run,root/"drive")
            self.assertIn("preview.html",r["files"])
            self.assertTrue(r["all_artifacts_read_back"])
            with self.assertRaises(ValueError):persist_receipt(run,root/"other")
            with patch("rsvt.data.annotation_qa.sha256_file",side_effect=["a","b"]):
                with self.assertRaisesRegex(ValueError,"readback"):persist_receipt(run,root/"drive")

    def test_review_requires_documented_valid_hypothesis_and_does_not_unlock_training(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"run";run.mkdir()
            (run/"points.json").write_text(json.dumps([{"point_id":"p","hypotheses":[{"hypothesis":"native","status":"review_pending"}]}]))
            d={"point_id":"p","status":"accepted","selected_hypothesis":"native","reviewer":"fixture","evidence":"procedural target"}
            self.assertFalse(record_coordinate_reviews(run,[d],drive_results_root=root)["training_ready"])
            for changed in ({"evidence":""},{"selected_hypothesis":"missing"},{"point_id":"unknown"}):
                with self.assertRaises(ValueError):record_coordinate_reviews(run,[{**d,**changed}],drive_results_root=root)

    def test_pipeline_excludes_test_pixels_preserves_manifest_and_persists_metadata_only_zip(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data";rows=[];records=[];labels=[]
            for scan,label,split in (("1","TB","train"),("2","TM","train"),("3","FB","validation"),("4","FM","test")):
                directory="Tampered Scans/Experiment 1 - Blind/"+scan
                rows.extend(source(data,directory,i) for i in (0,1))
                records.append({"case_key":"exp1:"+scan,"split":split})
                labels.append({"type":label,"uuid":scan,"slice":0 if label=="TB" else 1,"x":0 if label=="TB" else 2,"y":0 if label=="TB" else 3})
            def table(path, items):
                with path.open("w",newline="") as stream:
                    w=csv.DictWriter(stream,items[0].keys());w.writeheader();w.writerows(items)
            table(data/"Tampered Scans/labels_exp1.csv",labels)
            # A second empty experiment is represented by a header-only table.
            table(data/"Tampered Scans/labels_exp2.csv",labels[:1]);p=data/"Tampered Scans/labels_exp2.csv"
            p.write_text("type,uuid,slice,x,y\n")
            table(root/"files.csv",rows)
            manifest=root/"manifest.json";manifest.write_text(json.dumps({"records":records}));original=manifest.read_bytes()
            policy={"expected_scan_counts":{"exp1":{c:1 for c in ("TB","TM","FB","FM")},"exp2":{}},
                    "label_file_sha256":{e:sha256_file(data/f"Tampered Scans/labels_{e}.csv") for e in ("exp1","exp2")}}
            (root/"policy.json").write_text(json.dumps(policy));(root/"catalog.json").write_text("[]")
            real=checked_dicom;decoded=[]
            def guard(dataset_root,row,*,pixels=False):
                if pixels:decoded.append(row["relative_path"])
                if pixels and "/4/" in row["relative_path"]:raise AssertionError("test pixel access")
                return real(dataset_root,row,pixels=pixels)
            with patch("rsvt.data.annotation_qa.require_colab_runtime"),patch("rsvt.data.annotation_qa.checked_dicom",side_effect=guard):
                result=run_annotation_qa(dataset_root=data,manifest_path=manifest,files_path=root/"files.csv",catalog_path=root/"catalog.json",
                    policy_path=root/"policy.json",output_dir=root/"drive/run",drive_results_root=root/"drive",code_sha256="fixture")
            self.assertEqual(result["preview_points"],2);self.assertEqual(result["test_points_excluded"],1)
            self.assertTrue(decoded);self.assertFalse(result["training_ready"])
            self.assertEqual(manifest.read_bytes(),original)
            import zipfile
            with zipfile.ZipFile(root/"drive/run/metadata.review.zip") as z:
                self.assertTrue(all(n.endswith(".json") for n in z.namelist()))


from rsvt.data.acquisition import sha256_file

if __name__ == "__main__": unittest.main()
