"""Forensic classes are dataset labels, not independent clinical diagnoses."""

from enum import Enum


class ForensicClass(str, Enum):
    TRUE_BENIGN = "TB"
    TRUE_MALIGNANT = "TM"
    FAKE_BENIGN = "FB"
    FAKE_MALIGNANT = "FM"


MANIPULATION_BY_CLASS = {"TB": "none", "TM": "none", "FB": "removal", "FM": "insertion"}
