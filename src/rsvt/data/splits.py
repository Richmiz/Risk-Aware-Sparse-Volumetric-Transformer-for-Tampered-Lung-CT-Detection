"""Deterministic draft partitions over conservative evidence groups."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import PurePosixPath
import math

CLASSES = ("TB", "TM", "FB", "FM")
CLASS_INDEX = {label: index for index, label in enumerate(CLASSES)}


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def evidence_groups(records: list[dict], overlap: dict) -> dict[str, str]:
    """Union patient/series/source/content matches and shared DICOM-file evidence.

    Metadata patient keys remain provisional. Validity-mask hashes deliberately
    do not establish source identity: unrelated all-valid masks can be identical.
    """
    ids = [r["source_candidate_id"] for r in records]
    if len(ids) != len(set(ids)) or any(not key for key in ids):
        raise ValueError("Source candidates must be unique and nonempty")
    parent = {key: key for key in ids}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(keys):
        keys = list(keys)
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])

    seen, directories = {}, {}
    for row in records:
        key = row["source_candidate_id"]
        directory = row["relative_directory"].replace("\\", "/")
        if directory in directories and directories[directory] != key:
            raise ValueError("One source directory maps to multiple candidates")
        directories[directory] = key
        for kind in ("patient_metadata_keys", "series_metadata_key", "source_file_set_sha256",
                     "normalized_sha256", "f02_source_group_candidate"):
            for value in str(row.get(kind, "")).split(";"):
                if not value:
                    continue
                token = (kind, value)
                if token in seen:
                    union([seen[token], key])
                seen[token] = key
    for duplicate in overlap.get("exact_file_duplicates", []):
        matched = []
        for item in duplicate["files"]:
            if item["dataset"] == "ct_gan":
                directory = str(PurePosixPath(item["path"].replace("\\", "/")).parent)
                if directory in directories:
                    matched.append(directories[directory])
        union(matched)
    grouped = defaultdict(list)
    for key in ids:
        grouped[find(key)].append(key)
    return {key: "ctgan-f03-group-" + digest_json(sorted(members))[:20]
            for members in grouped.values() for key in members}


def validate_fractions(fractions: dict) -> None:
    if (len(fractions) < 2 or not set(fractions).issubset({"train", "validation", "test"})
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                   for v in fractions.values())
            or not math.isclose(sum(fractions.values()), 1., rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Partition fractions must be positive, finite, and sum to one")


def grouped_split(records: list[dict], fractions: dict, seed: int) -> list[dict]:
    """Minimize class/size imbalance without splitting a group or choosing on model outcomes."""
    validate_fractions(fractions)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")
    if len({r["case_key"] for r in records}) != len(records):
        raise ValueError("Duplicate case identity")
    groups = defaultdict(list)
    for row in records:
        if row["label_candidate"] not in CLASSES or not row["source_group_candidate"]:
            raise ValueError("Every row requires a known candidate label and evidence group")
        groups[row["source_group_candidate"]].append(row)
    groups = {key: sorted(members, key=lambda r: r["case_key"]) for key, members in sorted(groups.items())}
    parts = [name for name in ("train", "validation", "test") if name in fractions]
    if len(groups) < len(parts):
        raise ValueError("Too few independent evidence groups for requested partitions")
    total = Counter(r["label_candidate"] for r in records)
    support = Counter(label for members in groups.values() for label in {r["label_candidate"] for r in members})
    vectors = {key: Counter(r["label_candidate"] for r in members) for key, members in groups.items()}
    counts = {part: Counter() for part in parts}
    sizes, group_counts = Counter(), Counter()

    def objective():
        value = 0.
        for part in parts:
            for label in CLASSES:
                count = total[label]
                if not count:
                    continue
                target = count * fractions[part]
                value += (counts[part][label] - target)**2 / (target + 1.)
                if support[label] >= len(parts) and counts[part][label] == 0:
                    value += 5.
            target = len(records) * fractions[part]
            value += .2 * (sizes[part] - target)**2 / (target + 1.)
        return value

    def change(key, part, sign):
        for label, count in vectors[key].items():
            counts[part][label] += sign * count
        sizes[part] += sign * len(groups[key])
        group_counts[part] += sign

    order = sorted(groups, key=lambda key: (-len(groups[key]),
                   -sum(1 / total[r["label_candidate"]] for r in groups[key]), digest_json([seed, key])))
    assignments = {}
    for index, key in enumerate(order):
        empty = [part for part in parts if not group_counts[part]]
        eligible = empty if len(order) - index == len(empty) else parts
        options = []
        for part in eligible:
            change(key, part, 1)
            options.append((objective(), digest_json([seed, key, part]), part))
            change(key, part, -1)
        chosen = min(options)[2]
        change(key, chosen, 1)
        assignments[key] = chosen
    # Deterministic local improvements; no searching seeds or inspecting predictions.
    for _ in range(10):
        improved = False
        for key in order:
            before = assignments[key]
            if group_counts[before] <= 1:
                continue
            baseline, best = objective(), before
            change(key, before, -1)
            for after in parts:
                change(key, after, 1)
                cost = objective()
                change(key, after, -1)
                if cost < baseline - 1e-10:
                    baseline, best = cost, after
            change(key, best, 1)
            assignments[key] = best
            improved |= best != before
        if not improved:
            break
    return [{**row, "split": assignments[row["source_group_candidate"]]}
            for row in sorted(records, key=lambda r: r["case_key"])]


def audit_partitions(records: list[dict]) -> dict:
    seen = defaultdict(set)
    for row in records:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Invalid partition name")
        for kind in ("source_group_candidate", "f02_source_group_candidate", "source_candidate_id", "patient_metadata_keys",
                     "series_metadata_key", "source_file_set_sha256", "normalized_sha256"):
            for value in str(row.get(kind, "")).split(";"):
                if value:
                    seen[kind, value].add(row["split"])
    leaks = [{"evidence_type": kind, "identity": value, "partitions": sorted(parts)}
             for (kind, value), parts in sorted(seen.items()) if len(parts) > 1]
    distribution = {}
    for part in ("train", "validation", "test"):
        own = [r for r in records if r["split"] == part]
        distribution[part] = {"cases": len(own), "groups": len({r["source_group_candidate"] for r in own}),
                              "classes": dict(Counter(r["label_candidate"] for r in own)),
                              "generators": dict(Counter(r["generator_candidate"] or "authentic" for r in own))}
    return {"ok": not leaks, "known_evidence_leaks": leaks, "distribution": distribution,
            "independent_patient_count_verified": False,
            "missing_classes": {part: sorted(set(CLASSES) - set(info["classes"]))
                                for part, info in distribution.items() if info["cases"]}}


def leave_one_generator_out(records: list[dict], generator: str, *, seed=13, validation_fraction=.2) -> list[dict]:
    generators = {r["generator_candidate"] for r in records if r["generator_candidate"]}
    if len(generators) < 2 or generator not in generators:
        raise ValueError("LOGO requires at least two manipulated generator identities and a present holdout")
    held = {r["source_group_candidate"] for r in records if r["generator_candidate"] == generator}
    remaining = [r for r in records if r["source_group_candidate"] not in held]
    if not any(r["generator_candidate"] for r in remaining):
        raise ValueError("No manipulated training sources remain after holding out source groups")
    partitions = grouped_split(remaining, {"train": 1-validation_fraction, "validation": validation_fraction}, seed)
    partitions += [{**r, "split": "test"} for r in records if r["source_group_candidate"] in held]
    if not audit_partitions(partitions)["ok"]:
        raise ValueError("LOGO source evidence crosses partitions")
    return sorted(partitions, key=lambda r: r["case_key"])
