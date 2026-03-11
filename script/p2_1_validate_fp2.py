#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate OpenBabel FP2 outputs produced by p2_1_extract_2d_fp2.py.

Checks:
1) File-level integrity (memmap/meta/manifest/errors alignment)
2) Structural integrity (shape, key columns, uniqueness)
3) Fingerprint statistics (popcount distribution, zero vectors)
4) Exact bit-level verification by recomputing FP2 from sampled SMILES
5) Duplicate-SMILES consistency (Tanimoto must be 1.0)
6) Reproducibility support via sha256 baseline comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from openbabel import pybel
except ImportError:
    pybel = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = ROOT / "features_store"
DEFAULT_MEMMAP = DEFAULT_FEATURE_DIR / "fp2_uint64.memmap"
DEFAULT_META = DEFAULT_FEATURE_DIR / "fp2_meta.parquet"
DEFAULT_MANIFEST = DEFAULT_FEATURE_DIR / "manifest.json"
DEFAULT_ERRORS = DEFAULT_FEATURE_DIR / "fp2_errors.log"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            hasher.update(buf)
    return hasher.hexdigest()


def _infer_memmap_rows(memmap_path: Path, n_cols: int = 16, dtype=np.uint64) -> int:
    byte_size = memmap_path.stat().st_size
    row_bytes = np.dtype(dtype).itemsize * n_cols
    if byte_size % row_bytes != 0:
        raise ValueError(
            f"Memmap byte size ({byte_size}) is not divisible by row bytes ({row_bytes})."
        )
    return byte_size // row_bytes


def _make_popcount_lut() -> np.ndarray:
    return np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_rows_u64(chunk_u64: np.ndarray, lut: np.ndarray) -> np.ndarray:
    as_u8 = chunk_u64.view(np.uint8).reshape(chunk_u64.shape[0], -1)
    return lut[as_u8].sum(axis=1, dtype=np.uint16)


def _row_to_bitset(row_u64: np.ndarray) -> set[int]:
    bits: set[int] = set()
    for word_idx, word in enumerate(row_u64.tolist()):
        v = int(word)
        if v == 0:
            continue
        for offset in range(64):
            if (v >> offset) & 1:
                bits.add(word_idx * 64 + offset + 1)  # 1-indexed
    return bits


def _smiles_to_fp2_bitset(smiles: str, smiles_mode: str = "as_is", ph_value: float = 7.4) -> set[int]:
    input_smi = str(smiles)
    if smiles_mode == "ph74":
        ob_mol = pybel.readstring("smi", input_smi)
        ob_mol.OBMol.CorrectForPH(float(ph_value))
        input_smi = ob_mol.write("can").split()[0]
    mol = pybel.readstring("smi", input_smi)
    fp = mol.calcfp("fp2")
    return {int(b) for b in fp.bits if 1 <= int(b) <= 1024}


def _tanimoto_u64(a: np.ndarray, b: np.ndarray, lut: np.ndarray) -> float:
    inter = lut[np.bitwise_and(a, b).view(np.uint8)].sum(dtype=np.int64)
    union = lut[np.bitwise_or(a, b).view(np.uint8)].sum(dtype=np.int64)
    if union == 0:
        return 1.0
    return float(inter / union)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _count_error_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def validate(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {
        "validated_at": _now(),
        "inputs": {
            "memmap": str(args.memmap),
            "meta": str(args.meta),
            "manifest": str(args.manifest),
            "errors": str(args.errors),
            "smiles_col": args.smiles_col,
            "smiles_mode": args.smiles_mode,
            "ph_value": float(args.ph_value),
        },
        "checks": {},
        "warnings": [],
        "failed_checks": [],
    }

    required_files = [args.memmap, args.meta]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        report["failed_checks"].append("missing_required_files")
        report["checks"]["missing_required_files"] = {"ok": False, "missing": missing}
        return False, report

    manifest = _read_manifest(args.manifest)
    meta = pd.read_parquet(args.meta)
    rows = _infer_memmap_rows(args.memmap, n_cols=16, dtype=np.uint64)
    fp = np.memmap(args.memmap, dtype=np.uint64, mode="r", shape=(rows, 16))
    lut = _make_popcount_lut()

    shape_ok = True
    shape_details: dict[str, Any] = {
        "memmap_rows": int(rows),
        "memmap_cols": 16,
        "meta_rows": int(len(meta)),
    }
    if len(meta) != rows:
        shape_ok = False
    if manifest:
        shape_details["manifest_shape"] = manifest.get("shape")
        if "shape" in manifest and manifest["shape"] != [int(rows), 16]:
            shape_ok = False
        if "success" in manifest and int(manifest["success"]) != int(rows):
            shape_ok = False
    report["checks"]["shape_consistency"] = {"ok": shape_ok, **shape_details}
    if not shape_ok:
        report["failed_checks"].append("shape_consistency")

    required_cols = {"memmap_idx", "molregno", args.smiles_col}
    has_cols = required_cols.issubset(set(meta.columns))
    col_ok = bool(has_cols)
    if has_cols:
        idx_expected = np.arange(len(meta), dtype=np.int64)
        idx_ok = np.array_equal(meta["memmap_idx"].to_numpy(dtype=np.int64), idx_expected)
        unique_ok = meta["molregno"].nunique(dropna=False) == len(meta)
        null_ok = int(meta[list(required_cols)].isna().sum().sum()) == 0
        col_ok = idx_ok and unique_ok and null_ok
        report["checks"]["meta_integrity"] = {
            "ok": col_ok,
            "idx_sequential": bool(idx_ok),
            "molregno_unique": bool(unique_ok),
            "null_cells_required_columns": int(meta[list(required_cols)].isna().sum().sum()),
        }
    else:
        report["checks"]["meta_integrity"] = {
            "ok": False,
            "missing_columns": sorted(list(required_cols - set(meta.columns))),
        }
    if not col_ok:
        report["failed_checks"].append("meta_integrity")

    failed_n = _count_error_rows(args.errors)
    err_ok = True
    err_detail: dict[str, Any] = {"error_rows": int(failed_n)}
    if manifest and "failed" in manifest:
        manifest_failed = int(manifest["failed"])
        err_detail["manifest_failed"] = manifest_failed
        if manifest_failed != failed_n:
            err_ok = False
    report["checks"]["error_log_alignment"] = {"ok": err_ok, **err_detail}
    if not err_ok:
        report["failed_checks"].append("error_log_alignment")

    popcounts = np.empty(rows, dtype=np.uint16)
    for start in range(0, rows, args.chunk_rows):
        end = min(rows, start + args.chunk_rows)
        popcounts[start:end] = _popcount_rows_u64(fp[start:end], lut)

    zero_vec = int((popcounts == 0).sum())
    q = np.quantile(popcounts, [0.01, 0.5, 0.99]) if rows > 0 else np.array([0, 0, 0], dtype=float)
    stats_ok = zero_vec <= args.max_zero_vectors
    report["checks"]["popcount_stats"] = {
        "ok": stats_ok,
        "mean": float(popcounts.mean()) if rows else 0.0,
        "std": float(popcounts.std()) if rows else 0.0,
        "min": int(popcounts.min()) if rows else 0,
        "q01": float(q[0]),
        "median": float(q[1]),
        "q99": float(q[2]),
        "max": int(popcounts.max()) if rows else 0,
        "zero_vectors": zero_vec,
        "threshold_zero_vectors": int(args.max_zero_vectors),
    }
    if not stats_ok:
        report["failed_checks"].append("popcount_stats")

    sampled = min(args.sample_size, rows)
    exact_ok = True
    exact_detail: dict[str, Any] = {
        "sample_size": int(sampled),
        "mismatch_count": 0,
        "mismatch_examples": [],
    }
    if sampled > 0 and not args.skip_recompute:
        if pybel is None:
            exact_ok = False
            exact_detail["error"] = "openbabel.pybel is unavailable."
        else:
            rng = random.Random(args.seed)
            idxs = rng.sample(range(rows), sampled)
            for idx in idxs:
                row = fp[idx]
                smiles = str(meta.iloc[idx][args.smiles_col])
                molregno = int(meta.iloc[idx]["molregno"])
                recomputed = _smiles_to_fp2_bitset(
                    smiles,
                    smiles_mode=args.smiles_mode,
                    ph_value=args.ph_value,
                )
                packed = _row_to_bitset(row)
                if recomputed != packed:
                    exact_detail["mismatch_count"] += 1
                    if len(exact_detail["mismatch_examples"]) < 10:
                        only_recomputed = sorted(list(recomputed - packed))[:20]
                        only_packed = sorted(list(packed - recomputed))[:20]
                        exact_detail["mismatch_examples"].append(
                            {
                                "memmap_idx": int(idx),
                                "molregno": molregno,
                                "only_recomputed_first20": only_recomputed,
                                "only_packed_first20": only_packed,
                            }
                        )
            exact_ok = exact_detail["mismatch_count"] == 0
    else:
        exact_detail["skipped"] = bool(args.skip_recompute or sampled == 0)
    report["checks"]["exact_bit_recompute"] = {"ok": exact_ok, **exact_detail}
    if not exact_ok:
        report["failed_checks"].append("exact_bit_recompute")

    dup_ok = True
    dup_detail: dict[str, Any] = {
        "groups_with_duplicates": 0,
        "pairs_checked": 0,
        "non_unity_pairs": [],
    }
    if {args.smiles_col, "memmap_idx"}.issubset(meta.columns):
        vc = meta[args.smiles_col].value_counts()
        dup_smiles = vc[vc > 1].index.tolist()
        dup_detail["groups_with_duplicates"] = int(len(dup_smiles))
        if dup_smiles:
            for smi in dup_smiles:
                idx_list = meta.index[meta[args.smiles_col] == smi].tolist()
                if len(idx_list) < 2:
                    continue
                i0 = int(idx_list[0])
                for j0 in idx_list[1:]:
                    tan = _tanimoto_u64(fp[i0], fp[int(j0)], lut)
                    dup_detail["pairs_checked"] += 1
                    if not math.isclose(tan, 1.0, rel_tol=0.0, abs_tol=1e-12):
                        dup_ok = False
                        if len(dup_detail["non_unity_pairs"]) < 10:
                            dup_detail["non_unity_pairs"].append(
                                {
                                    "idx_a": i0,
                                    "idx_b": int(j0),
                                    "tanimoto": tan,
                                    "smiles": smi[:120],
                                }
                            )
                    if dup_detail["pairs_checked"] >= args.max_duplicate_pairs:
                        break
                if dup_detail["pairs_checked"] >= args.max_duplicate_pairs:
                    break
    report["checks"]["duplicate_smiles_consistency"] = {"ok": dup_ok, **dup_detail}
    if not dup_ok:
        report["failed_checks"].append("duplicate_smiles_consistency")

    is_monotonic = bool(meta["molregno"].is_monotonic_increasing) if "molregno" in meta.columns else False
    report["checks"]["ordering_observation"] = {
        "ok": True,
        "molregno_monotonic_increasing": is_monotonic,
        "note": (
            "If False, output ordering is likely affected by generator_unordered; "
            "this can break run-to-run deterministic row mapping."
        ),
    }

    hash_info = {
        "memmap_sha256": _sha256_file(args.memmap),
        "meta_sha256": _sha256_file(args.meta),
    }
    if args.manifest.exists():
        hash_info["manifest_sha256"] = _sha256_file(args.manifest)
    report["hashes"] = hash_info

    if args.baseline_json and args.baseline_json.exists():
        with args.baseline_json.open("r", encoding="utf-8") as f:
            baseline = json.load(f)
        base_hash = baseline.get("hashes", {})
        hash_match = all(hash_info.get(k) == base_hash.get(k) for k in hash_info.keys())
        report["checks"]["baseline_hash_match"] = {
            "ok": hash_match,
            "baseline": str(args.baseline_json),
            "baseline_hashes_present": sorted(list(base_hash.keys())),
        }
        if not hash_match:
            report["failed_checks"].append("baseline_hash_match")
    elif args.baseline_json:
        report["warnings"].append(
            f"Baseline file not found: {args.baseline_json}. Run with --write-baseline to create it."
        )

    ok = len(report["failed_checks"]) == 0
    report["overall_ok"] = ok
    return ok, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate FP2 memmap/meta/manifest outputs for sTP pipeline."
    )
    parser.add_argument("--memmap", type=Path, default=DEFAULT_MEMMAP, help="Path to fp2_uint64.memmap")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Path to fp2_meta.parquet")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to manifest.json")
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS, help="Path to fp2_errors.log")
    parser.add_argument("--sample-size", type=int, default=1000, help="Sample size for exact bit recompute checks.")
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="canonical_smiles",
        help="SMILES column in meta parquet to use for exact recompute and duplicate checks.",
    )
    parser.add_argument(
        "--smiles-mode",
        choices=["as_is", "ph74"],
        default="as_is",
        help="SMILES preprocessing mode before FP2 recompute. 'ph74' applies OpenBabel CorrectForPH().",
    )
    parser.add_argument(
        "--ph-value",
        type=float,
        default=7.4,
        help="pH value used when --smiles-mode=ph74.",
    )
    parser.add_argument("--seed", type=int, default=20260304, help="Random seed for sampling.")
    parser.add_argument("--chunk-rows", type=int, default=200000, help="Chunk rows for popcount scan.")
    parser.add_argument(
        "--max-zero-vectors",
        type=int,
        default=0,
        help="Maximum allowed all-zero fingerprints before failing.",
    )
    parser.add_argument(
        "--max-duplicate-pairs",
        type=int,
        default=5000,
        help="Maximum duplicate-SMILES pairs to check for tanimoto=1.",
    )
    parser.add_argument(
        "--skip-recompute",
        action="store_true",
        help="Skip exact recomputation from SMILES (OpenBabel check).",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="If provided, compare current hashes against baseline report JSON.",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        default=None,
        help="Write current report to this path for future reproducibility checks.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_FEATURE_DIR / "fp2_validation_report.json",
        help="Path to write validation report JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code if any check fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ok, report = validate(args)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.write_baseline is not None:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        with args.write_baseline.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== FP2 Validation Report ===")
    print(f"validated_at: {report['validated_at']}")
    print(f"overall_ok: {report['overall_ok']}")
    print(f"failed_checks: {report['failed_checks']}")
    print(f"report_json: {args.report_json}")
    if args.write_baseline is not None:
        print(f"baseline_written: {args.write_baseline}")

    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
