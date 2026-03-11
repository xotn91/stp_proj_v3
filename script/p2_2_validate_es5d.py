#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate ES5D outputs from p2_2_extract_3d_es5d_production.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Crippen

from openbabel import pybel

# Avoid nested BLAS/OpenMP threading (and SHM permission issues in restricted envs).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = ROOT / "features_store"
DEFAULT_MEMMAP = DEFAULT_FEATURE_DIR / "es5d_db_k20.memmap"
DEFAULT_META = DEFAULT_FEATURE_DIR / "es5d_meta_db.parquet"
DEFAULT_MANIFEST = DEFAULT_FEATURE_DIR / "es5d_db_manifest.json"
DEFAULT_ERRORS = DEFAULT_FEATURE_DIR / "es5d_db_errors.log"
DEFAULT_REPORT = DEFAULT_FEATURE_DIR / "es5d_validation_report.json"

TOP_K = 20
ES_DIM = 18
SEED = 42


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _count_error_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def _infer_memmap_rows(path: Path) -> int:
    row_bytes = np.dtype(np.float32).itemsize * TOP_K * ES_DIM
    size = path.stat().st_size
    if size % row_bytes != 0:
        raise ValueError(f"Invalid memmap size={size}, row_bytes={row_bytes}")
    return size // row_bytes


def _compute_es5d_reference(coords_5d: np.ndarray) -> np.ndarray:
    if coords_5d.shape[0] < 4:
        raise ValueError("Atoms too few for ES5D")

    c1 = coords_5d.mean(axis=0)
    d1 = np.sqrt(((coords_5d - c1) ** 2).sum(axis=1))
    c2 = coords_5d[int(np.argmax(d1))]
    d2 = np.sqrt(((coords_5d - c2) ** 2).sum(axis=1))
    c3 = coords_5d[int(np.argmax(d2))]

    va3 = (c2 - c1)[:3]
    vb3 = (c3 - c1)[:3]
    cp = np.cross(va3, vb3)
    cp_norm = np.linalg.norm(cp)
    if cp_norm > 1e-6:
        vc3 = (np.linalg.norm(va3) / (2.0 * cp_norm)) * cp
    else:
        vc3 = np.zeros(3, dtype=np.float32)

    c4 = c1.copy()
    c4[:3] = c1[:3] + vc3
    c4[3] = coords_5d[:, 3].max()

    c5 = c1.copy()
    c5[:3] = c1[:3] + vc3
    c5[3] = coords_5d[:, 3].min()

    c6 = c1.copy()
    c6[4] = c1[4] + np.linalg.norm(c2 - c1)

    centers = [c1, c2, c3, c4, c5, c6]
    out = np.zeros(ES_DIM, dtype=np.float32)
    for i, c in enumerate(centers):
        d = np.sqrt(((coords_5d - c) ** 2).sum(axis=1))
        m1 = d.mean()
        m2 = d.std(ddof=0)
        m3_raw = ((d - m1) ** 3).mean()
        m3 = np.cbrt(m3_raw) if m3_raw >= 0 else -np.cbrt(-m3_raw)
        out[i * 3 : i * 3 + 3] = [m1, m2, m3]
    return out


def _build_es5d_matrix_from_smiles(raw_smi: str) -> tuple[np.ndarray, int]:
    ob_mol = pybel.readstring("smi", raw_smi)
    ob_mol.OBMol.CorrectForPH(7.4)
    ph74 = ob_mol.write("can").split()[0]

    mol = Chem.MolFromSmiles(ph74)
    if not mol:
        raise ValueError("pH74 smiles parse failed")
    mol = Chem.AddHs(mol)

    p = AllChem.ETKDGv3()
    p.randomSeed = SEED
    if hasattr(p, "numThreads"):
        p.numThreads = 1
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=TOP_K, params=p)
    if not cids:
        raise ValueError("3D embedding failed")

    opt = AllChem.MMFFOptimizeMoleculeConfs(
        mol, maxIters=500, mmffVariant="MMFF94", numThreads=1
    )
    valid = [(cids[i], r[1]) for i, r in enumerate(opt) if r[0] == 0]
    if not valid:
        raise ValueError("MMFF convergence failed")
    valid.sort(key=lambda x: x[1])
    selected = valid[:TOP_K]

    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    if props is None:
        raise ValueError("MMFF props missing")
    q = np.array([props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    logp = np.array([v[0] for v in Crippen._GetAtomContribs(mol)], dtype=np.float32)

    out = np.full((TOP_K, ES_DIM), np.nan, dtype=np.float32)
    for idx, (cid, _) in enumerate(selected):
        c = np.zeros((mol.GetNumAtoms(), 5), dtype=np.float32)
        c[:, :3] = mol.GetConformer(cid).GetPositions()
        c[:, 3] = q * 25.0
        c[:, 4] = logp * 5.0
        out[idx] = _compute_es5d_reference(c)
    return out, len(selected)


def _rotation_matrix(rng: random.Random) -> np.ndarray:
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    r = np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ],
        dtype=np.float32,
    )
    return r


def _es5d_invariance_delta(coords_5d: np.ndarray, rng: random.Random) -> float:
    r = _rotation_matrix(rng)
    t = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)], dtype=np.float32)
    moved = coords_5d.copy()
    moved[:, :3] = coords_5d[:, :3] @ r.T + t
    a = _compute_es5d_reference(coords_5d)
    b = _compute_es5d_reference(moved)
    return float(np.max(np.abs(a - b)))


def _min_nonbond_distance(mol: Chem.Mol, conf_id: int = 0) -> float:
    conf = mol.GetConformer(conf_id)
    coords = np.array(conf.GetPositions(), dtype=np.float32)
    n = mol.GetNumAtoms()
    bonded = set()
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bonded.add((min(i, j), max(i, j)))
    min_d = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in bonded:
                continue
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < min_d:
                min_d = d
    return min_d if min_d < 1e8 else float("nan")


def _bond_length_outlier_count(mol: Chem.Mol, conf_id: int = 0) -> tuple[int, int]:
    conf = mol.GetConformer(conf_id)
    outliers = 0
    total = 0
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        pi = np.array(conf.GetAtomPosition(i))
        pj = np.array(conf.GetAtomPosition(j))
        d = float(np.linalg.norm(pi - pj))
        total += 1
        if d < 0.85 or d > 2.2:
            outliers += 1
    return outliers, total


def validate(args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    report: dict[str, Any] = {
        "validated_at": _now(),
        "inputs": {
            "memmap": str(args.memmap),
            "meta": str(args.meta),
            "manifest": str(args.manifest),
            "errors": str(args.errors),
        },
        "checks": {},
        "warnings": [],
        "failed_checks": [],
    }

    req = [args.memmap, args.meta]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        report["checks"]["missing_required_files"] = {"ok": False, "missing": missing}
        report["failed_checks"].append("missing_required_files")
        report["overall_ok"] = False
        return False, report

    meta = pd.read_parquet(args.meta)
    n_rows = _infer_memmap_rows(args.memmap)
    es = np.memmap(args.memmap, dtype=np.float32, mode="r", shape=(n_rows, TOP_K, ES_DIM))

    manifest = {}
    if args.manifest.exists():
        with args.manifest.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

    shape_ok = True
    shape_info = {"memmap_rows": int(n_rows), "meta_rows": int(len(meta)), "memmap_shape": [int(n_rows), TOP_K, ES_DIM]}
    if len(meta) != n_rows:
        shape_ok = False
    if "shape" in manifest:
        shape_info["manifest_shape"] = manifest["shape"]
        if manifest["shape"] != [int(n_rows), TOP_K, ES_DIM]:
            shape_ok = False
    report["checks"]["shape_consistency"] = {"ok": shape_ok, **shape_info}
    if not shape_ok:
        report["failed_checks"].append("shape_consistency")

    required_cols = {"memmap_idx", "molregno", "raw_smi", "ph74_smi", "converged_confs"}
    meta_ok = required_cols.issubset(set(meta.columns))
    if meta_ok:
        idx_ok = np.array_equal(meta["memmap_idx"].to_numpy(dtype=np.int64), np.arange(len(meta), dtype=np.int64))
        unique_ok = meta["molregno"].nunique(dropna=False) == len(meta)
        null_cells = int(meta[list(required_cols)].isna().sum().sum())
        null_ok = null_cells == 0
        meta_ok = idx_ok and unique_ok and null_ok
        report["checks"]["meta_integrity"] = {
            "ok": meta_ok,
            "idx_sequential": bool(idx_ok),
            "molregno_unique": bool(unique_ok),
            "null_cells_required_columns": null_cells,
        }
    else:
        report["checks"]["meta_integrity"] = {"ok": False, "missing_columns": sorted(list(required_cols - set(meta.columns)))}
    if not meta_ok:
        report["failed_checks"].append("meta_integrity")

    err_rows = _count_error_rows(args.errors)
    err_ok = True
    err_info = {"error_rows": int(err_rows)}
    if "failed" in manifest:
        err_info["manifest_failed"] = int(manifest["failed"])
        if int(manifest["failed"]) != int(err_rows):
            err_ok = False
    report["checks"]["error_log_alignment"] = {"ok": err_ok, **err_info}
    if not err_ok:
        report["failed_checks"].append("error_log_alignment")

    flat = es.reshape(-1)
    n_nan = int(np.isnan(flat).sum())
    n_inf = int(np.isinf(flat).sum())
    numeric_ok = n_inf == 0
    report["checks"]["numeric_finiteness"] = {
        "ok": numeric_ok,
        "nan_count_total": n_nan,
        "inf_count_total": n_inf,
        "nan_ratio": float(n_nan / flat.size),
    }
    if not numeric_ok:
        report["failed_checks"].append("numeric_finiteness")

    # Conformer fill-pattern: once NaN starts, remaining rows should be all NaN.
    cf_mismatch = 0
    monotonic_nan_break = 0
    valid_confs_observed = np.zeros(n_rows, dtype=np.int16)
    for i in range(n_rows):
        conf_all_nan = np.isnan(es[i]).all(axis=1)
        first_nan = np.argmax(conf_all_nan) if conf_all_nan.any() else TOP_K
        if conf_all_nan.any() and not conf_all_nan[first_nan:].all():
            monotonic_nan_break += 1
        valid = int((~conf_all_nan).sum())
        valid_confs_observed[i] = valid
        if "converged_confs" in meta.columns and int(meta.iloc[i]["converged_confs"]) != valid:
            cf_mismatch += 1
    cf_ok = cf_mismatch == 0 and monotonic_nan_break == 0
    report["checks"]["conformer_count_alignment"] = {
        "ok": cf_ok,
        "mismatch_rows": int(cf_mismatch),
        "non_tail_nan_rows": int(monotonic_nan_break),
        "valid_confs_min": int(valid_confs_observed.min()) if n_rows else 0,
        "valid_confs_median": float(np.median(valid_confs_observed)) if n_rows else 0.0,
        "valid_confs_max": int(valid_confs_observed.max()) if n_rows else 0,
    }
    if not cf_ok:
        report["failed_checks"].append("conformer_count_alignment")

    sampled = min(args.sample_size, n_rows)
    exact_mismatch = 0
    exact_examples = []
    rng = random.Random(args.seed)
    sample_idx = rng.sample(range(n_rows), sampled) if sampled > 0 else []
    inv_deltas = []
    geom_bad_clash = 0
    geom_bad_bond = 0
    geom_total = 0

    for idx in sample_idx:
        raw = str(meta.iloc[idx]["raw_smi"])
        stored = np.array(es[idx], dtype=np.float32)
        try:
            recomputed, converged = _build_es5d_matrix_from_smiles(raw)
            row_bad = False
            if int(converged) != int((~np.isnan(stored).all(axis=1)).sum()):
                row_bad = True
            if not np.allclose(
                np.nan_to_num(stored, nan=0.0),
                np.nan_to_num(recomputed, nan=0.0),
                rtol=args.rtol,
                atol=args.atol,
            ):
                row_bad = True
                if len(exact_examples) < 10:
                    delta = float(np.max(np.abs(np.nan_to_num(stored, nan=0.0) - np.nan_to_num(recomputed, nan=0.0))))
                    exact_examples.append({"memmap_idx": int(idx), "molregno": int(meta.iloc[idx]["molregno"]), "max_abs_delta": delta})
            if row_bad:
                exact_mismatch += 1

            # Geometry sanity on first valid conformer
            ob_m = pybel.readstring("smi", raw)
            ob_m.OBMol.CorrectForPH(7.4)
            ph74 = ob_m.write("can").split()[0]
            rm = Chem.MolFromSmiles(ph74)
            if rm:
                rm = Chem.AddHs(rm)
                p = AllChem.ETKDGv3()
                p.randomSeed = SEED
                if hasattr(p, "numThreads"):
                    p.numThreads = 1
                cids = AllChem.EmbedMultipleConfs(rm, numConfs=1, params=p)
                if cids:
                    AllChem.MMFFOptimizeMoleculeConfs(rm, maxIters=500, mmffVariant="MMFF94", numThreads=1)
                    min_nonbond = _min_nonbond_distance(rm, conf_id=0)
                    bo, bt = _bond_length_outlier_count(rm, conf_id=0)
                    geom_total += 1
                    if not np.isnan(min_nonbond) and min_nonbond < args.min_nonbond_distance:
                        geom_bad_clash += 1
                    if bt > 0 and (bo / bt) > args.max_bond_outlier_ratio:
                        geom_bad_bond += 1

                    props = AllChem.MMFFGetMoleculeProperties(rm, mmffVariant="MMFF94")
                    if props is not None:
                        q = np.array([props.GetMMFFPartialCharge(i) for i in range(rm.GetNumAtoms())], dtype=np.float32)
                        logp = np.array([v[0] for v in Crippen._GetAtomContribs(rm)], dtype=np.float32)
                        c5 = np.zeros((rm.GetNumAtoms(), 5), dtype=np.float32)
                        c5[:, :3] = rm.GetConformer(0).GetPositions()
                        c5[:, 3] = q * 25.0
                        c5[:, 4] = logp * 5.0
                        inv_deltas.append(_es5d_invariance_delta(c5, rng))
        except Exception:
            exact_mismatch += 1
            if len(exact_examples) < 10:
                exact_examples.append({"memmap_idx": int(idx), "molregno": int(meta.iloc[idx]["molregno"]), "max_abs_delta": None})

    exact_ok = exact_mismatch <= args.max_exact_mismatch
    report["checks"]["sample_recompute_exactness"] = {
        "ok": exact_ok,
        "sample_size": int(sampled),
        "mismatch_count": int(exact_mismatch),
        "allowed_mismatch": int(args.max_exact_mismatch),
        "examples": exact_examples,
    }
    if not exact_ok:
        report["failed_checks"].append("sample_recompute_exactness")

    inv_max = float(max(inv_deltas)) if inv_deltas else None
    inv_ok = inv_max is None or inv_max <= args.max_invariance_delta
    report["checks"]["rotation_translation_invariance"] = {
        "ok": inv_ok,
        "sampled_conformers": int(len(inv_deltas)),
        "max_abs_delta": inv_max,
        "threshold": float(args.max_invariance_delta),
    }
    if not inv_ok:
        report["failed_checks"].append("rotation_translation_invariance")

    geom_ok = True
    if geom_total > 0:
        clash_rate = geom_bad_clash / geom_total
        bond_rate = geom_bad_bond / geom_total
        geom_ok = clash_rate <= args.max_clash_rate and bond_rate <= args.max_bond_fail_rate
    else:
        clash_rate = 0.0
        bond_rate = 0.0
        report["warnings"].append("No geometry samples available.")
    report["checks"]["geometry_sanity"] = {
        "ok": geom_ok,
        "sampled_molecules": int(geom_total),
        "clash_fail_count": int(geom_bad_clash),
        "bond_fail_count": int(geom_bad_bond),
        "clash_fail_rate": float(clash_rate),
        "bond_fail_rate": float(bond_rate),
        "max_clash_rate": float(args.max_clash_rate),
        "max_bond_fail_rate": float(args.max_bond_fail_rate),
    }
    if not geom_ok:
        report["failed_checks"].append("geometry_sanity")

    hashes = {
        "memmap_sha256": _sha256_file(args.memmap),
        "meta_sha256": _sha256_file(args.meta),
    }
    if args.manifest.exists():
        hashes["manifest_sha256"] = _sha256_file(args.manifest)
    report["hashes"] = hashes

    if args.baseline_json and args.baseline_json.exists():
        with args.baseline_json.open("r", encoding="utf-8") as f:
            baseline = json.load(f)
        bh = baseline.get("hashes", {})
        match = all(hashes.get(k) == bh.get(k) for k in hashes.keys())
        report["checks"]["baseline_hash_match"] = {
            "ok": match,
            "baseline": str(args.baseline_json),
            "baseline_hashes_present": sorted(list(bh.keys())),
        }
        if not match:
            report["failed_checks"].append("baseline_hash_match")
    elif args.baseline_json:
        report["warnings"].append(f"Baseline not found: {args.baseline_json}")

    mono = bool(meta["molregno"].is_monotonic_increasing) if "molregno" in meta.columns else False
    report["checks"]["ordering_observation"] = {
        "ok": True,
        "molregno_monotonic_increasing": mono,
        "note": "False suggests generator_unordered affected row order.",
    }

    ok = len(report["failed_checks"]) == 0
    report["overall_ok"] = ok
    return ok, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate ES5D extraction outputs.")
    p.add_argument("--memmap", type=Path, default=DEFAULT_MEMMAP)
    p.add_argument("--meta", type=Path, default=DEFAULT_META)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    p.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--sample-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=20260304)
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--max-exact-mismatch", type=int, default=0)
    p.add_argument("--min-nonbond-distance", type=float, default=0.85)
    p.add_argument("--max-bond-outlier-ratio", type=float, default=0.10)
    p.add_argument("--max-clash-rate", type=float, default=0.01)
    p.add_argument("--max-bond-fail-rate", type=float, default=0.02)
    # float32 + rigid transform recomputation can introduce tiny numerical jitter.
    # 1e-4 keeps strict QC while avoiding false failures from machine-precision noise.
    p.add_argument("--max-invariance-delta", type=float, default=1e-4)
    p.add_argument("--baseline-json", type=Path, default=None)
    p.add_argument("--write-baseline", type=Path, default=None)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ok, report = validate(args)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with args.report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if args.write_baseline:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        with args.write_baseline.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== ES5D Validation Report ===")
    print(f"validated_at: {report['validated_at']}")
    print(f"overall_ok: {report['overall_ok']}")
    print(f"failed_checks: {report['failed_checks']}")
    print(f"report_json: {args.report_json}")
    if args.write_baseline:
        print(f"baseline_written: {args.write_baseline}")

    if args.strict and not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
