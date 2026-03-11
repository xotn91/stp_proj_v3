# -*- coding: utf-8 -*-
"""
P2.2 ES5D Extraction (Production)
- Input: features_store/chembl36_stp_training_set.parquet
- Output:
  - features_store/es5d_db_k20.memmap
  - features_store/es5d_meta_db.parquet
  - features_store/es5d_db_errors.log
  - features_store/es5d_db_manifest.json
"""

import argparse
import json
import multiprocessing
import os
import time
from collections import Counter
from pathlib import Path

# Avoid nested BLAS/OpenMP threading inside each worker process.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Crippen
from tqdm import tqdm

from openbabel import pybel

RDLogger.DisableLog("rdApp.*")
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "features_store" / "chembl36_stp_training_set.parquet"
DEFAULT_OUT_DIR = ROOT / "features_store"
TOP_K_CONFS = 20
SEED = 42


def compute_es5d_vector(coords_5d: np.ndarray) -> np.ndarray:
    """
    Convert N x 5 coordinates to 18D ES5D moment vector.
    """
    n_atoms = coords_5d.shape[0]
    if n_atoms < 4:
        raise ValueError("Atoms too few")

    c1 = np.mean(coords_5d, axis=0)
    d1 = np.linalg.norm(coords_5d - c1, axis=1)
    c2 = coords_5d[np.argmax(d1)]
    d2 = np.linalg.norm(coords_5d - c2, axis=1)
    c3 = coords_5d[np.argmax(d2)]

    v_a, v_b = c2 - c1, c3 - c1
    v_a3, v_b3 = v_a[:3], v_b[:3]
    cross_prod = np.cross(v_a3, v_b3)
    norm_cross = np.linalg.norm(cross_prod)
    v_c = (np.linalg.norm(v_a3) / (2.0 * norm_cross)) * cross_prod if norm_cross > 1e-6 else np.zeros(3)

    q_max = np.max(coords_5d[:, 3])
    q_min = np.min(coords_5d[:, 3])

    c4 = np.copy(c1)
    c4[:3] = c1[:3] + v_c
    c4[3] = q_max

    c5 = np.copy(c1)
    c5[:3] = c1[:3] + v_c
    c5[3] = q_min

    c6 = np.copy(c1)
    c6[4] = c1[4] + np.linalg.norm(v_a)

    centroids = [c1, c2, c3, c4, c5, c6]
    moments = np.zeros(18, dtype=np.float32)

    for i, c in enumerate(centroids):
        dists = np.linalg.norm(coords_5d - c, axis=1)
        m1 = np.mean(dists)
        m2 = np.std(dists, ddof=0)
        m3_raw = np.mean((dists - m1) ** 3)
        m3 = np.cbrt(m3_raw) if m3_raw >= 0 else -np.cbrt(-m3_raw)
        moments[i * 3 : i * 3 + 3] = [m1, m2, m3]

    return moments


def _suppress_worker_stderr_begin():
    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    return saved_stderr_fd


def _suppress_worker_stderr_end(saved_stderr_fd: int):
    os.dup2(saved_stderr_fd, 2)
    os.close(saved_stderr_fd)


def _process_3d_es5d(row):
    molregno, raw_smiles = row
    qc = {
        "raw_parse_ok": False,
        "raw_charge": np.nan,
        "ph74_smi": None,
        "ph74_parse_ok": False,
        "ph74_charge": np.nan,
        "is_changed": False,
        "q_diff": np.nan,
        "converged_confs": 0,
    }

    saved_stderr_fd = _suppress_worker_stderr_begin()
    try:
        mol_raw = Chem.MolFromSmiles(raw_smiles)
        if not mol_raw:
            return False, molregno, raw_smiles, None, qc, "Raw SMILES Parse Failed"
        qc["raw_parse_ok"] = True
        qc["raw_charge"] = Chem.GetFormalCharge(mol_raw)

        ob_mol = pybel.readstring("smi", raw_smiles)
        ob_mol.OBMol.CorrectForPH(7.4)
        ph74_smiles = ob_mol.write("can").split()[0]
        qc["ph74_smi"] = ph74_smiles
        qc["is_changed"] = bool(raw_smiles != ph74_smiles)

        mol_ph74 = Chem.MolFromSmiles(ph74_smiles)
        if not mol_ph74:
            return False, molregno, raw_smiles, None, qc, "pH74 SMILES Parse Failed"
        qc["ph74_parse_ok"] = True
        qc["ph74_charge"] = Chem.GetFormalCharge(mol_ph74)
        qc["q_diff"] = qc["ph74_charge"] - qc["raw_charge"]

        mol = Chem.AddHs(mol_ph74)
        embed_params = AllChem.ETKDGv3()
        embed_params.randomSeed = SEED
        if hasattr(embed_params, "numThreads"):
            embed_params.numThreads = 1

        cids = AllChem.EmbedMultipleConfs(mol, numConfs=TOP_K_CONFS, params=embed_params)
        if not cids:
            return False, molregno, raw_smiles, None, qc, "3D Embedding Failed"

        opt_results = AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            maxIters=500,
            mmffVariant="MMFF94",
            numThreads=1,
        )
        valid_confs = [(cids[i], r[1]) for i, r in enumerate(opt_results) if r[0] == 0]
        if not valid_confs:
            return False, molregno, raw_smiles, None, qc, "MMFF Convergence Failed"
        valid_confs.sort(key=lambda x: x[1])
        qc["converged_confs"] = len(valid_confs)

        selected = valid_confs[:TOP_K_CONFS]
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
        if mmff_props is None:
            return False, molregno, raw_smiles, None, qc, "MMFF Params Missing"

        charges = np.array([mmff_props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())], dtype=np.float32)
        logp = np.array([v[0] for v in Crippen._GetAtomContribs(mol)], dtype=np.float32)

        es5d_matrix = np.full((TOP_K_CONFS, 18), np.nan, dtype=np.float32)
        for idx, (cid, _) in enumerate(selected):
            coords_5d = np.zeros((mol.GetNumAtoms(), 5), dtype=np.float32)
            coords_5d[:, :3] = mol.GetConformer(cid).GetPositions()
            coords_5d[:, 3] = charges * 25.0
            coords_5d[:, 4] = logp * 5.0
            es5d_matrix[idx] = compute_es5d_vector(coords_5d)

        return True, molregno, raw_smiles, es5d_matrix, qc, ""
    except Exception as err:
        msg = str(err).replace("\n", " ")
        return False, molregno, raw_smiles, None, qc, f"Unexpected: {msg}"
    finally:
        _suppress_worker_stderr_end(saved_stderr_fd)


class ES5DProductionExtractor:
    def __init__(
        self,
        input_parquet: Path,
        out_dir: Path,
        n_jobs: int | None = None,
        backend: str = "loky",
        batch_size: str | int = "auto",
        max_molecules: int | None = None,
    ):
        self.input_parquet = Path(input_parquet)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.memmap_path = self.out_dir / "es5d_db_k20.memmap"
        self.meta_path = self.out_dir / "es5d_meta_db.parquet"
        self.error_log = self.out_dir / "es5d_db_errors.log"
        self.manifest_path = self.out_dir / "es5d_db_manifest.json"

        self.memmap_tmp_path = self.out_dir / "es5d_db_k20.memmap.tmp"
        self.meta_tmp_path = self.out_dir / "es5d_meta_db.parquet.tmp"
        self.error_tmp_path = self.out_dir / "es5d_db_errors.log.tmp"
        self.manifest_tmp_path = self.out_dir / "es5d_db_manifest.json.tmp"

        max_workers = max(1, multiprocessing.cpu_count() - 2)
        if n_jobs is None:
            self.n_jobs = max_workers
        else:
            self.n_jobs = max(1, min(max_workers, int(n_jobs)))

        if backend not in {"loky", "multiprocessing", "threading"}:
            raise ValueError("backend must be one of: loky, multiprocessing, threading")
        self.backend = backend

        if isinstance(batch_size, str):
            if batch_size != "auto":
                raise ValueError("batch_size string must be 'auto'")
            self.batch_size = batch_size
        else:
            self.batch_size = max(1, int(batch_size))

        self.max_molecules = None if max_molecules is None else max(1, int(max_molecules))

    @staticmethod
    def _safe_unlink(path: Path):
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            if path.exists():
                path.unlink()

    def _collect_unique(self) -> dict[int, str]:
        unique_mols: dict[int, str] = {}
        pf = pq.ParquetFile(self.input_parquet)
        for batch in pf.iter_batches(columns=["molregno", "canonical_smiles"]):
            d = batch.to_pydict()
            for molregno, smiles in zip(d["molregno"], d["canonical_smiles"]):
                if molregno is None or smiles is None:
                    continue
                if molregno not in unique_mols:
                    unique_mols[int(molregno)] = str(smiles)
        return unique_mols

    def run(self):
        if not self.input_parquet.exists():
            raise FileNotFoundError(f"Input parquet not found: {self.input_parquet}")

        print(f">> 1. Loading unique molecules from: {self.input_parquet}")
        unique_mols = self._collect_unique()
        tasks = sorted(unique_mols.items(), key=lambda x: x[0])
        if self.max_molecules is not None:
            tasks = tasks[: self.max_molecules]

        max_mols = len(tasks)
        if max_mols == 0:
            raise ValueError("No valid molecules found in input parquet.")
        print(f"   - Unique molecules: {max_mols:,}")

        self._safe_unlink(self.memmap_tmp_path)
        self._safe_unlink(self.meta_tmp_path)
        self._safe_unlink(self.error_tmp_path)
        self._safe_unlink(self.manifest_tmp_path)

        print(f">> 2. Initializing Memmap (N={max_mols}, K={TOP_K_CONFS}, 18D)...")
        es5d_memmap = np.memmap(self.memmap_tmp_path, dtype=np.float32, mode="w+", shape=(max_mols, TOP_K_CONFS, 18))

        error_counter: Counter[str] = Counter()
        valid_rows = []
        start_time = time.time()

        print(
            f">> 3. Running ES5D extraction with {self.n_jobs} workers "
            f"(backend={self.backend}, batch_size={self.batch_size})..."
        )
        with open(self.error_tmp_path, "w", encoding="utf-8") as ferr:
            ferr.write("molregno\traw_smiles\terror_reason\n")
            results = Parallel(
                n_jobs=self.n_jobs,
                return_as="generator_unordered",
                backend=self.backend,
                batch_size=self.batch_size,
                pre_dispatch=f"{self.n_jobs * 2}",
            )(
                delayed(_process_3d_es5d)(row) for row in tasks
            )

            write_idx = 0
            for success, molregno, raw_smi, data, qc, err in tqdm(results, total=max_mols):
                if success:
                    es5d_memmap[write_idx] = data
                    valid_rows.append(
                        {
                            "memmap_idx": write_idx,
                            "molregno": molregno,
                            "raw_smi": raw_smi,
                            "ph74_smi": qc["ph74_smi"],
                            "is_changed": qc["is_changed"],
                            "q_diff": qc["q_diff"],
                            "converged_confs": qc["converged_confs"],
                        }
                    )
                    write_idx += 1
                else:
                    error_counter[err] += 1
                    ferr.write(f"{molregno}\t{raw_smi}\t{err}\n")

        es5d_memmap.flush()

        if write_idx < max_mols:
            print(f">> 4. Truncating memmap to {write_idx} valid rows...")
            truncated_tmp = self.out_dir / "es5d_db_k20.memmap.truncated.tmp"
            self._safe_unlink(truncated_tmp)

            final_mm = np.memmap(truncated_tmp, dtype=np.float32, mode="w+", shape=(write_idx, TOP_K_CONFS, 18))
            final_mm[:] = es5d_memmap[:write_idx]
            final_mm.flush()
            del es5d_memmap, final_mm

            self._safe_unlink(self.memmap_tmp_path)
            os.replace(truncated_tmp, self.memmap_tmp_path)

        print(">> 5. Saving metadata and manifest...")
        meta_df = pd.DataFrame(valid_rows)
        meta_df.to_parquet(self.meta_tmp_path, index=False, compression="zstd")

        manifest = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": str(self.input_parquet),
            "method": "ElectroShape 5D (ES5D)",
            "shape": [int(write_idx), TOP_K_CONFS, 18],
            "success_rate": f"{write_idx}/{max_mols} ({write_idx / max_mols * 100:.2f}%)",
            "c6_offset_metric": "5D distance (||c2-c1|| in 5D space)",
            "v_c_offset_metric": "3D distance (||c2-c1|| in 3D space)",
            "conformer_policy": f"Store TOP {TOP_K_CONFS} by MMFF94 energy",
            "ph_engine": "OpenBabel CorrectForPH(7.4) (heuristic)",
            "elapsed_sec": round(time.time() - start_time, 2),
            "failed": int(max_mols - write_idx),
            "parallel": {
                "n_jobs": int(self.n_jobs),
                "backend": self.backend,
                "batch_size": self.batch_size,
            },
            "max_molecules": self.max_molecules,
        }
        with open(self.manifest_tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Atomic replacement at finalize step.
        os.replace(self.memmap_tmp_path, self.memmap_path)
        os.replace(self.meta_tmp_path, self.meta_path)
        os.replace(self.error_tmp_path, self.error_log)
        os.replace(self.manifest_tmp_path, self.manifest_path)

        print("\n=== P2.2 Completed ===")
        print(f"✅ {self.memmap_path}")
        print(f"✅ {self.meta_path}")
        print(f"✅ {self.error_log}")
        print(f"✅ {self.manifest_path}")
        print(f"   - Success: {write_idx:,} / Failed: {max_mols - write_idx:,}")
        if max_mols > 0:
            print(f"   - Success rate: {write_idx / max_mols * 100:.2f}%")
        if error_counter:
            print("\n[Top Failure Reasons]")
            for reason, count in error_counter.most_common(10):
                print(f" - {reason}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Extract ES5D (K=20) from training parquet.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input parquet path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Worker count (default: cpu_count-2).")
    parser.add_argument(
        "--backend",
        choices=["loky", "multiprocessing", "threading"],
        default="loky",
        help="Joblib backend.",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Joblib batch size: 'auto' or integer.",
    )
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=None,
        help="Optional limit for quick benchmark/debug run.",
    )
    args = parser.parse_args()

    batch_size: str | int
    if isinstance(args.batch_size, str) and args.batch_size != "auto":
        batch_size = int(args.batch_size)
    else:
        batch_size = args.batch_size

    extractor = ES5DProductionExtractor(
        input_parquet=Path(args.input),
        out_dir=Path(args.out_dir),
        n_jobs=args.n_jobs,
        backend=args.backend,
        batch_size=batch_size,
        max_molecules=args.max_molecules,
    )
    extractor.run()


if __name__ == "__main__":
    main()
