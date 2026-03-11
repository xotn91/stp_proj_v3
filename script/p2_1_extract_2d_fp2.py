# -*- coding: utf-8 -*-
"""
P2.1 FP2 Extraction
- Input: features_store/chembl36_stp_training_set.parquet
- Output:
  - features_store/fp2_uint64.memmap
  - features_store/fp2_meta.parquet
  - features_store/fp2_errors.log
  - features_store/manifest.json
"""

import argparse
import json
import multiprocessing
import os
import time
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
from tqdm import tqdm

try:
    from openbabel import pybel
except ImportError as exc:
    raise ImportError(
        "OpenBabel python wrapper is required. "
        "Install with: conda install -c conda-forge openbabel"
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "features_store" / "chembl36_stp_training_set.parquet"
DEFAULT_OUT_DIR = ROOT / "features_store"


def _compute_fp2_uint64(row):
    """
    Convert (molregno, smiles) into 1024-bit FP2 packed as uint64[16].
    """
    molregno, smiles = row
    try:
        mol = pybel.readstring("smi", smiles)
        fp = mol.calcfp("fp2")

        arr_uint64 = np.zeros(16, dtype=np.uint64)
        for bit in fp.bits:  # 1-indexed bit positions
            if 1 <= bit <= 1024:
                idx = (bit - 1) // 64
                offset = (bit - 1) % 64
                arr_uint64[idx] |= (np.uint64(1) << np.uint64(offset))

        return True, molregno, arr_uint64, ""
    except Exception as err:
        return False, molregno, None, str(err).replace("\n", " ")


class FP2Extractor:
    def __init__(
        self,
        parquet_file: Path,
        out_dir: Path,
        n_jobs: int | None = None,
        backend: str = "loky",
        batch_size: str | int = "auto",
    ):
        self.parquet_file = Path(parquet_file)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.memmap_path = self.out_dir / "fp2_uint64.memmap"
        self.meta_path = self.out_dir / "fp2_meta.parquet"
        self.error_log = self.out_dir / "fp2_errors.log"
        self.manifest_path = self.out_dir / "manifest.json"

        self.memmap_tmp_path = self.out_dir / "fp2_uint64.memmap.tmp"
        self.meta_tmp_path = self.out_dir / "fp2_meta.parquet.tmp"
        self.error_tmp_path = self.out_dir / "fp2_errors.log.tmp"
        self.manifest_tmp_path = self.out_dir / "manifest.json.tmp"

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

    @staticmethod
    def _safe_unlink(path: Path):
        try:
            path.unlink(missing_ok=True)
        except TypeError:
            if path.exists():
                path.unlink()

    def _collect_unique_molecules(self):
        """
        Stream parquet and deduplicate by molregno.
        """
        unique_mols: dict[int, str] = {}
        pf = pq.ParquetFile(self.parquet_file)
        for batch in pf.iter_batches(columns=["molregno", "canonical_smiles"]):
            data = batch.to_pydict()
            for molregno, smiles in zip(data["molregno"], data["canonical_smiles"]):
                if molregno is None or smiles is None:
                    continue
                if molregno not in unique_mols:
                    unique_mols[int(molregno)] = str(smiles)
        return unique_mols

    def run(self):
        if not self.parquet_file.exists():
            raise FileNotFoundError(f"Input parquet not found: {self.parquet_file}")

        print(f">> 1. Loading unique molecules from: {self.parquet_file}")
        unique_mols = self._collect_unique_molecules()
        n_total = len(unique_mols)
        if n_total == 0:
            raise ValueError("No valid (molregno, canonical_smiles) found in input parquet.")

        # Stable ordering for reproducible memmap index
        tasks = sorted(unique_mols.items(), key=lambda x: x[0])
        print(f"   - Unique molecules: {n_total:,}")

        self._safe_unlink(self.memmap_tmp_path)
        self._safe_unlink(self.meta_tmp_path)
        self._safe_unlink(self.error_tmp_path)
        self._safe_unlink(self.manifest_tmp_path)

        print(f">> 2. Initializing memmap: {self.memmap_path.name} ({n_total:,} x 16)")
        fp_memmap = np.memmap(self.memmap_tmp_path, dtype=np.uint64, mode="w+", shape=(n_total, 16))

        print(
            f">> 3. Calculating FP2 with {self.n_jobs} workers "
            f"(backend={self.backend}, batch_size={self.batch_size})..."
        )
        success_molregnos = []
        error_count = 0
        start = time.time()

        with open(self.error_tmp_path, "w", encoding="utf-8") as ferr:
            ferr.write("molregno\tcanonical_smiles\terror\n")
            results = Parallel(
                n_jobs=self.n_jobs,
                return_as="generator_unordered",
                backend=self.backend,
                batch_size=self.batch_size,
                pre_dispatch=f"{self.n_jobs * 2}",
            )(
                delayed(_compute_fp2_uint64)(row) for row in tasks
            )

            write_idx = 0
            for success, molregno, data, err_msg in tqdm(results, total=n_total):
                if success:
                    fp_memmap[write_idx] = data
                    success_molregnos.append(molregno)
                    write_idx += 1
                else:
                    error_count += 1
                    ferr.write(f"{molregno}\t{unique_mols[molregno]}\t{err_msg}\n")

        fp_memmap.flush()
        elapsed = time.time() - start
        print(f"   - Completed in {elapsed:.2f} sec")
        print(f"   - Success: {write_idx:,}, Failed: {error_count:,}")

        if error_count > 0:
            print(">> 4. Truncating memmap to successful rows...")
            truncated_tmp_path = self.out_dir / "fp2_uint64.memmap.truncated.tmp"
            self._safe_unlink(truncated_tmp_path)

            final_mm = np.memmap(truncated_tmp_path, dtype=np.uint64, mode="w+", shape=(write_idx, 16))
            final_mm[:] = fp_memmap[:write_idx]
            final_mm.flush()
            del fp_memmap, final_mm

            self._safe_unlink(self.memmap_tmp_path)
            os.replace(truncated_tmp_path, self.memmap_tmp_path)

        print(">> 5. Saving meta and manifest...")
        meta_df = pd.DataFrame(
            {
                "memmap_idx": np.arange(write_idx, dtype=np.int64),
                "molregno": success_molregnos,
                "canonical_smiles": [unique_mols[m] for m in success_molregnos],
            }
        )
        meta_df.to_parquet(self.meta_tmp_path, index=False, compression="zstd")

        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": str(self.parquet_file),
            "feature_type": "OpenBabel FP2",
            "data_type": "uint64",
            "shape": [int(write_idx), 16],
            "total_processed": int(n_total),
            "success": int(write_idx),
            "failed": int(error_count),
            "bitorder": "little/big independent logic (explicit bit shifting)",
            "parallel": {
                "n_jobs": int(self.n_jobs),
                "backend": self.backend,
                "batch_size": self.batch_size,
            },
        }
        with open(self.manifest_tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Atomic replacement at finalize step.
        os.replace(self.memmap_tmp_path, self.memmap_path)
        os.replace(self.meta_tmp_path, self.meta_path)
        os.replace(self.error_tmp_path, self.error_log)
        os.replace(self.manifest_tmp_path, self.manifest_path)

        print("=== P2.1 Completed ===")
        print(f"✅ {self.memmap_path}")
        print(f"✅ {self.meta_path}")
        print(f"✅ {self.manifest_path}")
        print(f"✅ {self.error_log}")


def main():
    parser = argparse.ArgumentParser(description="Extract OpenBabel FP2 (uint64[16]) from training parquet.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input parquet path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Number of parallel workers (default: cpu_count-2).")
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
    args = parser.parse_args()

    batch_size: str | int
    if isinstance(args.batch_size, str) and args.batch_size != "auto":
        batch_size = int(args.batch_size)
    else:
        batch_size = args.batch_size

    extractor = FP2Extractor(
        parquet_file=Path(args.input),
        out_dir=Path(args.out_dir),
        n_jobs=args.n_jobs,
        backend=args.backend,
        batch_size=batch_size,
    )
    extractor.run()


if __name__ == "__main__":
    main()
