# -*- coding: utf-8 -*-
"""
Generate leave-one-out prediction logs with the project's actual dual-GPU score engine.

The outer leave-one-out workflow matches the CV logging task, while the inner
adapter uses the project's FP2 / ES5D / coefficient assets so it can produce
combined_score, max_sim_2d, and max_sim_3d from the same core engine family as
the existing inference code.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401
import torch


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = DEFAULT_PROJECT_ROOT / "data"
DEFAULT_META_FILE = DEFAULT_PROJECT_ROOT / "features_store" / "final_training_meta.parquet"
DEFAULT_FP2 = DEFAULT_PROJECT_ROOT / "features_store" / "fp2_aligned.memmap"
DEFAULT_ES5D = DEFAULT_PROJECT_ROOT / "features_store" / "es5d_db_k20.memmap"
DEFAULT_COEF_JSON = (
    DEFAULT_PROJECT_ROOT
    / "features_store"
    / "p3_run_full_aug_scaf_off_20260311"
    / "p3_1_1_K1_paired_trainer_fast__K1__paired_sum__all__C10p0__Y2023__Hd6d53c2431"
    / "stp_coef_K1_paired_sum_all.json"
)
HUMAN_LABELS = {"human", "homo sapiens", "9606"}


def infer_size_bin(heavy_atom_count: int) -> str:
    """Assign the required heavy-atom bin label for a query molecule."""
    if pd.isna(heavy_atom_count):
        return "41_plus"
    heavy_atom_count = int(heavy_atom_count)
    if heavy_atom_count <= 15:
        return "01_15"
    if heavy_atom_count <= 20:
        return "16_20"
    if heavy_atom_count <= 25:
        return "21_25"
    if heavy_atom_count <= 30:
        return "26_30"
    if heavy_atom_count <= 35:
        return "31_35"
    if heavy_atom_count <= 40:
        return "36_40"
    return "41_plus"


def _normalize_species(value: object) -> str:
    """Convert a species value to a normalized comparison key."""
    return str(value).strip().lower()


def resolve_thresholds(thr_preset: str, thr2d: float, thr3d: float) -> Tuple[float, float]:
    """Resolve paper-style threshold presets to concrete 2D and 3D cutoffs."""
    if thr_preset == "stp2014":
        return 0.30, 0.65
    if thr_preset == "stp2019":
        return 0.65, 0.85
    return float(thr2d), float(thr3d)


def _load_meta_map(meta_path: str | Path) -> pd.DataFrame:
    """Load the minimal metadata required to resolve compound_id to memmap_idx."""
    meta_df = pd.read_parquet(meta_path, columns=["mol_chembl_id", "memmap_idx"]).copy()
    meta_df["compound_id"] = meta_df["mol_chembl_id"].astype(str).str.strip()
    meta_df["memmap_idx"] = pd.to_numeric(meta_df["memmap_idx"], errors="coerce")
    meta_df = meta_df.dropna(subset=["compound_id", "memmap_idx"]).copy()
    meta_df["memmap_idx"] = meta_df["memmap_idx"].astype(np.int64)
    meta_df = meta_df.sort_values(["compound_id", "memmap_idx"]).drop_duplicates("compound_id", keep="first")
    return meta_df[["compound_id", "memmap_idx"]].copy()


def load_inputs(
    compound_path: str | Path,
    target_path: str | Path,
    activity_path: str | Path,
    meta_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three required tables, keep human targets, and attach memmap_idx."""
    compounds = pd.read_parquet(compound_path).copy()
    targets = pd.read_parquet(target_path).copy()
    activities = pd.read_parquet(activity_path).copy()
    meta_map = _load_meta_map(meta_path)

    compound_required = ["compound_id", "canonical_smiles", "heavy_atom_count", "species"]
    target_required = ["target_id", "target_name", "species"]
    activity_required = ["compound_id", "target_id"]

    for column in compound_required:
        if column not in compounds.columns:
            raise KeyError(f"compound_table missing required column: {column}")
    for column in target_required:
        if column not in targets.columns:
            raise KeyError(f"target_table missing required column: {column}")
    for column in activity_required:
        if column not in activities.columns:
            raise KeyError(f"activity_table missing required column: {column}")

    compounds = compounds[compound_required].drop_duplicates(subset=["compound_id"]).copy()
    targets = targets[target_required].drop_duplicates(subset=["target_id"]).copy()
    activities = activities[activity_required].drop_duplicates().copy()

    compounds["compound_id"] = compounds["compound_id"].astype(str).str.strip()
    compounds["canonical_smiles"] = compounds["canonical_smiles"].fillna("").astype(str)
    compounds["heavy_atom_count"] = pd.to_numeric(
        compounds["heavy_atom_count"], errors="coerce"
    ).fillna(0).astype(np.int64)
    compounds["species"] = compounds["species"].map(_normalize_species)

    targets["target_id"] = targets["target_id"].astype(str).str.strip()
    targets["target_name"] = targets["target_name"].fillna("").astype(str)
    targets["species"] = targets["species"].map(_normalize_species)

    activities["compound_id"] = activities["compound_id"].astype(str).str.strip()
    activities["target_id"] = activities["target_id"].astype(str).str.strip()

    human_targets = targets[targets["species"].isin(HUMAN_LABELS)].copy()
    activities = activities[activities["target_id"].isin(human_targets["target_id"])].copy()
    compounds = compounds[compounds["compound_id"].isin(activities["compound_id"])].copy()
    compounds = compounds.merge(meta_map, on="compound_id", how="left")
    missing_memmap = int(compounds["memmap_idx"].isna().sum())
    if missing_memmap > 0:
        print(f">> Dropping {missing_memmap:,} compounds without memmap_idx")
        compounds = compounds.dropna(subset=["memmap_idx"]).copy()
    compounds["memmap_idx"] = compounds["memmap_idx"].astype(np.int64)

    activities = activities[activities["compound_id"].isin(compounds["compound_id"])].copy()
    compounds["size_bin"] = compounds["heavy_atom_count"].map(infer_size_bin)
    compounds["query_id"] = compounds["compound_id"]

    print(
        f">> Loaded human-only inputs: compounds={len(compounds):,}, "
        f"targets={len(human_targets):,}, activity_rows={len(activities):,}"
    )
    return compounds.reset_index(drop=True), human_targets.reset_index(drop=True), activities.reset_index(drop=True)


def build_truth_map(activity_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build the observed true-target mapping per compound."""
    return activity_df.groupby("compound_id")["target_id"].agg(lambda x: set(x)).to_dict()


def sample_queries_by_size_bin(
    compounds_df: pd.DataFrame,
    max_per_bin: int,
    random_seed: int,
) -> pd.DataFrame:
    """Sample up to a fixed number of query molecules within each size bin."""
    sampled_groups: List[pd.DataFrame] = []
    for size_bin, group in compounds_df.groupby("size_bin", sort=True):
        if len(group) <= max_per_bin:
            sampled_groups.append(group.copy())
            print(f">> Size bin {size_bin}: kept all {len(group):,} molecules")
            continue
        sampled = group.sample(n=max_per_bin, random_state=random_seed).copy()
        sampled_groups.append(sampled)
        print(
            f">> Size bin {size_bin}: sampled {len(sampled):,} of {len(group):,} molecules "
            f"(seed={random_seed})"
        )
    return pd.concat(sampled_groups, ignore_index=True)


class DualGPUSTPEngineAdapter:
    """Dual-GPU leave-one-out adapter built from the project's real score engine."""

    def __init__(
        self,
        reference_compounds_df: pd.DataFrame,
        reference_activity_df: pd.DataFrame,
        human_targets_df: pd.DataFrame,
        fp2_memmap_path: str | Path,
        es5d_memmap_path: str | Path,
        coef_json_path: str | Path,
        gpu_batch_size: int = 32,
        chunk_m: int = 1024,
        require_dual_gpu: bool = True,
        thr_preset: str = "none",
        thr2d: float = 0.65,
        thr3d: float = 0.85,
        disable_threshold_norm: bool = False,
        exclude_below_threshold: bool = False,
    ) -> None:
        self.reference_compounds_df = reference_compounds_df.reset_index(drop=True).copy()
        self.reference_activity_df = reference_activity_df.copy()
        self.human_targets_df = human_targets_df.copy()
        self.fp2_memmap_path = str(fp2_memmap_path)
        self.es5d_memmap_path = str(es5d_memmap_path)
        self.coef_json_path = str(coef_json_path)
        self.gpu_batch_size = int(gpu_batch_size)
        self.chunk_m = int(chunk_m)
        self.require_dual_gpu = bool(require_dual_gpu)
        self.thr_preset = str(thr_preset)
        self.disable_threshold_norm = bool(disable_threshold_norm)
        self.exclude_below_threshold = bool(exclude_below_threshold)
        self.thr2d, self.thr3d = resolve_thresholds(self.thr_preset, float(thr2d), float(thr3d))

        self._initialize_devices()
        self._load_coefficients()
        self._build_reference_tables()
        self._build_gpu_assets()

    def _initialize_devices(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        device_count = torch.cuda.device_count()
        if self.require_dual_gpu and device_count < 2:
            raise RuntimeError(f"Expected 2 GPUs but found {device_count}.")
        self.devices = ["cuda:0"]
        if device_count >= 2:
            self.devices.append("cuda:1")
        print(f">> GPU devices: {', '.join(self.devices)}")
        for device_id, device in enumerate(self.devices):
            print(f"   - {device}: {torch.cuda.get_device_name(device_id)}")
        print(
            f">> Threshold config: preset={self.thr_preset}, thr2d={self.thr2d:.3f}, "
            f"thr3d={self.thr3d:.3f}, normalize={not self.disable_threshold_norm}, "
            f"exclude_below_threshold={self.exclude_below_threshold}"
        )

    def _load_coefficients(self) -> None:
        with open(self.coef_json_path, "r", encoding="utf-8") as handle:
            coef_data = json.load(handle)
        self.k = int(coef_data.get("K", 5))
        self.policy = str(coef_data.get("Policy", "paired_sum"))
        self.model_family = str(coef_data.get("Model_Family", "unknown"))
        raw_coef = coef_data["coef"]
        sample_key = sorted(raw_coef.keys(), key=lambda x: int(x))[0]
        self.feature_dim = max(0, len(raw_coef[sample_key]) - 1)
        self.weights = torch.zeros((65, self.feature_dim + 1), dtype=torch.float32)
        for key, value in raw_coef.items():
            self.weights[int(key)] = torch.tensor(value, dtype=torch.float32)
        print(
            f">> Coefficient config: K={self.k}, policy={self.policy}, "
            f"feature_dim={self.feature_dim}, model_family={self.model_family}"
        )

    def _build_reference_tables(self) -> None:
        self.target_ids = self.human_targets_df["target_id"].astype(str).tolist()
        self.target_names = self.human_targets_df["target_name"].astype(str).tolist()
        self.target_id_to_index = {target_id: idx for idx, target_id in enumerate(self.target_ids)}
        self.num_targets = len(self.target_ids)

        self.compound_id_to_memmap_idx = dict(
            zip(
                self.reference_compounds_df["compound_id"],
                self.reference_compounds_df["memmap_idx"],
            )
        )
        activity = self.reference_activity_df.copy()
        activity["memmap_idx"] = activity["compound_id"].map(self.compound_id_to_memmap_idx)
        activity["target_index"] = activity["target_id"].map(self.target_id_to_index)
        activity = activity.dropna(subset=["memmap_idx", "target_index"]).copy()
        activity["memmap_idx"] = activity["memmap_idx"].astype(np.int64)
        activity["target_index"] = activity["target_index"].astype(np.int64)
        activity = activity.drop_duplicates(subset=["target_index", "memmap_idx"]).copy()

        self.row_idx_to_target_indices: Dict[int, np.ndarray] = {}
        for memmap_idx, group in activity.groupby("memmap_idx", sort=False):
            self.row_idx_to_target_indices[int(memmap_idx)] = group["target_index"].to_numpy(dtype=np.int64)

        self.target_to_memmap_indices: List[np.ndarray] = [
            np.zeros(0, dtype=np.int64) for _ in range(self.num_targets)
        ]
        for target_idx, group in activity.groupby("target_index", sort=False):
            self.target_to_memmap_indices[int(target_idx)] = group["memmap_idx"].to_numpy(dtype=np.int64)

    def _build_gpu_assets(self) -> None:
        print(">> Loading memmap assets into host memory")
        full_n = os.path.getsize(self.fp2_memmap_path) // (16 * np.dtype(np.uint64).itemsize)
        nconf = os.path.getsize(self.es5d_memmap_path) // (full_n * 18 * np.dtype(np.float32).itemsize)
        db_2d_ram = np.array(np.memmap(self.fp2_memmap_path, dtype=np.uint64, mode="r", shape=(full_n, 16)))
        db_3d_ram = np.array(np.memmap(self.es5d_memmap_path, dtype=np.float32, mode="r", shape=(full_n, nconf, 18)))

        print(">> Cloning score-engine assets to GPU memory")
        self.db2d: Dict[str, torch.Tensor] = {}
        self.db2d_ones: Dict[str, torch.Tensor] = {}
        self.db3d: Dict[str, torch.Tensor] = {}
        self.weights_gpu: Dict[str, torch.Tensor] = {}
        self.target_actives: Dict[str, List[torch.Tensor]] = {}
        for device in self.devices:
            fp_tensor = torch.from_numpy(np.unpackbits(db_2d_ram.view(np.uint8), axis=1)).to(
                device=device,
                dtype=torch.float16,
            )
            self.db2d[device] = fp_tensor
            self.db2d_ones[device] = fp_tensor.sum(dim=1).float()
            self.db3d[device] = torch.from_numpy(db_3d_ram).to(device=device)
            self.weights_gpu[device] = self.weights.to(device=device, non_blocking=True)
            self.target_actives[device] = [
                torch.tensor(indices, device=device, dtype=torch.int64)
                for indices in self.target_to_memmap_indices
            ]
        self.full_n = int(full_n)
        self.nconf = int(nconf)
        print(f">> Engine assets ready: fp_rows={self.full_n:,}, nconf={self.nconf}, K={self.k}")

    def _build_target_features(
        self,
        sim_2d: torch.Tensor,
        sim_3d: torch.Tensor,
        k_act: int,
    ) -> torch.Tensor:
        """Build target-level features compatible with the loaded coefficient layout."""
        batch_size = sim_2d.shape[0]
        if self.feature_dim == 2:
            k_actual = min(self.k, k_act)
            feats = torch.zeros((batch_size, 2), device=sim_2d.device, dtype=torch.float32)
            row_index = torch.arange(batch_size, device=sim_2d.device).unsqueeze(1).expand(batch_size, k_actual)
            force_independent_topk = self.exclude_below_threshold

            if self.policy == "paired_sum":
                score = sim_2d + sim_3d
            elif self.policy == "paired_2d":
                score = sim_2d
            elif self.policy == "paired_3d":
                score = sim_3d
            else:
                score = sim_2d + sim_3d

            if self.policy == "independent" or force_independent_topk:
                top_3d_idx = torch.topk(sim_3d, k_actual, dim=1).indices
                top_2d_idx = torch.topk(sim_2d, k_actual, dim=1).indices
                feats[:, 0] = sim_3d[row_index, top_3d_idx].max(dim=1).values
                feats[:, 1] = sim_2d[row_index, top_2d_idx].max(dim=1).values
            else:
                top_k_idx = torch.topk(score, k_actual, dim=1).indices
                feats[:, 0] = sim_3d[row_index, top_k_idx].max(dim=1).values
                feats[:, 1] = sim_2d[row_index, top_k_idx].max(dim=1).values

            return torch.clamp(feats, min=0.0)

        if self.feature_dim == 6:
            top_2d = torch.topk(sim_2d, k_act, dim=1).values.clamp(min=0.0)
            top_3d = torch.topk(sim_3d, k_act, dim=1).values.clamp(min=0.0)
            feats = torch.zeros((batch_size, 6), device=sim_2d.device, dtype=torch.float32)
            feats[:, 0] = top_3d.max(dim=1).values
            feats[:, 1] = top_3d.mean(dim=1)
            feats[:, 2] = torch.nan_to_num(top_3d.std(dim=1, unbiased=False), nan=0.0)
            feats[:, 3] = top_2d.max(dim=1).values
            feats[:, 4] = top_2d.mean(dim=1)
            feats[:, 5] = torch.nan_to_num(top_2d.std(dim=1, unbiased=False), nan=0.0)
            return feats

        expected_dim = self.k * 2
        if self.feature_dim != expected_dim:
            raise ValueError(
                f"Unsupported coefficient layout: feature_dim={self.feature_dim}, expected 6 or {expected_dim}"
            )

        if self.policy == "paired_sum":
            score = sim_2d + sim_3d
        elif self.policy == "paired_2d":
            score = sim_2d
        elif self.policy == "paired_3d":
            score = sim_3d
        else:
            score = sim_2d + sim_3d

        k_actual = min(self.k, k_act)
        feats = torch.zeros((batch_size, self.feature_dim), device=sim_2d.device, dtype=torch.float32)
        row_index = torch.arange(batch_size, device=sim_2d.device).unsqueeze(1).expand(batch_size, k_actual)
        force_independent_topk = self.exclude_below_threshold

        if self.policy == "independent" or force_independent_topk:
            top_3d_idx = torch.topk(sim_3d, k_actual, dim=1).indices
            top_2d_idx = torch.topk(sim_2d, k_actual, dim=1).indices
            feats[:, 0::2][:, :k_actual] = sim_3d[row_index, top_3d_idx]
            feats[:, 1::2][:, :k_actual] = sim_2d[row_index, top_2d_idx]
        else:
            top_k_idx = torch.topk(score, k_actual, dim=1).indices
            feats[:, 0::2][:, :k_actual] = sim_3d[row_index, top_k_idx]
            feats[:, 1::2][:, :k_actual] = sim_2d[row_index, top_k_idx]

        return torch.clamp(feats, min=0.0)

    def _run_gpu_half(
        self,
        query_memmap_idx: np.ndarray,
        query_heavy_atoms: np.ndarray,
        device: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if query_memmap_idx.size == 0:
            empty = np.zeros((0, self.num_targets), dtype=np.float32)
            return empty, empty, empty

        q_idx = torch.tensor(query_memmap_idx, device=device, dtype=torch.int64)
        q_ha = torch.tensor(query_heavy_atoms, device=device, dtype=torch.int64)
        ha_bins = torch.clamp(q_ha, 10, 60).long()
        local_weights = self.weights_gpu[device][ha_bins]
        w_int = local_weights[:, 0]
        w_feats = local_weights[:, 1:]

        q_2d = self.db2d[device][q_idx]
        q_ones = self.db2d_ones[device][q_idx].unsqueeze(1)
        q_3d = torch.nan_to_num(self.db3d[device][q_idx], nan=1.0e4).unsqueeze(1)

        max_sim_2d = torch.zeros((len(query_memmap_idx), self.num_targets), device=device, dtype=torch.float32)
        max_sim_3d = torch.zeros((len(query_memmap_idx), self.num_targets), device=device, dtype=torch.float32)
        combined = torch.zeros((len(query_memmap_idx), self.num_targets), device=device, dtype=torch.float32)

        for target_idx, actives in enumerate(self.target_actives[device]):
            m_size = len(actives)
            if m_size == 0:
                continue

            k_act = min(self.k, m_size)
            a_2d = self.db2d[device][actives]
            intersection = torch.matmul(q_2d, a_2d.T).float()
            union = q_ones + self.db2d_ones[device][actives].unsqueeze(0) - intersection
            sim_2d = torch.where(union > 0, intersection / union, torch.zeros_like(union))
            sim_2d.masked_fill_(q_idx.unsqueeze(1) == actives.unsqueeze(0), -1e9)

            a_3d = torch.nan_to_num(self.db3d[device][actives], nan=-1.0e4)
            sim_3d = torch.full((len(query_memmap_idx), m_size), -1e9, device=device, dtype=torch.float32)
            for start in range(0, m_size, self.chunk_m):
                chunk = a_3d[start : start + self.chunk_m].unsqueeze(0)
                dist = torch.cdist(q_3d, chunk, p=1.0)
                sim_3d[:, start : start + self.chunk_m] = (1.0 / (1.0 + (dist / 18.0))).amax(dim=(2, 3))
            sim_3d.masked_fill_(q_idx.unsqueeze(1) == actives.unsqueeze(0), -1e9)

            sim_2d_raw = sim_2d.clone()
            sim_3d_raw = sim_3d.clone()
            valid_2d = sim_2d_raw > -1.0e8
            valid_3d = sim_3d_raw > -1.0e8
            raw_max_2d = torch.where(valid_2d, sim_2d_raw, -1.0).amax(dim=1).clamp(min=0.0)
            raw_max_3d = torch.where(valid_3d, sim_3d_raw, -1.0).amax(dim=1).clamp(min=0.0)

            if self.exclude_below_threshold:
                below_2d = valid_2d & (sim_2d_raw < self.thr2d)
                below_3d = valid_3d & (sim_3d_raw < self.thr3d)
                sim_2d = sim_2d.masked_fill(below_2d, -1.0)
                sim_3d = sim_3d.masked_fill(below_3d, -1.0)

            if not self.disable_threshold_norm:
                if self.thr2d < 1.0:
                    sim_2d = torch.where(
                        valid_2d,
                        torch.clamp((sim_2d - self.thr2d) / (1.0 - self.thr2d), 0.0, 1.0),
                        sim_2d,
                    )
                if self.thr3d < 1.0:
                    sim_3d = torch.where(
                        valid_3d,
                        torch.clamp((sim_3d - self.thr3d) / (1.0 - self.thr3d), 0.0, 1.0),
                        sim_3d,
                    )

            sim_2d.masked_fill_(~valid_2d, -1.0)
            sim_3d.masked_fill_(~valid_3d, -1.0)

            feats = self._build_target_features(sim_2d=sim_2d, sim_3d=sim_3d, k_act=k_act)

            logits = w_int + torch.sum(w_feats * feats, dim=1)
            combined[:, target_idx] = torch.sigmoid(logits)
            max_sim_2d[:, target_idx] = raw_max_2d
            max_sim_3d[:, target_idx] = raw_max_3d

        torch.cuda.synchronize(device=device)
        return (
            max_sim_2d.cpu().numpy(),
            max_sim_3d.cpu().numpy(),
            combined.cpu().numpy(),
        )

    def predict_query_batch(
        self,
        query_batch_df: pd.DataFrame,
        top_k: int,
        min_score: float,
    ) -> List[pd.DataFrame]:
        memmap_idx = query_batch_df["memmap_idx"].to_numpy(dtype=np.int64)
        heavy_atoms = query_batch_df["heavy_atom_count"].to_numpy(dtype=np.int64)
        if len(self.devices) == 1:
            max2d, max3d, combined = self._run_gpu_half(memmap_idx, heavy_atoms, self.devices[0])
        else:
            split_point = (len(memmap_idx) + 1) // 2
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_left = executor.submit(
                    self._run_gpu_half,
                    memmap_idx[:split_point],
                    heavy_atoms[:split_point],
                    self.devices[0],
                )
                future_right = executor.submit(
                    self._run_gpu_half,
                    memmap_idx[split_point:],
                    heavy_atoms[split_point:],
                    self.devices[1],
                )
                left_max2d, left_max3d, left_combined = future_left.result()
                right_max2d, right_max3d, right_combined = future_right.result()
            max2d = np.concatenate([left_max2d, right_max2d], axis=0)
            max3d = np.concatenate([left_max3d, right_max3d], axis=0)
            combined = np.concatenate([left_combined, right_combined], axis=0)

        batch_results: List[pd.DataFrame] = []
        target_ids_arr = np.asarray(self.target_ids, dtype=object)
        target_names_arr = np.asarray(self.target_names, dtype=object)
        for batch_idx in range(len(query_batch_df)):
            combined_row = combined[batch_idx]
            order = np.argsort(-combined_row, kind="stable")
            ranks = np.arange(1, len(order) + 1, dtype=np.int64)
            keep_mask = (ranks <= int(top_k)) | (combined_row[order] >= float(min_score))
            kept = order[keep_mask]
            batch_results.append(
                pd.DataFrame(
                    {
                        "target_id": target_ids_arr[kept],
                        "target_name": target_names_arr[kept],
                        "max_sim_2d": max2d[batch_idx][kept].astype(np.float32),
                        "max_sim_3d": max3d[batch_idx][kept].astype(np.float32),
                        "combined_score": combined_row[kept].astype(np.float32),
                        "rank": ranks[keep_mask].astype(np.int64),
                    }
                )
            )
        return batch_results


def predict_targets_leave_one_out(
    query_row: pd.Series,
    reference_compounds_df: pd.DataFrame,
    reference_activity_df: pd.DataFrame,
    human_targets_df: pd.DataFrame,
    adapter: DualGPUSTPEngineAdapter,
    top_k: int = 100,
    min_score: float = 0.02,
) -> pd.DataFrame:
    """Return target-level scores for one leave-one-out query molecule."""
    return adapter.predict_query_batch(pd.DataFrame([query_row.to_dict()]), top_k=top_k, min_score=min_score)[0]


def run_leave_one_out(
    query_compounds_df: pd.DataFrame,
    truth_map: Dict[str, Set[str]],
    adapter: DualGPUSTPEngineAdapter,
    keep_top_rank: int,
    keep_min_score: float,
    progress_interval: int,
) -> pd.DataFrame:
    """Generate query-level leave-one-out prediction logs."""
    result_frames: List[pd.DataFrame] = []
    total_queries = len(query_compounds_df)
    started_at = time.time()
    last_reported = 0

    for start_idx in range(0, total_queries, adapter.gpu_batch_size):
        end_idx = min(start_idx + adapter.gpu_batch_size, total_queries)
        processed = end_idx
        should_report = (
            start_idx == 0
            or processed == total_queries
            or (processed - last_reported) >= max(1, int(progress_interval))
        )
        if should_report:
            elapsed = max(time.time() - started_at, 1.0e-9)
            rate = processed / elapsed
            remaining = max(total_queries - processed, 0)
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta_minutes = eta_seconds / 60.0 if np.isfinite(eta_seconds) else float("inf")
            print(
                f">> Progress: {processed:,}/{total_queries:,} queries "
                f"({processed / total_queries:.1%}) | elapsed={elapsed / 60.0:.1f} min "
                f"| rate={rate:.2f} q/s | eta={eta_minutes:.1f} min",
                flush=True,
            )
            last_reported = processed

        query_batch_df = query_compounds_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        batch_predictions = adapter.predict_query_batch(
            query_batch_df=query_batch_df,
            top_k=keep_top_rank,
            min_score=keep_min_score,
        )

        for local_idx, prediction_df in enumerate(batch_predictions):
            query_row = query_batch_df.iloc[local_idx]
            prediction_df["query_id"] = str(query_row["query_id"])
            prediction_df["query_smiles"] = str(query_row["canonical_smiles"])
            prediction_df["query_heavy_atom_count"] = int(query_row["heavy_atom_count"])
            prediction_df["size_bin"] = str(query_row["size_bin"])
            prediction_df["cv_scheme"] = "loo"
            prediction_df["is_true_target"] = prediction_df["target_id"].isin(
                truth_map.get(str(query_row["compound_id"]), set())
            )
            result_frames.append(prediction_df)

    if not result_frames:
        return pd.DataFrame(
            columns=[
                "query_id",
                "query_smiles",
                "query_heavy_atom_count",
                "size_bin",
                "cv_scheme",
                "target_id",
                "target_name",
                "max_sim_2d",
                "max_sim_3d",
                "combined_score",
                "rank",
                "is_true_target",
            ]
        )

    output_df = pd.concat(result_frames, ignore_index=True)
    ordered_columns = [
        "query_id",
        "query_smiles",
        "query_heavy_atom_count",
        "size_bin",
        "cv_scheme",
        "target_id",
        "target_name",
        "max_sim_2d",
        "max_sim_3d",
        "combined_score",
        "rank",
        "is_true_target",
    ]
    return output_df[ordered_columns].copy()


def save_cv_predictions(predictions_df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist the final leave-one-out log as parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(output_path, index=False)
    print(f">> Saved {len(predictions_df):,} rows to {output_path}")


def summarize_query_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-query summary statistics from the leave-one-out prediction log."""
    if predictions_df.empty:
        return pd.DataFrame(
            columns=[
                "query_id",
                "size_bin",
                "rows_kept",
                "true_targets_kept",
                "true_targets_top1_count",
                "true_targets_top5_count",
                "true_targets_top10_count",
                "true_targets_top15_count",
                "true_targets_top100_count",
                "best_true_rank",
                "top1_hit",
                "top5_hit",
                "top10_hit",
                "top15_hit",
                "top100_hit",
            ]
        )

    work = predictions_df.copy()
    work["rank"] = pd.to_numeric(work["rank"], errors="coerce").fillna(0).astype(np.int64)
    grouped = work.groupby(["query_id", "size_bin"], sort=False)
    summary = grouped.agg(
        rows_kept=("target_id", "size"),
        true_targets_kept=("is_true_target", "sum"),
    ).reset_index()

    true_only = work[work["is_true_target"]].copy()
    if true_only.empty:
        summary["true_targets_top1_count"] = 0
        summary["true_targets_top5_count"] = 0
        summary["true_targets_top10_count"] = 0
        summary["true_targets_top15_count"] = 0
        summary["true_targets_top100_count"] = 0
        summary["best_true_rank"] = np.nan
    else:
        true_rank_counts = (
            true_only.assign(
                true_targets_top1_count=true_only["rank"].le(1).astype(np.int64),
                true_targets_top5_count=true_only["rank"].le(5).astype(np.int64),
                true_targets_top10_count=true_only["rank"].le(10).astype(np.int64),
                true_targets_top15_count=true_only["rank"].le(15).astype(np.int64),
                true_targets_top100_count=true_only["rank"].le(100).astype(np.int64),
            )
            .groupby("query_id", sort=False)
            .agg(
                true_targets_top1_count=("true_targets_top1_count", "sum"),
                true_targets_top5_count=("true_targets_top5_count", "sum"),
                true_targets_top10_count=("true_targets_top10_count", "sum"),
                true_targets_top15_count=("true_targets_top15_count", "sum"),
                true_targets_top100_count=("true_targets_top100_count", "sum"),
            )
            .reset_index()
        )
        best_true = true_only.groupby("query_id", sort=False)["rank"].min().rename("best_true_rank")
        summary = summary.merge(true_rank_counts, on="query_id", how="left")
        summary = summary.merge(best_true, on="query_id", how="left")

    count_columns = [
        "true_targets_top1_count",
        "true_targets_top5_count",
        "true_targets_top10_count",
        "true_targets_top15_count",
        "true_targets_top100_count",
    ]
    for column in count_columns:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(np.int64)

    summary["top1_hit"] = summary["best_true_rank"].le(1).fillna(False)
    summary["top5_hit"] = summary["best_true_rank"].le(5).fillna(False)
    summary["top10_hit"] = summary["best_true_rank"].le(10).fillna(False)
    summary["top15_hit"] = summary["best_true_rank"].le(15).fillna(False)
    summary["top100_hit"] = summary["best_true_rank"].le(100).fillna(False)
    return summary


def summarize_size_bin_precision(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Build empirical precision-by-rank tables for each size bin."""
    if predictions_df.empty:
        return pd.DataFrame(
            columns=[
                "size_bin",
                "rank",
                "n_predictions",
                "n_true_positives",
                "precision_at_rank",
            ]
        )

    work = predictions_df.copy()
    work["rank"] = pd.to_numeric(work["rank"], errors="coerce").fillna(0).astype(np.int64)
    work["is_true_target"] = work["is_true_target"].astype(bool)
    precision_df = (
        work.groupby(["size_bin", "rank"], sort=True)
        .agg(
            n_predictions=("target_id", "size"),
            n_true_positives=("is_true_target", "sum"),
        )
        .reset_index()
    )
    precision_df["precision_at_rank"] = np.where(
        precision_df["n_predictions"] > 0,
        precision_df["n_true_positives"] / precision_df["n_predictions"],
        0.0,
    )
    return precision_df


def build_run_summary(
    predictions_df: pd.DataFrame,
    sampled_queries_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    adapter: DualGPUSTPEngineAdapter,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build a compact run-level metadata summary for downstream review."""
    query_summary = summarize_query_predictions(predictions_df)
    rows_per_query = query_summary["rows_kept"] if not query_summary.empty else pd.Series(dtype=np.int64)
    return {
        "sampled_queries": int(len(sampled_queries_df)),
        "human_targets": int(len(targets_df)),
        "output_rows": int(len(predictions_df)),
        "coef_json": str(args.coef_json),
        "coef_k": int(adapter.k),
        "coef_policy": adapter.policy,
        "coef_feature_dim": int(adapter.feature_dim),
        "thr_preset": args.thr_preset,
        "thr2d": float(adapter.thr2d),
        "thr3d": float(adapter.thr3d),
        "disable_threshold_norm": bool(args.disable_threshold_norm),
        "exclude_below_threshold": bool(args.exclude_below_threshold),
        "keep_top_rank": int(args.keep_top_rank),
        "keep_min_score": float(args.keep_min_score),
        "rows_per_query_min": int(rows_per_query.min()) if not rows_per_query.empty else 0,
        "rows_per_query_median": float(rows_per_query.median()) if not rows_per_query.empty else 0.0,
        "rows_per_query_max": int(rows_per_query.max()) if not rows_per_query.empty else 0,
    }


def save_supporting_outputs(
    predictions_df: pd.DataFrame,
    sampled_queries_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    adapter: DualGPUSTPEngineAdapter,
    args: argparse.Namespace,
) -> None:
    """Save summary sidecar outputs that support later precision-curve fitting work."""
    output_path = Path(args.output)
    stem = output_path.with_suffix("")

    query_summary = summarize_query_predictions(predictions_df)
    precision_summary = summarize_size_bin_precision(predictions_df)
    run_summary = build_run_summary(predictions_df, sampled_queries_df, targets_df, adapter, args)

    query_summary_path = Path(f"{stem}.query_summary.csv")
    precision_path = Path(f"{stem}.size_bin_precision.csv")
    summary_json_path = Path(f"{stem}.run_summary.json")

    query_summary.to_csv(query_summary_path, index=False)
    precision_summary.to_csv(precision_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    print(f">> Saved query summary: {query_summary_path}")
    print(f">> Saved size-bin precision summary: {precision_path}")
    print(f">> Saved run summary: {summary_json_path}")


def resummarize_existing_output(parquet_path: str | Path) -> None:
    """Rebuild sidecar summary files from an existing prediction parquet."""
    parquet_path = Path(parquet_path)
    print(f">> Loading existing parquet for resummarization: {parquet_path}")
    predictions_df = pd.read_parquet(parquet_path)
    stem = parquet_path.with_suffix("")
    query_summary = summarize_query_predictions(predictions_df)
    precision_summary = summarize_size_bin_precision(predictions_df)
    query_summary_path = Path(f"{stem}.query_summary.csv")
    precision_path = Path(f"{stem}.size_bin_precision.csv")
    query_summary.to_csv(query_summary_path, index=False)
    precision_summary.to_csv(precision_path, index=False)
    print(f">> Rebuilt query summary: {query_summary_path}")
    print(f">> Rebuilt size-bin precision summary: {precision_path}")


def main() -> None:
    """Parse arguments and run the leave-one-out prediction workflow."""
    if hasattr(os.sys.stdout, "reconfigure"):
        os.sys.stdout.reconfigure(line_buffering=True)
    if hasattr(os.sys.stderr, "reconfigure"):
        os.sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Generate SwissTargetPrediction-style leave-one-out prediction logs."
    )
    parser.add_argument("--compound-table", default=str(DEFAULT_DATA_DIR / "compound_table.parquet"))
    parser.add_argument("--target-table", default=str(DEFAULT_DATA_DIR / "target_table.parquet"))
    parser.add_argument("--activity-table", default=str(DEFAULT_DATA_DIR / "activity_table.parquet"))
    parser.add_argument("--meta-file", default=str(DEFAULT_META_FILE))
    parser.add_argument("--fp2-memmap", default=str(DEFAULT_FP2))
    parser.add_argument("--es5d-memmap", default=str(DEFAULT_ES5D))
    parser.add_argument("--coef-json", default=str(DEFAULT_COEF_JSON))
    parser.add_argument("--output", default=str(DEFAULT_DATA_DIR / "cv_predictions.parquet"))
    parser.add_argument("--max-queries-per-size-bin", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=20260312)
    parser.add_argument("--keep-top-rank", type=int, default=100)
    parser.add_argument("--keep-min-score", type=float, default=0.02)
    parser.add_argument("--gpu-batch-size", type=int, default=32)
    parser.add_argument("--chunk-m", type=int, default=1024)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--allow-single-gpu", action="store_true")
    parser.add_argument("--thr-preset", choices=["none", "stp2014", "stp2019"], default="stp2019")
    parser.add_argument("--thr2d", type=float, default=0.65)
    parser.add_argument("--thr3d", type=float, default=0.85)
    parser.add_argument("--disable-threshold-norm", action="store_true")
    parser.add_argument("--exclude-below-threshold", action="store_true")
    parser.add_argument(
        "--resummarize-only",
        default="",
        help="Existing parquet path to rebuild query_summary and size_bin_precision without rerunning inference.",
    )
    args = parser.parse_args()

    if args.resummarize_only:
        resummarize_existing_output(args.resummarize_only)
        return

    print(">> Loading leave-one-out inputs")
    compounds_df, targets_df, activities_df = load_inputs(
        compound_path=args.compound_table,
        target_path=args.target_table,
        activity_path=args.activity_table,
        meta_path=args.meta_file,
    )
    sampled_queries_df = sample_queries_by_size_bin(
        compounds_df=compounds_df,
        max_per_bin=args.max_queries_per_size_bin,
        random_seed=args.random_seed,
    )
    truth_map = build_truth_map(activities_df)

    print(">> Initializing dual-GPU STP engine adapter")
    adapter = DualGPUSTPEngineAdapter(
        reference_compounds_df=compounds_df,
        reference_activity_df=activities_df,
        human_targets_df=targets_df,
        fp2_memmap_path=args.fp2_memmap,
        es5d_memmap_path=args.es5d_memmap,
        coef_json_path=args.coef_json,
        gpu_batch_size=args.gpu_batch_size,
        chunk_m=args.chunk_m,
        require_dual_gpu=not args.allow_single_gpu,
        thr_preset=args.thr_preset,
        thr2d=args.thr2d,
        thr3d=args.thr3d,
        disable_threshold_norm=args.disable_threshold_norm,
        exclude_below_threshold=args.exclude_below_threshold,
    )

    print(">> Running leave-one-out prediction log generation")
    predictions_df = run_leave_one_out(
        query_compounds_df=sampled_queries_df,
        truth_map=truth_map,
        adapter=adapter,
        keep_top_rank=args.keep_top_rank,
        keep_min_score=args.keep_min_score,
        progress_interval=args.progress_interval,
    )

    print(">> Writing output parquet")
    save_cv_predictions(predictions_df=predictions_df, output_path=args.output)
    print(">> Writing supporting summaries")
    save_supporting_outputs(
        predictions_df=predictions_df,
        sampled_queries_df=sampled_queries_df,
        targets_df=targets_df,
        adapter=adapter,
        args=args,
    )
    print(
        f">> Completed: sampled_queries={len(sampled_queries_df):,}, "
        f"targets={len(targets_df):,}, output_rows={len(predictions_df):,}"
    )


if __name__ == "__main__":
    main()

# Example:
# python /mnt/d/stp_proj_v3/script/p4_0_generate_cv_predictions.py \
#   --compound-table /mnt/d/stp_proj_v3/data/compound_table.parquet \
#   --target-table /mnt/d/stp_proj_v3/data/target_table.parquet \
#   --activity-table /mnt/d/stp_proj_v3/data/activity_table.parquet \
#   --meta-file /mnt/d/stp_proj_v3/features_store/final_training_meta.parquet \
#   --fp2-memmap /mnt/d/stp_proj_v3/features_store/fp2_aligned.memmap \
#   --es5d-memmap /mnt/d/stp_proj_v3/features_store/es5d_db_k20.memmap \
#   --coef-json /mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_20260311/p3_1_1_K1_paired_trainer_fast__K1__paired_sum__all__C10p0__Y2023__Hd6d53c2431/stp_coef_K1_paired_sum_all.json \
#   --max-queries-per-size-bin 300 \
#   --progress-interval 25 \
#   --thr-preset stp2014 \
#   --output /mnt/d/stp_proj_v3/data/cv_predictions.parquet
