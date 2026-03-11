# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:37:17 2026

@author: KIOM_User
"""

# -*- coding: utf-8 -*-
"""
Phase 3 v3.0 (FULL SCRIPT): DATA-PARALLEL AUC-BOOSTED DUAL-GPU STP TRAINING
- [DATA PARALLEL] Full DB cloned to both GPUs. Batch is split 50/50 for true 100% Dual-GPU load.
- [FIX-A..F] Scaffold masking, StandardScaler original space mapping, Gaussian smoothing included.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import time
import json
import warnings
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


class STPAucBoostedTrainer:
    def __init__(
        self,
        meta_file: str,
        fp2_memmap: str,
        es5d_memmap: str,
        out_dir: str = "features_store",
        k_mode: int = 5,
        k_policy: str = "paired_sum",
        c_reg: float = 1.0,
        cutoff_year: int = 2023,
        batch_size: int = 4096,
        chunk_3d_b: int = 128,
        smooth_sigma: float = 1.5,
        seed: int = 42,
    ):
        self.meta_file = meta_file
        self.fp2_path = fp2_memmap
        self.es5d_path = es5d_memmap
        self.out_dir = out_dir

        self.K = int(k_mode)
        self.POLICY = str(k_policy)
        self.C_REG = float(c_reg)
        self.CUTOFF_YEAR = int(cutoff_year)

        self.BATCH_SIZE = int(batch_size)
        self.CHUNK_3D_B = int(chunk_3d_b)
        self.SMOOTH_SIGMA = float(smooth_sigma)
        self.SEED = int(seed)

        os.makedirs(self.out_dir, exist_ok=True)
        self._initialize_gpu_engine()

    # -------------------------
    # Utils
    # -------------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _coef_to_original_space(coef_scaled: np.ndarray, intercept_scaled: float, scaler: StandardScaler) -> tuple[np.ndarray, float]:
        scale = scaler.scale_.astype(np.float64)
        mean = scaler.mean_.astype(np.float64)
        w_prime = coef_scaled.astype(np.float64)
        b_prime = float(intercept_scaled)

        w = w_prime / scale
        b = b_prime - float(np.sum((w_prime * mean) / scale))
        return w.astype(np.float64), float(b)

    # -------------------------
    # GPU engine init (TRUE DATA PARALLEL)
    # -------------------------
    def _initialize_gpu_engine(self):
        print(f"\n>> 1. Initializing TRUE DATA-PARALLEL Dual-GPU Engine (K={self.K})...")

        full_df = pd.read_parquet(self.meta_file)

        if "scaffold_smiles" not in full_df.columns:
            raise KeyError("meta parquet must contain 'scaffold_smiles' column for scaffold masking.")
        full_df["scaffold_hash"] = pd.util.hash_pandas_object(full_df["scaffold_smiles"], index=False).astype(np.int64)

        if "publication_year" in full_df.columns:
            full_df["publication_year"] = pd.to_numeric(full_df["publication_year"], errors="coerce").fillna(-1).astype(int)

        self.FULL_N = os.path.getsize(self.fp2_path) // 128
        self.NCONF = os.path.getsize(self.es5d_path) // (self.FULL_N * 72)
        print(f"   - FULL_N={self.FULL_N:,}, NCONF={self.NCONF}")

        print(f"   - [RAM] Loading memmaps into System RAM...")
        db_2d_ram = np.array(np.memmap(self.fp2_path, dtype=np.uint64, mode="r", shape=(self.FULL_N, 16)))
        db_3d_ram = np.array(np.memmap(self.es5d_path, dtype=np.float32, mode="r", shape=(self.FULL_N, self.NCONF, 18)))

        print("   - [GPU] Cloning full database to BOTH cuda:0 and cuda:1 for Data Parallelism...")
        # ==========================================
        # GPU 0 메모리 적재
        # ==========================================
        self.db2d_0 = torch.from_numpy(np.unpackbits(db_2d_ram.view(np.uint8), axis=1)).to(device="cuda:0", dtype=torch.float16)
        self.db2d_ones_0 = self.db2d_0.sum(dim=1).float()
        self.db3d_0 = torch.from_numpy(db_3d_ram).to(device="cuda:0")

        # ==========================================
        # GPU 1 메모리 복제 (통째로 넘김)
        # ==========================================
        self.db2d_1 = self.db2d_0.to(device="cuda:1", non_blocking=True)
        self.db2d_ones_1 = self.db2d_ones_0.to(device="cuda:1", non_blocking=True)
        self.db3d_1 = self.db3d_0.to(device="cuda:1", non_blocking=True)

        mem_idx = full_df["memmap_idx"].values.astype(np.int64)
        if mem_idx.min() < 0 or mem_idx.max() >= self.FULL_N:
            raise ValueError("memmap_idx out of range for FULL_N.")

        scaf_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        scaf_arr[mem_idx] = full_df["scaffold_hash"].values.astype(np.int64)
        self.gpu_scaf_0 = torch.tensor(scaf_arr, device="cuda:0", dtype=torch.int64)
        self.gpu_scaf_1 = self.gpu_scaf_0.to(device="cuda:1", non_blocking=True)

        folds_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        if "cv_fold" in full_df.columns:
            folds_arr[mem_idx] = full_df["cv_fold"].fillna(-1).astype(int).values
        self.gpu_fold_0 = torch.tensor(folds_arr, device="cuda:0", dtype=torch.int64)
        self.gpu_fold_1 = self.gpu_fold_0.to(device="cuda:1", non_blocking=True)

        if "publication_year" in full_df.columns:
            self.meta_df = full_df[full_df["publication_year"] <= self.CUTOFF_YEAR].copy()
            self.future_df = full_df[full_df["publication_year"] > self.CUTOFF_YEAR].copy()
            print(f"   - ⏳ Temporal Split: Train(<= {self.CUTOFF_YEAR}) vs OOT(> {self.CUTOFF_YEAR}); NaNs treated as past (-1)")
        else:
            self.meta_df = full_df.copy()
            self.future_df = pd.DataFrame()

        positives = self.meta_df[self.meta_df["set_type"] == "Positive"]
        tmap = positives.groupby("target_chembl_id")["memmap_idx"].apply(list)

        self.target_actives_cuda0 = {k: torch.tensor(v, device="cuda:0", dtype=torch.int64) for k, v in tmap.items()}
        self.target_actives_cuda1 = {k: torch.tensor(v, device="cuda:1", dtype=torch.int64) for k, v in tmap.items()}

        torch.cuda.synchronize()
        print(f"   - Setup complete. Targets with actives (train): {len(self.target_actives_cuda0):,}")

    # -------------------------
    # Feature extraction (Data Parallel)
    # -------------------------
    def _run_gpu_half(self, q_idx, q_scaf, q_fold, target_id, device_id):
        B = len(q_idx)
        if B == 0: return None
            
        if device_id == 0:
            db2d, db2d_ones, db3d = self.db2d_0, self.db2d_ones_0, self.db3d_0
            gpu_scaf, gpu_fold = self.gpu_scaf_0, self.gpu_fold_0
            actives = self.target_actives_cuda0[target_id]
            device = "cuda:0"
        else:
            db2d, db2d_ones, db3d = self.db2d_1, self.db2d_ones_1, self.db3d_1
            gpu_scaf, gpu_fold = self.gpu_scaf_1, self.gpu_fold_1
            actives = self.target_actives_cuda1[target_id]
            device = "cuda:1"

        M = len(actives)

        # 2D 
        q_2d = db2d[q_idx]
        a_2d = db2d[actives]
        intersection = torch.matmul(q_2d, a_2d.T).float()
        union = db2d_ones[q_idx].unsqueeze(1) + db2d_ones[actives].unsqueeze(0) - intersection
        sim_2d = torch.where(union > 0, intersection / union, torch.zeros_like(union))

        # 3D
        q_3d = torch.nan_to_num(db3d[q_idx], nan=1e4)
        a_3d = torch.nan_to_num(db3d[actives], nan=-1e4)
        sim_3d = torch.empty((B, M), device=device, dtype=torch.float32)
        a_3d_exp = a_3d.unsqueeze(0)

        chunk_size = max(128, 250_000_000 // (M * 400 + 1))
        for i in range(0, B, chunk_size):
            q_chunk = q_3d[i:i + chunk_size].unsqueeze(1)
            dist = torch.cdist(q_chunk, a_3d_exp, p=1.0)
            sim_3d[i:i + chunk_size] = (1.0 / (1.0 + (dist / 18.0))).amax(dim=(2, 3))

        # 마스킹
        a_scaf_exp = gpu_scaf[actives].unsqueeze(0)
        mask = (q_idx.unsqueeze(1) != actives.unsqueeze(0)) & \
               (q_scaf.unsqueeze(1) != a_scaf_exp) & \
               (a_scaf_exp != -1)

        if q_fold is not None:
            mask &= (q_fold.unsqueeze(1) != gpu_fold[actives].unsqueeze(0))

        sim_2d.masked_fill_(~mask, -1e9)
        sim_3d.masked_fill_(~mask, -1e9)

        k_actual = min(self.K, M)
        top_2d_vals = torch.topk(sim_2d, k_actual, dim=1).values.clamp(min=0.0)
        top_3d_vals = torch.topk(sim_3d, k_actual, dim=1).values.clamp(min=0.0)

        return top_2d_vals, top_3d_vals

    def _extract_summary_features_batched(self, query_indices, target_id, query_scafs, query_folds=None):
        B = len(query_indices)
        if target_id not in self.target_actives_cuda0:
            return torch.zeros((B, 6), dtype=torch.float32, device="cpu")

        mid = B // 2

        q_idx_0, q_scaf_0 = query_indices[:mid], query_scafs[:mid]
        q_fold_0 = query_folds[:mid] if query_folds is not None else None

        q_idx_1 = query_indices[mid:].to("cuda:1", non_blocking=True)
        q_scaf_1 = query_scafs[mid:].to("cuda:1", non_blocking=True)
        q_fold_1 = query_folds[mid:].to("cuda:1", non_blocking=True) if query_folds is not None else None

        res_0 = self._run_gpu_half(q_idx_0, q_scaf_0, q_fold_0, target_id, 0)
        res_1 = self._run_gpu_half(q_idx_1, q_scaf_1, q_fold_1, target_id, 1)

        feats = torch.zeros((B, 6), dtype=torch.float32, device="cpu")

        if res_0 is not None:
            t2d, t3d = res_0
            feats[:mid, 0], feats[:mid, 1], feats[:mid, 2] = t3d.max(dim=1).values.cpu(), t3d.mean(dim=1).cpu(), torch.nan_to_num(t3d.std(dim=1, unbiased=False), nan=0.0).cpu()
            feats[:mid, 3], feats[:mid, 4], feats[:mid, 5] = t2d.max(dim=1).values.cpu(), t2d.mean(dim=1).cpu(), torch.nan_to_num(t2d.std(dim=1, unbiased=False), nan=0.0).cpu()

        if res_1 is not None:
            t2d, t3d = res_1
            feats[mid:, 0], feats[mid:, 1], feats[mid:, 2] = t3d.max(dim=1).values.cpu(), t3d.mean(dim=1).cpu(), torch.nan_to_num(t3d.std(dim=1, unbiased=False), nan=0.0).cpu()
            feats[mid:, 3], feats[mid:, 4], feats[mid:, 5] = t2d.max(dim=1).values.cpu(), t2d.mean(dim=1).cpu(), torch.nan_to_num(t2d.std(dim=1, unbiased=False), nan=0.0).cpu()

        return feats

    def _build_dataset_batched(self, df: pd.DataFrame, desc: str) -> pd.DataFrame:
        print(f"\n>> 2. Generating Summary Features for {desc} (N={len(df):,}) ...")
        start_time = time.time()
        is_oot = (desc.lower().find("oot") >= 0)

        df = df.copy()
        if "scaffold_hash" not in df.columns:
            df["scaffold_hash"] = pd.util.hash_pandas_object(df["scaffold_smiles"], index=False).astype(np.int64)

        df["sort_idx"] = np.arange(len(df), dtype=np.int64)
        grouped = df.groupby("target_chembl_id", sort=False)

        all_features = np.zeros((len(df), 6), dtype=np.float32)

        with tqdm(total=len(df), desc="Matrix Generation", mininterval=5.0) as pbar:
            for target_id, group in grouped:
                group_indices = group["sort_idx"].values
                q_memmap = torch.tensor(group["memmap_idx"].values, device="cuda:0", dtype=torch.int64)
                q_scafs = torch.tensor(group["scaffold_hash"].values, device="cuda:0", dtype=torch.int64)

                if (not is_oot) and ("cv_fold" in group.columns):
                    q_folds = torch.tensor(group["cv_fold"].fillna(-1).astype(int).values, device="cuda:0", dtype=torch.int64)
                else:
                    q_folds = None

                for i in range(0, len(q_memmap), self.BATCH_SIZE):
                    batch_memmap = q_memmap[i:i + self.BATCH_SIZE]
                    batch_scafs = q_scafs[i:i + self.BATCH_SIZE]
                    batch_folds = None if q_folds is None else q_folds[i:i + self.BATCH_SIZE]

                    batch_feats = self._extract_summary_features_batched(
                        batch_memmap, target_id, batch_scafs, batch_folds
                    )
                    all_features[group_indices[i:i + self.BATCH_SIZE]] = batch_feats.numpy()
                    pbar.update(len(batch_memmap))

        cols = ["s_3d_max", "s_3d_mean", "s_3d_std", "s_2d_max", "s_2d_mean", "s_2d_std"]

        out_df = pd.DataFrame({
            "heavy_atoms": df["heavy_atoms"].values.astype(np.int32),
            "ha_bin": np.clip(df["heavy_atoms"].values.astype(np.int32), 10, 60),
            "label": (df["set_type"] == "Positive").astype(int).values,
            "cv_fold": df["cv_fold"].fillna(-1).astype(int).values if "cv_fold" in df.columns else np.full(len(df), -1, dtype=int),
        })
        out_df = pd.concat([out_df, pd.DataFrame(all_features, columns=cols)], axis=1)

        print(f"   ✅ Feature matrix generated in {time.time() - start_time:.2f}s")
        return out_df

    # -------------------------
    # Training + evaluation
    # -------------------------
    def execute_cv_and_oot_evaluation(self):
        self.train_df = self._build_dataset_batched(self.meta_df, "Past Training Data")
        feature_cols = ["s_3d_max", "s_3d_mean", "s_3d_std", "s_2d_max", "s_2d_mean", "s_2d_std"]

        unique_bins = sorted(self.train_df["ha_bin"].unique().tolist())
        unique_folds = np.sort(self.train_df["cv_fold"].unique())

        print("\n>> 3. Executing CV Training (balanced LR + per-bin scaling -> original space)...")
        cv_reports = []

        self.production_params = {}
        raw_coefficients = np.zeros((65, 7), dtype=np.float64) 
        trained_mask = np.zeros(65, dtype=np.int8)

        for ha_bin in tqdm(unique_bins, desc="Training per HA-bin"):
            subset = self.train_df[self.train_df["ha_bin"] == ha_bin]
            X = subset[feature_cols].values.astype(np.float64)
            y = subset["label"].values.astype(int)
            folds = subset["cv_fold"].values.astype(int)

            bin_auc, bin_prauc = [], []
            for f in unique_folds:
                if f == -1: continue
                train_mask_fold = (folds != f)
                test_mask_fold = (folds == f)

                X_train, y_train = X[train_mask_fold], y[train_mask_fold]
                X_test, y_test = X[test_mask_fold], y[test_mask_fold]

                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = LogisticRegression(class_weight="balanced", penalty="l2", C=self.C_REG, solver="lbfgs", max_iter=1000, random_state=self.SEED)
                clf.fit(X_train_s, y_train)

                preds = clf.predict_proba(X_test_s)[:, 1]
                bin_auc.append(roc_auc_score(y_test, preds))
                bin_prauc.append(average_precision_score(y_test, preds))

            cv_reports.append({
                "ha_bin": int(ha_bin),
                "mean_AUC_Combined": float(np.mean(bin_auc) if bin_auc else 0.0),
                "mean_PRAUC_Combined": float(np.mean(bin_prauc) if bin_prauc else 0.0),
                "N_bin": int(len(y)),
            })

            if len(np.unique(y)) >= 2:
                scaler_full = StandardScaler()
                X_s = scaler_full.fit_transform(X)

                clf_full = LogisticRegression(class_weight="balanced", penalty="l2", C=self.C_REG, solver="lbfgs", max_iter=1000, random_state=self.SEED)
                clf_full.fit(X_s, y)

                w_orig, b_orig = self._coef_to_original_space(clf_full.coef_[0], float(clf_full.intercept_[0]), scaler_full)

                self.production_params[int(ha_bin)] = {"intercept": float(b_orig), "coef": w_orig.tolist(), "N": int(len(y))}
                raw_coefficients[int(ha_bin), 0] = b_orig
                raw_coefficients[int(ha_bin), 1:] = w_orig
                trained_mask[int(ha_bin)] = 1

        cv_path = os.path.join(self.out_dir, f"stp_cv_report_FusionK{self.K}_{self.POLICY}.csv")
        pd.DataFrame(cv_reports).to_csv(cv_path, index=False)
        print(f"   ✅ CV report saved: {cv_path}")

        print("\n>> 4. Applying Gaussian smoothing to coefficients (valid bins only)...")
        smoothed = raw_coefficients.copy()
        valid_bins = np.where(trained_mask == 1)[0].tolist()

        if len(valid_bins) >= 3:
            for col_idx in range(7):
                series = raw_coefficients[:, col_idx].copy()
                mask = np.zeros_like(series)
                mask[valid_bins] = 1.0
                sm = gaussian_filter1d(series * mask, sigma=self.SMOOTH_SIGMA)
                m = gaussian_filter1d(mask, sigma=self.SMOOTH_SIGMA)
                smoothed[:, col_idx] = np.where(m > 1e-8, sm / m, 0.0)
        else:
            print("   - Not enough valid bins for smoothing; skipping.")

        coef_dict = {}
        for ha_bin in range(10, 61):
            if trained_mask[ha_bin] == 1:
                coef_dict[str(ha_bin)] = smoothed[ha_bin].tolist()
            else:
                if len(valid_bins) == 0:
                    coef_dict[str(ha_bin)] = [0.0] * 7
                else:
                    closest = min(valid_bins, key=lambda x: abs(x - ha_bin))
                    coef_dict[str(ha_bin)] = smoothed[closest].tolist()

            b = float(coef_dict[str(ha_bin)][0])
            w = [float(x) for x in coef_dict[str(ha_bin)][1:]]
            self.production_params[ha_bin] = {"intercept": b, "coef": w, "N": int(self.production_params.get(ha_bin, {}).get("N", 0))}

        coef_path = os.path.join(self.out_dir, f"stp_coef_FusionK{self.K}_{self.POLICY}.json")
        with open(coef_path, "w", encoding="utf-8") as f:
            json.dump({"K": self.K, "Policy": self.POLICY, "Temporal_Cutoff": self.CUTOFF_YEAR, "FeatureCols": feature_cols, "CoefSpace": "original_feature_space", "coef": coef_dict}, f, indent=2)
        print(f"   ✅ Coefficient JSON saved: {coef_path}")

        if self.future_df is None or self.future_df.empty:
            print("\n>> 5. No future/OOT rows found; skipping OOT evaluation.")
            return

        self.future_features_df = self._build_dataset_batched(self.future_df, "Future OOT")
        print("\n>> 5. Evaluating OOT performance (smoothed coefficients, no scaler needed)...")

        oot_reports = []
        for ha_bin in tqdm(sorted(self.future_features_df["ha_bin"].unique().tolist()), desc="OOT bins"):
            subset = self.future_features_df[self.future_features_df["ha_bin"] == ha_bin]
            if len(subset) == 0: continue
            y_oot = subset["label"].values.astype(int)
            if len(np.unique(y_oot)) < 2: continue

            X_oot = subset[feature_cols].values.astype(np.float64)
            params = self.production_params.get(int(ha_bin))
            if params is None: continue

            w = np.array(params["coef"], dtype=np.float64)
            b = float(params["intercept"])
            preds = self._sigmoid(b + X_oot @ w)

            oot_reports.append({
                "ha_bin": int(ha_bin), "N_Samples": int(len(y_oot)),
                "OOT_AUC_Combined": float(roc_auc_score(y_oot, preds)),
                "OOT_PRAUC": float(average_precision_score(y_oot, preds)),
            })

        oot_path = os.path.join(self.out_dir, f"stp_OOT_report_FusionK{self.K}_{self.POLICY}.csv")
        if oot_reports:
            pd.DataFrame(oot_reports).to_csv(oot_path, index=False)
            print(f"   ✅ OOT report saved: {oot_path}")
        else:
            print("   - OOT report empty.")

if __name__ == "__main__":
    trainer = STPAucBoostedTrainer(
        meta_file="features_store/final_training_meta.parquet",
        fp2_memmap="features_store/fp2_aligned.memmap",
        es5d_memmap="features_store/es5d_db_k20.memmap",
        out_dir="features_store",
        k_mode=5,
        k_policy="paired_sum",
        c_reg=1.0,
        cutoff_year=2023,
        batch_size=4096,
        chunk_3d_b=128,
        smooth_sigma=1.5,
        seed=42,
    )
    trainer.execute_cv_and_oot_evaluation()
