# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 13:12:25 2026

@author: KIOM_User
"""



# -*- coding: utf-8 -*-
"""
Phase 3: TRUE ULTRA-FAST Batched STP-Grade Dual-GPU Scoring
- Tensor Matrix Multiplication with Pre-Cast FP16
- [FIXED] Reverse-Chunked Broadcasting for 3D cdist (Eliminates VRAM bandwidth choke)
- Masking fixed (-1.0) to prevent Top-K tie pollution
"""

# ===== [필수 추가] 라이브러리 로드 전 OpenMP 충돌 방지 =====
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ============================================================

import time
import json
import argparse
import hashlib
import sys
import traceback
import warnings
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from tqdm import tqdm

# Numpy 읽기 전용 경고 무시
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


def _validate_pair_rule(
    df,
    context,
    check_cv_consistency=True,
    min_neg_per_pair=10,
    max_neg_per_pair=10,
):
    required = ["pair_id", "set_type", "target_chembl_id", "cv_fold"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{context}: missing required columns for pair rule check: {missing}")

    pair_sizes = df.groupby("pair_id").size()
    pos_sizes = (
        df[df["set_type"] == "Positive"]
        .groupby("pair_id")
        .size()
        .reindex(pair_sizes.index, fill_value=0)
    )
    neg_sizes = (
        df[df["set_type"] == "Negative"]
        .groupby("pair_id")
        .size()
        .reindex(pair_sizes.index, fill_value=0)
    )
    target_nunique = df.groupby("pair_id")["target_chembl_id"].nunique()
    cv_nunique = (
        df.groupby("pair_id")["cv_fold"].nunique()
        if "cv_fold" in df.columns
        else pd.Series(0, index=pair_sizes.index)
    )
    min_rows = 1 + int(min_neg_per_pair)
    max_rows = 1 + int(max_neg_per_pair)
    violations = {
        "pair_rows_below_min_count": int((pair_sizes < min_rows).sum()),
        "pair_rows_above_max_count": int((pair_sizes > max_rows).sum()),
        "pair_pos_not_1_count": int((pos_sizes != 1).sum()),
        "pair_neg_below_min_count": int((neg_sizes < int(min_neg_per_pair)).sum()),
        "pair_neg_above_max_count": int((neg_sizes > int(max_neg_per_pair)).sum()),
        "pair_target_mismatch_count": int((target_nunique != 1).sum()),
        "pair_cv_leakage_count": int((cv_nunique != 1).sum()) if check_cv_consistency else 0,
    }
    if any(v > 0 for v in violations.values()):
        raise ValueError(f"{context}: pair rule violation detected: {violations}")
    print(
        f"   - Pair rule verified [{context}]: rows={len(df):,}, "
        f"pairs={len(pair_sizes):,} (1 positive + {int(min_neg_per_pair)}~{int(max_neg_per_pair)} negative)"
    )


class STPUltraFastTrainer:
    def __init__(self, meta_file, fp2_memmap, es5d_memmap, out_dir="features_store", 
                 k_mode=1, k_policy='paired_sum', c_reg=10.0, cutoff_year=2023,
                 scaffold_mask=True, assay_mask=False, batch_size=4096,
                 chunk_3d_b=128, use_tf32=True, disable_threshold_norm=False,
                 thr_preset="custom", thr2d=0.30, thr3d=0.65,
                 exclude_below_threshold=False, keep_negative_features=False,
                 stp_mode_2014=False, cv_scheme="preassigned",
                 single_modal_eval="both", model_family="stack_hgb",
                 hgb_max_iter=300, hgb_learning_rate=0.05, hgb_max_leaf_nodes=31,
                 min_neg_per_pair=7, max_neg_per_pair=10, show_progress=True,
                 feature_cache_dir=None, feature_cache_signature=None):
        self.meta_file = meta_file
        self.fp2_path = fp2_memmap
        self.es5d_path = es5d_memmap
        self.out_dir = out_dir
        
        self.K = k_mode
        self.POLICY = k_policy  
        self.C_REG = c_reg      
        self.CUTOFF_YEAR = cutoff_year 
        self.scaffold_mask = bool(scaffold_mask)
        # assay mask path intentionally disabled for p3_1_1/p3_1_2
        self.assay_mask = False
        self.use_tf32 = bool(use_tf32)
        self.eval_mode = self._compose_eval_mode(self.scaffold_mask, self.assay_mask)
        self.disable_threshold_norm = bool(disable_threshold_norm)
        self.thr_preset = str(thr_preset)
        self.thr2d, self.thr3d = self._resolve_thresholds(self.thr_preset, float(thr2d), float(thr3d))
        self.exclude_below_threshold = bool(exclude_below_threshold)
        self.keep_negative_features = bool(keep_negative_features)
        self.stp_mode_2014 = bool(stp_mode_2014)
        self.cv_scheme = str(cv_scheme)
        self.single_modal_eval = str(single_modal_eval)
        if self.single_modal_eval not in {"max", "lr", "both"}:
            raise ValueError("single_modal_eval must be one of: max, lr, both")
        self.model_family = str(model_family)
        if self.model_family not in {"lr", "stack_hgb"}:
            raise ValueError("model_family must be one of: lr, stack_hgb")
        self.hgb_max_iter = int(hgb_max_iter)
        self.hgb_learning_rate = float(hgb_learning_rate)
        self.hgb_max_leaf_nodes = int(hgb_max_leaf_nodes)
        self.min_neg_per_pair = int(min_neg_per_pair)
        self.max_neg_per_pair = int(max_neg_per_pair)
        self.show_progress = bool(show_progress)
        self.feature_cache_dir = feature_cache_dir
        self.feature_cache_signature = feature_cache_signature
        
        self.BATCH_SIZE = int(batch_size)
        # [핵심] 쿼리(B)를 자르는 청크 사이즈 (L2 캐시 최적화 사이즈)
        self.CHUNK_3D_B = int(chunk_3d_b)
        
        self._initialize_gpu_engine()

    @staticmethod
    def _feature_columns_for_k(k_mode):
        cols = []
        for i in range(1, int(k_mode) + 1):
            cols.extend([f"s1_3d_{i}", f"s2_2d_{i}"])
        return cols

    def _slice_feature_df_for_k(self, df, k_mode):
        base_cols = ["heavy_atoms", "ha_bin", "label", "cv_fold", "pair_id"]
        feature_cols = self._feature_columns_for_k(k_mode)
        missing = [c for c in base_cols + feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Cached feature matrix missing required columns: {missing}")
        return df.loc[:, base_cols + feature_cols].copy()

    def _feature_cache_subdir(self):
        if not self.feature_cache_dir or not self.feature_cache_signature:
            return None
        return os.path.join(self.feature_cache_dir, self.feature_cache_signature)

    def _cache_file_path(self, split_name, k_mode):
        cache_root = self._feature_cache_subdir()
        if cache_root is None:
            return None
        return os.path.join(cache_root, f"{split_name}_features_K{k_mode}.parquet")

    def _try_load_feature_cache(self, split_name):
        cache_root = self._feature_cache_subdir()
        if cache_root is None or not os.path.isdir(cache_root):
            return None

        available = []
        for name in os.listdir(cache_root):
            if not name.startswith(f"{split_name}_features_K") or not name.endswith(".parquet"):
                continue
            k_str = name[len(f"{split_name}_features_K"):-len(".parquet")]
            if k_str.isdigit():
                available.append(int(k_str))
        viable = sorted(k for k in available if k >= self.K)
        if not viable:
            return None

        cache_k = viable[0]
        cache_path = self._cache_file_path(split_name, cache_k)
        print(f"   - Reusing cached feature matrix [{split_name}] from: {cache_path}")
        cached_df = pd.read_parquet(cache_path)
        if cache_k != self.K:
            print(f"   - Cached K={cache_k} -> slicing to requested K={self.K}")
        return self._slice_feature_df_for_k(cached_df, self.K)

    def _save_feature_cache(self, split_name, df):
        cache_path = self._cache_file_path(split_name, self.K)
        if cache_path is None:
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"   - Saved feature matrix cache [{split_name}] to: {cache_path}")

    def _load_or_build_dataset(self, df, desc, split_name):
        cached = self._try_load_feature_cache(split_name)
        if cached is not None:
            return cached
        built = self._build_dataset_batched(df, desc)
        self._save_feature_cache(split_name, built)
        return built

    @staticmethod
    def _compose_eval_mode(scaffold_mask, assay_mask):
        if scaffold_mask:
            return "distinct_scaffolds"
        return "all"

    @staticmethod
    def _resolve_thresholds(thr_preset, thr2d, thr3d):
        if thr_preset == "stp2014":
            return 0.30, 0.65
        if thr_preset == "stp2019":
            return 0.65, 0.85
        return thr2d, thr3d

    def _cv_check_required(self):
        return self.cv_scheme == "preassigned"

    def _assign_cv_folds(self, df):
        if self.cv_scheme == "preassigned":
            if "cv_fold" not in df.columns:
                raise KeyError("cv_scheme=preassigned requires 'cv_fold' column.")
            out = df.copy()
            out["cv_fold"] = out["cv_fold"].astype(np.int64)
            return out

        out = df.copy()
        if self.cv_scheme == "groupkfold_pairid":
            groups = out["pair_id"].astype(str).values
        elif self.cv_scheme == "groupkfold_scaffold":
            groups = out["scaffold_hash"].astype(str).values
        elif self.cv_scheme == "groupkfold_scaffold_assay":
            if "assay_id" not in out.columns:
                raise KeyError("cv_scheme=groupkfold_scaffold_assay requires 'assay_id' column.")
            groups = pd.util.hash_pandas_object(
                out[["scaffold_hash", "assay_id"]].astype(str), index=False
            ).astype(np.int64).values
        else:
            raise ValueError(f"Unknown cv_scheme: {self.cv_scheme}")

        gkf = GroupKFold(n_splits=10)
        y = (out["set_type"] == "Positive").astype(int).values
        X_dummy = np.zeros((len(out), 1), dtype=np.int8)
        fold = np.full(len(out), -1, dtype=np.int64)
        for f_idx, (_, test_idx) in enumerate(gkf.split(X_dummy, y, groups=groups)):
            fold[test_idx] = f_idx
        if np.any(fold < 0):
            raise RuntimeError("Failed to assign all rows to GroupKFold folds.")
        out["cv_fold"] = fold
        return out

    @staticmethod
    def _compute_fold_stats(df):
        rows = []
        for f in sorted(pd.unique(df["cv_fold"])):
            sub = df[df["cv_fold"] == f]
            n_pos = int((sub["label"] == 1).sum())
            n_neg = int((sub["label"] == 0).sum())
            ratio = (n_pos / n_neg) if n_neg > 0 else np.nan
            rows.append(
                {
                    "cv_fold": int(f),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "pos_neg_ratio": float(ratio) if np.isfinite(ratio) else np.nan,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _balanced_sample_weight(y):
        y = np.asarray(y, dtype=np.int64)
        n = len(y)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return np.ones(n, dtype=np.float32)
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        return np.where(y == 1, w_pos, w_neg).astype(np.float32)

    def _build_hgb_model(self):
        return HistGradientBoostingClassifier(
            learning_rate=self.hgb_learning_rate,
            max_iter=self.hgb_max_iter,
            max_leaf_nodes=self.hgb_max_leaf_nodes,
            min_samples_leaf=20,
            l2_regularization=1e-3,
            random_state=42,
        )

    def _fit_predict_combined(self, X_train, y_train, X_test):
        if self.model_family == "lr":
            clf = LogisticRegression(
                class_weight='balanced', penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000
            )
            clf.fit(X_train, y_train)
            return clf.predict_proba(X_test)[:, 1], clf

        x3_tr = X_train[:, :self.K]
        x2_tr = X_train[:, self.K:]
        x3_te = X_test[:, :self.K]
        x2_te = X_test[:, self.K:]
        n_train = len(y_train)
        if n_train < 40 or len(np.unique(y_train)) < 2:
            clf = LogisticRegression(
                class_weight='balanced', penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000
            )
            clf.fit(X_train, y_train)
            return clf.predict_proba(X_test)[:, 1], clf

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_meta = np.zeros((n_train, 2), dtype=np.float32)
        for inner_tr, inner_va in skf.split(X_train, y_train):
            y_inner = y_train[inner_tr]
            m3 = self._build_hgb_model()
            m2 = self._build_hgb_model()
            sw_inner = self._balanced_sample_weight(y_inner)
            m3.fit(x3_tr[inner_tr], y_inner, sample_weight=sw_inner)
            m2.fit(x2_tr[inner_tr], y_inner, sample_weight=sw_inner)
            oof_meta[inner_va, 0] = m3.predict_proba(x3_tr[inner_va])[:, 1]
            oof_meta[inner_va, 1] = m2.predict_proba(x2_tr[inner_va])[:, 1]

        meta = LogisticRegression(
            class_weight='balanced', penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000
        )
        meta.fit(oof_meta, y_train)

        base3 = self._build_hgb_model()
        base2 = self._build_hgb_model()
        sw_full = self._balanced_sample_weight(y_train)
        base3.fit(x3_tr, y_train, sample_weight=sw_full)
        base2.fit(x2_tr, y_train, sample_weight=sw_full)

        test_meta = np.column_stack([
            base3.predict_proba(x3_te)[:, 1],
            base2.predict_proba(x2_te)[:, 1],
        ])
        preds = meta.predict_proba(test_meta)[:, 1]
        model_bundle = {"base3d": base3, "base2d": base2, "meta": meta}
        return preds, model_bundle

    def _predict_combined(self, model, X):
        if self.model_family == "lr":
            return model.predict_proba(X)[:, 1]
        x3 = X[:, :self.K]
        x2 = X[:, self.K:]
        meta_x = np.column_stack([
            model["base3d"].predict_proba(x3)[:, 1],
            model["base2d"].predict_proba(x2)[:, 1],
        ])
        return model["meta"].predict_proba(meta_x)[:, 1]
        
    def _initialize_gpu_engine(self):
        print(
            f"\n>> 1. Initializing Ultra-Fast GPU Engine "
            f"(K={self.K}, Policy={self.POLICY}, EvalMode={self.eval_mode}, "
            f"scaffold_mask={self.scaffold_mask})..."
        )
        if self.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        
        full_df = pd.read_parquet(self.meta_file)
        
        required_cols = ['memmap_idx', 'target_chembl_id', 'set_type', 'heavy_atoms', 'scaffold_smiles', 'pair_id']
        if self.cv_scheme == "preassigned":
            required_cols.append("cv_fold")
        if self.cv_scheme == "groupkfold_scaffold_assay":
            required_cols.append("assay_id")
        for c in required_cols:
            assert c in full_df.columns, f"치명적 오류: 필수 컬럼 '{c}'이 누락되었습니다."
        _validate_pair_rule(
            full_df,
            "meta_file",
            check_cv_consistency=self._cv_check_required(),
            min_neg_per_pair=self.min_neg_per_pair,
            max_neg_per_pair=self.max_neg_per_pair,
        )
            
        full_df['scaffold_hash'] = pd.util.hash_pandas_object(full_df['scaffold_smiles'], index=False).astype(np.int64)
        
        size_fp2 = os.path.getsize(self.fp2_path)
        size_es5d = os.path.getsize(self.es5d_path)
        
        row_fp2 = 128 
        assert size_fp2 % row_fp2 == 0, "치명적 오류: FP2 memmap misaligned"
        self.FULL_N = size_fp2 // row_fp2
        
        row_es5d_atom = 72 
        assert size_es5d % (self.FULL_N * row_es5d_atom) == 0, "치명적 오류: ES5D memmap misaligned"
        self.NCONF = size_es5d // (self.FULL_N * row_es5d_atom)
        assert self.NCONF in (1, 20), f"치명적 오류: Unexpected NCONF={self.NCONF}"
        
        print(f"   - Validated Memmap: N={self.FULL_N:,}, Conformations={self.NCONF}")
        
        db_2d_cpu = np.memmap(self.fp2_path, dtype=np.uint64, mode='r', shape=(self.FULL_N, 16))
        db_3d_cpu = np.memmap(self.es5d_path, dtype=np.float32, mode='r', shape=(self.FULL_N, self.NCONF, 18))
        
        print("   - Pre-Casting 2D FP2 to float16 for instant Matmul (approx 1.76 GB VRAM)...")
        db_2d_bits = np.unpackbits(db_2d_cpu.view(np.uint8), axis=1) 
        self.db2d_gpu = torch.from_numpy(db_2d_bits).to(device='cuda:0', dtype=torch.float16)
        self.db3d_gpu = torch.from_numpy(db_3d_cpu).to(device='cuda:0') 
        self.db2d_ones = self.db2d_gpu.sum(dim=1).float()
        
        scaf_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        scaf_arr[full_df['memmap_idx'].values] = full_df['scaffold_hash'].values
        self.gpu_scaffolds = torch.tensor(scaf_arr, device='cuda:0')
        print(f"   - Mask config: scaffold_mask={self.scaffold_mask}, assay_mask=disabled")
        
        if "publication_year" in full_df.columns:
            # Pair-wise temporal split: keep all rows of the same pair_id together.
            pos_rows = full_df[full_df["set_type"] == "Positive"][["pair_id", "publication_year"]].drop_duplicates(
                subset=["pair_id"]
            )
            if len(pos_rows) == full_df["pair_id"].nunique():
                pair_year = pos_rows.set_index("pair_id")["publication_year"]
            else:
                pair_year = full_df.groupby("pair_id")["publication_year"].max()
            pair_is_future = pair_year > self.CUTOFF_YEAR
            full_df["__is_future_pair"] = full_df["pair_id"].map(pair_is_future).fillna(False).astype(bool)
            self.future_df = full_df[full_df["__is_future_pair"]].drop(columns=["__is_future_pair"]).copy()
            self.meta_df = full_df[~full_df["__is_future_pair"]].drop(columns=["__is_future_pair"]).copy()
        else:
            self.future_df = pd.DataFrame()
            self.meta_df = full_df.copy()
        self.meta_df = self._assign_cv_folds(self.meta_df)
        if not self.future_df.empty:
            self.future_df = self.future_df.copy()
            self.future_df["cv_fold"] = -1
        print(f"   - CV scheme: {self.cv_scheme}")
        folds_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        folds_arr[self.meta_df['memmap_idx'].values] = self.meta_df['cv_fold'].values
        self.gpu_folds = torch.tensor(folds_arr, device='cuda:0')
        # Temporal split 이후에도 pair 규칙 유지 여부 검증 (11/1/10, target/cv_fold 일관성)
        _validate_pair_rule(
            self.meta_df,
            f"train_split<=Y{self.CUTOFF_YEAR}",
            check_cv_consistency=self._cv_check_required(),
            min_neg_per_pair=self.min_neg_per_pair,
            max_neg_per_pair=self.max_neg_per_pair,
        )
        if not self.future_df.empty:
            _validate_pair_rule(
                self.future_df,
                f"oot_split>Y{self.CUTOFF_YEAR}",
                check_cv_consistency=self._cv_check_required(),
                min_neg_per_pair=self.min_neg_per_pair,
                max_neg_per_pair=self.max_neg_per_pair,
            )

        positives = self.meta_df[self.meta_df['set_type'] == 'Positive']
        tmap = positives.groupby('target_chembl_id')['memmap_idx'].apply(list)
        self.target_actives_tensor = {k: torch.tensor(v, device='cuda:0', dtype=torch.int64) for k, v in tmap.items()}

    def _extract_features_batched(self, query_indices, target_id, query_scafs, query_folds=None):
        B = len(query_indices)
        if target_id not in self.target_actives_tensor:
            return torch.zeros((B, self.K * 2), dtype=torch.float32, device='cpu')
            
        actives_tensor = self.target_actives_tensor[target_id]
        M = len(actives_tensor)
        
        # --- (B, M) 2D 마스킹 ---
        q_idx_exp = query_indices.unsqueeze(1)
        a_idx_exp = actives_tensor.unsqueeze(0)
        mask = (q_idx_exp != a_idx_exp)
        if self.scaffold_mask:
            q_scaf_exp = query_scafs.unsqueeze(1)
            a_scaf_exp = self.gpu_scaffolds[actives_tensor].unsqueeze(0)
            mask &= (q_scaf_exp != a_scaf_exp) & (a_scaf_exp != -1)
        if query_folds is not None:
            mask &= (query_folds.unsqueeze(1) != self.gpu_folds[actives_tensor].unsqueeze(0))
            
        # --- 초고속 2D Tanimoto (Matmul) ---
        q_2d_fp16 = self.db2d_gpu[query_indices] 
        a_2d_fp16 = self.db2d_gpu[actives_tensor]
        
        intersection = torch.matmul(q_2d_fp16, a_2d_fp16.T).float()
        q_ones = self.db2d_ones[query_indices].unsqueeze(1)
        a_ones = self.db2d_ones[actives_tensor].unsqueeze(0)
        
        union = q_ones + a_ones - intersection
        sim_2d_matrix = torch.where(union > 0, intersection / union, 0.0)
        
        # --- [초고속 패치] L2 Cache-Optimized 3D cdist ---
        q_3d = torch.nan_to_num(self.db3d_gpu[query_indices], nan=1e4) # (B, 20, 18)
        a_3d = torch.nan_to_num(self.db3d_gpu[actives_tensor], nan=-1e4) # (M, 20, 18)
        
        sim_3d_matrix = torch.empty((B, M), device='cuda:0', dtype=torch.float32)
        a_3d_exp = a_3d.unsqueeze(0) # (1, M, 20, 18)
        
        # 쿼리(B)를 잘게 썰어서 VRAM 병목 제거
        for i in range(0, B, self.CHUNK_3D_B):
            q_chunk = q_3d[i:i+self.CHUNK_3D_B].unsqueeze(1) # (C, 1, 20, 18)
            dist = torch.cdist(q_chunk, a_3d_exp, p=1.0)     # (C, M, 20, 20)
            sim_3d_matrix[i:i+self.CHUNK_3D_B] = (1.0 / (1.0 + (dist / 18.0))).amax(dim=(2, 3))
            
        sim_2d_raw = sim_2d_matrix
        sim_3d_raw = sim_3d_matrix

        if self.exclude_below_threshold:
            below_2d = sim_2d_raw < self.thr2d
            below_3d = sim_3d_raw < self.thr3d
            # Modality-wise thresholding only: do not couple 2D/3D invalidation.
            sim_2d_matrix = sim_2d_matrix.masked_fill(below_2d, -1.0)
            sim_3d_matrix = sim_3d_matrix.masked_fill(below_3d, -1.0)

        if not self.disable_threshold_norm:
            # paper-style threshold normalization
            sim_2d_matrix = (sim_2d_matrix - self.thr2d) / (1.0 - self.thr2d)
            sim_2d_matrix = torch.clamp(sim_2d_matrix, 0.0, 1.0)
            sim_3d_matrix = (sim_3d_matrix - self.thr3d) / (1.0 - self.thr3d)
            sim_3d_matrix = torch.clamp(sim_3d_matrix, 0.0, 1.0)

        # --- 무효 후보는 -1.0으로 채워 오염 방지 ---
        sim_2d_matrix.masked_fill_(~mask, -1.0)
        sim_3d_matrix.masked_fill_(~mask, -1.0)
        
        if self.POLICY == 'paired_sum': score = sim_2d_matrix + sim_3d_matrix
        elif self.POLICY == 'paired_2d': score = sim_2d_matrix
        elif self.POLICY == 'paired_3d': score = sim_3d_matrix
        else: score = sim_2d_matrix + sim_3d_matrix 
        
        k_actual = min(self.K, M)
        if k_actual == 0:
            return torch.zeros((B, self.K * 2), dtype=torch.float32, device='cpu')
            
        features = torch.zeros((B, self.K * 2), dtype=torch.float32, device='cuda:0')
        b_idx = torch.arange(B, device='cuda:0').unsqueeze(1).expand(B, k_actual)
        
        force_independent_topk = self.exclude_below_threshold
        if self.POLICY == 'independent' or force_independent_topk:
            top_3d_idx = torch.topk(sim_3d_matrix, k_actual, dim=1).indices
            top_2d_idx = torch.topk(sim_2d_matrix, k_actual, dim=1).indices
            features[:, 0::2][:, :k_actual] = sim_3d_matrix[b_idx, top_3d_idx]
            features[:, 1::2][:, :k_actual] = sim_2d_matrix[b_idx, top_2d_idx]
        else:
            top_k_indices = torch.topk(score, k_actual, dim=1).indices
            features[:, 0::2][:, :k_actual] = sim_3d_matrix[b_idx, top_k_indices]
            features[:, 1::2][:, :k_actual] = sim_2d_matrix[b_idx, top_k_indices]
        
        if not self.keep_negative_features:
            features = torch.clamp(features, min=0.0)
        return features.cpu()

    def _build_dataset_batched(self, df, desc):
        print(f"\n>> Generating Feature Matrix for {desc} (N={len(df):,}) using Ultra-Fast Batched Engine...")
        _validate_pair_rule(
            df,
            f"pre-batch:{desc}",
            check_cv_consistency=self._cv_check_required(),
            min_neg_per_pair=self.min_neg_per_pair,
            max_neg_per_pair=self.max_neg_per_pair,
        )
        start_time = time.time()
        
        is_oot = (desc == "Future OOT")
        
        df = df.copy()
        df['sort_idx'] = np.arange(len(df))
        grouped = df.groupby('target_chembl_id')
        total_targets = len(grouped)
        
        all_features = np.zeros((len(df), self.K * 2), dtype=np.float32)
        print(f"   - Processing {total_targets} unique targets...")
        
        with tqdm(total=len(df), desc="Rows Processed", unit="rows", disable=(not self.show_progress)) as pbar:
            for target_id, group in grouped:
                group_indices = group['sort_idx'].values
                q_memmap = torch.tensor(group['memmap_idx'].values, device='cuda:0')
                q_scafs = torch.tensor(group['scaffold_hash'].values, device='cuda:0')
                q_folds = None if is_oot else torch.tensor(group['cv_fold'].values, device='cuda:0')
                
                for i in range(0, len(q_memmap), self.BATCH_SIZE):
                    batch_memmap = q_memmap[i:i+self.BATCH_SIZE]
                    batch_scafs = q_scafs[i:i+self.BATCH_SIZE]
                    batch_folds = None if q_folds is None else q_folds[i:i+self.BATCH_SIZE]
                    
                    batch_feats = self._extract_features_batched(batch_memmap, target_id, batch_scafs, batch_folds)
                    all_features[group_indices[i:i+self.BATCH_SIZE]] = batch_feats.numpy()
                    pbar.update(len(batch_memmap))
        
        cols = []
        for i in range(1, self.K + 1):
            cols.extend([f"s1_3d_{i}", f"s2_2d_{i}"])
            
        out_df = pd.DataFrame({
            'heavy_atoms': df['heavy_atoms'].values, 
            'ha_bin': np.clip(df['heavy_atoms'].values, 10, 60),
            'label': (df['set_type'] == 'Positive').astype(int).values, 
            'cv_fold': df.get('cv_fold', pd.Series(np.full(len(df), -1))).values,
            'pair_id': df['pair_id'].values
        })
        out_df = pd.concat([out_df, pd.DataFrame(all_features, columns=cols)], axis=1)
        print(f"   [OK] Batched Matrix generated in {time.time()-start_time:.2f}s")
        return out_df

    def execute_cv_and_oot_evaluation(self):
        _validate_pair_rule(
            self.meta_df,
            "pre-training",
            check_cv_consistency=self._cv_check_required(),
            min_neg_per_pair=self.min_neg_per_pair,
            max_neg_per_pair=self.max_neg_per_pair,
        )
        self.train_df = self._load_or_build_dataset(self.meta_df, "Past Training Data", "train")
        fold_stats_df = self._compute_fold_stats(self.train_df)
        fold_stats_df["cv_scheme"] = self.cv_scheme
        print("\n>> CV fold class distribution")
        for r in fold_stats_df.itertuples(index=False):
            print(
                f"   - fold={r.cv_fold}: n_pos={r.n_pos}, n_neg={r.n_neg}, "
                f"pos:neg={r.pos_neg_ratio:.6f}"
            )
        
        print(
            f"\n>> 3. Executing True CV & Training "
            f"(model={self.model_family}, L2 C={self.C_REG}, EvalMode={self.eval_mode})..."
        )
        feature_cols = [c for c in self.train_df.columns if c.startswith('s1_') or c.startswith('s2_')]
        
        # Reorder to [3D_1..K, 2D_1..K] for modality-only LR evaluation.
        X_reorder_cols = [f"s1_3d_{i}" for i in range(1, self.K + 1)] + [f"s2_2d_{i}" for i in range(1, self.K + 1)]
        three_d_indices = list(range(self.K))
        two_d_indices = list(range(self.K, 2 * self.K))
        
        unique_bins = sorted(self.train_df['ha_bin'].unique())
        unique_folds = np.sort(self.train_df['cv_fold'].unique())
        
        cv_reports = []
        self.production_models = {} 
        self.production_models_3d = {}
        self.production_models_2d = {}
        final_coefficients = {}
        
        for ha_bin in tqdm(unique_bins, desc="Training Models", disable=(not self.show_progress)):
            subset = self.train_df[self.train_df['ha_bin'] == ha_bin]
            X, y, folds = subset[feature_cols].values, subset['label'].values, subset['cv_fold'].values
            X_reordered = subset[X_reorder_cols].values
            
            bin_auc, bin_prauc = [], []
            bin_auc_3d_max, bin_auc_2d_max = [], []
            bin_auc_3d_lr, bin_auc_2d_lr = [], []
            for f in unique_folds:
                if f == -1: continue 
                train_mask, test_mask = (folds != f), (folds == f)
                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]
                Xr_train = X_reordered[train_mask]
                Xr_test = X_reordered[test_mask]
                
                if len(np.unique(y_test)) > 1 and self.single_modal_eval in {"max", "both"}:
                    pred_3d_max = Xr_test[:, three_d_indices].max(axis=1)
                    pred_2d_max = Xr_test[:, two_d_indices].max(axis=1)
                    bin_auc_3d_max.append(roc_auc_score(y_test, pred_3d_max))
                    bin_auc_2d_max.append(roc_auc_score(y_test, pred_2d_max))

                if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
                    preds, _ = self._fit_predict_combined(Xr_train, y_train, Xr_test)
                    bin_auc.append(roc_auc_score(y_test, preds))
                    bin_prauc.append(average_precision_score(y_test, preds))
                    if self.single_modal_eval in {"lr", "both"}:
                        clf_3d = LogisticRegression(class_weight='balanced', penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000)
                        clf_3d.fit(Xr_train[:, three_d_indices], y_train)
                        pred_3d = clf_3d.predict_proba(Xr_test[:, three_d_indices])[:, 1]
                        bin_auc_3d_lr.append(roc_auc_score(y_test, pred_3d))
                        clf_2d = LogisticRegression(class_weight='balanced', penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000)
                        clf_2d.fit(Xr_train[:, two_d_indices], y_train)
                        pred_2d = clf_2d.predict_proba(Xr_test[:, two_d_indices])[:, 1]
                        bin_auc_2d_lr.append(roc_auc_score(y_test, pred_2d))
            
            cv_reports.append({
                'ha_bin': ha_bin, 
                'eval_mode': self.eval_mode,
                'cv_scheme': self.cv_scheme,
                'model_family': self.model_family,
                'mean_AUC_Combined': np.mean(bin_auc) if bin_auc else 0.0,
                'mean_AUC_3D_Max': np.mean(bin_auc_3d_max) if bin_auc_3d_max else 0.0,
                'mean_AUC_3D_LR': np.mean(bin_auc_3d_lr) if bin_auc_3d_lr else 0.0,
                'mean_AUC_2D_Max': np.mean(bin_auc_2d_max) if bin_auc_2d_max else 0.0,
                'mean_AUC_2D_LR': np.mean(bin_auc_2d_lr) if bin_auc_2d_lr else 0.0,
                'mean_PRAUC_Combined': np.mean(bin_prauc) if bin_prauc else 0.0
            })
            
            if len(np.unique(y)) > 1:
                _, model_prod = self._fit_predict_combined(X_reordered, y, X_reordered)
                self.production_models[ha_bin] = model_prod
                clf_prod_3d = LogisticRegression(class_weight='balanced', penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000)
                clf_prod_3d.fit(X_reordered[:, three_d_indices], y)
                self.production_models_3d[ha_bin] = clf_prod_3d
                clf_prod_2d = LogisticRegression(class_weight='balanced', penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000)
                clf_prod_2d.fit(X_reordered[:, two_d_indices], y)
                self.production_models_2d[ha_bin] = clf_prod_2d
                if self.model_family == "lr":
                    final_coefficients[str(ha_bin)] = [float(model_prod.intercept_[0])] + [float(c) for c in model_prod.coef_[0]]
                else:
                    meta = model_prod["meta"]
                    final_coefficients[str(ha_bin)] = [float(meta.intercept_[0])] + [float(c) for c in meta.coef_[0]]
            else:
                self.production_models[ha_bin] = None
                self.production_models_3d[ha_bin] = None
                self.production_models_2d[ha_bin] = None
                final_coefficients[str(ha_bin)] = [-5.0] + [3.0, 7.0] * self.K

        pd.DataFrame(cv_reports).to_csv(
            os.path.join(self.out_dir, f"stp_cv_report_K{self.K}_{self.POLICY}_{self.eval_mode}.csv"),
            index=False,
        )
        fold_stats_df.to_csv(
            os.path.join(self.out_dir, f"stp_cv_fold_stats_K{self.K}_{self.POLICY}_{self.eval_mode}.csv"),
            index=False,
        )
        with open(os.path.join(self.out_dir, f"stp_coef_K{self.K}_{self.POLICY}_{self.eval_mode}.json"), "w") as f:
            json.dump({
                "K": self.K,
                "Policy": self.POLICY,
                "Eval_Mode": self.eval_mode,
                "Model_Family": self.model_family,
                "Scaffold_Mask_Applied": self.scaffold_mask,
                "Assay_Mask_Applied": self.assay_mask,
                "Temporal_Cutoff": self.CUTOFF_YEAR,
                "coef": final_coefficients
            }, f, indent=4)

        coef_rows = []
        for ha_bin_key, coef_values in final_coefficients.items():
            row = {
                "ha_bin": int(ha_bin_key),
                "eval_mode": self.eval_mode,
                "policy": self.POLICY,
                "K": self.K,
                "intercept": float(coef_values[0]),
            }
            if len(coef_values) == 3:
                row["a_shape"] = float(coef_values[1])
                row["a_fp"] = float(coef_values[2])
            else:
                for i in range(self.K):
                    row[f"a_shape_{i+1}"] = float(coef_values[1 + (2 * i)])
                    row[f"a_fp_{i+1}"] = float(coef_values[2 + (2 * i)])
            coef_rows.append(row)

        coef_by_ha_path = os.path.join(
            self.out_dir, f"stp_coef_by_ha_K{self.K}_{self.POLICY}_{self.eval_mode}.csv"
        )
        pd.DataFrame(coef_rows).sort_values("ha_bin").to_csv(coef_by_ha_path, index=False)
        print(f"   - Coef-by-ha CSV saved: {coef_by_ha_path}")
        
        if not self.future_df.empty:
            _validate_pair_rule(
                self.future_df,
                "pre-oot",
                check_cv_consistency=self._cv_check_required(),
                min_neg_per_pair=self.min_neg_per_pair,
                max_neg_per_pair=self.max_neg_per_pair,
            )
            self.future_features_df = self._load_or_build_dataset(self.future_df, "Future OOT", "future")
            print("\n>> 4. Evaluating Out-Of-Time (OOT) Performance...")
            
            oot_reports = []
            for ha_bin in tqdm(unique_bins, desc="OOT Evaluation", disable=(not self.show_progress)):
                if ha_bin not in self.production_models or self.production_models[ha_bin] is None: continue
                
                subset = self.future_features_df[self.future_features_df['ha_bin'] == ha_bin]
                if len(subset) == 0: continue
                
                X_oot, y_oot = subset[feature_cols].values, subset['label'].values
                Xr_oot = subset[X_reorder_cols].values
                
                if len(np.unique(y_oot)) > 1:
                    preds_oot = self._predict_combined(self.production_models[ha_bin], Xr_oot)
                    pred_3d_oot = self.production_models_3d[ha_bin].predict_proba(Xr_oot[:, three_d_indices])[:, 1]
                    pred_2d_oot = self.production_models_2d[ha_bin].predict_proba(Xr_oot[:, two_d_indices])[:, 1]
                    oot_reports.append({
                        'ha_bin': ha_bin, 'N_Samples': len(y_oot),
                        'eval_mode': self.eval_mode,
                        'OOT_AUC_Combined': roc_auc_score(y_oot, preds_oot),
                        'OOT_AUC_3D_Only': roc_auc_score(y_oot, pred_3d_oot),
                        'OOT_AUC_2D_Only': roc_auc_score(y_oot, pred_2d_oot),
                        'OOT_PRAUC': average_precision_score(y_oot, preds_oot)
                    })
                    
            if oot_reports:
                pd.DataFrame(oot_reports).to_csv(
                    os.path.join(self.out_dir, f"stp_OOT_report_K{self.K}_{self.POLICY}_{self.eval_mode}.csv"),
                    index=False,
                )
                print("   [OK] OOT Evaluation Completed! Report saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-configurable paired trainer")
    parser.add_argument("--meta_file", default="features_store/final_training_meta.parquet")
    parser.add_argument("--fp2_memmap", default="features_store/fp2_aligned.memmap")
    parser.add_argument("--es5d_memmap", default=None)
    parser.add_argument("--out_dir", default="features_store")
    parser.add_argument("--k_mode", type=int, default=1)
    parser.add_argument(
        "--k_policy",
        default="paired_sum",
        choices=["paired_sum", "paired_2d", "paired_3d", "independent"],
    )
    parser.add_argument("--c_reg", type=float, default=10.0)
    parser.add_argument("--cutoff_year", type=int, default=2023)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--chunk_3d_b", type=int, default=128)
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--disable_threshold_norm", action="store_true")
    parser.add_argument(
        "--thr_preset",
        type=str,
        default="custom",
        choices=["custom", "stp2014", "stp2019"],
    )
    parser.add_argument("--thr2d", type=float, default=0.30)
    parser.add_argument("--thr3d", type=float, default=0.65)
    parser.add_argument("--exclude_below_threshold", action="store_true")
    parser.add_argument(
        "--stp_mode_2014",
        action="store_true",
        help="Use STP2014 behavior: force thr_preset=stp2014 and independent policy.",
    )
    parser.add_argument("--keep_negative_features", action="store_true")
    parser.add_argument(
        "--scaffold_mask",
        default="on",
        choices=["on", "off"],
    )
    parser.add_argument(
        "--assay_mask",
        default="off",
        choices=["on", "off"],
        help="Ignored. assay_ID mask is disabled in this script.",
    )
    parser.add_argument(
        "--cv_scheme",
        default="preassigned",
        choices=[
            "preassigned",
            "groupkfold_pairid",
            "groupkfold_scaffold",
            "groupkfold_scaffold_assay",
        ],
    )
    parser.add_argument(
        "--single_modal_eval",
        default="both",
        choices=["max", "lr", "both"],
        help="Single-modality CV metric mode for 2D/3D AUC.",
    )
    parser.add_argument(
        "--model_family",
        default="stack_hgb",
        choices=["lr", "stack_hgb"],
        help="Combined model family: baseline LR or 2D/3D HGB + meta LR stacking.",
    )
    parser.add_argument("--hgb_max_iter", type=int, default=300)
    parser.add_argument("--hgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--hgb_max_leaf_nodes", type=int, default=31)
    parser.add_argument("--min_neg_per_pair", type=int, default=7)
    parser.add_argument("--max_neg_per_pair", type=int, default=10)
    parser.add_argument(
        "--feature_cache_dir",
        default=None,
        help="Optional directory for reusing feature matrices across K-only reruns.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress output. Use this for nohup/background runs.",
    )
    args = parser.parse_args()

    es5d_path = args.es5d_memmap or "features_store/es5d_db_k20.memmap"
    scaffold_mask = args.scaffold_mask == "on"
    assay_mask = False
    if args.stp_mode_2014:
        if args.thr_preset != "stp2014":
            print("[WARN] --stp_mode_2014 enabled: overriding --thr_preset to stp2014")
            args.thr_preset = "stp2014"
        if "--k_policy" in sys.argv and args.k_policy != "independent":
            raise ValueError("--stp_mode_2014 requires --k_policy independent (paired policies are not allowed).")
        if "--k_policy" not in sys.argv:
            args.k_policy = "independent"
        if args.k_policy != "independent":
            raise ValueError("--stp_mode_2014 requires k_policy=independent.")
    final_thr2d, final_thr3d = STPUltraFastTrainer._resolve_thresholds(
        args.thr_preset, args.thr2d, args.thr3d
    )
    eval_mode_label = STPUltraFastTrainer._compose_eval_mode(scaffold_mask, assay_mask)
    script_path = os.path.abspath(__file__)
    script_stem = os.path.splitext(os.path.basename(script_path))[0]
    c_reg_tag = str(args.c_reg).replace(".", "p")
    run_cfg_for_hash = {
        "script_stem": script_stem,
        "meta_file": os.path.abspath(args.meta_file),
        "fp2_memmap": os.path.abspath(args.fp2_memmap),
        "es5d_memmap": os.path.abspath(es5d_path),
        "k_mode": args.k_mode,
        "k_policy": args.k_policy,
        "eval_mode": eval_mode_label,
        "scaffold_mask": scaffold_mask,
        "assay_mask": assay_mask,
        "batch_size": args.batch_size,
        "chunk_3d_b": args.chunk_3d_b,
        "thr_preset": args.thr_preset,
        "thr2d_final": final_thr2d,
        "thr3d_final": final_thr3d,
        "disable_threshold_norm": args.disable_threshold_norm,
        "exclude_below_threshold": args.exclude_below_threshold,
        "stp_mode_2014": args.stp_mode_2014,
        "cv_scheme": args.cv_scheme,
        "single_modal_eval": args.single_modal_eval,
        "model_family": args.model_family,
        "hgb_max_iter": args.hgb_max_iter,
        "hgb_learning_rate": args.hgb_learning_rate,
        "hgb_max_leaf_nodes": args.hgb_max_leaf_nodes,
        "min_neg_per_pair": args.min_neg_per_pair,
        "max_neg_per_pair": args.max_neg_per_pair,
        "keep_negative_features": args.keep_negative_features,
        "c_reg": args.c_reg,
        "cutoff_year": args.cutoff_year,
    }
    feature_cache_cfg = {
        "script_stem": script_stem,
        "meta_file": os.path.abspath(args.meta_file),
        "fp2_memmap": os.path.abspath(args.fp2_memmap),
        "es5d_memmap": os.path.abspath(es5d_path),
        "k_policy": args.k_policy,
        "scaffold_mask": scaffold_mask,
        "assay_mask": assay_mask,
        "batch_size": args.batch_size,
        "chunk_3d_b": args.chunk_3d_b,
        "thr_preset": args.thr_preset,
        "thr2d_final": final_thr2d,
        "thr3d_final": final_thr3d,
        "disable_threshold_norm": args.disable_threshold_norm,
        "exclude_below_threshold": args.exclude_below_threshold,
        "stp_mode_2014": args.stp_mode_2014,
        "cv_scheme": args.cv_scheme,
        "min_neg_per_pair": args.min_neg_per_pair,
        "max_neg_per_pair": args.max_neg_per_pair,
        "keep_negative_features": args.keep_negative_features,
        "cutoff_year": args.cutoff_year,
    }
    feature_cache_signature = hashlib.sha1(
        json.dumps(feature_cache_cfg, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]
    run_hash = hashlib.sha1(
        json.dumps(run_cfg_for_hash, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:10]
    run_dir_name = (
        f"{script_stem}__K{args.k_mode}__{args.k_policy}__{eval_mode_label}"
        f"__C{c_reg_tag}__Y{args.cutoff_year}__H{run_hash}"
    )
    run_out_dir = os.path.join(args.out_dir, run_dir_name)
    os.makedirs(run_out_dir, exist_ok=True)
    run_log_path = os.path.join(run_out_dir, "run_config.log")

    with open(run_log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"start_time={datetime.now().isoformat()}\n")
        log_f.write(f"script={script_path}\n")
        log_f.write(f"command={' '.join(sys.argv)}\n")
        log_f.write(f"meta_file={args.meta_file}\n")
        log_f.write(f"fp2_memmap={args.fp2_memmap}\n")
        log_f.write(f"es5d_memmap={es5d_path}\n")
        log_f.write(f"k_mode={args.k_mode}\n")
        log_f.write(f"k_policy={args.k_policy}\n")
        log_f.write(f"eval_mode={eval_mode_label}\n")
        log_f.write(f"scaffold_mask={scaffold_mask}\n")
        log_f.write(f"assay_mask={assay_mask}\n")
        log_f.write(f"assay_mask_arg_ignored={args.assay_mask}\n")
        log_f.write(f"batch_size={args.batch_size}\n")
        log_f.write(f"chunk_3d_b={args.chunk_3d_b}\n")
        log_f.write(f"use_tf32={not args.disable_tf32}\n")
        log_f.write(f"disable_threshold_norm={args.disable_threshold_norm}\n")
        log_f.write(f"thr_preset={args.thr_preset}\n")
        log_f.write(f"thr2d_input={args.thr2d}\n")
        log_f.write(f"thr3d_input={args.thr3d}\n")
        log_f.write(f"thr2d_final={final_thr2d}\n")
        log_f.write(f"thr3d_final={final_thr3d}\n")
        log_f.write(f"exclude_below_threshold={args.exclude_below_threshold}\n")
        log_f.write(f"stp_mode_2014={args.stp_mode_2014}\n")
        log_f.write(f"cv_scheme={args.cv_scheme}\n")
        log_f.write(f"single_modal_eval={args.single_modal_eval}\n")
        log_f.write(f"model_family={args.model_family}\n")
        log_f.write(f"hgb_max_iter={args.hgb_max_iter}\n")
        log_f.write(f"hgb_learning_rate={args.hgb_learning_rate}\n")
        log_f.write(f"hgb_max_leaf_nodes={args.hgb_max_leaf_nodes}\n")
        log_f.write(f"min_neg_per_pair={args.min_neg_per_pair}\n")
        log_f.write(f"max_neg_per_pair={args.max_neg_per_pair}\n")
        log_f.write(f"show_progress={not args.no_progress}\n")
        log_f.write(f"feature_cache_dir={args.feature_cache_dir}\n")
        log_f.write(f"feature_cache_signature={feature_cache_signature}\n")
        log_f.write(f"keep_negative_features={args.keep_negative_features}\n")
        log_f.write(f"c_reg={args.c_reg}\n")
        log_f.write("class_weight=balanced\n")
        log_f.write(f"cutoff_year={args.cutoff_year}\n")
        log_f.write(f"run_hash={run_hash}\n")
        log_f.write("pair_id_governance=enabled\n")
        log_f.write("pair_sampling_rule=pair_id_only(no_row_sampling)\n")
        log_f.write(
            f"pair_structure_rule=rows[{1+args.min_neg_per_pair}..{1+args.max_neg_per_pair}]"
            f"(1_pos+{args.min_neg_per_pair}..{args.max_neg_per_pair}_neg)_per_pair\n"
        )
        log_f.write("pair_precheck_stages=meta_load,train_split,pre_batch,pre_training,pre_oot\n")
        log_f.write("status=started\n")
    print(f"   - Run output dir: {run_out_dir}")
    print(f"   - Run config log: {run_log_path}")

    try:
        trainer = STPUltraFastTrainer(
            args.meta_file,
            args.fp2_memmap,
            es5d_path,
            out_dir=run_out_dir,
            k_mode=args.k_mode,
            k_policy=args.k_policy,
            c_reg=args.c_reg,
            cutoff_year=args.cutoff_year,
            scaffold_mask=scaffold_mask,
            batch_size=args.batch_size,
            chunk_3d_b=args.chunk_3d_b,
            use_tf32=(not args.disable_tf32),
            disable_threshold_norm=args.disable_threshold_norm,
            thr_preset=args.thr_preset,
            thr2d=args.thr2d,
            thr3d=args.thr3d,
            exclude_below_threshold=args.exclude_below_threshold,
            keep_negative_features=args.keep_negative_features,
            stp_mode_2014=args.stp_mode_2014,
            cv_scheme=args.cv_scheme,
            single_modal_eval=args.single_modal_eval,
            model_family=args.model_family,
            hgb_max_iter=args.hgb_max_iter,
            hgb_learning_rate=args.hgb_learning_rate,
            hgb_max_leaf_nodes=args.hgb_max_leaf_nodes,
            min_neg_per_pair=args.min_neg_per_pair,
            max_neg_per_pair=args.max_neg_per_pair,
            show_progress=(not args.no_progress),
            feature_cache_dir=args.feature_cache_dir,
            feature_cache_signature=feature_cache_signature,
        )
        trainer.execute_cv_and_oot_evaluation()
        with open(run_log_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"end_time={datetime.now().isoformat()}\n")
            log_f.write("status=success\n")
    except Exception as exc:
        with open(run_log_path, "a", encoding="utf-8") as log_f:
            log_f.write(f"end_time={datetime.now().isoformat()}\n")
            log_f.write("status=failed\n")
            log_f.write(f"error={repr(exc)}\n")
            log_f.write(traceback.format_exc())
        raise
