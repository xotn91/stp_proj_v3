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
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Numpy 읽기 전용 경고 무시
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

class STPUltraFastTrainer:
    def __init__(self, meta_file, fp2_memmap, es5d_memmap, out_dir="features_store", 
                 k_mode=1, k_policy='independent', c_reg=10.0, cutoff_year=2023,
                 eval_mode='distinct_scaffolds'):
        self.meta_file = meta_file
        self.fp2_path = fp2_memmap
        self.es5d_path = es5d_memmap
        self.out_dir = out_dir
        
        self.K = k_mode
        self.POLICY = k_policy  
        self.C_REG = c_reg      
        self.CUTOFF_YEAR = cutoff_year 
        self.eval_mode = eval_mode
        
        self.BATCH_SIZE = 4096 
        # [핵심] 쿼리(B)를 자르는 청크 사이즈 (L2 캐시 최적화 사이즈)
        self.CHUNK_3D_B = 128 
        
        self._initialize_gpu_engine()
        
    def _initialize_gpu_engine(self):
        print(
            f"\n>> 1. Initializing Ultra-Fast GPU Engine "
            f"(K={self.K}, Policy={self.POLICY}, EvalMode={self.eval_mode})..."
        )
        
        full_df = pd.read_parquet(self.meta_file)
        
        required_cols = ['memmap_idx', 'target_chembl_id', 'set_type', 'heavy_atoms', 'scaffold_smiles']
        for c in required_cols:
            assert c in full_df.columns, f"치명적 오류: 필수 컬럼 '{c}'이 누락되었습니다."
            
        full_df['scaffold_hash'] = pd.util.hash_pandas_object(full_df['scaffold_smiles'], index=False).astype(np.int64)
        if 'assay_id' in full_df.columns:
            assay_col = 'assay_id'
        elif 'assay_chembl_id' in full_df.columns:
            assay_col = 'assay_chembl_id'
        else:
            raise AssertionError("치명적 오류: assay_id 또는 assay_chembl_id 컬럼이 누락되었습니다.")
        
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
        
        folds_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        folds_arr[full_df['memmap_idx'].values] = full_df.get('cv_fold', pd.Series(np.full(len(full_df), -1))).values
        self.gpu_folds = torch.tensor(folds_arr, device='cuda:0')
        
        scaf_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        scaf_arr[full_df['memmap_idx'].values] = full_df['scaffold_hash'].values
        self.gpu_scaffolds = torch.tensor(scaf_arr, device='cuda:0')

        assay_codes, _ = pd.factorize(full_df[assay_col], sort=False)
        assay_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        assay_arr[full_df['memmap_idx'].values] = assay_codes.astype(np.int64)
        self.gpu_assays = torch.tensor(assay_arr, device='cuda:0')
        apply_scaffold_mask = self.eval_mode in ('distinct_scaffolds', 'distinct_scaffolds_assays')
        apply_assay_mask = self.eval_mode == 'distinct_scaffolds_assays'
        print(
            f"   - Mask config: scaffold_mask={apply_scaffold_mask}, "
            f"assay_mask={apply_assay_mask} (assay_col={assay_col})"
        )
        
        if 'publication_year' in full_df.columns:
            self.future_df = full_df[full_df['publication_year'] > self.CUTOFF_YEAR].copy()
            self.meta_df = full_df[full_df['publication_year'] <= self.CUTOFF_YEAR].copy()
        else:
            self.future_df = pd.DataFrame()
            self.meta_df = full_df.copy()

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
        apply_scaffold_mask = self.eval_mode in ('distinct_scaffolds', 'distinct_scaffolds_assays')
        if apply_scaffold_mask:
            q_scaf_exp = query_scafs.unsqueeze(1)
            a_scaf_exp = self.gpu_scaffolds[actives_tensor].unsqueeze(0)
            mask &= (q_scaf_exp != a_scaf_exp) & (a_scaf_exp != -1)
        if self.eval_mode == 'distinct_scaffolds_assays':
            q_assay = self.gpu_assays[query_indices].unsqueeze(1)
            a_assay = self.gpu_assays[actives_tensor].unsqueeze(0)
            assay_mask = (q_assay != a_assay)
            mask &= assay_mask
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
            
        # paper-style threshold normalization
        THR_2D = 0.65
        THR_3D = 0.30
        sim_2d_matrix = (sim_2d_matrix - THR_2D) / (1.0 - THR_2D)
        sim_2d_matrix = torch.clamp(sim_2d_matrix, 0.0, 1.0)
        sim_3d_matrix = (sim_3d_matrix - THR_3D) / (1.0 - THR_3D)
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
        
        if self.POLICY == 'independent':
            top_3d_idx = torch.topk(sim_3d_matrix, k_actual, dim=1).indices
            top_2d_idx = torch.topk(sim_2d_matrix, k_actual, dim=1).indices
            features[:, 0::2][:, :k_actual] = sim_3d_matrix[b_idx, top_3d_idx]
            features[:, 1::2][:, :k_actual] = sim_2d_matrix[b_idx, top_2d_idx]
        else:
            top_k_indices = torch.topk(score, k_actual, dim=1).indices
            features[:, 0::2][:, :k_actual] = sim_3d_matrix[b_idx, top_k_indices]
            features[:, 1::2][:, :k_actual] = sim_2d_matrix[b_idx, top_k_indices]
        
        features.masked_fill_(features < 0.0, 0.0)
        return features.cpu()

    def _build_dataset_batched(self, df, desc):
        print(f"\n>> Generating Feature Matrix for {desc} (N={len(df):,}) using Ultra-Fast Batched Engine...")
        start_time = time.time()
        
        is_oot = (desc == "Future OOT")
        
        df = df.copy()
        df['sort_idx'] = np.arange(len(df))
        grouped = df.groupby('target_chembl_id')
        total_targets = len(grouped)
        
        all_features = np.zeros((len(df), self.K * 2), dtype=np.float32)
        print(f"   - Processing {total_targets} unique targets...")
        
        with tqdm(total=len(df), desc="Rows Processed", unit="rows") as pbar:
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
            'cv_fold': df.get('cv_fold', pd.Series(np.full(len(df), -1))).values
        })
        out_df = pd.concat([out_df, pd.DataFrame(all_features, columns=cols)], axis=1)
        print(f"   ✅ Batched Matrix generated in {time.time()-start_time:.2f}s")
        return out_df

    def execute_cv_and_oot_evaluation(self):
        self.train_df = self._build_dataset_batched(self.meta_df, "Past Training Data")
        
        print(f"\n>> 3. Executing True CV & Training (L2, C={self.C_REG}, EvalMode={self.eval_mode})...")
        feature_cols = [c for c in self.train_df.columns if c.startswith('s1_') or c.startswith('s2_')]
        
        idx_3d = [i for i, c in enumerate(feature_cols) if '3d' in c]
        idx_2d = [i for i, c in enumerate(feature_cols) if '2d' in c]
        
        unique_bins = sorted(self.train_df['ha_bin'].unique())
        unique_folds = np.sort(self.train_df['cv_fold'].unique())
        
        cv_reports = []
        self.production_models = {} 
        final_coefficients = {}
        
        for ha_bin in tqdm(unique_bins, desc="Training LR"):
            subset = self.train_df[self.train_df['ha_bin'] == ha_bin]
            X, y, folds = subset[feature_cols].values, subset['label'].values, subset['cv_fold'].values
            
            bin_auc, bin_prauc, bin_auc_3d, bin_auc_2d = [], [], [], []
            for f in unique_folds:
                if f == -1: continue 
                train_mask, test_mask = (folds != f), (folds == f)
                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]
                
                if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
                    clf = LogisticRegression(penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000)
                    clf.fit(X_train, y_train)
                    preds = clf.predict_proba(X_test)[:, 1]
                    bin_auc.append(roc_auc_score(y_test, preds))
                    bin_prauc.append(average_precision_score(y_test, preds))
                    bin_auc_3d.append(roc_auc_score(y_test, X_test[:, idx_3d].max(axis=1)))
                    bin_auc_2d.append(roc_auc_score(y_test, X_test[:, idx_2d].max(axis=1)))
            
            cv_reports.append({
                'ha_bin': ha_bin, 
                'eval_mode': self.eval_mode,
                'mean_AUC_Combined': np.mean(bin_auc) if bin_auc else 0.0,
                'mean_AUC_3D_Only': np.mean(bin_auc_3d) if bin_auc_3d else 0.0,
                'mean_AUC_2D_Only': np.mean(bin_auc_2d) if bin_auc_2d else 0.0,
                'mean_PRAUC_Combined': np.mean(bin_prauc) if bin_prauc else 0.0
            })
            
            if len(np.unique(y)) > 1:
                clf_prod = LogisticRegression(penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000)
                clf_prod.fit(X, y)
                self.production_models[ha_bin] = clf_prod
                final_coefficients[str(ha_bin)] = [float(clf_prod.intercept_[0])] + [float(c) for c in clf_prod.coef_[0]]
            else:
                self.production_models[ha_bin] = None
                final_coefficients[str(ha_bin)] = [-5.0] + [3.0, 7.0] * self.K

        pd.DataFrame(cv_reports).to_csv(
            os.path.join(self.out_dir, f"stp_cv_report_K{self.K}_{self.POLICY}_{self.eval_mode}.csv"),
            index=False,
        )
        with open(os.path.join(self.out_dir, f"stp_coef_K{self.K}_{self.POLICY}_{self.eval_mode}.json"), "w") as f:
            json.dump({
                "K": self.K,
                "Policy": self.POLICY,
                "Eval_Mode": self.eval_mode,
                "Assay_Mask_Applied": self.eval_mode == 'distinct_scaffolds_assays',
                "Temporal_Cutoff": self.CUTOFF_YEAR,
                "coef": final_coefficients
            }, f, indent=4)
        
        if not self.future_df.empty:
            self.future_features_df = self._build_dataset_batched(self.future_df, "Future OOT")
            print("\n>> 4. Evaluating Out-Of-Time (OOT) Performance...")
            
            oot_reports = []
            for ha_bin in tqdm(unique_bins, desc="OOT Evaluation"):
                if ha_bin not in self.production_models or self.production_models[ha_bin] is None: continue
                
                subset = self.future_features_df[self.future_features_df['ha_bin'] == ha_bin]
                if len(subset) == 0: continue
                
                X_oot, y_oot = subset[feature_cols].values, subset['label'].values
                
                if len(np.unique(y_oot)) > 1:
                    preds_oot = self.production_models[ha_bin].predict_proba(X_oot)[:, 1]
                    oot_reports.append({
                        'ha_bin': ha_bin, 'N_Samples': len(y_oot),
                        'eval_mode': self.eval_mode,
                        'OOT_AUC_Combined': roc_auc_score(y_oot, preds_oot),
                        'OOT_AUC_3D_Only': roc_auc_score(y_oot, X_oot[:, idx_3d].max(axis=1)),
                        'OOT_AUC_2D_Only': roc_auc_score(y_oot, X_oot[:, idx_2d].max(axis=1)),
                        'OOT_PRAUC': average_precision_score(y_oot, preds_oot)
                    })
                    
            if oot_reports:
                pd.DataFrame(oot_reports).to_csv(
                    os.path.join(self.out_dir, f"stp_OOT_report_K{self.K}_{self.POLICY}_{self.eval_mode}.csv"),
                    index=False,
                )
                print("   ✅ OOT Evaluation Completed! Report saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-configurable unpaired trainer")
    parser.add_argument("--meta_file", default="features_store/final_training_meta.parquet")
    parser.add_argument("--fp2_memmap", default="features_store/fp2_aligned.memmap")
    parser.add_argument("--es5d_memmap", default=None)
    parser.add_argument("--out_dir", default="features_store")
    parser.add_argument("--k_mode", type=int, default=1)
    parser.add_argument(
        "--k_policy",
        default="independent",
        choices=["independent", "paired_sum", "paired_2d", "paired_3d"],
    )
    parser.add_argument("--c_reg", type=float, default=10.0)
    parser.add_argument("--cutoff_year", type=int, default=2023)
    parser.add_argument(
        "--eval_mode",
        default="distinct_scaffolds",
        choices=["all", "distinct_scaffolds", "distinct_scaffolds_assays"],
    )
    args = parser.parse_args()

    es5d_path = args.es5d_memmap or f"features_store/es5d_db_k{args.k_mode}.memmap"

    trainer = STPUltraFastTrainer(
        args.meta_file,
        args.fp2_memmap,
        es5d_path,
        out_dir=args.out_dir,
        k_mode=args.k_mode,
        k_policy=args.k_policy,
        c_reg=args.c_reg,
        cutoff_year=args.cutoff_year,
        eval_mode=args.eval_mode,
    )
    trainer.execute_cv_and_oot_evaluation()
