# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 14:26:24 2026

@author: KIOM_User
"""

# -*- coding: utf-8 -*-
"""
Phase 3 v2.0: AUC-BOOSTED TRUE DUAL-GPU STP TRAINING
- [FIX 1] Class Imbalance Fix: Applied class_weight='balanced' to Logistics.
- [FIX 2] Feature Representation: K=5 Summary Statistics (Max, Mean, Std) Fusion.
- [FIX 3] Coefficient Stability: Applied Gaussian 1D Smoothing across HA-bins.
- [FIX 4] NaN Handling: Safe temporal split for missing publication_year.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import json
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

class STPAucBoostedTrainer:
    def __init__(self, meta_file, fp2_memmap, es5d_memmap, out_dir="features_store", 
                 k_mode=5, k_policy='paired_sum', c_reg=1.0, cutoff_year=2023): # C_REG 1.0으로 조정 (안정성)
        self.meta_file = meta_file
        self.fp2_path = fp2_memmap
        self.es5d_path = es5d_memmap
        self.out_dir = out_dir
        
        self.K = k_mode
        self.POLICY = k_policy  
        self.C_REG = c_reg      
        self.CUTOFF_YEAR = cutoff_year 
        
        self.BATCH_SIZE = 4096 
        self.CHUNK_3D_B = 128 
        
        self._initialize_gpu_engine()
        
def _initialize_gpu_engine(self):
    print(f"\n>> 1. Initializing AUC-Boosted Dual-GPU Engine (K={self.K} Fusion)...")
    
    full_df = pd.read_parquet(self.meta_file)
    full_df['scaffold_hash'] = pd.util.hash_pandas_object(
        full_df['scaffold_smiles'], index=False
    ).astype(np.int64)

    if 'publication_year' in full_df.columns:
        full_df['publication_year'] = full_df['publication_year'].fillna(-1)

    self.FULL_N = os.path.getsize(self.fp2_path) // 128
    self.NCONF = os.path.getsize(self.es5d_path) // (self.FULL_N * 72)

    db_2d_ram = np.array(np.memmap(self.fp2_path, dtype=np.uint64,
                                   mode='r', shape=(self.FULL_N, 16)))
    db_3d_ram = np.array(np.memmap(self.es5d_path, dtype=np.float32,
                                   mode='r', shape=(self.FULL_N, self.NCONF, 18)))

    self.db2d_gpu = torch.from_numpy(
        np.unpackbits(db_2d_ram.view(np.uint8), axis=1)
    ).to(device='cuda:0', dtype=torch.float16)

    self.db2d_ones = self.db2d_gpu.sum(dim=1).float()
    self.db3d_gpu = torch.from_numpy(db_3d_ram).to(device='cuda:1')

    # ✅ FULL_N 기준 scaffold 배열 재구성 (핵심 수정)
    scaf_arr = np.full(self.FULL_N, -1, dtype=np.int64)
    scaf_arr[full_df['memmap_idx'].values] = full_df['scaffold_hash'].values
    self.gpu_scaffolds = torch.tensor(scaf_arr, device='cuda:0')

    # fold도 동일 방식 유지
    folds_arr = np.full(self.FULL_N, -1, dtype=np.int64)
    folds_arr[full_df['memmap_idx'].values] = \
        full_df.get('cv_fold', pd.Series(-1)).values
    self.gpu_folds = torch.tensor(folds_arr, device='cuda:0')

    # ✅ OOT 정의 개선
    if 'publication_year' in full_df.columns:
        self.future_df = full_df[full_df['publication_year'] > self.CUTOFF_YEAR].copy()
        self.meta_df = full_df[full_df['publication_year'] <= self.CUTOFF_YEAR].copy()
    else:
        self.future_df = pd.DataFrame()
        self.meta_df = full_df.copy()

    positives = self.meta_df[self.meta_df['set_type'] == 'Positive']
    tmap = positives.groupby('target_chembl_id')['memmap_idx'].apply(list)

    self.target_actives_cuda0 = {
        k: torch.tensor(v, device='cuda:0') for k, v in tmap.items()
    }
    self.target_actives_cuda1 = {
        k: torch.tensor(v, device='cuda:1') for k, v in tmap.items()
    }

def _extract_summary_features_batched(self, query_indices, target_id, query_scafs, query_folds=None):
        B = len(query_indices)
        if target_id not in self.target_actives_cuda0:
            return torch.zeros((B, 6), dtype=torch.float32, device='cpu')
            
        actives_0 = self.target_actives_cuda0[target_id]
        actives_1 = self.target_actives_cuda1[target_id]
        M = len(actives_0)
        
        q_idx_0, q_scaf_0 = query_indices, query_scafs
        a_scaf_exp = self.gpu_scaffolds[actives_0].unsqueeze(0)
        
        mask = (q_idx_0.unsqueeze(1) != actives_0.unsqueeze(0)) & \
               (q_scaf_0.unsqueeze(1) != a_scaf_exp) & \
               (a_scaf_exp != -1)
        if query_folds is not None:
            mask &= (query_folds.unsqueeze(1) != self.gpu_folds[actives_0].unsqueeze(0))
            
        # --- 2D 연산 (cuda:0) ---
        q_2d, a_2d = self.db2d_gpu[q_idx_0], self.db2d_gpu[actives_0]
        intersection = torch.matmul(q_2d, a_2d.T).float()
        union = self.db2d_ones[q_idx_0].unsqueeze(1) + self.db2d_ones[actives_0].unsqueeze(0) - intersection
        sim_2d_matrix = torch.where(union > 0, intersection / union, 0.0)
        
        # --- 3D 연산 (cuda:1) ---
        q_idx_1 = query_indices.to('cuda:1')
        q_3d, a_3d = torch.nan_to_num(self.db3d_gpu[q_idx_1], nan=1e4), torch.nan_to_num(self.db3d_gpu[actives_1], nan=-1e4)
        sim_3d_matrix_1 = torch.empty((B, M), device='cuda:1', dtype=torch.float32)
        a_3d_exp = a_3d.unsqueeze(0) 
        
        for i in range(0, B, self.CHUNK_3D_B):
            q_chunk = q_3d[i:i+self.CHUNK_3D_B].unsqueeze(1) 
            dist = torch.cdist(q_chunk, a_3d_exp, p=1.0)     
            sim_3d_matrix_1[i:i+self.CHUNK_3D_B] = (1.0 / (1.0 + (dist / 18.0))).amax(dim=(2, 3))
            
        sim_3d_matrix = sim_3d_matrix_1.to('cuda:0', non_blocking=True)
        
        # --- 마스킹 적용 ---
        sim_2d_matrix.masked_fill_(~mask, -1e9)
        sim_3d_matrix.masked_fill_(~mask, -1e9)
      
  
        
        k_actual = min(self.K, M)
        if k_actual == 0:
            return torch.zeros((B, 6), dtype=torch.float32, device='cpu')
            
        # =====================================================================
        # 🚨 [핵심 패치] 2D와 3D를 "각각 독립적으로" 정렬하고 Top-K를 추출합니다.
        # =====================================================================
        top_3d_vals = torch.topk(sim_3d_matrix, k_actual, dim=1).values.clamp(min=0.0)
        top_2d_vals = torch.topk(sim_2d_matrix, k_actual, dim=1).values.clamp(min=0.0)
        
        features = torch.zeros((B, 6), dtype=torch.float32, device='cpu')
        
        # 3D 고유 요약 통계 (독립적 3D 타겟 스코어)
        features[:, 0] = top_3d_vals.max(dim=1).values
        features[:, 1] = top_3d_vals.mean(dim=1)
        features[:, 2] = torch.nan_to_num(top_3d_vals.std(dim=1, unbiased=False), nan=0.0)
        
        # 2D 고유 요약 통계 (독립적 2D 타겟 스코어)
        features[:, 3] = top_2d_vals.max(dim=1).values
        features[:, 4] = top_2d_vals.mean(dim=1)
        features[:, 5] = torch.nan_to_num(top_2d_vals.std(dim=1, unbiased=False), nan=0.0)
        
        return features


    def _build_dataset_batched(self, df, desc):
        print(f"\n>> Generating Summary Fusion Features for {desc} (N={len(df):,})...")
        start_time = time.time()
        is_oot = (desc == "Future OOT")
        
        df = df.copy()
        df['sort_idx'] = np.arange(len(df))
        grouped = df.groupby('target_chembl_id')
        
        all_features = np.zeros((len(df), 6), dtype=np.float32) # 피처 6개 고정
        
        with tqdm(total=len(df), desc="Matrix Generation", mininterval=5.0) as pbar:
            for target_id, group in grouped:
                group_indices = group['sort_idx'].values
                q_memmap = torch.tensor(group['memmap_idx'].values, device='cuda:0')
                q_scafs = torch.tensor(group['scaffold_hash'].values, device='cuda:0')
                q_folds = None if is_oot else torch.tensor(group['cv_fold'].values, device='cuda:0')
                
                for i in range(0, len(q_memmap), self.BATCH_SIZE):
                    batch_memmap, batch_scafs = q_memmap[i:i+self.BATCH_SIZE], q_scafs[i:i+self.BATCH_SIZE]
                    batch_folds = None if q_folds is None else q_folds[i:i+self.BATCH_SIZE]
                    
                    batch_feats = self._extract_summary_features_batched(batch_memmap, target_id, batch_scafs, batch_folds)
                    all_features[group_indices[i:i+self.BATCH_SIZE]] = batch_feats.numpy()
                    pbar.update(len(batch_memmap))
        
        cols = ['s_3d_max', 's_3d_mean', 's_3d_std', 's_2d_max', 's_2d_mean', 's_2d_std']
        out_df = pd.DataFrame({
            'heavy_atoms': df['heavy_atoms'].values, 
            'ha_bin': np.clip(df['heavy_atoms'].values, 10, 60),
            'label': (df['set_type'] == 'Positive').astype(int).values, 
            'cv_fold': df.get('cv_fold', pd.Series(np.full(len(df), -1))).values
        })
        out_df = pd.concat([out_df, pd.DataFrame(all_features, columns=cols)], axis=1)
        print(f"   ✅ Matrix generated in {time.time()-start_time:.2f}s")
        return out_df

    def execute_cv_and_oot_evaluation(self):
        self.train_df = self._build_dataset_batched(self.meta_df, "Past Training Data")
        
        print(f"\n>> 3. Executing Imbalance-Aware CV Training (class_weight='balanced')...")
        feature_cols = ['s_3d_max', 's_3d_mean', 's_3d_std', 's_2d_max', 's_2d_mean', 's_2d_std']
        
        unique_bins = sorted(self.train_df['ha_bin'].unique())
        unique_folds = np.sort(self.train_df['cv_fold'].unique())
        
        cv_reports = []
        self.production_models = {} 
        raw_coefficients = np.zeros((65, 7)) # Intercept + 6 Features
        
        for ha_bin in tqdm(unique_bins, desc="Training LR"):
            subset = self.train_df[self.train_df['ha_bin'] == ha_bin]
            X, y, folds = subset[feature_cols].values, subset['label'].values, subset['cv_fold'].values
            
            bin_auc, bin_prauc = [], []
            for f in unique_folds:
                if f == -1: continue 
                train_mask, test_mask = (folds != f), (folds == f)
                X_train, y_train, X_test, y_test = X[train_mask], y[train_mask], X[test_mask], y[test_mask]
                
                if len(np.unique(y_train)) > 1 and len(np.unique(y_test)) > 1:
                    # [핵심 수술 1] 10:1 불균형 해소: class_weight='balanced'
                    clf = LogisticRegression(class_weight='balanced', penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000)
                    clf.fit(X_train, y_train)
                    preds = clf.predict_proba(X_test)[:, 1]
                    bin_auc.append(roc_auc_score(y_test, preds))
                    bin_prauc.append(average_precision_score(y_test, preds))
            
            cv_reports.append({
                'ha_bin': ha_bin, 
                'mean_AUC_Combined': np.mean(bin_auc) if bin_auc else 0.0,
                'mean_PRAUC_Combined': np.mean(bin_prauc) if bin_prauc else 0.0
            })
            
            if len(np.unique(y)) > 1:
                clf_prod = LogisticRegression(class_weight='balanced', penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000)
                clf_prod.fit(X, y)
                self.production_models[ha_bin] = clf_prod
                raw_coefficients[ha_bin, 0] = clf_prod.intercept_[0]
                raw_coefficients[ha_bin, 1:] = clf_prod.coef_[0]

        pd.DataFrame(cv_reports).to_csv(os.path.join(self.out_dir, f"stp_cv_report_FusionK{self.K}_{self.POLICY}.csv"), index=False)
        
        # [핵심 수술 3] Gaussian Smoothing: HA-Bin 간의 가중치 급변(노이즈) 제거
        print("\n>> 4. Applying Gaussian Smoothing to Coefficients...")
        smoothed_coefficients = {}
        
        for col_idx in range(7):
            raw_coefficients[:, col_idx] = gaussian_filter1d(raw_coefficients[:, col_idx], sigma=1.5)
            
        for ha_bin in range(10, 61):
            if np.all(raw_coefficients[ha_bin] == 0):
                # 데이터가 아예 없는 빈 Bin은 가장 가까운 값 복사 (안전장치)
                valid_idx = np.nonzero(raw_coefficients[:, 1])[0]
                closest_idx = valid_idx[np.argmin(np.abs(valid_idx - ha_bin))]
                smoothed_coefficients[str(ha_bin)] = raw_coefficients[closest_idx].tolist()
            else:
                smoothed_coefficients[str(ha_bin)] = raw_coefficients[ha_bin].tolist()
                
            # 부드러워진 가중치를 실제 모델 객체에도 강제 이식
            if ha_bin in self.production_models and self.production_models[ha_bin] is not None:
                self.production_models[ha_bin].intercept_ = np.array([smoothed_coefficients[str(ha_bin)][0]])
                self.production_models[ha_bin].coef_ = np.array([smoothed_coefficients[str(ha_bin)][1:]])

        with open(os.path.join(self.out_dir, f"stp_coef_FusionK{self.K}_{self.POLICY}.json"), "w") as f:
            json.dump({"K": self.K, "Policy": self.POLICY, "Temporal_Cutoff": self.CUTOFF_YEAR, "coef": smoothed_coefficients}, f, indent=4)
        
        if not self.future_df.empty:
            self.future_features_df = self._build_dataset_batched(self.future_df, "Future OOT")
            print("\n>> 5. Evaluating OOT Performance with Smoothed Imbalance-Aware Models...")
            
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
                        'OOT_AUC_Combined': roc_auc_score(y_oot, preds_oot),
                        'OOT_PRAUC': average_precision_score(y_oot, preds_oot)
                    })
                    
            if oot_reports:
                pd.DataFrame(oot_reports).to_csv(os.path.join(self.out_dir, f"stp_OOT_report_FusionK{self.K}_{self.POLICY}.csv"), index=False)
                print("   ✅ OOT Evaluation Completed! Report saved.")

if __name__ == "__main__":
    trainer = STPAucBoostedTrainer(
        "features_store/final_training_meta.parquet", 
        "features_store/fp2_aligned.memmap", 
        "features_store/es5d_db_k20.memmap", 
        k_mode=5, k_policy='paired_sum', c_reg=1.0, cutoff_year=2023 
    )
    trainer.execute_cv_and_oot_evaluation()