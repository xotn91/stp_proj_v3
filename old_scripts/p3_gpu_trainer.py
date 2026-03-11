# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 09:55:19 2026

@author: KIOM_User
"""

# -*- coding: utf-8 -*-
"""
Phase 3: FINAL STP-Grade Dual-GPU Scoring & Temporal Validation
- Fixed: PyTorch bitwise_count Absence (Solved via Boolean Unpacking)
- Fixed: Numpy Non-writable Tensor Warning (Solved via .copy())
- Added: Strict Byte-level Memmap Alignment & Dynamic NCONF
- Maintained: True OOT (Out-Of-Time) Future Data Evaluation
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import json
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

class STPFinalTrainer:
    def __init__(self, meta_file, fp2_memmap, es5d_memmap, out_dir="features_store", 
                 k_mode=1, k_policy='paired_sum', c_reg=10.0, cutoff_year=2023):
        self.meta_file = meta_file
        self.fp2_path = fp2_memmap
        self.es5d_path = es5d_memmap
        self.out_dir = out_dir
        
        self.K = k_mode
        self.POLICY = k_policy  
        self.C_REG = c_reg      
        self.CUTOFF_YEAR = cutoff_year 
        
        self._initialize_gpu_engine()
        
    def _initialize_gpu_engine(self):
        print(f"\n>> 1. Initializing GPU Engine (K={self.K}, Policy={self.POLICY})...")
        
        full_df = pd.read_parquet(self.meta_file)
        
        required_cols = ['memmap_idx', 'target_chembl_id', 'set_type', 'heavy_atoms', 'scaffold_smiles']
        for c in required_cols:
            assert c in full_df.columns, f"치명적 오류: 필수 컬럼 '{c}'이 누락되었습니다."
            
        full_df['scaffold_hash'] = pd.util.hash_pandas_object(full_df['scaffold_smiles'], index=False).astype(np.int64)
        
        size_fp2 = os.path.getsize(self.fp2_path)
        size_es5d = os.path.getsize(self.es5d_path)
        
        row_fp2 = 16 * 8  
        assert size_fp2 % row_fp2 == 0, "FP2 memmap file size is not aligned (Corrupted)."
        self.FULL_N = size_fp2 // row_fp2
        
        row_es5d_atom = 18 * 4  
        assert size_es5d % (self.FULL_N * row_es5d_atom) == 0, "ES5D memmap file size is not aligned with FULL_N."
        self.NCONF = size_es5d // (self.FULL_N * row_es5d_atom)
        assert self.NCONF in (1, 20), f"치명적 오류: 예상치 못한 NCONF 값 ({self.NCONF})"
        
        print(f"   - Validated Memmap: N={self.FULL_N:,}, Conformations={self.NCONF}")
        
        db_2d_cpu = np.memmap(self.fp2_path, dtype=np.uint64, mode='r', shape=(self.FULL_N, 16))
        db_3d_cpu = np.memmap(self.es5d_path, dtype=np.float32, mode='r', shape=(self.FULL_N, self.NCONF, 18))
        
        # [핵심 수정 1] bitwise_count 에러 해결을 위한 1024-bit Boolean Unpacking
        print("   - Unpacking 2D FP2 bits for ultra-fast Boolean Tanimoto (approx 884 MB VRAM)...")
        # uint64를 uint8로 본 다음 비트 단위로 풀어서 0과 1의 2차원 배열(FULL_N, 1024)로 만듦
        db_2d_bits = np.unpackbits(db_2d_cpu.view(np.uint8), axis=1) 
        
        # [핵심 수정 2] .copy()를 추가하여 PyTorch 읽기 전용 경고 완벽 제거
        self.db2d_gpu = torch.from_numpy(db_2d_bits).to(device='cuda:0', dtype=torch.bool)
        self.db3d_gpu = torch.from_numpy(db_3d_cpu.copy()).to(device='cuda:0') 
        
        # 미리 분자별 1의 개수(Popcount)를 float 형태로 계산해둠
        self.db2d_ones = self.db2d_gpu.sum(dim=1).float()
        
        folds_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        folds_arr[full_df['memmap_idx'].values] = full_df.get('cv_fold', pd.Series(np.full(len(full_df), -1))).values
        self.gpu_folds = torch.tensor(folds_arr, device='cuda:0')
        
        scaf_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        scaf_arr[full_df['memmap_idx'].values] = full_df['scaffold_hash'].values
        self.gpu_scaffolds = torch.tensor(scaf_arr, device='cuda:0')
        
        if 'publication_year' in full_df.columns:
            print(f"   ⏳ Applying Temporal Split (Cut-off <= {self.CUTOFF_YEAR})")
            self.future_df = full_df[full_df['publication_year'] > self.CUTOFF_YEAR].copy()
            self.meta_df = full_df[full_df['publication_year'] <= self.CUTOFF_YEAR].copy()
            print(f"   - Reserved {len(self.future_df):,} future molecules for OOT Evaluation.")
        else:
            self.future_df = pd.DataFrame()
            self.meta_df = full_df.copy()

        print("   - Building Past Target-to-Actives Tensor Cache...")
        positives = self.meta_df[self.meta_df['set_type'] == 'Positive']
        tmap = positives.groupby('target_chembl_id')['memmap_idx'].apply(list)
        self.target_actives_tensor = {k: torch.tensor(v, device='cuda:0', dtype=torch.int64) for k, v in tmap.items()}

    def _extract_features(self, query_idx, target_id, query_scaf_hash, query_fold=None):
        if target_id not in self.target_actives_tensor:
            return [0.0] * (self.K * 2)
            
        actives_tensor = self.target_actives_tensor[target_id]
        
        mask = (actives_tensor != query_idx) & \
               (self.gpu_scaffolds[actives_tensor] != query_scaf_hash) & \
               (self.gpu_scaffolds[actives_tensor] != -1)
               
        if query_fold is not None and query_fold != -1:
            mask &= (self.gpu_folds[actives_tensor] != query_fold)
               
        valid_indices = actives_tensor[mask]
        
        if valid_indices.numel() == 0:
            return [0.0] * (self.K * 2)
            
        # --- 2D Boolean Tanimoto ---
        q_2d = self.db2d_gpu[query_idx].unsqueeze(0)
        q_ones = self.db2d_ones[query_idx]
        db_2d = self.db2d_gpu[valid_indices]
        db_ones = self.db2d_ones[valid_indices]
        
        # [핵심 수정 3] 비트 연산(&) 후 합계를 구하는 가장 빠르고 안전한 방법 적용
        intersection = (q_2d & db_2d).sum(dim=1).float()
        union = q_ones + db_ones - intersection
        sim_2d = torch.where(union > 0, intersection / union, 0.0)
        
        # --- 3D Manhattan ---
        q_3d = torch.nan_to_num(self.db3d_gpu[query_idx], nan=1e4)
        db_3d = torch.nan_to_num(self.db3d_gpu[valid_indices], nan=-1e4)
        
        dist_matrix = torch.cdist(db_3d, q_3d.unsqueeze(0), p=1.0)
        sim_3d_matrix = 1.0 / (1.0 + (dist_matrix / 18.0))
        sim_3d = sim_3d_matrix.amax(dim=(1, 2))  
        
        # --- Policy Selection ---
        features = []
        k_actual = min(self.K, sim_2d.numel())
        
        if self.POLICY == 'independent':
            top_3d = torch.topk(sim_3d, k_actual).values.tolist()
            top_2d = torch.topk(sim_2d, k_actual).values.tolist()
            for s3, s2 in zip(top_3d, top_2d):
                features.extend([s3, s2])
        else:
            if self.POLICY == 'paired_sum': score = sim_2d + sim_3d
            elif self.POLICY == '2d': score = sim_2d
            else: score = sim_3d
            
            top_k_indices = torch.topk(score, k_actual).indices
            for idx in top_k_indices:
                features.extend([sim_3d[idx].item(), sim_2d[idx].item()])
            
        while len(features) < (self.K * 2):
            features.extend([0.0, 0.0])
            
        return features

    def _build_dataset(self, df, desc):
        print(f"\n>> Generating Feature Matrix for {desc} (N={len(df):,})...")
        start_time = time.time()
        
        all_features, labels, ha_list, cv_folds = [], [], [], []
        is_oot = (desc == "Future OOT")
        
        for row in tqdm(df.itertuples(), total=len(df)):
            q_fold = None if is_oot else getattr(row, 'cv_fold', -1)
            feats = self._extract_features(row.memmap_idx, row.target_chembl_id, row.scaffold_hash, q_fold)
            
            all_features.append(feats)
            labels.append(1 if row.set_type == 'Positive' else 0)
            ha_list.append(row.heavy_atoms)
            cv_folds.append(getattr(row, 'cv_fold', -1))
            
        cols = []
        for i in range(1, self.K + 1):
            cols.extend([f"s1_3d_{i}", f"s2_2d_{i}"])
            
        out_df = pd.DataFrame({
            'heavy_atoms': ha_list, 'ha_bin': np.clip(ha_list, 10, 60),
            'label': labels, 'cv_fold': cv_folds
        })
        out_df = pd.concat([out_df, pd.DataFrame(all_features, columns=cols)], axis=1)
        print(f"   ✅ Matrix generated in {time.time()-start_time:.2f}s")
        return out_df

    def execute_cv_and_oot_evaluation(self):
        self.train_df = self._build_dataset(self.meta_df, "Past Training Data")
        
        print(f"\n>> 3. Executing True CV & Training (L2, C={self.C_REG})...")
        feature_cols = [c for c in self.train_df.columns if c.startswith('s1_') or c.startswith('s2_')]
        
        idx_3d = [i for i, c in enumerate(feature_cols) if '3d' in c]
        idx_2d = [i for i, c in enumerate(feature_cols) if '2d' in c]
        
        unique_bins = sorted(self.train_df['ha_bin'].unique())
        unique_folds = np.sort(self.train_df['cv_fold'].unique())
        
        cv_reports = []
        self.production_models = {} 
        final_coefficients = {}
        
        for ha_bin in tqdm(unique_bins):
            subset = self.train_df[self.train_df['ha_bin'] == ha_bin]
            X = subset[feature_cols].values
            y = subset['label'].values
            folds = subset['cv_fold'].values
            
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
                    
                    # Ablation
                    s1_max = X_test[:, idx_3d].max(axis=1)
                    s2_max = X_test[:, idx_2d].max(axis=1)
                    bin_auc_3d.append(roc_auc_score(y_test, s1_max))
                    bin_auc_2d.append(roc_auc_score(y_test, s2_max))
            
            cv_reports.append({
                'ha_bin': ha_bin, 
                'mean_AUC_Combined': np.mean(bin_auc) if bin_auc else 0.0,
                'mean_AUC_3D_Only': np.mean(bin_auc_3d) if bin_auc_3d else 0.0,
                'mean_AUC_2D_Only': np.mean(bin_auc_2d) if bin_auc_2d else 0.0,
                'mean_PRAUC_Combined': np.mean(bin_prauc) if bin_prauc else 0.0
            })
            
            # 배포용 모델 학습
            if len(np.unique(y)) > 1:
                clf_prod = LogisticRegression(penalty='l2', C=self.C_REG, solver='lbfgs', max_iter=1000)
                clf_prod.fit(X, y)
                self.production_models[ha_bin] = clf_prod
                final_coefficients[str(ha_bin)] = [float(clf_prod.intercept_[0])] + [float(c) for c in clf_prod.coef_[0]]
            else:
                final_coefficients[str(ha_bin)] = [-5.0] + [3.0, 7.0] * self.K
                
        pd.DataFrame(cv_reports).to_csv(os.path.join(self.out_dir, f"stp_cv_report_K{self.K}_{self.POLICY}.csv"), index=False)
        with open(os.path.join(self.out_dir, f"stp_coef_K{self.K}_{self.POLICY}.json"), "w") as f:
            json.dump({"K": self.K, "Policy": self.POLICY, "Temporal_Cutoff": self.CUTOFF_YEAR, "coef": final_coefficients}, f, indent=4)
        
        # ---------------------------------------------------------
        # True OOT (Out-of-Time) Evaluation
        # ---------------------------------------------------------
        if not self.future_df.empty:
            self.future_features_df = self._build_dataset(self.future_df, "Future OOT")
            print("\n>> 4. Evaluating Out-Of-Time (OOT) Performance...")
            
            oot_reports = []
            for ha_bin in tqdm(unique_bins):
                if ha_bin not in self.production_models: continue
                
                subset = self.future_features_df[self.future_features_df['ha_bin'] == ha_bin]
                if len(subset) == 0: continue
                
                X_oot = subset[feature_cols].values
                y_oot = subset['label'].values
                
                if len(np.unique(y_oot)) > 1:
                    preds_oot = self.production_models[ha_bin].predict_proba(X_oot)[:, 1]
                    s1_max_oot = X_oot[:, idx_3d].max(axis=1)
                    s2_max_oot = X_oot[:, idx_2d].max(axis=1)
                    
                    oot_reports.append({
                        'ha_bin': ha_bin, 'N_Samples': len(y_oot),
                        'OOT_AUC_Combined': roc_auc_score(y_oot, preds_oot),
                        'OOT_AUC_3D_Only': roc_auc_score(y_oot, s1_max_oot),
                        'OOT_AUC_2D_Only': roc_auc_score(y_oot, s2_max_oot),
                        'OOT_PRAUC': average_precision_score(y_oot, preds_oot)
                    })
                    
            if oot_reports:
                pd.DataFrame(oot_reports).to_csv(os.path.join(self.out_dir, f"stp_OOT_report_K{self.K}_{self.POLICY}.csv"), index=False)
                print("   ✅ OOT Evaluation Completed! Report saved.")

if __name__ == "__main__":
    trainer = STPFinalTrainer(
        "features_store/final_training_meta.parquet", 
        "features_store/fp2_aligned.memmap", 
        "features_store/es5d_db_k1.memmap", 
        k_mode=1, k_policy='paired_sum', c_reg=10.0, cutoff_year=2023
    )
    trainer.execute_cv_and_oot_evaluation()