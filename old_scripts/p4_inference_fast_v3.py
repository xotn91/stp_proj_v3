# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 15:19:17 2026

@author: KIOM_User
"""


# -*- coding: utf-8 -*-
"""
Phase 4 v3.2: TRUE DATA-PARALLEL OOT INFERENCE ENGINE (FINAL PATCHED)
- [ADD] Query Molecule ChEMBL ID mapping added to the final CSV output.
- [PATCH] Temporal Split synchronized with Phase 3 (> cutoff_year).
- [PATCH] GPU-Resident Weights: Eliminates CPU->GPU PCIe transfer bottleneck.
- [SYNC] Fully synchronized with K=5 Summary Fusion and Dual-GPU Data Parallelism.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

class STPUltraFastRecommender:
    def __init__(self, meta_file, fp2_memmap, es5d_memmap, coef_json, target_dict_csv=None, cutoff_year=2023):
        self.meta_file = meta_file
        self.fp2_path = fp2_memmap
        self.es5d_path = es5d_memmap
        self.coef_file = coef_json
        self.target_dict_csv = target_dict_csv
        self.CUTOFF_YEAR = cutoff_year
        self.out_dir = os.path.dirname(coef_json)
        
        self.BATCH_SIZE = 512       
        self.CHUNK_M = 1024         
        
        self._load_brain()
        self._initialize_gpu_engine()
        self._load_target_metadata()
        
    def _load_brain(self):
        print(f"\n>> 1. Loading Trained Brain: {os.path.basename(self.coef_file)}")
        with open(self.coef_file, 'r') as f:
            data = json.load(f)
        self.K = data.get("K", 5)
        self.POLICY = data.get("Policy", "paired_sum")
        
        # 가중치 텐서 구조: [HA_Bin, 7] -> 0: Intercept, 1~3: 3D (Max,Mean,Std), 4~6: 2D (Max,Mean,Std)
        self.weights = torch.zeros((65, 7), dtype=torch.float32)
        
        for k, v in data['coef'].items():
            ha_bin = int(k)
            self.weights[ha_bin] = torch.tensor(v, dtype=torch.float32)
            
        # 가중치를 CPU에 두지 않고 미리 두 GPU의 VRAM에 상주(Resident)시킴
        self.weights_0 = self.weights.to('cuda:0', non_blocking=True)
        self.weights_1 = self.weights.to('cuda:1', non_blocking=True)
            
        print(f"   - Loaded & GPU-Resident weights for Fusion Features (K={self.K}).")

    def _initialize_gpu_engine(self):
        print(f"\n>> 2. Initializing TRUE DATA-PARALLEL Inference Engine...")
        
        full_df = pd.read_parquet(self.meta_file)
        full_df['scaffold_hash'] = pd.util.hash_pandas_object(full_df['scaffold_smiles'], index=False).astype(np.int64)
        
        self.FULL_N = os.path.getsize(self.fp2_path) // 128
        self.NCONF = os.path.getsize(self.es5d_path) // (self.FULL_N * 72)
        
        print("   - [RAM] Loading Memmap into System RAM...")
        db_2d_ram = np.array(np.memmap(self.fp2_path, dtype=np.uint64, mode='r', shape=(self.FULL_N, 16)))
        db_3d_ram = np.array(np.memmap(self.es5d_path, dtype=np.float32, mode='r', shape=(self.FULL_N, self.NCONF, 18)))
        
        print("   - [GPU] Cloning full database to BOTH cuda:0 and cuda:1...")
        self.db2d_0 = torch.from_numpy(np.unpackbits(db_2d_ram.view(np.uint8), axis=1)).to(device='cuda:0', dtype=torch.float16)
        self.db2d_ones_0 = self.db2d_0.sum(dim=1).float()
        self.db3d_0 = torch.from_numpy(db_3d_ram).to(device='cuda:0')
        
        self.db2d_1 = self.db2d_0.to(device='cuda:1', non_blocking=True)
        self.db2d_ones_1 = self.db2d_ones_0.to(device='cuda:1', non_blocking=True)
        self.db3d_1 = self.db3d_0.to(device='cuda:1', non_blocking=True)
        
        mem_idx = full_df["memmap_idx"].values.astype(np.int64)
        scaf_arr = np.full(self.FULL_N, -1, dtype=np.int64)
        scaf_arr[mem_idx] = full_df["scaffold_hash"].values.astype(np.int64)
        
        self.gpu_scaf_0 = torch.tensor(scaf_arr, device="cuda:0", dtype=torch.int64)
        self.gpu_scaf_1 = self.gpu_scaf_0.to(device="cuda:1", non_blocking=True)
        
        # OOT 평가셋 기준을 Phase 3와 완벽히 동기화 (> cutoff_year)
        if 'publication_year' in full_df.columns:
            full_df['publication_year'] = pd.to_numeric(full_df['publication_year'], errors='coerce').fillna(-1).astype(int)
            self.future_df = full_df[full_df['publication_year'] > self.CUTOFF_YEAR].copy()
            past_df = full_df[full_df['publication_year'] <= self.CUTOFF_YEAR].copy()
            print(f"   - Temporal Split: References(<= {self.CUTOFF_YEAR}) vs Test(> {self.CUTOFF_YEAR})")
        else:
            self.future_df = pd.DataFrame()
            past_df = full_df.copy()

        past_positives = past_df[past_df['set_type'] == 'Positive']
        tmap = past_positives.groupby('target_chembl_id')['memmap_idx'].apply(list)
        
        self.target_list = list(tmap.keys())
        self.num_targets = len(self.target_list)
        
        self.target_actives_0 = [torch.tensor(tmap[tid], device='cuda:0', dtype=torch.int64) for tid in self.target_list]
        self.target_actives_1 = [torch.tensor(tmap[tid], device='cuda:1', dtype=torch.int64) for tid in self.target_list]
        torch.cuda.synchronize()
        print(f"   - Cached Active References for {self.num_targets} Targets.")

    def _load_target_metadata(self):
        self.t_meta = {}
        if self.target_dict_csv and os.path.exists(self.target_dict_csv):
            tdf = pd.read_csv(self.target_dict_csv)
            for _, row in tdf.iterrows():
                self.t_meta[row.get('target_chembl_id', '')] = {
                    'target_name': row.get('pref_name', 'Unknown'),
                    'uniprot_id': row.get('uniprot_id', 'Unknown'),
                    'common_name': row.get('common_name', 'Unknown')
                }

    def _run_gpu_half_inference(self, q_idx, q_scaf, q_ha, device_id):
        B = len(q_idx)
        if B == 0: return None
        
        device = f"cuda:{device_id}"
        ha_bins = torch.clamp(q_ha, 10, 60).long()
        
        if device_id == 0:
            db2d, db2d_ones, db3d, gpu_scaf = self.db2d_0, self.db2d_ones_0, self.db3d_0, self.gpu_scaf_0
            actives_list = self.target_actives_0
            local_weights = self.weights_0[ha_bins] # (B, 7)
        else:
            db2d, db2d_ones, db3d, gpu_scaf = self.db2d_1, self.db2d_ones_1, self.db3d_1, self.gpu_scaf_1
            actives_list = self.target_actives_1
            local_weights = self.weights_1[ha_bins] # (B, 7)

        w_int = local_weights[:, 0]
        w_feats = local_weights[:, 1:] # (B, 6)
        
        q_2d = db2d[q_idx]              
        q_ones = db2d_ones[q_idx].unsqueeze(1)   
        q_3d = torch.nan_to_num(db3d[q_idx], nan=1e4).unsqueeze(1)
        
        probs = torch.zeros((B, self.num_targets), device=device, dtype=torch.float32)
        feats = torch.zeros((B, 6), device=device, dtype=torch.float32)
        
        for t_idx, actives in enumerate(actives_list):
            M = len(actives)
            if M == 0: continue
            
            k_act = min(self.K, M)
            if k_act <= 0: continue
            
            a_scaf_exp = gpu_scaf[actives].unsqueeze(0)
            mask = (q_idx.unsqueeze(1) != actives.unsqueeze(0)) & \
                   (q_scaf.unsqueeze(1) != a_scaf_exp) & \
                   (a_scaf_exp != -1)
                   
            a_2d = db2d[actives]
            intersection = torch.matmul(q_2d, a_2d.T).float()
            union = q_ones + db2d_ones[actives].unsqueeze(0) - intersection
            sim_2d = torch.where(union > 0, intersection / union, torch.zeros_like(union))
            sim_2d.masked_fill_(~mask, -1e9)
            
            a_3d = torch.nan_to_num(db3d[actives], nan=-1e4) 
            sim_3d = torch.zeros((B, M), device=device, dtype=torch.float32)
            
            for m in range(0, M, self.CHUNK_M):
                a_chunk = a_3d[m:m+self.CHUNK_M].unsqueeze(0) 
                dist = torch.cdist(q_3d, a_chunk, p=1.0)      
                sim_3d[:, m:m+self.CHUNK_M] = (1.0 / (1.0 + (dist / 18.0))).amax(dim=(2, 3))
            sim_3d.masked_fill_(~mask, -1e9)
            
            top_2d = torch.topk(sim_2d, k_act, dim=1).values.clamp(min=0.0)
            top_3d = torch.topk(sim_3d, k_act, dim=1).values.clamp(min=0.0)
            
            feats[:, 0] = top_3d.max(dim=1).values
            feats[:, 1] = top_3d.mean(dim=1)
            feats[:, 2] = torch.nan_to_num(top_3d.std(dim=1, unbiased=False), nan=0.0)
            feats[:, 3] = top_2d.max(dim=1).values
            feats[:, 4] = top_2d.mean(dim=1)
            feats[:, 5] = torch.nan_to_num(top_2d.std(dim=1, unbiased=False), nan=0.0)
            
            logit = w_int + torch.sum(w_feats * feats, dim=1)
            probs[:, t_idx] = torch.sigmoid(logit)
            
        return probs 

    def evaluate_oot_top_n(self):
        if self.future_df.empty:
            print(">> No future data found for OOT evaluation.")
            return
            
        print(f"\n>> 3. Evaluating Top-N Hit Rate on OOT Data (Data-Parallel Engine)...")
        
        pos_df = self.future_df[self.future_df['set_type'] == 'Positive']
        unique_queries = pos_df.drop_duplicates(subset=['memmap_idx']).copy()
        
        # [추가된 로직] memmap_idx와 화합물 ChEMBL ID 매핑
        mol_id_map = {}
        if 'mol_chembl_id' in unique_queries.columns:
            mol_id_map = unique_queries.set_index('memmap_idx')['mol_chembl_id'].to_dict()
        
        q_indices = torch.tensor(unique_queries['memmap_idx'].values, dtype=torch.int64)
        q_scafs = torch.tensor(unique_queries['scaffold_hash'].values, dtype=torch.int64)
        q_has = torch.tensor(unique_queries['heavy_atoms'].values, dtype=torch.int64)
        
        total_queries = len(q_indices)
        print(f"   - Total Unique OOT Molecules to test: {total_queries:,}")
        
        true_targets_dict = pos_df.groupby('memmap_idx')['target_chembl_id'].apply(set).to_dict()
        top1, top5, top10, top15 = 0, 0, 0, 0
        csv_results = []
        
        with tqdm(total=total_queries, desc="Dual-GPU Screening") as pbar:
            for i in range(0, total_queries, self.BATCH_SIZE):
                b_idx = q_indices[i:i+self.BATCH_SIZE]
                b_scaf = q_scafs[i:i+self.BATCH_SIZE]
                b_ha = q_has[i:i+self.BATCH_SIZE]
                
                mid = len(b_idx) // 2
                probs_0 = self._run_gpu_half_inference(b_idx[:mid].to("cuda:0"), b_scaf[:mid].to("cuda:0"), b_ha[:mid].to("cuda:0"), 0)
                probs_1 = self._run_gpu_half_inference(b_idx[mid:].to("cuda:1"), b_scaf[mid:].to("cuda:1"), b_ha[mid:].to("cuda:1"), 1)
                
                batch_probs = torch.zeros((len(b_idx), self.num_targets), dtype=torch.float32)
                if probs_0 is not None: batch_probs[:mid] = probs_0.cpu()
                if probs_1 is not None: batch_probs[mid:] = probs_1.cpu()
                
                topk_res = torch.topk(batch_probs, 15, dim=1)
                top15_pred_indices = topk_res.indices.numpy()
                top15_pred_probs = topk_res.values.numpy() 
                
                for j, q_memmap_tensor in enumerate(b_idx):
                    q_memmap = q_memmap_tensor.item()
                    true_targets = true_targets_dict.get(q_memmap, set())
                    pred_targets = [self.target_list[idx] for idx in top15_pred_indices[j]]
                    
                    # 쿼리 분자의 ChEMBL ID 가져오기
                    q_chembl_id = mol_id_map.get(q_memmap, "Unknown")
                    
                    if len(true_targets.intersection(pred_targets[:1])) > 0: top1 += 1
                    if len(true_targets.intersection(pred_targets[:5])) > 0: top5 += 1
                    if len(true_targets.intersection(pred_targets[:10])) > 0: top10 += 1
                    if len(true_targets.intersection(pred_targets[:15])) > 0: top15 += 1
                    
                    for rank in range(15):
                        t_id = pred_targets[rank]
                        t_info = self.t_meta.get(t_id, {})
                        csv_results.append({
                            'query_memmap_idx': q_memmap,
                            'query_mol_chembl_id': q_chembl_id, # ChEMBL ID 컬럼 추가
                            'rank': rank + 1,
                            'target_chembl_id': t_id,
                            'target_name': t_info.get('target_name', 'Unknown'),
                            'probability': round(float(top15_pred_probs[j, rank]), 4),
                            'is_true_target': t_id in true_targets
                        })
                    
                pbar.update(len(b_idx))
                
        print("\n==================================================")
        print(" 🎯 FINAL OOT PREDICTIVE PERFORMANCE (TOP-N HIT RATE)")
        print("==================================================")
        if total_queries > 0:
            print(f" - Top-1  Hit Rate: {top1 / total_queries * 100:.2f}%")
            print(f" - Top-5  Hit Rate: {top5 / total_queries * 100:.2f}%")
            print(f" - Top-10 Hit Rate: {top10 / total_queries * 100:.2f}%")
            print(f" - Top-15 Hit Rate: {top15 / total_queries * 100:.2f}%")
        print("==================================================")

        if csv_results:
            out_file = os.path.join(self.out_dir, f"stp_OOT_predictions_Top15_FusionK{self.K}.csv")
            pd.DataFrame(csv_results).to_csv(out_file, index=False)
            print(f"\n   ✅ Predictions saved successfully to: {out_file}")

if __name__ == "__main__":
    recommender = STPUltraFastRecommender(
        meta_file="features_store/final_training_meta.parquet", 
        fp2_memmap="features_store/fp2_aligned.memmap", 
        es5d_memmap="features_store/es5d_db_k20.memmap", 
        coef_json="features_store/stp_coef_FusionK5_paired_sum.json", 
        target_dict_csv=None,  
        cutoff_year=2023 
    )
    
    recommender.evaluate_oot_top_n()