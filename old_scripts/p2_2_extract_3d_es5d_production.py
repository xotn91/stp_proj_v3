# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 11:08:35 2026

@author: KIOM_User
"""

# -*- coding: utf-8 -*-
"""
Phase 2.2: STP-Grade ES5D Extraction (Production-Ready)
Features: K=1 Conformer Policy, Memmap Truncation, pH 7.4 QC Metrics, Error Summary
"""

import os
import time
import json
import multiprocessing
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit import RDLogger  # <-- 여기로 분리하여 수정합니다.


# RDKit C++ 에러 로그 억제 (파이썬 레벨에서 예외 처리)
RDLogger.DisableLog('rdApp.*')

# [설계 정책] DB 저장용 대표 Conformer 개수 (STP 논문 원본의 20x20 비교 재현)
TOP_K_CONFS = 20 

# ==========================================
# 1. ES5D Calculation Engine
# ==========================================
def compute_es5d_vector(coords_5d):
    """
    N x 5 Coordinates -> 18D Moment Vector
    """
    N = coords_5d.shape[0]
    if N < 4: raise ValueError("Atoms too few")

    c1 = np.mean(coords_5d, axis=0)
    dists_to_c1 = np.linalg.norm(coords_5d - c1, axis=1)
    c2 = coords_5d[np.argmax(dists_to_c1)]
    dists_to_c2 = np.linalg.norm(coords_5d - c2, axis=1)
    c3 = coords_5d[np.argmax(dists_to_c2)]
    
    # 카이랄 중심점 외적: 3D Norm 기반 스케일링 보장
    v_a, v_b = c2 - c1, c3 - c1
    v_aS, v_bS = v_a[:3], v_b[:3]
    cross_prod = np.cross(v_aS, v_bS)
    norm_cross = np.linalg.norm(cross_prod)
    
    v_c = (np.linalg.norm(v_aS) / (2.0 * norm_cross)) * cross_prod if norm_cross > 1e-6 else np.zeros(3)
        
    q_max, q_min = np.max(coords_5d[:, 3]), np.min(coords_5d[:, 3])
    c4 = np.copy(c1); c4[:3] = c1[:3] + v_c; c4[3] = q_max
    c5 = np.copy(c1); c5[:3] = c1[:3] + v_c; c5[3] = q_min
    
    # c6 오프셋: 현재 5D 거리(||c2 - c1||) 사용 명시
    c6 = np.copy(c1)
    c6[4] = c1[4] + np.linalg.norm(v_a) 
    
    centroids = [c1, c2, c3, c4, c5, c6]
    
    moments = np.zeros(18, dtype=np.float32)
    for i, c in enumerate(centroids):
        dists = np.linalg.norm(coords_5d - c, axis=1)
        m1 = np.mean(dists)
        m2 = np.std(dists, ddof=0)
        m3_raw = np.mean((dists - m1)**3)
        m3 = np.cbrt(m3_raw) if m3_raw >= 0 else -np.cbrt(-m3_raw)
        moments[i*3 : i*3+3] = [m1, m2, m3]
        
    return moments

# ==========================================
# 2. Worker Function (Quality Gates & Extraction)
# ==========================================
def _process_3d_es5d_production(row):
    molregno, raw_smiles = row
    
    # QC 지표 초기화
    qc = {
        'raw_parse_ok': False, 'raw_charge': np.nan,
        'ph74_smi': None, 'ph74_parse_ok': False, 'ph74_charge': np.nan,
        'is_changed': False, 'q_diff': np.nan, 'converged_confs': 0
    }
    
    try:
        # [QC 1] Raw SMILES RDKit Parsing
        mol_raw = Chem.MolFromSmiles(raw_smiles)
        if mol_raw:
            qc['raw_parse_ok'] = True
            qc['raw_charge'] = Chem.GetFormalCharge(mol_raw)
        else:
            return False, molregno, raw_smiles, None, qc, "Raw SMILES Parse Failed"

        # [QC 2] OpenBabel pH 7.4 Correction (Heuristic)
        ob_mol = pybel.readstring("smi", raw_smiles)
        ob_mol.OBMol.CorrectForPH(7.4)
        ph74_smiles = ob_mol.write("can").split()[0]
        qc['ph74_smi'] = ph74_smiles
        qc['is_changed'] = bool(raw_smiles != ph74_smiles)

        # [QC 3] pH Adjusted SMILES RDKit Parsing
        mol_ph74 = Chem.MolFromSmiles(ph74_smiles)
        if not mol_ph74: return False, molregno, raw_smiles, None, qc, "pH74 SMILES Parse Failed"
        
        qc['ph74_parse_ok'] = True
        qc['ph74_charge'] = Chem.GetFormalCharge(mol_ph74)
        qc['q_diff'] = qc['ph74_charge'] - qc['raw_charge']

# [STEP 4] 3D Conformer Generation (Generate 20)
        mol = Chem.AddHs(mol_ph74)
        
        # 파라미터 객체를 먼저 생성하고 시드(Seed)를 내부에 할당합니다.
        embed_params = AllChem.ETKDGv3()
        embed_params.randomSeed = 42
        
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=20, params=embed_params)
        if not cids: return False, molregno, raw_smiles, None, qc, "3D Embedding Failed"


        # [STEP 5] MMFF94 Optimization & Filtering
        opt_results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500, mmffVariant='MMFF94')
        valid_confs = [(cids[i], res[1]) for i, res in enumerate(opt_results) if res[0] == 0]
        if not valid_confs: return False, molregno, raw_smiles, None, qc, "MMFF Convergence Failed"
        
        valid_confs.sort(key=lambda x: x[1]) # 에너지가 낮은 순 정렬
        qc['converged_confs'] = len(valid_confs)
        
        # 대표 Conformer만 선택 (TOP_K_CONFS)
        selected_confs = valid_confs[:TOP_K_CONFS]
        
        # [STEP 6] Extract Properties & ES5D
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
        if mmff_props is None: return False, molregno, raw_smiles, None, qc, "MMFF Params Missing"
        
        charges = np.array([mmff_props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())])
        logp_contribs = np.array([contrib[0] for contrib in Crippen._GetAtomContribs(mol)])
        
        es5d_matrix = np.full((TOP_K_CONFS, 18), np.nan, dtype=np.float32)
        for arr_idx, (cid, _) in enumerate(selected_confs):
            coords_5d = np.zeros((mol.GetNumAtoms(), 5), dtype=np.float32)
            coords_5d[:, :3] = mol.GetConformer(cid).GetPositions()
            coords_5d[:, 3] = charges * 25.0
            coords_5d[:, 4] = logp_contribs * 5.0
            es5d_matrix[arr_idx] = compute_es5d_vector(coords_5d)
            
        return True, molregno, raw_smiles, es5d_matrix, qc, ""
        
    except Exception as e:
        return False, molregno, raw_smiles, None, qc, f"Unexpected: {str(e).replace(chr(10), ' ')}"

# ==========================================
# 3. Main Pipeline Class
# ==========================================
class ES5DProductionExtractor:
    def __init__(self, input_meta, out_dir="features_store"):
        self.input_meta = input_meta
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.memmap_path = os.path.join(out_dir, "es5d_db_k1.memmap")
        self.meta_path = os.path.join(out_dir, "es5d_meta_db.parquet")
        self.error_log = os.path.join(out_dir, "es5d_db_errors.log")
        self.manifest_path = os.path.join(out_dir, "es5d_db_manifest.json")
        
    def run(self):
        df_meta = pd.read_parquet(self.input_meta)
        tasks = zip(df_meta['molregno'].to_numpy(), df_meta['canonical_smiles'].to_numpy())
        max_mols = len(df_meta)
        
        print(f">> Initializing Memmap (N={max_mols}, K={TOP_K_CONFS}, 18D)...")
        es5d_memmap = np.memmap(self.memmap_path, dtype=np.float32, mode='w+', shape=(max_mols, TOP_K_CONFS, 18))
        
        n_cores = max(1, min(48, multiprocessing.cpu_count() - 2))
        
        valid_rows = []
        error_counter = Counter()
        start_time = time.time()
        
        with open(self.error_log, "w", encoding='utf-8') as f_err:
            f_err.write("molregno\traw_smiles\terror_reason\n")
            
            results = Parallel(n_jobs=n_cores, return_as="generator")(
                delayed(_process_3d_es5d_production)(row) for row in tasks
            )
            
            w_idx = 0
            for success, molregno, raw_smi, data, qc, err in tqdm(results, total=max_mols):
                if success:
                    es5d_memmap[w_idx] = data
                    valid_rows.append({
                        'memmap_idx': w_idx, 'molregno': molregno, 'raw_smi': raw_smi, 
                        'ph74_smi': qc['ph74_smi'], 'is_changed': qc['is_changed'], 
                        'q_diff': qc['q_diff'], 'converged_confs': qc['converged_confs']
                    })
                    w_idx += 1
                else:
                    error_counter[err] += 1
                    f_err.write(f"{molregno}\t{raw_smi}\t{err}\n")
        
        es5d_memmap.flush()
        
        # [CRITICAL] Truncate Memmap to valid rows
        if w_idx < max_mols:
            print(f">> Truncating Memmap to {w_idx} valid rows (safely removing {max_mols - w_idx} junk rows)...")
            final_mm_path = self.memmap_path + ".tmp"
            final_mm = np.memmap(final_mm_path, dtype=np.float32, mode='w+', shape=(w_idx, TOP_K_CONFS, 18))
            final_mm[:] = es5d_memmap[:w_idx]
            final_mm.flush()
            del es5d_memmap, final_mm
            os.remove(self.memmap_path)
            os.rename(final_mm_path, self.memmap_path)
            
        # Metadata
        meta_df = pd.DataFrame(valid_rows)
        meta_df.to_parquet(self.meta_path, index=False, compression='zstd')
        
        # Manifest
        manifest = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": "ElectroShape 5D (ES5D)",
            "shape": [w_idx, TOP_K_CONFS, 18],
            "success_rate": f"{w_idx}/{max_mols} ({w_idx/max_mols*100:.2f}%)",
            "c6_offset_metric": "5D distance (||c2-c1|| in 5D space)",
            "v_c_offset_metric": "3D distance (||c2-c1|| in 3D space)",
            "conformer_policy": f"Store TOP {TOP_K_CONFS} by MMFF94 energy",
            "ph_engine": "OpenBabel CorrectForPH(7.4) (heuristic)"
        }
        with open(self.manifest_path, "w") as f: json.dump(manifest, f, indent=4)
        
        # [QC Summary Report]
        print("\n" + "="*40)
        print("📊 QC SUMMARY REPORT")
        print("="*40)
        print(f"Total Processed: {max_mols:,}")
        print(f"Success:         {w_idx:,}")
        print(f"Failed:          {max_mols - w_idx:,}")
        print("\n[Failure Reasons]")
        for reason, count in error_counter.most_common():
            print(f" - {reason}: {count:,} ({count/max_mols*100:.2f}%)")
            
        if valid_rows:
            changed_pct = (meta_df['is_changed'].sum() / w_idx) * 100
            print(f"\n[pH 7.4 Correction Impact]")
            print(f" - Molecules structure altered by pH correction: {changed_pct:.2f}%")
        print("="*40 + "\n")

if __name__ == "__main__":
    extractor = ES5DProductionExtractor("features_store/fp2_meta.parquet")
    extractor.run()