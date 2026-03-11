# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:53:43 2026

@author: KIOM_User
"""
# -*- coding: utf-8 -*-
"""
Phase 2.1: STP-Grade OpenBabel FP2 Extraction
Architecture: PyArrow Streaming + uint64[16] Memmap + Parquet Meta + Manifest
"""

import os
import time
import json
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed

# OpenBabel Import (표준 conda-forge 경로 우선)
try:
    from openbabel import pybel
except ImportError:
    print("❌ OpenBabel 파이썬 래퍼를 찾을 수 없습니다.")
    print("   설치 안내: conda install -c conda-forge openbabel")
    raise

# ==========================================
# 1. Worker Function (Return uint64 array for fast popcount)
# ==========================================
def _compute_fp2_uint64(row):
    """
    SMILES를 받아 1024-bit FP2를 생성하고,
    후속 Tanimoto(Popcount) 연산에 최적화된 16개의 uint64 배열로 반환.
    """
    molregno, smiles = row
    
    try:
        mol = pybel.readstring("smi", smiles)
        fp = mol.calcfp("fp2")
        
        # 16개의 64-bit 정수 배열 (총 1024 bits) 생성
        arr_uint64 = np.zeros(16, dtype=np.uint64)
        
        # FP2 bits는 1-indexed (1 ~ 1024)
        for bit in fp.bits:
            if 1 <= bit <= 1024:
                idx = (bit - 1) // 64
                offset = (bit - 1) % 64
                # 해당 위치의 비트를 1로 설정 (Little/Big endian 독립적 논리 연산)
                arr_uint64[idx] |= (np.uint64(1) << np.uint64(offset))
                
        # 무거운 문자열(smiles)은 IPC 오버헤드 방지를 위해 반환하지 않음
        return True, molregno, arr_uint64, ""
    
    except Exception as e:
        return False, molregno, None, str(e).replace('\n', ' ')

# ==========================================
# 2. Main Pipeline Class
# ==========================================
class FP2UltimateExtractor:
    def __init__(self, parquet_file, out_dir="features_store"):
        self.parquet_file = parquet_file
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        # 출력 파일 경로 설정
        self.memmap_path = os.path.join(self.out_dir, "fp2_uint64.memmap")
        self.meta_path = os.path.join(self.out_dir, "fp2_meta.parquet")
        self.error_log = os.path.join(self.out_dir, "fp2_errors.log")
        self.manifest_path = os.path.join(self.out_dir, "manifest.json")
        
        # 코어 수 안전 장치 (최소 1, 최대 48)
        max_workers = 48
        self.n_cores = max(1, min(max_workers, multiprocessing.cpu_count() - 2))

    def run(self):
        print(f">> 1. Streaming & Deduplicating by 'molregno' from {self.parquet_file}...")
        
        # Pandas 메모리 폭발 방지: PyArrow로 청크 단위 스트리밍 후 순수 Python Dict 생성
        unique_mols = {}
        try:
            parquet_file_reader = pq.ParquetFile(self.parquet_file)
            for batch in parquet_file_reader.iter_batches(columns=['molregno', 'canonical_smiles']):
                d = batch.to_pydict()
                # molregno를 Key로 사용하여 자동 중복 제거 (가장 빠르고 메모리 효율적)
                for m, s in zip(d['molregno'], d['canonical_smiles']):
                    if m not in unique_mols:
                        unique_mols[m] = s
        except Exception as e:
            print(f"❌ Parquet 읽기 오류: {e}")
            print("   'molregno' 및 'canonical_smiles' 컬럼이 존재하는지 확인하세요.")
            return

        max_mols = len(unique_mols)
        print(f"   Extracted {max_mols:,} unique molecules.")
        
        tasks = list(unique_mols.items()) # [(molregno, smiles), ...]
        
        print(f"\n>> 2. Initializing uint64 Memmap (Shape: {max_mols} x 16)...")
        # 16개의 uint64 = 128 bytes. Popcount에 완벽히 정렬됨.
        fp_memmap = np.memmap(self.memmap_path, dtype=np.uint64, mode='w+', shape=(max_mols, 16))
        
        print(f"\n>> 3. Extracting & Packing FP2 (Using {self.n_cores} workers)...")
        valid_molregnos = []
        error_count = 0
        
        start_time = time.time()
        
        with open(self.error_log, "w", encoding='utf-8') as f_err:
            # return_as="generator"로 결과를 즉시 소비하여 RAM 점유 방지
            results = Parallel(n_jobs=self.n_cores, return_as="generator")(
                delayed(_compute_fp2_uint64)(row) for row in tasks
            )
            
            write_idx = 0
            for success, molregno, data, err_msg in tqdm(results, total=max_mols):
                if success:
                    fp_memmap[write_idx] = data
                    valid_molregnos.append(molregno)
                    write_idx += 1
                else:
                    error_count += 1
                    # 원본 smiles는 tasks 딕셔너리에서 찾아 로깅
                    f_err.write(f"{molregno}\t{unique_mols[molregno]}\t{err_msg}\n")
        
        fp_memmap.flush()
        elapsed_time = time.time() - start_time
        print(f"   Completed in {elapsed_time:.2f} seconds.")
        print(f"   Success: {write_idx:,} / Failures: {error_count:,}")
        
        # 실패 건수만큼 Truncate
        if error_count > 0:
            print(f">> 4. Truncating memmap to {write_idx} valid rows...")
            final_memmap_path = os.path.join(self.out_dir, "fp2_uint64_final.memmap")
            final_memmap = np.memmap(final_memmap_path, dtype=np.uint64, mode='w+', shape=(write_idx, 16))
            final_memmap[:] = fp_memmap[:write_idx]
            final_memmap.flush()
            
            del fp_memmap, final_memmap
            os.remove(self.memmap_path)
            os.rename(final_memmap_path, self.memmap_path)
        
        print("\n>> 5. Saving Metadata & Manifest...")
        
        # 메타데이터 (zstd 압축 권장 적용)
        valid_smiles_list = [unique_mols[m] for m in valid_molregnos]
        meta_df = pd.DataFrame({
            'memmap_idx': np.arange(write_idx), # Memmap의 row index와 1:1 매칭
            'molregno': valid_molregnos,
            'canonical_smiles': valid_smiles_list
        })
        meta_df.to_parquet(self.meta_path, index=False, compression='zstd')
        
        # Manifest 기록
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_file": self.parquet_file,
            "feature_type": "OpenBabel FP2",
            "data_type": "uint64",
            "shape": [write_idx, 16],
            "total_processed": max_mols,
            "success": write_idx,
            "failed": error_count,
            "bitorder": "little/big independent logic (explicit bit shifting)"
        }
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
            
        print("\n=== Phase 2.1 Completed Perfectly ===")
        print(f"✅ Features: {self.memmap_path}")
        print(f"✅ Metadata: {self.meta_path}")
        print(f"✅ Manifest: {self.manifest_path}")

if __name__ == "__main__":
    INPUT_PARQUET = "chembl36_stp_training_set_final_v2.parquet"
    OUTPUT_DIR = "features_store"
    
    extractor = FP2UltimateExtractor(INPUT_PARQUET, OUTPUT_DIR)
    extractor.run()