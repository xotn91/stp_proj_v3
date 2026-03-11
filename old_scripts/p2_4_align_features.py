# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 12:38:02 2026

@author: KIOM_User
"""


# -*- coding: utf-8 -*-
"""
Phase 2.4 (Corrected): Unique Tensor Alignment
마스터 테이블(Pair)이 아닌 유니크 분자(es5d_meta) 기준으로 FP2와 ES5D를 1:1 정렬합니다.
"""
import os
import numpy as np
import pandas as pd

def align_unique_fp2_to_es5d():
    print(">> 1. Loading Unique Metadata...")
    # [핵심 수정] 쌍(Pair) 데이터가 아닌 유니크 분자 메타데이터 사용
    df_es5d = pd.read_parquet("features_store/es5d_meta_db.parquet")
    df_fp2 = pd.read_parquet("features_store/fp2_meta.parquet")
    
    df_fp2 = df_fp2.rename(columns={'memmap_idx': 'fp2_idx'})
    
    print(">> 2. Aligning indices by 'molregno'...")
    df_merged = pd.merge(df_es5d, df_fp2[['molregno', 'fp2_idx']], on='molregno', how='inner')
    df_merged = df_merged.sort_values('memmap_idx').reset_index(drop=True)
    
    N_final = len(df_merged)
    N_fp2_original = len(df_fp2)
    
    print(f"   - Target Size (Unique 3D Molecules): {N_final:,} (Should be ~863,811)")
    print(f"   - Original FP2 Size: {N_fp2_original:,}")
    
    print("\n>> 3. Re-building FP2 Memmap...")
    fp2_original = np.memmap("features_store/fp2_uint64.memmap", dtype=np.uint64, mode='r', shape=(N_fp2_original, 16))
    
    # 덮어쓰기 생성
    fp2_aligned = np.memmap("features_store/fp2_aligned.memmap", dtype=np.uint64, mode='w+', shape=(N_final, 16))
    
    original_indices = df_merged['fp2_idx'].values
    fp2_aligned[:] = fp2_original[original_indices]
    
    fp2_aligned.flush()
    print(f"\n✅ Alignment Completed Perfectly!")
    print("   - 이제 Phase 3 코드를 다시 실행하셔도 좋습니다.")

if __name__ == "__main__":
    align_unique_fp2_to_es5d()