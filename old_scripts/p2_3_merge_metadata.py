# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 09:49:19 2026

@author: KIOM_User
"""


import pandas as pd
import os

def merge_master_and_features(master_file, es5d_meta_file, out_dir="features_store"):
    print(">> 1. Loading Master Table (Phase 1)...")
    df_master = pd.read_parquet(master_file)
    
    print("\n>> 2. Loading 3D ES5D Metadata (Phase 2)...")
    df_es5d = pd.read_parquet(es5d_meta_file)
    
    print("\n>> 3. Merging Datasets on 'molregno'...")
    # [수정된 부분] scaffold_smiles와 publication_year(연도)를 반드시 포함하여 가져옵니다.
    cols_to_bring = [
        'molregno', 'mol_chembl_id', 'target_chembl_id', 
        'heavy_atoms', 'set_type', 'pair_id', 'cv_fold', 
        'scaffold_smiles', 'publication_year'
    ]
    
    # 만약 publication_year가 마스터에 없다면 제외하고 가져오는 안전 장치
    cols_to_bring = [c for c in cols_to_bring if c in df_master.columns]
    
    df_merged = pd.merge(
        df_es5d, 
        df_master[cols_to_bring], 
        on='molregno', 
        how='inner'
    )
    
    # memmap_idx 기준으로 정렬하여 텐서 배열과의 1:1 매칭 보장
    df_merged = df_merged.sort_values('memmap_idx').reset_index(drop=True)
    
    # 4. 결과 덮어쓰기 저장
    output_path = os.path.join(out_dir, "final_training_meta.parquet")
    df_merged.to_parquet(output_path, index=False, compression='zstd')
    
    print(f"\n✅ Merge Completed! Saved to: {output_path}")
    print(f"   - 확인된 컬럼들: {list(df_merged.columns)}")

if __name__ == "__main__":
    MASTER_FILE = "chembl36_stp_training_set_final_v2.parquet"
    ES5D_META = "features_store/es5d_meta_db.parquet"
    merge_master_and_features(MASTER_FILE, ES5D_META)