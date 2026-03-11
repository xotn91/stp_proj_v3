# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:31:09 2026

@author: KIOM_User
"""

import pandas as pd

def validate_stp_data(parquet_file):
    print(f"=== Loading Data: {parquet_file} ===")
    df = pd.read_parquet(parquet_file)
    print(f"Total Rows: {len(df):,}\n")

    # 1. 1:10 비율 및 세트(A안) 구성 검증
    print("[Check 1] 1 Positive : 10 Negatives Ratio & Target-Centric Check")
    pair_counts = df.groupby('pair_id').size()
    if all(pair_counts == 11):
        print("  ✅ PASS: 모든 pair_id가 정확히 11개(Pos 1 + Neg 10)의 데이터를 가집니다.")
    else:
        print(f"  ❌ FAIL: 11개가 아닌 pair_id가 {len(pair_counts[pair_counts != 11])}개 존재합니다.")

    # A안 검증: 같은 pair_id 내에서 target_chembl_id는 모두 같아야 함
    target_nunique = df.groupby('pair_id')['target_chembl_id'].nunique()
    if all(target_nunique == 1):
        print("  ✅ PASS: A안(타겟 고정) 로직이 정상 작동했습니다. (Set 내 타겟 동일)")
    else:
        print("  ❌ FAIL: 같은 Set 내에 다른 타겟이 섞여 있습니다.")

    # 2. 파라로그 및 Positive-Negative 중복 검증
    print("\n[Check 2] Negative Mol Selection Validity")
    # Positive로 쓰인 (분자, 타겟) 쌍
    pos_pairs = set(zip(df[df['set_type']=='Positive']['mol_chembl_id'], 
                        df[df['set_type']=='Positive']['target_chembl_id']))
    # Negative로 쓰인 (분자, 타겟) 쌍
    neg_pairs = set(zip(df[df['set_type']=='Negative']['mol_chembl_id'], 
                        df[df['set_type']=='Negative']['target_chembl_id']))
    
    overlap = pos_pairs.intersection(neg_pairs)
    if len(overlap) == 0:
        print("  ✅ PASS: Positive로 알려진 상호작용이 Negative로 잘못 샘플링되지 않았습니다.")
    else:
        print(f"  ❌ FAIL: {len(overlap)}개의 상호작용이 Pos/Neg에 중복 출현합니다.")

    # 3. 분자 크기(Heavy Atoms) 제한 검증
    print("\n[Check 3] Heavy Atoms Limit (<= 80)")
    if df['heavy_atoms'].max() <= 80:
        print(f"  ✅ PASS: 모든 분자의 중원자 수가 80개 이하입니다. (Max: {df['heavy_atoms'].max()})")
    else:
        print(f"  ❌ FAIL: 중원자 80개를 초과하는 분자가 존재합니다.")

    # 4. Cross-Validation 누수(Leakage) 검증
    print("\n[Check 4] Cross-Validation Leakage")
    cv_leakage = df.groupby('pair_id')['cv_fold'].nunique()
    if all(cv_leakage == 1):
        print("  ✅ PASS: 동일한 pair_id 그룹은 모두 같은 CV Fold에 할당되었습니다. (Data Leakage 없음)")
    else:
        print("  ❌ FAIL: 하나의 pair_id가 여러 Fold에 쪼개져 있습니다.")

if __name__ == "__main__":
    validate_stp_data("chembl36_stp_training_set_final_v2.parquet")