# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:32:09 2026

@author: KIOM_User
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_paper_figures(parquet_file):
    print(">> Loading Data for Summary...")
    df = pd.read_parquet(parquet_file)
    
    # Positive 데이터(실제 실험 데이터)만 추출하여 통계 생성
    pos_df = df[df['set_type'] == 'Positive']

    # ---------------------------------------------------------
    # 1. NAR 2019 Table 1 재현: 종별(Organism) 통계
    # ---------------------------------------------------------
    print("\n=== [Table 1] Summary of Bioactivity Data (cf. NAR 2019 Table 1) ===")
    summary_data = []
    organisms = ['Homo sapiens', 'Rattus norvegicus', 'Mus musculus']
    
    for org in organisms:
        sub_df = pos_df[pos_df['organism'] == org]
        num_targets = sub_df['target_chembl_id'].nunique()
        num_compounds = sub_df['mol_chembl_id'].nunique()
        num_interactions = len(sub_df) # Positive 1행이 1개의 interaction
        
        summary_data.append({
            'Species': org,
            'Number of targets': num_targets,
            'Number of active compounds': num_compounds,
            'Number of interactions': num_interactions
        })
    
    # All (Total) 계산
    summary_data.append({
        'Species': 'All (Total)',
        'Number of targets': pos_df['target_chembl_id'].nunique(),
        'Number of active compounds': pos_df['mol_chembl_id'].nunique(),
        'Number of interactions': len(pos_df)
    })
    
    summary_table = pd.DataFrame(summary_data)
    print(summary_table.to_string(index=False))

    # ---------------------------------------------------------
    # 2. Bioinformatics 2013 Fig 5 재현: 중원자(Heavy Atoms) 분포
    # ---------------------------------------------------------
    print("\n=== Generating Figure (cf. Bioinformatics 2013 Fig 5A) ===")
    plt.figure(figsize=(10, 6))
    
    # 중복 분자 제거 후 순수 화합물 집합의 분포 계산
    unique_mols = pos_df.drop_duplicates(subset=['mol_chembl_id'])
    
    sns.histplot(unique_mols['heavy_atoms'], bins=range(0, 85, 2), kde=False, color='gray')
    median_val = unique_mols['heavy_atoms'].median()
    
    plt.axvline(median_val, color='black', linestyle='dashed', linewidth=2, label=f'Median ({median_val})')
    plt.title('Distribution of the number of heavy atoms (ChEMBL 36 Actives)')
    plt.xlabel('Number of heavy atoms')
    plt.ylabel('Distribution (Counts)')
    plt.xlim(0, 80)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('heavy_atoms_distribution.png', dpi=300)
    print("  ✅ Saved plot as 'heavy_atoms_distribution.png'")

    # ---------------------------------------------------------
    # 3. 연도별 데이터 분포 (시계열 검증(Temporal Split) 전략 수립용)
    # ---------------------------------------------------------
    print("\n=== Publication Year Distribution (For Temporal Validation) ===")
    year_dist = pos_df['publication_year'].value_counts().sort_index()
    print(year_dist.tail(10)) # 최근 10년 통계 출력

if __name__ == "__main__":
    generate_paper_figures("chembl36_stp_training_set_final_v2.parquet")