# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 15:32:08 2026

@author: KIOM_User
"""

# -*- coding: utf-8 -*-
"""
Phase 1.5: Target Homology Mapping for Cross-Species Prediction
이미 수행된 BLAST 결과와 ChEMBL DB를 연동하여 상동성(Ortholog/Paralog) 참조용 JSON을 생성합니다.
"""

import pandas as pd
import json
from sqlalchemy import create_engine
import os

def create_homology_map(db_url, blast_file="blast_work/blast_results.xml", output_file="target_homology_map.json"):
    print(">> 1. ChEMBL Database에서 Target 메타데이터 로드 중...")
    engine = create_engine(db_url)
    
    # 인간, 마우스, 랫드의 타겟 정보만 추출
    query = """
    SELECT chembl_id, pref_name, organism 
    FROM target_dictionary 
    WHERE organism IN ('Homo sapiens', 'Mus musculus', 'Rattus norvegicus')
      AND target_type IN ('SINGLE PROTEIN', 'PROTEIN COMPLEX');
    """
    target_meta = pd.read_sql(query, engine).set_index('chembl_id').to_dict('index')
    print(f"   로드된 타겟 수: {len(target_meta):,} 개")

    print("\n>> 2. BLAST 결과 파싱 및 상동성 네트워크(Homology Network) 구축 중...")
    if not os.path.exists(blast_file):
        raise FileNotFoundError(f"BLAST 결과 파일이 없습니다: {blast_file}")
        
    # Phase 1에서 만든 BLAST 결과 읽기
    blast_df = pd.read_csv(blast_file, sep='\t', header=None, 
                           names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 
                                  'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])
    
    # 매우 강력한 상동성(E-value <= 1e-10) 및 자기 자신 제외
    homology_df = blast_df[(blast_df['evalue'] <= 1e-10) & (blast_df['qseqid'] != blast_df['sseqid'])]
    
    homology_map = {}
    
    for _, row in homology_df.iterrows():
        source_id = row['qseqid']
        homolog_id = row['sseqid']
        
        # 메타데이터에 존재하는 타겟들만 처리
        if source_id in target_meta and homolog_id in target_meta:
            if source_id not in homology_map:
                homology_map[source_id] = []
                
            homology_map[source_id].append({
                "homolog_chembl_id": homolog_id,
                "pref_name": target_meta[homolog_id]['pref_name'],
                "organism": target_meta[homolog_id]['organism'],
                "evalue": row['evalue'],
                "pident": row['pident'] # 서열 일치도(%)
            })

    print(f"   상동성 맵핑이 완료된 타겟 수: {len(homology_map):,} 개")
    
    print(f"\n>> 3. JSON 파일로 저장 중: {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(homology_map, f, indent=4, ensure_ascii=False)
        
    print("✅ 성공적으로 생성되었습니다. 이후 Phase 4(예측 엔진)에서 이 파일을 로드하여 사용합니다.")

if __name__ == "__main__":
    # 데이터베이스 접속 정보 (선생님의 환경에 맞게 수정)
    DB_URL = "postgresql://postgres:99pqpeqt@localhost:5432/chembl_36"
    create_homology_map(DB_URL)