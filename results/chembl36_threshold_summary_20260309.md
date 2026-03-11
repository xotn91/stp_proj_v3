# ChEMBL 36 조건별 활성값 집계 (2026-03-09)

## 적용 조건
- Source: ChEMBL 36 (`chembl_36_sqlite/chembl_36.db`, dump와 동일 릴리스)
- Organism: `Homo sapiens`, `Mus musculus`, `Rattus norvegicus`
- Ligand: `compound_properties.heavy_atoms <= 80`
- Target Type: `SINGLE PROTEIN`, `PROTEIN COMPLEX`
- Direct binding: `assays.assay_type = 'B'`
- Activity type: `Ki`, `Kd`, `IC50`, `EC50`
- Activity row 필터: `standard_value IS NOT NULL`, `standard_relation IN ('=','<','<=')`
- 단위 정규화: `M/mM/uM(µM)/nM/pM/fM -> uM`

## 임계값 집계 (activity record 기준)

| metric | total_records | <1 uM | <10 uM | <100 uM | >=100 uM |
|---|---:|---:|---:|---:|---:|
| ALL | 2,005,051 | 1,476,950 | 1,824,258 | 1,969,895 | 35,156 |
| EC50 | 101,514 | 72,248 | 93,148 | 100,212 | 1,302 |
| IC50 | 1,384,122 | 1,004,971 | 1,250,613 | 1,361,592 | 22,530 |
| Kd | 69,464 | 46,566 | 61,679 | 66,606 | 2,858 |
| Ki | 449,951 | 353,165 | 418,818 | 441,485 | 8,466 |

