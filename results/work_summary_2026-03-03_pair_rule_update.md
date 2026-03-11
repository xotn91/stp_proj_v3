# 작업 요약 (2026-03-03)

## 1) 재검증 수행
- P1 검증 재실행:
  - 스크립트: `script/p1_2_validate.py`
  - 결과: `results/p1_validation_report.json`
  - 주요 항목: `pair_not_11_count=0`, `cv_leakage_count=0`, threshold 위반 0건
- P3 subset 품질 검증 재실행:
  - 스크립트: `script/p3_0_validate_subset_quality.py`
  - 결과 폴더: `features_store/p3_0_subset_validation_20260303_133450`

## 2) before_assay_id 데이터 샘플/분석
- 원본: `features_store/chembl36_stp_training_set.before_assay_id.parquet`
- 샘플 5000행 CSV 저장:
  - `results/before_assay_id_sample_5000.csv`
- 양성/PAIR 분석 결과:
  - `results/before_assay_id_positive_pair_summary.json`
  - `results/before_assay_id_positive_pair_counts.csv`
- 핵심 수치:
  - 양성 총 행: 1,365,929
  - 양성 pair_id 개수: 1,365,929
  - 양성 기준 pair_id당 행수: 전부 1
  - 전체 기준(해당 pair_id의 전체 행수): 전부 11

## 3) pair_id 1:10 강제 규칙 반영 코드 수정
- 수정 파일:
  - `script/p3_0_build_subset_meta.py`
  - `script/p3_0_validate_subset_quality.py`
  - `script/p3_1_0_run_mask_grid.py`
  - `script/p3_1_1_K1_paired_trainer_fast.py`
  - `script/p3_1_2_K1_unpaired_trainer_fast.py`

### 반영 내용
- subset 생성을 row 단위에서 pair_id 단위 샘플링으로 변경
  - positive anchor(`set_type=Positive`) 기준 pair_id를 층화 샘플링 후, 해당 pair의 11행 전체 포함
- 학습/배치 실행 전 pair 규칙 강제 검증 추가
  - pair_size=11
  - positive=1
  - negative=10
  - pair 내 target 일관성
  - pair 내 cv_fold 일관성
- subset 검증 리포트에 pair 무결성 지표 추가

## 4) 20% subset 재생성 (수정 로직 적용)
- 실행 스크립트: `script/p3_0_build_subset_meta.py`
- 실행 시각: 2026-03-03
- 생성 경로:
  - `features_store/p3_0_subset_r20_s20260301_y2023_20260303_135710/final_training_meta_subset.parquet`
- 로그:
  - `features_store/p3_0_subset_r20_s20260301_y2023_20260303_135710/subset_build_log.txt`

### 재생성 결과
- `rows_total=14,695,681`
- `rows_subset=4,564,043`
- `pairs_total=1,335,971`
- `pairs_subset=414,913`
- `pair_not_11_count_subset=0`
- `pair_pos_not_1_count_subset=0`
- `pair_neg_not_10_count_subset=0`

## 5) P3 문서 규칙 명시
- 문서 업데이트:
  - `p3.md`
- 추가된 핵심 규칙:
  - 학습/샘플링/검증 전 과정 pair_id 기준 관리
  - pair_id별 양성 1 + 음성 10(총 11) 유지
  - subset 생성 시 pair_id 단위 샘플링 강제
  - 배치/학습 전 pair 규칙 사전 검증 필수
