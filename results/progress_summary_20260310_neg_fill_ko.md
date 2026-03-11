# STP v3 진행 요약 (2026-03-10)

## 1) 운영 정책(현재 적용)
- Positive: `activity_uM <= 1`
- Negative 보강: `activity_uM >= 100` 우선
- 부족 시 fallback: 기존 `activity_uM = NaN` 음성 사용
- 목표: pair당 음성 10개(`1:10`)까지 채우되,
- 10개 미달 pair는 **drop하지 않고 유지**
- 중복 방지: 동일 `pair_id` 내 동일 `molregno` 음성 중복 금지

## 2) 기준 데이터 및 1차 보강
- 기준 meta:
  - `features_store/p3_ready_pos_le1uM_neg7plus_20260309/final_training_meta.parquet`
- 1차 운영모드 보강 결과:
  - 출력 폴더: `features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/`
  - `added_rows`: 230,117
  - `pairs_neg_eq10`: 1,079,383
  - `pairs_neg_lt10`: 272
  - `dup_negative_within_pair`: 0
- 근거 파일:
  - `augment_manifest.json`
  - `delta_added_negatives.parquet`
  - `shortage_pairs_to_fill10.csv`
  - `shortage_targets_unique.csv`

## 3) DMP(원천 dump) 기반 부족분 보강
- 대상 dump:
  - `D:\STP_Projt\Chembl raw data\chembl_36\chembl_36_postgresql\chembl_36_postgresql.dmp`
- 추출 조건:
  - assay_type `B`
  - organism: human/mouse/rat
  - target_type: `SINGLE PROTEIN`, `PROTEIN COMPLEX`
  - activity type: `Ki/Kd/IC50/EC50`
  - `activity_uM >= 100`
  - heavy atoms `<=80` 스크리닝
- 결과:
  - DMP raw candidates: 303
  - heavy_atoms<=80: 270
  - 즉시 추가 가능(기존 ES5D 보유): 20
  - 신규 피처 필요: 16
  - 즉시 추가 20 반영 후 `pairs_neg_lt10`: 255
- 근거 파일:
  - `dmp_candidates_raw.csv`
  - `dmp_fill_manifest.json`
  - `dmp_delta_addable.parquet`
  - `dmp_candidates_need_feature.csv`
  - `dmp_need_feature_molregno_unique.csv`
  - `dmp_fill_apply_report.json`
  - `shortage_pairs_after_dmp.csv`

## 4) 신규 16개 분자 FP2/ES5D 증분 생성 및 반영
- 신규 16개 분자 입력:
  - `dmp_need_feature_input.parquet`
- 생성 결과:
  - FP2: 16/16 성공
  - ES5D: 16/16 성공
- 생성 산출물:
  - `dmp_new16_fp2/fp2_uint64.memmap`
  - `dmp_new16_fp2/fp2_meta.parquet`
  - `dmp_new16_es5d/es5d_db_k20.memmap`
  - `dmp_new16_es5d/es5d_meta_db.parquet`

## 5) 최종 확장 파일(현재 최신)
- feature 확장(append):
  - `fp2_aligned.with_dmp16.memmap`
  - `es5d_db_k20.with_dmp16.memmap`
  - `es5d_meta_db.with_dmp16.parquet`
- meta 최신:
  - `final_training_meta.with_dmpfill_plus16.parquet`
- shortage 최신:
  - `shortage_pairs_after_dmp_plus16.csv`
- 리포트:
  - `dmp_plus16_apply_report.json`

## 6) 최신 집계(최종)
- 전체 pair: 1,079,655
- `neg == 10`: 1,079,409
- `neg < 10`: 246
- `1:10 미충족 비율`: 0.022785%
- 중복 음성(`pair_id,molregno`) 문제: 0
- memmap 정합성:
  - `N_fp2 = N_es5d = 827,131`
  - meta의 `memmap_idx` 범위 정상

## 7) 학습 시 사용 권장 입력(최신)
- Meta:
  - `/mnt/d/stp_proj_v3/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/final_training_meta.with_dmpfill_plus16.parquet`
- FP2:
  - `/mnt/d/stp_proj_v3/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/fp2_aligned.with_dmp16.memmap`
- ES5D:
  - `/mnt/d/stp_proj_v3/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/es5d_db_k20.with_dmp16.memmap`

## 8) 실행/보강에 사용한 주요 스크립트
- 운영모드 음성 보강:
  - `/mnt/d/stp_proj_v3/script/p1_4_augment_negatives_current_mode.py`
- DMP 후보 추출/적용(작업용):
  - `/home/xotn91/extract_dmp_candidates_for_shortage.py`
  - `/home/xotn91/fill_shortage_from_dmp_candidates.py`
  - `/home/xotn91/apply_new16_features_and_fill.py`
- 기존 피처 생성기:
  - `/mnt/d/stp_proj_v3/script/p2_1_extract_2d_fp2.py`
  - `/mnt/d/stp_proj_v3/script/p2_2_extract_3d_es5d_production.py`

