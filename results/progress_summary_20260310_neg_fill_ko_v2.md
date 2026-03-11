# STP v3 진행상황 요약 (2026-03-10, 최신)

## 1) 현재 운영 정책
- Positive: `activity_uM <= 1`
- Negative: `activity_uM >= 100` 우선 추가
- 부족 시 fallback: 기존 `activity_uM = NaN` 음성 사용
- 목표: pair당 음성 10개(1:10)
- 10개 미달 pair는 drop 없이 유지
- 동일 `pair_id` 내 동일 `molregno` 음성 중복 금지

## 2) 음성 보강 진행 결과
- 1차 보강 결과
  - base meta: `p3_ready_pos_le1uM_neg7plus_20260309/final_training_meta.parquet`
  - 추가 음성: 230,117행
  - `neg<10` pair: 272
- DMP 연계 보강
  - DMP 후보 raw: 303
  - heavy_atoms<=80 통과: 270
  - 즉시 추가 가능(기존 ES5D 보유): 20
  - 신규 피처 필요: 16
  - 반영 후 `neg<10`: 255
- 신규 16개 분자 FP2/ES5D 생성(16/16 성공) 후 추가 반영
  - 추가 반영: 10행
  - 최종 `neg<10`: 246

## 3) 최신 학습용 파일
- Meta:
  - `/mnt/d/stp_proj_v3/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/final_training_meta.with_dmpfill_plus16.parquet`
- FP2:
  - `/mnt/d/stp_proj_v3/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/fp2_aligned.with_dmp16.memmap`
- ES5D:
  - `/mnt/d/stp_proj_v3/features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/es5d_db_k20.with_dmp16.memmap`

## 4) 최신 집계
- 전체 pair: 1,079,655
- `neg==10` pair: 1,079,409
- `neg<10` pair: 246
- 1:10 미충족 비율: 0.022785%
- pair 내 음성 중복: 0
- memmap 정합성:
  - `N_fp2 = N_es5d = 827,131`
  - meta `memmap_idx` 범위 정상

## 5) p3 스크립트 반영 상태(확인 완료)
- `--min_neg_per_pair`, `--max_neg_per_pair` 옵션 반영됨
- `10/10`으로 실행 시 1:10 강제 검증 가능
- 주의: 자동 보강 기능은 아니며, 조건 미충족 pair 존재 시 에러로 중단

## 6) 주요 산출물/리포트
- `features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/augment_manifest.json`
- `features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/dmp_fill_manifest.json`
- `features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/dmp_fill_apply_report.json`
- `features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/dmp_plus16_apply_report.json`
- `features_store/p3_ready_pos_le1uM_neg10_keepall_20260310/shortage_pairs_after_dmp_plus16.csv`

