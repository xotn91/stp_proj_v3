# STP 프로젝트 진행 요약 (2026-03-10)

## 1) 데이터셋 정책 변경/확정
- 기준 데이터셋: `chembl36_stp_training_set.pos_le1uM.neg_priority10.parquet`
- 양성(Positive): `activity_uM <= 1`
- 음성(Negative): 신규 `activity_uM >= 100` 우선 반영
- pair 규칙 완화: 기존 `1 pos + 10 neg` 고정에서
  - 현재 학습 허용 범위: `1 pos + 7~10 neg`
  - `7 미만`은 제외
  - `10 미만(7~9)`은 사용하되 로그 기록

## 2) FP2/ES5D/Meta 정합화
- 신규 FP2 생성 완료 (전용 경로)
- ES5D 증분 재사용 + 신규 생성 완료
- ES5D 상세검증 결과:
  - 기존 기준(5e-5)에서 회전/이동 불변성 1항목만 미세 초과
  - 최대 편차 `6.198883e-05`
  - 실사용 영향 Low로 판단, 검증 기본 임계값 `1e-4`로 조정
- 의사결정 파일:
  - `/mnt/d/stp_proj_v3/features_store/es5d_pos_le1uM_neg_priority10/es5d_validation_decision_20260309.json`

## 3) p3 진입용 신규 산출물
생성 위치:
- `/mnt/d/stp_proj_v3/features_store/p3_ready_pos_le1uM_neg7plus_20260309/`

주요 파일:
- `final_training_meta.parquet`
- `fp2_aligned.memmap`
- `pair_neg_not10_log.csv` (7~9 포함 + dropped 표시)
- `pair_dropped_log.csv` (7 미만 등 제외 사유)
- `build_manifest.json`

요약 수치(Manifest 기준):
- `es5d_rows = 827,115`
- `fp2_aligned_rows = 827,115` (정합)
- `meta_pairs = 1,079,655`
- `pairs_dropped = 24,354`
- `pairs_neg_not10_logged = 209,341`
- `kept_7to9 = 209,296`, `dropped(<7) = 45`

## 4) p3 코드 반영 사항
수정 파일:
- `/mnt/d/stp_proj_v3/script/p3_1_1_K1_paired_trainer_fast.py`
- `/mnt/d/stp_proj_v3/script/p3_1_2_K1_unpaired_trainer_fast.py`
- `/mnt/d/stp_proj_v3/script/p3_1_0_run_mask_grid.py`
- `/mnt/d/stp_proj_v3/script/p3_0_build_subset_meta.py`
- `/mnt/d/stp_proj_v3/script/p3_0_validate_subset_quality.py`

반영 내용:
- `--min_neg_per_pair`, `--max_neg_per_pair` 도입 (기본 7/10)
- pair-rule 검증을 `7~10` 허용으로 변경
- time split을 row 기준에서 `pair_id` 기준으로 변경
  - 동일 pair가 train/oot로 찢어지는 문제 해결

## 5) 20% 시험 실행 결과
- 20% 요청이었지만 pair-계층샘플 + min_per_group 영향으로 실제 약 `31.19%` 샘플 사용
- 시험 배치(1 job) 성공 완료
- 결과 경로:
  - `/mnt/d/stp_proj_v3/features_store/p3_0_batch_mask_grid_20260309_164801/`
- 보고서:
  - `stp_cv_report_K1_paired_sum_distinct_scaffolds.csv`
  - `stp_OOT_report_K1_paired_sum_distinct_scaffolds.csv`

## 6) 전체 데이터 scaffold on/off 실행 이력
- 병렬(on/off 동시) 실행 시도 후 장시간 진행
- 사용자 요청에 따라 해당 병렬 실행은 중단
- 이후 순차 실행(OFF -> ON) 안내 제공

## 7) 현재 상태 (저장 시점)
- 현재 동작 중: `scaffold OFF` 단독 학습 1건
- 실행 프로세스 TTY: `?` (터미널 비연결, 독립 실행)
- 실행 명령 핵심:
  - `p3_1_1_K1_paired_trainer_fast.py`
  - `--scaffold_mask off --assay_mask on`
  - `--meta_file /mnt/d/stp_proj_v3/features_store/p3_ready_pos_le1uM_neg7plus_20260309/final_training_meta.parquet`
- 출력 경로:
  - `/mnt/d/stp_proj_v3/features_store/p3_seq_full_scaf_off_on_20260310/`

## 8) 실시간 확인 명령
```bash
# 로그 추적
tail -f /mnt/d/stp_proj_v3/features_store/p3_seq_full_scaf_off_on_20260310/scaffold_off.log

# 진행률 한 줄
watch -n 5 "grep -a 'Rows Processed:' /mnt/d/stp_proj_v3/features_store/p3_seq_full_scaf_off_on_20260310/scaffold_off.log | tail -n 1"

# GPU 모니터링
watch -n 5 nvidia-smi
```

## 9) 다음 실행 계획 (권장)
1. 현재 OFF 완료 대기
2. 완료 후 ON 실행
3. OFF/ON CV + OOT 지표 비교표 생성
4. 최종 정책 확정 후 full 재학습/모델 고정
