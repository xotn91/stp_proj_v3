# P3 작업 요약 (현재 기준)

## 1) 작업 경로/규칙
- 작업 경로: `D:\stp_proj_v3` (WSL: `/mnt/d/stp_proj_v3`)
- P3 규칙 문서: `p3.md`
- 추가 반영 규칙:
  - P3 코드 수정은 `script` 폴더 파일만 대상 (`old_scripts` 수정 금지)

## 2) 현재 P3 핵심 스크립트
- `script/p3_1_1_K1_paired_trainer_fast.py`
- `script/p3_1_2_K1_unpaired_trainer_fast.py`

### 공통 반영사항
- 논문식 threshold normalization 반영
  - 2D threshold: 0.65
  - 3D threshold: 0.30
  - clamp: `[0, 1]`
- K 동적 처리 가능 (`--k_mode`)
- 결과 파일명에 모드 정보 반영
- `ha_bin`별 coef CSV 추가 저장
  - `stp_coef_by_ha_K{K}_{POLICY}_{eval_mode}.csv`

### 마스크 옵션(독립 제어)
- `--scaffold_mask {on,off}`
- `--assay_mask {on,off}`
- 내부 `eval_mode` 라벨 자동 계산:
  - `all`
  - `distinct_scaffolds`
  - `assays_only`
  - `distinct_scaffolds_assays`

### 실행별 결과 분리 저장
- 실행 시 옵션 조합별 전용 폴더 생성
- 각 실행 폴더에 `run_config.log` 저장
  - 스크립트/명령/옵션/성공·실패/에러 기록

## 3) 20% 샘플링 스크립트
- `script/p3_0_build_subset_meta.py`
- 목적:
  - `final_training_meta.parquet`에서 subset(기본 20%) 생성
  - 층화 기준: `target_chembl_id`, `set_type`, `ha_bin`, `cv_fold`, `time_split`
- 출력:
  - `final_training_meta_subset.parquet`
  - `subset_build_log.json`
  - `subset_build_log.txt`

## 4) 8개 조합 자동 실행 배치
- `script/p3_1_0_run_mask_grid.py`
- 조합:
  - 스크립트 2개 x (`scaffold_mask` on/off) x (`assay_mask` on/off) = 8개
- 배치 로그(`p3_0 스타일`):
  - `batch_run_log.json`
  - `batch_run_log.txt`
- 각 job별로 예상 실행 폴더/`run_config.log` 존재 및 옵션 키 기록 여부 검증 필드 포함

## 5) P2 통합 스크립트(기존 old p2_3+p2_4 통합)
- `script/p2_3_merge_and_align.py`
- 기능:
  - `final_training_meta.parquet` 생성(merge)
  - `fp2_aligned.memmap` 생성(ES5D index 정렬)
  - manifest 저장

## 6) 실행 순서 가이드
1. (선택) 전체 메타 갱신 필요 시:
   - `p2_3_merge_and_align.py`
2. (선택) 시간 단축용 subset 필요 시:
   - `p3_0_build_subset_meta.py`
3. 옵션 조합 실행:
   - `p3_1_0_run_mask_grid.py`

## 7) 자주 묻는 사항 정리
- `p3_1_0_run_mask_grid.py` 전에 `p3_0_build_subset_meta.py`는 필수가 아님
  - 전체 메타로 바로 실행 가능
  - subset으로 돌릴 때만 먼저 실행 후 `--meta_file`로 subset parquet **파일 경로** 지정
  - `--meta_file`에 디렉터리를 넣으면 자동 보정되지만, 가능하면 `final_training_meta_subset.parquet` 파일 경로를 직접 지정 권장
- `p2_3_merge_and_align.py`는 `inner join` 기반이므로 조인 결과 행 수는 원본과 달라질 수 있음
