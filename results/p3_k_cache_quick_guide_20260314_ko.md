# P3 K-Cache 빠른 실행 가이드

## 목적

- `p3_1_1`을 먼저 `K=5`로 실행
- 생성된 Feature Matrix 캐시를 `K=1` 실행에서 재사용
- `K`만 달라지는 경우에만 재사용

## 실행 스크립트

- 실행 파일:
  - `/mnt/d/stp_proj_v3/script/run_p3_1_1_full_aug_scaf_off_k5_then_k1_reuse_20260314.sh`

## 실행 방법

외부 터미널에서 아래와 같이 실행합니다.

```bash
cd /mnt/d/stp_proj_v3/script
./run_p3_1_1_full_aug_scaf_off_k5_then_k1_reuse_20260314.sh
```

## 수행 순서

1. `p3_1_1`을 `K=5`로 실행합니다.
2. 이때 생성된 Feature Matrix를 캐시에 저장합니다.
3. 이어서 `p3_1_1`을 `K=1`로 실행합니다.
4. `K=1` 실행 시 `K=5`에서 만든 캐시를 재사용합니다.

## 결과 저장 위치

- K=5 결과:
  - `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_k5_then_k1_reuse_20260314/k5_run`
- K=1 결과:
  - `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_k5_then_k1_reuse_20260314/k1_reuse_run`
- 캐시:
  - `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_k5_then_k1_reuse_20260314/feature_cache`
- 로그:
  - `/mnt/d/stp_proj_v3/features_store/logs/`

## 중요한 제한 사항

- 재사용은 `K`만 바뀌는 경우에만 동작합니다.
- 아래 조건이 바뀌면 캐시를 재사용하지 않고 다시 계산합니다.
  - `scaffold_mask`
  - threshold 설정
  - `k_policy`
  - `cv_scheme`
  - `cutoff_year`
  - 입력 데이터셋 경로

## 참고

- `K=5`로 만든 캐시는 `K=1`에서 재사용할 수 있습니다.
- 반대로 `K=1` 결과만으로 `K=5`를 복원할 수는 없습니다.
- 현재 구조에서는 `scaffold on/off` 변경 시 Feature Matrix를 다시 생성해야 합니다.

