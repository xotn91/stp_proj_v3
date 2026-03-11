# P3 실행 체크리스트 (보조 문서)

## 문서 목적
- 본 문서는 `p3.md`를 대체하지 않는 **보조 실행 가이드**이다.
- 규칙의 우선순위는 항상 `p3.md`와 `AGENTS.md`를 따른다.
- 본 문서는 실행 순서/점검 항목/명령 예시만 제공한다.

## 충돌 방지 원칙
- `old_scripts/`는 수정하지 않는다. (P3 코드는 `script/`만 수정)
- 로그 생성/검증 항목은 `p3.md` 최소 요건을 반드시 따른다.
- ES5D Conformer K는 `AGENTS.md` 규칙에 따라 **20 고정**(`es5d_db_k20.memmap`)을 사용한다.
- P3 실행 시 Top-K(모델 K)는 런타임 옵션으로 관리하되, P2의 ES5D 파일명 규칙(K=20 고정)과 혼동하지 않는다.

## 사전 점검
1. 작업 경로: `/mnt/d/stp_proj_v3`
2. 실행 환경: `micromamba run -n stp python ...`
3. 필수 입력 파일 존재 확인
- `features_store/final_training_meta.parquet`
- `features_store/fp2_aligned.memmap`
- `features_store/es5d_db_k20.memmap`
4. GPU 상태 확인 (외부 터미널 권장)
- `nvidia-smi`
5. 기존 장시간 실행 프로세스 중복 여부 확인
- `pgrep -af p3_1_`

## 현재 사용 대상 스크립트
- `script/p3_1_1_K1_paired_trainer_fast.py`
- `script/p3_1_2_K1_unpaired_trainer_fast.py`

## 실행 시 주의사항
- 위 두 스크립트는 `--k_mode 1/5`를 받을 수 있다.
- 현재 코드 기본 ES5D 경로는 `features_store/es5d_db_k20.memmap`로 설정되어 있다.
- 명시적으로 지정하려면 아래 옵션 사용:
- `--es5d_memmap features_store/es5d_db_k20.memmap`
- `--meta_file`는 디렉터리가 아닌 parquet 파일 경로를 권장한다.
  - 예: `.../final_training_meta_subset.parquet`

## 권장 실행 조합 (K x scaffold x assay)
- K: `1`, `5`
- scaffold mask: `on`, `off`
- assay mask: `on`, `off`
- 총 8개 조합(스크립트 1개 기준), paired+unpaired 합계 16개

## 배치 실행 예시
```bash
cd /mnt/d/stp_proj_v3

META="features_store/final_training_meta.parquet"
FP2="features_store/fp2_aligned.memmap"
ES5D="features_store/es5d_db_k20.memmap"
OUT="features_store"

for K in 1 5; do
  for SCAF in on off; do
    for ASSAY in on off; do
      echo "=== PAIRED: K=$K scaffold=$SCAF assay=$ASSAY ==="
      micromamba run -n stp python script/p3_1_1_K1_paired_trainer_fast.py \
        --meta_file "$META" \
        --fp2_memmap "$FP2" \
        --es5d_memmap "$ES5D" \
        --out_dir "$OUT" \
        --k_mode "$K" \
        --k_policy paired_sum \
        --scaffold_mask "$SCAF" \
        --assay_mask "$ASSAY"

      echo "=== UNPAIRED: K=$K scaffold=$SCAF assay=$ASSAY ==="
      micromamba run -n stp python script/p3_1_2_K1_unpaired_trainer_fast.py \
        --meta_file "$META" \
        --fp2_memmap "$FP2" \
        --es5d_memmap "$ES5D" \
        --out_dir "$OUT" \
        --k_mode "$K" \
        --k_policy independent \
        --scaffold_mask "$SCAF" \
        --assay_mask "$ASSAY"
    done
  done
done
```

## 실행 후 확인 체크리스트
1. 각 실행마다 `run_config.log` 생성 여부 확인
2. 실행 결과 폴더명에 다음 정보 포함 여부 확인
- 스크립트명, K, policy, eval_mode, C, cutoff year
3. 산출물 확인
- `stp_cv_report_*.csv`
- `stp_coef_*.json`
- `stp_coef_by_ha_*.csv`
- (해당 시) `stp_OOT_report_*.csv`
4. 로그 최소 요건 검증
- 에러 내용
- 누락 정보
- 수행 통계 요약
- 중요 기록 사항

## 실패 시 재실행 원칙
- 실패한 조합 1건만 동일 옵션으로 단건 재실행
- `run_config.log`의 `status=failed`와 오류 스택을 우선 확인
- 입력 파일 불일치(FP2/ES5D/meta shape mismatch) 먼저 점검 후 재실행

## 문서 버전
- 작성일: 2026-03-01
- 성격: `p3.md` 보조 문서 (규칙 대체 아님)
