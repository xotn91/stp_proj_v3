# STP Project 작업 요약 (2026-03-01)

## 1) 요청/검토 범위
- `D:\stp_proj_v3`(WSL: `/mnt/d/stp_proj_v3`)에서 현재 규칙(`md`)과 진행 상태를 점검
- `p2_` 스크립트 실행 가능성 및 리스크 검토
- 자원 활용(속도 최적화) 관점에서 `p2_1`, `p2_2` 코드 개선
- 이후 `K=1/5`, `scaffold on/off`, `assay on/off` 조합 실행 방법 정리

## 2) 규칙/문서 확인 결과
- 확인 문서:
  - `AGENTS.md`
  - `p1.md`
  - `p3.md`
  - `results/work_summary_2026-03-01.md`
- 확인 핵심:
  - P2는 표준화된 `canonical_smiles` 사용 규칙과 충돌 없음
  - ES5D conformer K=20 고정 규칙 확인
  - `p2.md`, `note.md`는 현재 프로젝트 내에 없음(참조 규칙은 있으나 파일 부재)

## 3) P2 사전 점검 결과
- 스크립트 위치 확인:
  - `script/p2_1_extract_2d_fp2.py`
  - `script/p2_2_extract_3d_es5d_production.py`
  - (레거시) `script/p2_es5d.py`
- 기존 산출물 정합성 이슈 확인:
  - 최신 training parquet 갱신 시각 이후에도 기존 FP2/ES5D 산출물이 과거 시점 파일이어서 재생성 필요
  - ES5D memmap/meta 상호 불일치 이력 확인
- 환경 점검:
  - 기본 셸 `python` 없음
  - `stp-rdkit` env에서 `openbabel` import 실패
  - `stp` env에서 `rdkit/openbabel` import 정상
  - 결론: `micromamba run -n stp python ...`로 실행 필요

## 4) P2 코드 개선 내역

### 4.1 `script/p2_1_extract_2d_fp2.py`
- 병렬/성능 옵션 추가:
  - `--backend {loky,multiprocessing,threading}`
  - `--batch-size`
- CPU 상한 48 고정 제거:
  - 기본 `cpu_count()-2` 사용
- 병렬 결과 수집 병목 완화:
  - `return_as="generator_unordered"` 적용
- 안정성 강화:
  - `*.tmp` 파일에 작성 후 마지막 단계에서 원자적 `os.replace`
- OpenMP/BLAS 과다 스레드 억제 환경변수 추가

### 4.2 `script/p2_2_extract_3d_es5d_production.py`
- 병렬/성능 옵션 추가:
  - `--backend`, `--batch-size`
- CPU 상한 48 고정 제거(기본 `cpu_count()-2`)
- 병렬 결과 수집 병목 완화:
  - `return_as="generator_unordered"`
- 안정성 강화:
  - `*.tmp` 작성 후 원자적 `os.replace`
- RDKit/OpenMP 관련 스레드 억제
- 워커 stderr 억제 로직 추가:
  - RDKit/OpenBabel 경고 출력 폭주로 인한 I/O 병목 완화
- 벤치/디버깅 옵션 추가:
  - `--max-molecules`

## 5) 실행 결과

### 5.1 `p2_1` 재실행
- 개선 코드로 재실행 완료
- 결과 파일 갱신:
  - `features_store/fp2_uint64.memmap`
  - `features_store/fp2_meta.parquet`
  - `features_store/manifest.json`
  - `features_store/fp2_errors.log`

### 5.2 `p2_2` 실행 상태
- 인터랙티브/벤치 실행으로 동작 검증 완료
- 본 환경에서는 장시간 백그라운드 프로세스 유지가 불안정하여 완주 고정은 어려움
- 중단 테스트에서 남은 `es5d *.tmp` 임시파일은 정리 완료
- 기존 최종 ES5D 파일은 보존 상태

## 6) SMILES 사용 컬럼 확인
- `p2_1`, `p2_2` 모두 입력 parquet의 `canonical_smiles` 컬럼 사용
- `canonical_smiles_original`은 P2에서 직접 사용하지 않음
- `p2_2`는 `canonical_smiles`를 입력으로 받아 내부에서 pH 7.4 보정(`CorrectForPH`) 후 사용

## 7) 외부 터미널 실행 권장
- 장시간 작업은 VS Code/Windows Terminal 등 외부 터미널이 안정적
- 진행 상황은 `tqdm` 진행바로 실시간 표시됨

## 8) P3 조합 실행 가이드 (K/scaffold/assay)
- 질문하신 `K=1/5`, `scaffold on/off`, `assay on/off` 조합은 P2가 아니라 P3 학습 옵션
- 대상 스크립트:
  - `script/p3_1_1_K1_paired_trainer_fast.py`
  - `script/p3_1_2_K1_unpaired_trainer_fast.py`
- 중요 주의사항:
  - 스크립트 기본 ES5D 경로가 `es5d_db_k{K}.memmap`이므로,
  - 프로젝트 규칙(K=20 고정) 환경에서는 반드시 `--es5d_memmap features_store/es5d_db_k20.memmap` 명시 필요

### 8.1 16개 조합 연속 실행 예시
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

## 9) 현재 권장 다음 단계
1. 외부 터미널에서 `p2_2` 전체 본 실행 완주
2. 완주 후 `es5d_db_manifest.json`의 shape/success_rate 확인
3. 이후 P3 조합 배치 실행
