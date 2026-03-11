# Project Rules (stp_proj_v3)

## Project Goal
- SwissTargetPrediction 모델 구현
- ChEMBL 36 데이터 활용

## Working Directory
- Windows: `D:\stp_proj_v3`
- Linux(WSL): `/mnt/d/stp_proj_v3`

## Directory Structure
- `stp_proj_v3/script`: 작성 스크립트 저장
- `stp_proj_v3/old_scripts`: 구 버전 스크립트 저장
- `stp_proj_v3/features_store`: 수행 결과물 저장 (json, parquet, memmap, log 등 스크립트 수행에 필요한 결과 위주)
- `stp_proj_v3/results`: 수행 결과물 저장 (csv, 그림 등 스크립트 수행 후 결과 확인 용도 파일 위주)
- `stp_proj_v3/blast_work`: 타겟 간 paralog 분석 관련 자료

## Conformer K
- ES5D Conformer K는 항상 20으로 고정한다.
- ES5D 출력 파일명은 `es5d_db_k20.memmap`로 통일한다.

## Model Top-K
- Model Top-K는 실행 시 항상 프롬프트로 입력받는다.
- 실행 시 K 입력이 필수다.

## Script Naming & Phase Rules
- P1 (DATA collection): 파일명 `p1_`로 시작
  - 작성 시 `p1.md` 및 `note.md` 규정 참조
- P2 (2D FP + 3D conformer construction): 파일명 `p2_`로 시작
  - 작성 시 `p2.md` 및 `note.md` 규정 참조
- P3 (Training): 파일명 `p3_`로 시작
  - 작성 시 `p3.md` 및 `note.md` 규정 참조
- P4 (Test): 파일명 `p4_`로 시작
  - 작성 시 `p4.md` 및 `note.md` 규정 참조
- P5 (Prediction): 파일명 `p5_`로 시작
  - 작성 시 `p5.md` 및 `note.md` 규정 참조

## Script Series Naming Rules
- 각 Phase 내부에서 과정/단계를 구분해야 하는 경우 시리즈 번호를 사용한다.
- 기본 형식: `p{phase}_{series}_{task}.py`
  - 예시: `p1_1_validate.py`, `p1_2_summarize.py`, `p2_1_fp2_extract.py`
- 최소 형식: `p{phase}_{series}.py` 도 허용한다.
  - 예시: `p3_1.py`, `p4_2.py`
- `series`는 1부터 시작하는 정수이며, Phase 내에서 중복 없이 순차적으로 증가시킨다.
- 동일 Phase에서 생성되는 복수 스크립트는 반드시 시리즈 번호(`p1_1`, `p1_2` 등)로 구분한다.

## Script Examples
- `stp_proj_v3/script/p2_es5d.py`
- `stp_proj_v3/script/p3_train.py`
- `stp_proj_v3/script/p4_infer.py`

## Logging & Validation
- 스크립트 작성 시 `old_scripts/` 내 스크립트를 기반으로 한다.
- 스크립트 수행 후 log 기록 생성
- log 최소 요건:
  - 에러 발생 내용
  - 누락 정보
  - 수행 통계 요약
  - 스크립트 검토 후 중요 기록 사항
- 수행 결과물 검증 방법 제안 및 확보
- 현재 구축된 컴퓨팅 환경 활용 극대화
  - CPU: AMD Ryzen Threadripper PRO 7975WX 32-Cores (4.00 GHz)
  - RAM: 128 GB
  - GPU: NVIDIA RTX 4000 Ada VRAM 20 GB x2

## P1 SMILES Handling Rules
- `p1_1_data_curation.py`는 원본 SMILES를 반드시 `canonical_smiles_original` 컬럼으로 보존한다.
- 표준화 처리(standardization, counter ion/solvent 제거, neutralization, kekulization) 후 SMILES는 `canonical_smiles` 컬럼에 저장한다.
- 이후 단계(P2 이상) 스크립트는 기본적으로 `canonical_smiles`(표준화 SMILES)를 사용한다.
- 수행 완료 시 원본/표준화 SMILES 비교 통계(변경 건수, 실패 건수, 샘플)를 로그에 기록해야 한다.

## P5 Query SMILES Normalization Rule
- P5 (Prediction) 단계의 `p5_`로 시작하는 모든 스크립트는 신규 입력 쿼리 분자 SMILES에 대해 반드시 정규화 과정을 수행해야 한다.
- 필수 정규화 과정: standardization, counter ion/solvent 제거, neutralization, kekulization.
- 정규화 전 원본 SMILES는 보존해야 하며(예: `canonical_smiles_original`), 예측/특징 추출에는 정규화된 SMILES를 사용한다.
