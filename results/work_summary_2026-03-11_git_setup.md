# STP 프로젝트 작업 요약 (2026-03-11)

## 1) 프로젝트 및 현재 실행 상태
- 작업 경로:
  - Windows: `D:\stp_proj_v3`
  - WSL: `/mnt/d/stp_proj_v3`
- 현재 프로젝트는 Git 저장소로 초기화되었고 GitHub 원격 저장소와 연동 완료됨
- 확인된 장기 실행 작업:
  - 프로세스: `script/p3_1_1_K1_paired_trainer_fast.py`
  - 시작 시각: `2026-03-11T08:01`
  - 출력 경로: `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_20260311/`
  - 실행 로그: `/mnt/d/stp_proj_v3/features_store/p3_run_full_aug_scaf_off_20260311/train_scaffold_off.log`

## 2) P3 코드 점검 중 확인 사항
- `p3_1_1_K1_paired_trainer_fast.py` 실행 명령에는 `--assay_mask on`이 포함되어 있었음
- 그러나 실제 코드에서는 `assay_mask`가 강제로 `False`로 고정되어 인자가 무시됨
- 확인 위치:
  - `script/p3_1_1_K1_paired_trainer_fast.py`
  - `script/p3_1_2_K1_unpaired_trainer_fast.py`
- 현재 `run_config.log`에도 아래처럼 기록됨
  - `assay_mask=False`
  - `assay_mask_arg_ignored=on`
- 따라서 2026-03-11 기준 실행 중인 full run은 assay mask 적용 실험이 아님

## 3) Git 로컬 저장소 설정
- `/mnt/d/stp_proj_v3`에서 `git init` 수행
- 기본 브랜치 `main`으로 설정
- 로컬 작성자 정보 설정:
  - `user.name = xotn91`
  - `user.email = xotn91@gmail.com`

## 4) .gitignore 정책 정리
- 대용량 및 생성 산출물은 Git 추적 대상에서 제외하도록 `.gitignore` 작성
- 주요 제외 대상:
  - `features_store/`
  - `blast_work/`
  - `TEMP/`
  - `*.parquet`
  - `*.memmap`
  - `*.log`
  - `results/*.csv`
  - `results/*.json`
  - `results/*.png`
  - `results/*.xlsx`
  - `results/summary_*/`

## 5) Git 커밋 이력
- 초기 커밋:
  - `6d004e7` `Initialize stp_proj_v3 repository`
- 결과 요약 폴더 제외 정책 추가:
  - `ad5abcd` `Ignore generated summary result directories`
- 프로젝트 README 추가:
  - `eb19226` `Add project README`

## 6) GitHub 연동 작업
- GitHub 원격 저장소:
  - `git@github.com:xotn91/stp_proj_v3.git`
- HTTPS push는 인증 부재로 실패
- SSH 키 신규 생성:
  - 경로: `/home/xotn91/.ssh/id_ed25519`
  - 공개키 GitHub 등록 완료
- 일반 SSH `github.com:22` 연결은 현재 환경에서 불가
- 해결:
  - `/home/xotn91/.ssh/config`에 GitHub SSH over 443 설정 추가
  - `ssh.github.com:443` 경유로 인증 성공
- 최종 결과:
  - `origin/main` push 완료
  - 로컬 `main`이 `origin/main` 추적 중

## 7) 추가된 문서
- `README.md` 추가 완료
- 주요 내용:
  - 프로젝트 목적
  - 작업 경로
  - 디렉터리 설명
  - Git 추적 정책
  - 권장 Git 작업 흐름
  - P3 작업 규칙 요약

## 8) 현재 상태 요약
- GitHub 연동 완료
- 작업 트리 정리 완료
- README 반영 완료
- 현재 남은 주요 실무 항목:
  - `assay_mask` 무시 문제를 별도 브랜치에서 수정할지 결정
  - 실행 중인 full run 지속 모니터링
  - 필요 시 후속 브랜치 전략 예:
    - `fix/assay-mask`
    - `exp/p3-full-run-analysis`
