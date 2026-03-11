# STP Project V3 진행 요약 (2026-03-09)

## 1) ChEMBL36 조건 필터링 집계 결과
적용 조건:
- ChEMBL 36
- Organism: `Homo sapiens`, `Mus musculus`, `Rattus norvegicus`
- Ligand: `heavy_atoms <= 80`
- Target type: `SINGLE PROTEIN`, `PROTEIN COMPLEX`
- Assay type: `B`(Binding)
- Activity type: `Ki`, `Kd`, `IC50`, `EC50`
- `standard_relation IN ('=','<','<=')`, 단위는 μM 기준으로 정규화

집계(활성 레코드 기준):
- `< 1 μM`: **1,476,950**
- `< 10 μM`: **1,824,258**
- `< 100 μM`: **1,969,895**
- `>= 100 μM`: **35,156**

타입별 주요 수치:
- `EC50`: 101,514 (>=100 μM: 1,302)
- `IC50`: 1,384,122 (>=100 μM: 22,530)
- `Kd`: 69,464 (>=100 μM: 2,858)
- `Ki`: 449,951 (>=100 μM: 8,466)

관련 파일:
- `/mnt/d/stp_proj_v3/results/chembl36_threshold_summary_20260309.md`
- `/mnt/d/stp_proj_v3/results/chembl36_threshold_summary_20260309.csv`

---

## 2) 타겟 겹침 분석 (`>100 μM` vs `<=10 μM`)
동일 조건에서 계산 결과:
- `>100 μM` 데이터와 `<=10 μM` 데이터가 공통으로 가지는 타겟 수: **1,717**
- `>100 μM` 전체 레코드: **32,464**
- 그중 공통 타겟(1,717개)에 속한 레코드: **31,672**

---

## 3) 학습 parquet 음성 보강 수행 결과
입력 파일:
- `/mnt/d/stp_proj_v3/features_store/chembl36_stp_training_set.parquet`

보강 규칙:
- 동일 기본 필터 유지
- `activity_uM >= 100`인 `Ki/Kd/IC50/EC50` binding 실험 레코드를 음성(`Assumed_Inactive`, `Negative`)으로 추가
- 기존 `pair_id` 체계 유지
- 기존 `(target, molecule)` 중복/충돌 제거

출력 파일:
- `/mnt/d/stp_proj_v3/features_store/chembl36_stp_training_set.neg_ge100_aug.parquet`
- `/mnt/d/stp_proj_v3/features_store/chembl36_stp_training_set.neg_ge100_aug_report_20260309.json`

검증 요약:
- SQL 원시 후보: 36,804
- 기존 pair 중복 제외: 2,590
- (중복 중) 기존 Positive 충돌 제외: 2,558
- 최종 신규 추가: **25,766**
- 신규 내부 중복: 0
- 기존 데이터와 신규 추가 겹침: 0
- 전체 행수: 15,025,219 -> **15,050,985**
- `pair_id`별 Positive=1 유지, Negative 최소 10 유지
  - `pair_negative_min=10`, `pair_negative_max=63`

---

## 4) 모델 학습 관련 결정사항
요청사항 반영 방향:
- 음성 학습에서 **신규 `>=100 μM` 음성 우선 사용**
- pair별 10개를 못 채우면 기존 음성으로 보충

추가 기술 판단:
- 새 학습 parquet 반영 시 `fp2/es5d/memmap`은 정합성 때문에 원칙적으로 재구성 권장
- 다만 `es5d`는 시간 비용이 크므로,
  - 기존 생성된 분자의 ES5D를 **캐시 재사용**하고
  - 신규 분자만 증분 계산하는 방식 권장

---

## 5) 다음 실행 권장 순서
1. pair별 음성 10개 재구성 로직 확정 (`>=100 μM` 우선 + 기존 음성 보충)
2. 새 학습 parquet 확정
3. `fp2` 재생성
4. `es5d`는 기존 캐시 매칭 후 미매칭 분자만 증분 생성
5. `meta/memmap/manifest` 재정렬 생성
6. 학습/검증 재실행 및 비교 리포트 생성

