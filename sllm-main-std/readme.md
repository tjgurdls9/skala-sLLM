# The following practice code is intended for educational purposes only. For contact : audit@korea.ac.kr, Sungryel Lim Ph.D

# This practice code is not a completed commercial version but has been developed for educational purposes; supplementation is required depending on the deployment objective for use as a commercial service.

# 가상환경 생성 (VS-Code의 root 디렉토리에서)

python3.11 -m venv sllm
source ./sllm/bin/activate
cd sllm-main-std
(sllm) python -m pip install -r requirements.txt


# 프로젝트 디렉토리 구조 확인

sllm-std/
│
├── requirements.txt                    # SFT 실습에 필요한 Python 패키지 목록
│
├── dataset/
│   │
│   ├── training/                       # SFT 학습용 데이터
│   │   ├── hr_sft_train.jsonl          # LoRA 학습에 사용하는 Train 데이터
│   │   └── hr_sft_validation.jsonl     # 학습 중 성능 검증용 Validation 데이터
│   │
│   └── evaluation/                     # 학습 완료 후 모델 평가용 데이터
│       └── hr_eval.jsonl               # Base 모델과 SFT 모델 비교 평가 데이터
│
├── scripts/
│   │
│   ├── train_lora.py                   # Qwen2.5 모델 LoRA SFT 학습 → 학습 완료 후 Adapter 생성
│   │
│   └── evaluate_model.py               # Base 모델 ↔ SFT 모델 성능 비교
│                                       # → Keyword / F1 / Hallucination 평가
│
├── models/
│   └── hr-qwen-lora/                   # 학습된 LoRA Adapter 저장 위치
│                                       # → 최초 배포 시에는 비어 있음
│
└── outputs/                             # 학습·평가 결과 파일 저장 위치
                                         # 최초 배포 시에는 비어 있음
                                         # training_log.json
                                         # training_summary.json
                                         # evaluation_results.jsonl
                                         # evaluation_summary.json


# 실습환경 확인

Qwen2.5-1.5B-Instruct + PEFT LoRA + TRL SFTTrainer


# SFT 학습 및 평가 (sllm-main-std 디렉토리에서)

1) 데이터셋 내용 확인
dataset/training/hr_sft_all.jsonl
dataset/training/hr_sft_train.jsonl
dataset/training/hr_sft_validation.jsonl

2) 학습 실행
항목 섦명 - SFT_MAX_LENGTH : 한 샘플의 최대 토큰 길이, SFT_EPOCHS : 전체 학습 데이터 반복 횟수, SFT_GRAD_ACCUM : 몇 스텝의 gradient를 모아 갱신할지 의미

SFT_MAX_LENGTH=384 \
SFT_EPOCHS=2 \
SFT_LEARNING_RATE=1e-4 \
python scripts/train_lora.py

3) 결과 평가
python scripts/evaluate_model.py

4) 재학습 시, 기존 학습된 모델 삭제
rm -rf models/hr-qwen-lora  

5) 데모 사이트 접속하여 체크하기
담당 교수가 준비한 데모 사이트에 접속 
- Base Model vs. SFT Model 비교 평가
- RAG 부착 여부가 미치는 영향 비교

내부망에서 담당 교수 IP (172.16.21.96)는 변경될 수 있으며
이 때에는 별도 안내합니다.

http://172.16.21.96:5173/


