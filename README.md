# skala-sLLM

SKALA sLLM 파인튜닝 실습. Qwen2.5-1.5B-Instruct에 LoRA SFT를 적용해 HR 정책 QA에 특화시키고, Base 모델과 성능을 비교합니다.

## 결과

10건 평가 기준으로 파인튜닝 모델이 **9승 1패**로 앞섰습니다.

| 지표 | Base | Fine-tuned | 개선 |
|---|---|---|---|
| Keyword Score | 0.060 | 0.260 | +0.200 |
| Token F1 | 0.077 | 0.275 | +0.197 |
| Total Score | 0.065 | 0.264 | +0.199 |
| Hallucination Rate | 0.0 | 0.0 | — |

학습 설정: Apple Silicon MPS / FP32 LoRA / 3 epochs / max_length 512 / lr 1e-4 / LoRA rank 16, alpha 32 / grad accum 8. 최종 train loss 0.427, 219 스텝.

평가 데이터 10건이 전부 알려진 정책(known policy)이라 hallucination 지표는 양쪽 모두 0으로 나왔습니다. 이 지표를 제대로 보려면 unknown 케이스를 섞어야 합니다.

## 구조

```
sllm-main-std/
├── dataset/
│   ├── training/     hr_sft_train.jsonl · hr_sft_validation.jsonl · hr_sft_all.jsonl
│   └── evaluation/   hr_eval.jsonl
├── scripts/
│   ├── train_lora.py       LoRA SFT 학습 (CUDA는 4bit QLoRA, MPS·CPU는 FP32 LoRA로 자동 분기)
│   └── evaluate_model.py   Base ↔ SFT 비교 — Keyword / F1 / Hallucination
├── outputs/          학습·평가 결과 JSON
└── models/           학습된 LoRA Adapter (저장소에서는 제외)
```

## 실행

```bash
python3.11 -m venv sllm
source ./sllm/bin/activate
cd sllm-main-std
pip install -r requirements.txt

# 학습
SFT_MAX_LENGTH=384 SFT_EPOCHS=2 SFT_LEARNING_RATE=1e-4 python scripts/train_lora.py

# 평가
python scripts/evaluate_model.py

# 재학습 시 기존 어댑터 삭제
rm -rf models/hr-qwen-lora
```

## 저장소에서 제외한 것

가상환경(`sllm/`, 약 1.5G)과 학습된 LoRA 어댑터·체크포인트(`models/`, 약 527M)는 위 명령으로 재생성 가능해 추적하지 않습니다. 강의 제공 서브노트·과제 템플릿도 제외했습니다.

`sllm-main-std/readme.md`는 강의에서 배포된 원본 안내 문서입니다.
