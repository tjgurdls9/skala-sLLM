"""
Base Qwen과 Fine-tuned LoRA 모델 비교 평가

평가 항목:
- 회사 고유 규정 Keyword 반영
- Reference Answer와 Token F1 비교
- Unknown Policy Hallucination 검사
- Base / Fine-tuned 모델 성능 비교

입력:
    dataset/evaluation/hr_eval.jsonl

출력:
    outputs/evaluation_results.jsonl
    outputs/evaluation_summary.json

기본 정책:
- 최대 20개 평가
- Known Policy + Unknown Policy 평가
- 동일한 생성 조건으로 Base와 LoRA 비교
- Total Score 차이가 0.01 이하이면 Tie

실행:
    python scripts/evaluate_model.py

평가 개수 변경:
    python scripts/evaluate_model.py --max-evaluation 10
"""


# 초보자를 위한 코드 읽기 가이드
#
# 이 파일은 위에서 아래까지 2,000줄의 코드가
# 모두 차례대로 실행되는 구조가 아닙니다.
#
# 위쪽에서는 평가에 필요한 함수들을 먼저 정의하고,
# 파일 아래쪽의 main() 함수에서 필요한 함수들을
# 순서대로 호출합니다.
#
# 따라서 처음부터 모든 함수를 읽기보다는
# 먼저 main()을 확인하여 전체 실행 흐름을 파악한 뒤,
# main()에서 호출되는 함수의 정의를 찾아가며 확인하는 것을 권장합니다.
#
#
# 추천 코드 읽기 순서
#
# main()
#   │
#   ├─ validate_adapter()
#   │    → 학습된 LoRA Adapter 파일이 정상적으로 존재하는지 확인
#   │
#   ├─ get_device() / get_dtype()
#   │    → CUDA / MPS / CPU 중 어떤 장치에서 실행할지 결정
#   │
#   ├─ AutoTokenizer.from_pretrained()
#   │    → Qwen Tokenizer 로드
#   │
#   ├─ AutoModelForCausalLM.from_pretrained()
#   │    → Fine-Tuning 이전의 Base Qwen 모델 로드
#   │
#   ├─ PeftModel.from_pretrained()
#   │    → Base Qwen에 학습된 LoRA Adapter 연결
#   │
#   ├─ read_jsonl()
#   │    → 평가 Dataset 로드
#   │
#   ├─ generate_answer()
#   │    → 질문을 모델에 전달하고 답변 생성
#   │
#   ├─ evaluate_answer()
#   │    → 생성된 답변을 평가
#   │      │
#   │      ├─ keyword_score()
#   │      │    → 필수 Keyword가 포함되었는지 평가
#   │      │
#   │      ├─ token_f1_score()
#   │      │    → Reference Answer와 Token 단위 유사도 평가
#   │      │
#   │      └─ hallucination_score()
#   │           → Unknown Policy에서 잘못된 규정을 생성했는지 평가
#   │
#   ├─ determine_winner()
#   │    → Base와 Fine-tuned 모델의 점수를 비교
#   │
#   ├─ update_stats()
#   │    → 각 평가 결과를 전체 통계에 누적
#   │
#   └─ build_summary()
#        → 전체 평가 결과의 평균과 개선 정도 계산
#
#
# 핵심적으로 먼저 볼 부분
#
# 1. main()
#    → 전체 평가 프로그램의 실행 순서
#
# 2. generate_answer()
#    → 실제로 모델이 어떻게 답변을 생성하는지 확인
#
# 3. evaluate_answer()
#    → 생성된 답변을 어떤 기준으로 평가하는지 확인
#
# 4. keyword_score()
#    token_f1_score()
#    hallucination_score()
#    → 세부 평가 방식 확인
#
# 5. determine_winner()
#    → Base와 Fine-tuned 모델의 최종 비교 방식 확인
#
#
# 처음에는 다음 보조 함수까지 모두 이해할 필요는 없습니다.
#
# format_score()
# print_model_result()
# print_case_report()
# average()
# save_jsonl()
# save_summary()
# print_summary()
# cleanup_runtime()
# parse_args()
#
# 이러한 함수들은 콘솔 출력, 결과 저장, 메모리 정리 등
# 평가 프로그램을 보조하는 역할을 합니다.
#
#
# VS Code 사용 팁
#
# 함수 이름에 커서를 놓고 'Go to Definition'을 사용하면
# 해당 함수가 정의된 위치로 바로 이동할 수 있습니다.
#
# 함수 내용을 확인한 뒤 이전 위치로 돌아와
# main()의 다음 실행 단계를 계속 따라가면 됩니다.

from __future__ import annotations

# Terminal에서 전달되는 --max-evaluation 등의 옵션을 처리합니다.
import argparse

# 평가 종료 후 Python 및 GPU 메모리 정리에 사용합니다.
import gc

# JSONL 평가 Dataset을 읽고 결과를 JSON으로 저장합니다.
import json

# 평가 문장의 공백, 특수문자 등을 정규화할 때 사용합니다.
import re

# 평가 결과 통계를 하나의 객체로 관리하기 위해 사용합니다.
from dataclasses import dataclass, field

# 운영체제에 관계없이 파일 경로를 처리합니다.
from pathlib import Path

# Dictionary 등에 다양한 타입이 들어갈 수 있도록 타입 힌트에 사용합니다.
from typing import Any

# PyTorch: 모델 실행 및 CUDA / MPS / CPU 장치 제어
import torch

# 학습된 LoRA Adapter를 Base Model에 연결하기 위해 사용합니다.
from peft import PeftModel

# Qwen Base Model과 Tokenizer를 로드합니다.
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# 1. 기본 설정

# 현재 evaluate_model.py 파일을 기준으로
# 프로젝트의 최상위 디렉토리를 찾습니다.
#
# 예:
#
# project/
# ├── dataset/
# ├── models/
# ├── outputs/
# └── scripts/
#     └── evaluate_model.py
#
# parent        -> scripts/
# parent.parent -> project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 평가에 사용할 원본 Base Model
#
# Fine-Tuning 시 사용했던 Base Model과
# 반드시 동일한 모델을 사용해야 합니다.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# Fine-Tuning으로 생성된 LoRA Adapter 위치
ADAPTER_PATH = (
    PROJECT_ROOT
    / "models"
    / "hr-qwen-lora"
)


# 평가용 Dataset 위치
EVALUATION_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "evaluation"
    / "hr_eval.jsonl"
)


# 평가 항목별 상세 결과 저장 위치
RESULT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_results.jsonl"
)


# 전체 평가 결과 요약 저장 위치
SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_summary.json"
)


# 사용자 질문 + System Prompt를 포함한
# 최대 입력 Token 길이
MAX_INPUT_TOKENS = 2048


# 모델이 답변으로 새롭게 생성할 수 있는
# 최대 Token 수
MAX_NEW_TOKENS = 256


# 별도의 옵션이 없다면 최대 20개의 평가 항목을 사용합니다.
DEFAULT_MAX_EVALUATION = 20


# 두 모델의 Total Score 차이가 0.01 이하이면
# 실제 성능 차이가 크지 않은 것으로 보고 Tie로 처리합니다.
WIN_MARGIN = 0.01


# Base Model과 Fine-tuned Model 모두에게
# 동일하게 적용할 System Prompt입니다.
#
# 두 모델에 동일한 Prompt를 사용해야
# Fine-Tuning 효과를 공정하게 비교할 수 있습니다.
SYSTEM_PROMPT = """
당신은 Human AI Corporation의 인사 규정 안내 도우미입니다.

다음 원칙을 반드시 지키세요.

- Human AI Corporation의 회사 규정에 근거하여 정확하게 답변하세요.
- 회사 고유 코드, 기한, 승인 절차, 조건, 예외가 있으면 정확히 설명하세요.
- 일반적인 기업 관행을 회사 규정처럼 추측하지 마세요.
- 규정에 없는 내용은 임의로 만들어내지 마세요.
- 절차가 있으면 순서대로 설명하세요.
- 적용 조건과 예외가 있으면 함께 안내하세요.
- 확인할 수 없는 내용은 인사팀 또는 담당 부서 확인이 필요하다고 안내하세요.
""".strip()


# Unknown Policy 질문에서
# "모른다", "확인이 필요하다"는 안전한 답변을 했는지
# 확인하기 위한 표현 목록입니다.
UNKNOWN_SAFE_TERMS = [
    "확인",
    "확인할 수 없습니다",
    "규정에 없습니다",
    "규정에서는",
    "인사팀",
    "담당 부서",
    "추측",
]


# Unknown Policy인데도 존재하지 않는 규정을
# 실제 규정처럼 만들어냈는지 검사하기 위한 표현입니다.
UNKNOWN_RISKY_PATTERNS = [
    "전액 지원합니다",
    "무조건 지원합니다",
    "반드시 지급합니다",
    "보상해 줍니다",
    "지원됩니다",
    "지급됩니다",
]


# 2. 평가 통계

@dataclass
class EvaluationStats:
    """
    여러 평가 항목에서 나온 결과를 누적해서 저장합니다.

    예:
    - Base Model 승리 횟수
    - Fine-tuned Model 승리 횟수
    - 평균 Keyword Score
    - 평균 Token F1
    - Hallucination 발생 횟수
    """

    # Base Model이 더 높은 점수를 받은 횟수
    base_wins: int = 0

    # Fine-tuned Model이 더 높은 점수를 받은 횟수
    tuned_wins: int = 0

    # 두 모델의 차이가 WIN_MARGIN 이하인 횟수
    ties: int = 0

    # 실제 회사 규정이 존재하는 평가 항목 수
    known_count: int = 0

    # 학습 데이터에 존재하지 않는 규정 질문 수
    unknown_count: int = 0

    # Unknown Policy에서 Base Model이
    # 존재하지 않는 규정을 만들어낸 횟수
    base_unknown_hallucinations: int = 0

    # Unknown Policy에서 Fine-tuned Model이
    # 존재하지 않는 규정을 만들어낸 횟수
    tuned_unknown_hallucinations: int = 0

    # 각 평가 항목의 Keyword Score를 저장합니다.
    #
    # default_factory=list를 사용하면
    # EvaluationStats 객체마다 새로운 list가 생성됩니다.
    base_keyword_scores: list[float] = field(
        default_factory=list
    )

    tuned_keyword_scores: list[float] = field(
        default_factory=list
    )

    # Reference Answer와 비교한 Token F1 점수
    base_f1_scores: list[float] = field(
        default_factory=list
    )

    tuned_f1_scores: list[float] = field(
        default_factory=list
    )

    # Keyword / F1 / Hallucination을 종합한 최종 점수
    base_total_scores: list[float] = field(
        default_factory=list
    )

    tuned_total_scores: list[float] = field(
        default_factory=list
    )


# 3. 실행 장치

def get_device() -> torch.device:
    """
    현재 컴퓨터에서 모델 추론에 사용할 장치를 자동으로 결정합니다.

    우선순위:
        NVIDIA CUDA -> Apple MPS -> CPU
    """

    # NVIDIA GPU가 있으면 CUDA 사용
    if torch.cuda.is_available():
        return torch.device("cuda:0")

    # Apple Silicon GPU를 사용할 수 있으면 MPS 사용
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    # GPU가 없다면 CPU 사용
    return torch.device("cpu")


def get_dtype(
    device: torch.device,
) -> torch.dtype:
    """
    실행 장치에 따라 Base Model의 데이터 타입을 결정합니다.

    GPU에서는 메모리를 줄이기 위해 FP16을 사용하고,
    CPU에서는 호환성을 위해 FP32를 사용합니다.
    """

    if device.type in {
        "cuda",
        "mps",
    }:
        return torch.float16

    return torch.float32


# 4. 데이터와 Adapter 확인

def read_jsonl(
    path: Path,
    max_records: int,
) -> list[dict[str, Any]]:
    """
    평가용 JSONL 파일을 읽고 기본 형식을 검사합니다.

    필요한 주요 필드:
    - id
    - question

    max_records만큼 읽으면 중단합니다.
    """

    # 평가 Dataset이 실제로 존재하는지 먼저 확인합니다.
    if not path.is_file():
        raise FileNotFoundError(
            f"평가 데이터셋을 찾을 수 없습니다: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        # 한 줄씩 읽으면서 실제 파일 줄 번호도 함께 관리합니다.
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            # 빈 줄은 평가 대상에서 제외합니다.
            if not line:
                continue

            try:
                # JSON 문자열을 Python Dictionary로 변환합니다.
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{line_number}번째 줄의 JSON 형식이 "
                    "잘못되었습니다."
                ) from exc

            # 각 평가 항목을 구분하기 위한 id가 반드시 필요합니다.
            if not record.get("id"):
                raise ValueError(
                    f"{line_number}번째 평가 항목에 "
                    "id가 없습니다."
                )

            # 평가 질문을 문자열로 가져옵니다.
            question = str(
                record.get(
                    "question",
                    "",
                )
            ).strip()

            if not question:
                raise ValueError(
                    f"{line_number}번째 평가 항목의 "
                    "question이 비어 있습니다."
                )

            records.append(record)

            # 지정된 최대 평가 수만큼 읽었다면 종료합니다.
            if len(records) >= max_records:
                break

    if not records:
        raise ValueError(
            "평가 데이터가 없습니다."
        )

    return records


def validate_adapter() -> None:
    """
    Fine-Tuning으로 생성된 LoRA Adapter가
    정상적으로 존재하는지 확인합니다.

    최소한 다음 파일이 필요합니다.

    adapter_config.json
    adapter_model.safetensors 또는 adapter_model.bin
    """

    if not ADAPTER_PATH.is_dir():
        raise FileNotFoundError(
            f"LoRA Adapter를 찾을 수 없습니다: "
            f"{ADAPTER_PATH}"
        )

    # LoRA 구조와 설정 정보
    config_path = (
        ADAPTER_PATH
        / "adapter_config.json"
    )

    # PEFT 버전이나 저장 방식에 따라
    # safetensors 또는 bin 형태가 존재할 수 있습니다.
    weight_paths = [
        ADAPTER_PATH
        / "adapter_model.safetensors",

        ADAPTER_PATH
        / "adapter_model.bin",
    ]

    if not config_path.is_file():
        raise FileNotFoundError(
            f"adapter_config.json을 찾을 수 없습니다: "
            f"{config_path}"
        )

    # 두 가지 가중치 파일 중 하나라도 존재해야 합니다.
    if not any(
        path.is_file()
        for path in weight_paths
    ):
        raise FileNotFoundError(
            "Adapter 가중치 파일을 찾을 수 없습니다."
        )


# 5. 모델 추론

def generate_answer(
    model,
    tokenizer,
    device: torch.device,
    question: str,
) -> str:
    """
    하나의 질문을 모델에 전달하고
    생성된 답변 문자열을 반환합니다.

    Base Model과 Fine-tuned Model 모두
    이 함수를 사용하기 때문에 생성 조건이 동일합니다.
    """

    # Qwen Chat Template에 전달할 대화 구조를 생성합니다.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    # system / user 메시지를
    # Qwen이 학습할 때 사용했던 Chat Template 형식으로 변환합니다.
    #
    # add_generation_prompt=True
    # -> 이제 assistant가 답변해야 한다는 위치까지 Prompt를 생성합니다.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Prompt 문자열을 실제 Token ID로 변환합니다.
    inputs = tokenizer(
        prompt,

        # PyTorch Tensor 형태로 반환
        return_tensors="pt",

        # 너무 긴 입력은 잘라냅니다.
        truncation=True,

        max_length=MAX_INPUT_TOKENS,

    # Token Tensor를 실제 모델 장치로 이동합니다.
    ).to(device)

    # 평가에서는 Gradient 계산이 필요하지 않습니다.
    #
    # inference_mode()를 사용하면 메모리 사용량과
    # 추론 오버헤드를 줄일 수 있습니다.
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,

            # 최대 생성 길이
            max_new_tokens=MAX_NEW_TOKENS,

            # Sampling을 끄고 결정적인 방식으로 생성합니다.
            #
            # Base / Fine-tuned 모델을 같은 조건에서
            # 비교하기 위한 설정입니다.
            do_sample=False,

            # 추론 시 KV Cache 사용
            use_cache=True,

            # 동일한 단어나 문장이 지나치게 반복되는 것을
            # 약하게 억제합니다.
            repetition_penalty=1.05,

            # 동일한 3-token 조합을 반복하지 않도록 제한합니다.
            no_repeat_ngram_size=3,

            pad_token_id=(
                tokenizer.pad_token_id
            ),

            eos_token_id=(
                tokenizer.eos_token_id
            ),
        )

    # outputs에는 입력 Prompt Token까지 포함되어 있습니다.
    input_length = (
        inputs["input_ids"].shape[1]
    )

    # 입력 부분을 제외하고
    # 모델이 새롭게 생성한 Token만 가져옵니다.
    generated_tokens = (
        outputs[0][input_length:]
    )

    # Token ID를 사람이 읽을 수 있는 문자열로 변환합니다.
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# 6. Token F1

def normalize_text(
    text: str,
) -> str:
    """
    모델 답변과 Reference Answer를 비교하기 전에
    문자열 형식을 최대한 동일하게 정리합니다.

    예:
    - 대문자 -> 소문자
    - 여러 공백 -> 하나의 공백
    - 불필요한 특수문자 제거
    """

    normalized = (
        text
        .lower()
        .strip()
    )

    # 연속된 여러 개의 공백을 하나로 변경합니다.
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    # 문자, 숫자, 한글, -, 공백을 제외한
    # 대부분의 특수문자를 제거합니다.
    normalized = re.sub(
        r"[^\w가-힣\- ]",
        "",
        normalized,
    )

    return normalized.strip()


def token_f1_score(
    prediction: str,
    reference: str | None,
) -> float:
    """
    모델 답변(prediction)과 정답(reference)에
    공통으로 등장하는 Token을 이용해 F1 Score를 계산합니다.

    Precision:
        모델이 생성한 Token 중
        Reference에도 있는 비율

    Recall:
        Reference의 Token 중
        모델 답변에도 포함된 비율

    F1:
        Precision과 Recall의 조화 평균
    """

    # Reference Answer가 없다면 비교할 수 없습니다.
    if not reference:
        return 0.0

    # 문자열을 정규화한 뒤 공백 기준 Token으로 분리합니다.
    prediction_tokens = (
        normalize_text(
            prediction
        ).split()
    )

    reference_tokens = (
        normalize_text(
            reference
        ).split()
    )

    if (
        not prediction_tokens
        or not reference_tokens
    ):
        return 0.0

    # 각 Token이 몇 번 등장했는지 저장합니다.
    prediction_counts: dict[
        str,
        int,
    ] = {}

    reference_counts: dict[
        str,
        int,
    ] = {}

    for token in prediction_tokens:
        prediction_counts[token] = (
            prediction_counts.get(
                token,
                0,
            )
            + 1
        )

    for token in reference_tokens:
        reference_counts[token] = (
            reference_counts.get(
                token,
                0,
            )
            + 1
        )

    # 두 문장에 공통으로 등장한 Token 수를 계산합니다.
    #
    # 한쪽에 같은 단어가 더 많이 등장하더라도
    # 공통으로 존재하는 개수까지만 인정합니다.
    overlap = sum(
        min(
            prediction_counts.get(
                token,
                0,
            ),
            count,
        )
        for token, count
        in reference_counts.items()
    )

    if overlap == 0:
        return 0.0

    # 생성 답변 기준 정확도
    precision = (
        overlap
        / len(
            prediction_tokens
        )
    )

    # Reference 기준 재현율
    recall = (
        overlap
        / len(
            reference_tokens
        )
    )

    # Precision과 Recall을 하나의 점수로 결합합니다.
    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


# 7. Keyword 평가

def keyword_score(
    answer: str,
    must_include: list[str],
    must_not_include: list[str],
) -> dict[str, Any]:
    """
    회사 고유 규정에 필요한 핵심 Keyword가
    모델 답변에 포함되어 있는지 검사합니다.

    must_include:
        반드시 등장해야 하는 표현

    must_not_include:
        답변에 등장하면 안 되는 표현
    """

    # 정답에 반드시 필요한 Keyword 중
    # 실제 답변에 들어간 항목
    included = [
        keyword
        for keyword in must_include
        if keyword in answer
    ]

    # 필요한 Keyword이지만 답변에서 빠진 항목
    missing = [
        keyword
        for keyword in must_include
        if keyword not in answer
    ]

    # 등장해서는 안 되는 표현 중
    # 실제 답변에 포함된 항목
    forbidden = [
        keyword
        for keyword in must_not_include
        if keyword in answer
    ]

    # 필수 Keyword 자체가 없는 평가 항목이면
    # Keyword Score를 계산하지 않습니다.
    if not must_include:
        score = None

    else:
        # 필수 Keyword 중 몇 %를 맞혔는지 계산합니다.
        score = (
            len(included)
            / len(must_include)
        )

        # 금지 Keyword가 발견되면 0.5 감점합니다.
        if forbidden:
            score = max(
                0.0,
                score - 0.5,
            )

        score = round(
            score,
            4,
        )

    return {
        "score": score,
        "included": included,
        "missing": missing,
        "forbidden_found": forbidden,
    }


# 8. Hallucination 평가

def hallucination_score(
    answer: str,
    category: str | None,
    must_not_include: list[str],
) -> dict[str, Any]:
    """
    Unknown Policy 질문에서
    모델이 존재하지 않는 회사 규정을 만들어냈는지 검사합니다.

    Known Policy 질문에는 적용하지 않습니다.
    """

    # Unknown Policy가 아니라면 Hallucination 평가 대상이 아닙니다.
    if category != "unknown_policy":
        return {
            "applicable": False,
            "safe_response": None,
            "hallucinated": None,
            "score": None,
            "safe_terms_found": [],
            "risky_terms_found": [],
        }

    # "확인이 필요하다", "인사팀에 문의" 등
    # 안전하게 답변한 표현을 찾습니다.
    safe_terms = [
        term
        for term in UNKNOWN_SAFE_TERMS
        if term in answer
    ]

    # 공통 위험 표현과 Dataset별 금지 표현을 합칩니다.
    #
    # dict.fromkeys()를 사용해 중복 표현을 제거합니다.
    risky_patterns = list(
        dict.fromkeys(
            UNKNOWN_RISKY_PATTERNS
            + must_not_include
        )
    )

    # 실제 답변에서 위험 표현을 찾습니다.
    risky_terms = [
        term
        for term in risky_patterns
        if term in answer
    ]

    # 안전 표현이 하나라도 존재하는지 확인
    safe_response = bool(
        safe_terms
    )

    # 위험 표현이 하나라도 존재하면
    # Hallucination이 발생한 것으로 간주합니다.
    hallucinated = bool(
        risky_terms
    )

    # 안전 표현 O + 위험 표현 X
    # -> 가장 안전한 답변
    if (
        safe_response
        and not hallucinated
    ):
        score = 1.0

    # 안전 표현도 있지만 잘못된 정보도 포함
    elif (
        safe_response
        and hallucinated
    ):
        score = 0.5

    # 안전하게 회피하지 못함
    else:
        score = 0.0

    return {
        "applicable": True,
        "safe_response": safe_response,
        "hallucinated": hallucinated,
        "score": score,
        "safe_terms_found": safe_terms,
        "risky_terms_found": risky_terms,
    }


# 9. 종합 평가

def calculate_total_score(
    category: str | None,
    keyword: dict[str, Any],
    token_f1: float,
    hallucination: dict[str, Any],
) -> float:
    """
    Keyword / Token F1 / Hallucination Score를
    하나의 Total Score로 결합합니다.

    Known Policy:
        Keyword 70%
        Token F1 30%

    Unknown Policy:
        Keyword 30%
        Token F1 20%
        Safety 50%
    """

    # Keyword 평가가 없는 경우 0점으로 처리합니다.
    keyword_value = (
        keyword["score"]
        if keyword["score"] is not None
        else 0.0
    )

    # Unknown Policy는
    # Hallucination 방지가 가장 중요하므로
    # Safety Score에 50%의 비중을 둡니다.
    if category == "unknown_policy":
        safety_value = (
            hallucination["score"]
            if hallucination["score"] is not None
            else 0.0
        )

        total = (
            keyword_value * 0.30
            + token_f1 * 0.20
            + safety_value * 0.50
        )

    # Known Policy는
    # 회사 고유 Keyword를 정확히 설명하는 것이 중요하므로
    # Keyword Score에 70% 비중을 둡니다.
    else:
        total = (
            keyword_value * 0.70
            + token_f1 * 0.30
        )

    return round(
        total,
        4,
    )


def evaluate_answer(
    answer: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    하나의 모델 답변에 대해

    1. Keyword
    2. Token F1
    3. Hallucination
    4. Total Score

    를 순서대로 계산합니다.
    """

    category = record.get(
        "category"
    )

    # 답변에 반드시 포함되어야 하는 Keyword
    must_include = list(
        record.get(
            "must_include",
            [],
        )
    )

    # 답변에 포함되어서는 안 되는 표현
    must_not_include = list(
        record.get(
            "must_not_include",
            [],
        )
    )

    keyword = keyword_score(
        answer,
        must_include,
        must_not_include,
    )

    f1 = token_f1_score(
        answer,
        record.get(
            "reference_answer"
        ),
    )

    hallucination = hallucination_score(
        answer,
        category,
        must_not_include,
    )

    total = calculate_total_score(
        category,
        keyword,
        f1,
        hallucination,
    )

    return {
        "keyword": keyword,
        "token_f1": round(
            f1,
            4,
        ),
        "hallucination": hallucination,
        "total": total,
    }


def determine_winner(
    base_total: float,
    tuned_total: float,
) -> tuple[str, float]:
    """
    Base Model과 Fine-tuned Model의 Total Score를 비교합니다.

    Fine-tuned - Base 점수 차이를 계산하며,
    절대 차이가 WIN_MARGIN 이하이면 Tie로 처리합니다.
    """

    difference = round(
        tuned_total
        - base_total,
        4,
    )

    # Fine-tuned가 의미 있게 높은 경우
    if difference > WIN_MARGIN:
        return (
            "fine_tuned",
            difference,
        )

    # Base Model이 의미 있게 높은 경우
    if difference < -WIN_MARGIN:
        return (
            "base",
            difference,
        )

    # 차이가 0.01 이하라면 실질적으로 동일하다고 판단
    return (
        "tie",
        difference,
    )


# 10. 콘솔 출력

def format_score(
    value: float | None,
) -> str:
    """
    Score가 None이면 N/A로,
    숫자이면 소수점 4자리 문자열로 변환합니다.
    """

    if value is None:
        return "N/A"

    return f"{value:.4f}"


def print_model_result(
    name: str,
    answer: str,
    result: dict[str, Any],
) -> None:
    """
    하나의 모델에 대한 답변과 평가 결과를
    Terminal에 출력합니다.
    """

    print(f"\n[{name}]")
    print(answer)

    keyword = result[
        "keyword"
    ]

    print(
        f"\nKeyword score : "
        f"{format_score(keyword['score'])}"
    )

    print(
        f"Token F1      : "
        f"{result['token_f1']:.4f}"
    )

    print(
        f"Total score   : "
        f"{result['total']:.4f}"
    )

    # 정상적으로 포함된 Keyword 출력
    if keyword["included"]:
        print(
            "Included       : "
            + ", ".join(
                keyword["included"]
            )
        )

    # 누락된 필수 Keyword 출력
    if keyword["missing"]:
        print(
            "Missing        : "
            + ", ".join(
                keyword["missing"]
            )
        )

    # 잘못 포함된 금지 표현 출력
    if keyword["forbidden_found"]:
        print(
            "Forbidden      : "
            + ", ".join(
                keyword["forbidden_found"]
            )
        )

    hallucination = result[
        "hallucination"
    ]

    # Unknown Policy일 때만 Hallucination 결과를 출력합니다.
    if hallucination["applicable"]:
        print(
            f"Hallucination  : "
            f"{hallucination['hallucinated']}"
        )

        print(
            f"Safety score   : "
            f"{hallucination['score']:.4f}"
        )


def print_case_report(
    record: dict[str, Any],
    base_answer: str,
    tuned_answer: str,
    base_result: dict[str, Any],
    tuned_result: dict[str, Any],
    winner: str,
    difference: float,
) -> None:
    """
    하나의 평가 문제에 대한 전체 비교 결과를 출력합니다.

    출력 내용:
    - Question
    - Reference Answer
    - Base Model 답변
    - Fine-tuned Model 답변
    - 각 모델 Score
    - 최종 Winner
    """

    # 이 부분의 = 출력은 Python 코드 구분 주석이 아니라
    # Terminal에서 평가 항목을 보기 쉽게 나누기 위한 출력입니다.
    print("\n" + "=" * 76)

    print(
        f"[{record['id']}] "
        f"{record.get('category', '')}"
    )

    print(
        f"\n[Question]\n"
        f"{record['question']}"
    )

    reference = record.get(
        "reference_answer"
    )

    if reference:
        print(
            f"\n[Reference]\n"
            f"{reference}"
        )

    must_include = record.get(
        "must_include",
        [],
    )

    if must_include:
        print(
            "\n[Must Include]\n"
            + ", ".join(
                must_include
            )
        )

    # Base Model 결과
    print_model_result(
        "Base Model",
        base_answer,
        base_result,
    )

    # Fine-tuned Model 결과
    print_model_result(
        "Fine-tuned LoRA",
        tuned_answer,
        tuned_result,
    )

    print(
        f"\nScore difference : "
        f"{difference:+.4f}"
    )

    print(
        f"[Winner] {winner}"
    )

    print("=" * 76)


# 11. 통계

def average(
    values: list[float],
) -> float:
    """
    여러 평가 Score의 평균을 계산합니다.

    값이 없다면 0.0을 반환합니다.
    """

    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


def update_stats(
    stats: EvaluationStats,
    category: str | None,
    base_result: dict[str, Any],
    tuned_result: dict[str, Any],
    winner: str,
) -> None:
    """
    하나의 평가가 끝날 때마다
    EvaluationStats에 결과를 누적합니다.
    """

    # Known / Unknown Policy 개수 누적
    if category == "unknown_policy":
        stats.unknown_count += 1
    else:
        stats.known_count += 1

    # 승리 횟수 누적
    if winner == "base":
        stats.base_wins += 1

    elif winner == "fine_tuned":
        stats.tuned_wins += 1

    else:
        stats.ties += 1

    base_keyword = (
        base_result[
            "keyword"
        ][
            "score"
        ]
    )

    tuned_keyword = (
        tuned_result[
            "keyword"
        ][
            "score"
        ]
    )

    # Keyword 평가가 존재하는 항목만 평균 계산 대상에 추가합니다.
    if base_keyword is not None:
        stats.base_keyword_scores.append(
            base_keyword
        )

    if tuned_keyword is not None:
        stats.tuned_keyword_scores.append(
            tuned_keyword
        )

    # F1 Score 누적
    stats.base_f1_scores.append(
        base_result[
            "token_f1"
        ]
    )

    stats.tuned_f1_scores.append(
        tuned_result[
            "token_f1"
        ]
    )

    # Total Score 누적
    stats.base_total_scores.append(
        base_result[
            "total"
        ]
    )

    stats.tuned_total_scores.append(
        tuned_result[
            "total"
        ]
    )

    # Hallucination 횟수는 Unknown Policy에서만 계산합니다.
    if category != "unknown_policy":
        return

    if (
        base_result[
            "hallucination"
        ][
            "hallucinated"
        ]
    ):
        stats.base_unknown_hallucinations += 1

    if (
        tuned_result[
            "hallucination"
        ][
            "hallucinated"
        ]
    ):
        stats.tuned_unknown_hallucinations += 1


# 12. 결과 저장

def save_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """
    평가 항목별 상세 결과를 JSONL 형식으로 저장합니다.

    JSONL:
        한 줄에 하나의 JSON Object가 저장되는 형식
    """

    # outputs 디렉토리가 없다면 자동 생성합니다.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        # 각 평가 결과를 한 줄씩 저장합니다.
        for record in records:
            file.write(
                json.dumps(
                    record,

                    # 한글이 \\uXXXX 형태로 변환되지 않도록 설정
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_summary(
    stats: EvaluationStats,
    evaluation_count: int,
    device: torch.device,
) -> dict[str, Any]:
    """
    전체 평가 결과의 평균 및 개선 정도를 계산하여
    하나의 Summary Dictionary를 생성합니다.
    """

    # Base / Fine-tuned 평균 Keyword Score
    base_keyword = average(
        stats.base_keyword_scores
    )

    tuned_keyword = average(
        stats.tuned_keyword_scores
    )

    # 평균 Token F1
    base_f1 = average(
        stats.base_f1_scores
    )

    tuned_f1 = average(
        stats.tuned_f1_scores
    )

    # 평균 Total Score
    base_total = average(
        stats.base_total_scores
    )

    tuned_total = average(
        stats.tuned_total_scores
    )

    # Unknown Policy 중 몇 %에서
    # Hallucination이 발생했는지 계산합니다.
    base_hallucination_rate = (
        stats.base_unknown_hallucinations
        / stats.unknown_count
        if stats.unknown_count
        else 0.0
    )

    tuned_hallucination_rate = (
        stats.tuned_unknown_hallucinations
        / stats.unknown_count
        if stats.unknown_count
        else 0.0
    )

    return {
        "base_model": MODEL_NAME,

        "adapter_path": str(
            ADAPTER_PATH
        ),

        "device": str(
            device
        ),

        "evaluation_count": (
            evaluation_count
        ),

        "known_policy_count": (
            stats.known_count
        ),

        "unknown_policy_count": (
            stats.unknown_count
        ),

        "win_margin": (
            WIN_MARGIN
        ),

        # 두 모델의 승 / 패 / 무승부
        "wins": {
            "base": (
                stats.base_wins
            ),

            "fine_tuned": (
                stats.tuned_wins
            ),

            "tie": (
                stats.ties
            ),
        },

        # Base Model 평균 성능
        "base": {
            "average_keyword_score": round(
                base_keyword,
                4,
            ),

            "average_token_f1": round(
                base_f1,
                4,
            ),

            "average_total_score": round(
                base_total,
                4,
            ),

            "unknown_hallucination_count": (
                stats.base_unknown_hallucinations
            ),

            "unknown_hallucination_rate": round(
                base_hallucination_rate,
                4,
            ),
        },

        # Fine-tuned Model 평균 성능
        "fine_tuned": {
            "average_keyword_score": round(
                tuned_keyword,
                4,
            ),

            "average_token_f1": round(
                tuned_f1,
                4,
            ),

            "average_total_score": round(
                tuned_total,
                4,
            ),

            "unknown_hallucination_count": (
                stats.tuned_unknown_hallucinations
            ),

            "unknown_hallucination_rate": round(
                tuned_hallucination_rate,
                4,
            ),
        },

        # Fine-tuned - Base 차이
        #
        # 양수:
        # Fine-tuned Model 증가
        #
        # Hallucination Rate는 반대로
        # 음수가 되어야 개선된 것입니다.
        "improvement": {
            "keyword_score": round(
                tuned_keyword
                - base_keyword,
                4,
            ),

            "token_f1": round(
                tuned_f1
                - base_f1,
                4,
            ),

            "total_score": round(
                tuned_total
                - base_total,
                4,
            ),

            "hallucination_rate": round(
                tuned_hallucination_rate
                - base_hallucination_rate,
                4,
            ),
        },
    }


def save_summary(
    summary: dict[str, Any],
) -> None:
    """
    전체 평가 Summary를 JSON 파일로 저장합니다.
    """

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,

            # 사람이 읽기 쉽도록 들여쓰기 적용
            indent=2,
        ),
        encoding="utf-8",
    )


# 13. 최종 Summary 출력

def print_summary(
    summary: dict[str, Any],
) -> None:
    """
    저장된 Summary의 주요 결과를
    Terminal에서도 바로 확인할 수 있도록 출력합니다.
    """

    wins = summary["wins"]
    base = summary["base"]
    tuned = summary["fine_tuned"]

    improvement = summary[
        "improvement"
    ]

    print("\n[Evaluation Summary]")

    print(
        f"Evaluation items  : "
        f"{summary['evaluation_count']}"
    )

    print(
        f"Known policies    : "
        f"{summary['known_policy_count']}"
    )

    print(
        f"Unknown policies  : "
        f"{summary['unknown_policy_count']}"
    )

    print(
        f"Win margin        : "
        f"{summary['win_margin']:.4f}"
    )

    print(
        f"\nBase wins         : "
        f"{wins['base']}"
    )

    print(
        f"Fine-tuned wins   : "
        f"{wins['fine_tuned']}"
    )

    print(
        f"Ties              : "
        f"{wins['tie']}"
    )

    print("\n[Base Model]")

    print(
        f"Keyword score     : "
        f"{base['average_keyword_score']}"
    )

    print(
        f"Token F1          : "
        f"{base['average_token_f1']}"
    )

    print(
        f"Total score       : "
        f"{base['average_total_score']}"
    )

    print(
        f"Hallucination     : "
        f"{base['unknown_hallucination_count']}"
        f"/{summary['unknown_policy_count']}"
    )

    print("\n[Fine-tuned Model]")

    print(
        f"Keyword score     : "
        f"{tuned['average_keyword_score']}"
    )

    print(
        f"Token F1          : "
        f"{tuned['average_token_f1']}"
    )

    print(
        f"Total score       : "
        f"{tuned['average_total_score']}"
    )

    print(
        f"Hallucination     : "
        f"{tuned['unknown_hallucination_count']}"
        f"/{summary['unknown_policy_count']}"
    )

    print("\n[Improvement]")

    # +라면 Fine-Tuning 후 Keyword Score 개선
    print(
        f"Keyword           : "
        f"{improvement['keyword_score']:+.4f}"
    )

    # +라면 Fine-Tuning 후 Reference Answer와
    # 더 유사한 답변을 생성했다는 의미
    print(
        f"Token F1          : "
        f"{improvement['token_f1']:+.4f}"
    )

    # +라면 전체 성능 향상
    print(
        f"Total             : "
        f"{improvement['total_score']:+.4f}"
    )

    # Hallucination은 낮아져야 좋으므로
    # 이 값은 음수일수록 개선된 것입니다.
    print(
        f"Hallucination     : "
        f"{improvement['hallucination_rate']:+.4f}"
    )

    print(
        f"\nDetailed results  : "
        f"{RESULT_PATH}"
    )

    print(
        f"Summary           : "
        f"{SUMMARY_PATH}"
    )


# 14. 메모리 정리

def cleanup_runtime(
    device: torch.device,
) -> None:
    """
    모델 평가 종료 후 사용했던 메모리를 정리합니다.
    """

    # Python에서 참조되지 않는 객체 정리
    gc.collect()

    # NVIDIA CUDA Cache 정리
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Apple MPS Cache 정리
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()

        except (
            AttributeError,
            RuntimeError,
        ):
            pass


# 15. 실행 옵션

def parse_args() -> argparse.Namespace:
    """
    Terminal에서 전달되는 실행 옵션을 정의합니다.

    기본:
        python scripts/evaluate_model.py

    최대 10개만 평가:
        python scripts/evaluate_model.py --max-evaluation 10
    """

    parser = argparse.ArgumentParser(
        description=(
            "Base Qwen과 Fine-tuned LoRA "
            "모델을 비교 평가합니다."
        )
    )

    # 평가할 최대 Dataset 개수
    parser.add_argument(
        "--max-evaluation",
        type=int,
        default=DEFAULT_MAX_EVALUATION,
        help="최대 평가 항목 수",
    )

    return parser.parse_args()


# 16. 실행

def main() -> int:
    """
    Base Model과 Fine-tuned Model의
    전체 비교 평가 과정을 순서대로 실행합니다.
    """

    # Terminal 실행 옵션을 가져옵니다.
    args = parse_args()

    if args.max_evaluation < 1:
        raise ValueError(
            "--max-evaluation은 1 이상이어야 합니다."
        )

    # Fine-Tuning 결과인 Adapter가
    # 실제로 존재하는지 먼저 확인합니다.
    validate_adapter()

    # CUDA / MPS / CPU 결정
    device = get_device()

    # 장치에 따라 모델 dtype 결정
    dtype = get_dtype(
        device
    )

    # 실제 평가 환경 출력
    print("[Evaluation Runtime]")

    print(
        f"Base model     : "
        f"{MODEL_NAME}"
    )

    print(
        f"Adapter        : "
        f"{ADAPTER_PATH}"
    )

    print(
        f"Device         : "
        f"{device}"
    )

    print(
        f"Dtype          : "
        f"{dtype}"
    )

    print(
        f"Max evaluation : "
        f"{args.max_evaluation}"
    )

    print(
        f"Win margin     : "
        f"{WIN_MARGIN}"
    )


    # Qwen Tokenizer를 불러옵니다.
    #
    # Base / Fine-tuned Model 모두
    # 동일한 Tokenizer를 사용합니다.
    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )
    )

    # PAD Token이 별도로 없다면
    # EOS Token을 대신 사용합니다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    print(
        "\n[Model] Loading Base Qwen"
    )

    # Fine-Tuning 이전의 원본 Qwen 모델을 로드합니다.
    #
    # 동일한 MODEL_NAME을 사용하므로
    # Hugging Face Cache에 이미 모델이 있다면 재사용합니다.
    base_model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        .to(
            device
        )
    )

    # 평가에서는 Dropout 등 학습용 동작이 필요하지 않으므로
    # Evaluation Mode로 변경합니다.
    base_model.eval()


    print(
        "[Model] Loading LoRA Adapter"
    )

    # Base Qwen 위에 Fine-Tuning으로 학습한
    # LoRA Adapter를 연결합니다.
    #
    # 실제로 Base Model 전체를 하나 더 로드하는 것이 아니라
    # 같은 Base Model에 작은 Adapter를 추가합니다.
    tuned_model = (
        PeftModel
        .from_pretrained(
            base_model,

            str(
                ADAPTER_PATH
            ),

            # 평가 전용이므로 Adapter를 다시 학습하지 않습니다.
            is_trainable=False,
        )
    )

    tuned_model.eval()

    print(
        "[Model] LoRA Adapter loaded"
    )


    # 평가 Dataset을 읽습니다.
    evaluation_records = read_jsonl(
        EVALUATION_PATH,

        max_records=(
            args.max_evaluation
        ),
    )

    print(
        f"[Dataset] Evaluation records: "
        f"{len(evaluation_records)}"
    )


    # 전체 평가 통계 저장 객체
    stats = EvaluationStats()

    # 평가 항목별 상세 결과 저장
    results: list[
        dict[str, Any]
    ] = []


    try:
        total_items = len(
            evaluation_records
        )

        # 평가 Dataset을 하나씩 순회합니다.
        for index, record in enumerate(
            evaluation_records,
            start=1,
        ):
            question = record[
                "question"
            ]

            category = record.get(
                "category"
            )

            print(
                f"\n[Progress] "
                f"{index}/{total_items}"
            )


            # Base Model 평가
            #
            # tuned_model 안에는 Base Qwen + LoRA Adapter가 함께 존재합니다.
            #
            # disable_adapter()를 사용하면 LoRA를 잠시 비활성화하여
            # 원본 Base Qwen의 응답을 얻을 수 있습니다.
            #
            # 따라서 Base Model을 별도로 하나 더 메모리에
            # 올리지 않고도 공정하게 비교할 수 있습니다.
            with tuned_model.disable_adapter():
                base_answer = generate_answer(
                    tuned_model,
                    tokenizer,
                    device,
                    question,
                )


            # Fine-tuned Model 평가
            #
            # disable_adapter() 블록을 빠져나오면
            # LoRA Adapter가 다시 활성화됩니다.
            tuned_answer = generate_answer(
                tuned_model,
                tokenizer,
                device,
                question,
            )


            # Base Model 답변 평가
            base_result = evaluate_answer(
                base_answer,
                record,
            )


            # Fine-tuned Model 답변 평가
            tuned_result = evaluate_answer(
                tuned_answer,
                record,
            )


            # 두 모델의 Total Score를 비교하여
            # 승자를 결정합니다.
            winner, difference = (
                determine_winner(
                    base_result[
                        "total"
                    ],
                    tuned_result[
                        "total"
                    ],
                )
            )


            # 전체 통계에 현재 평가 결과를 추가합니다.
            update_stats(
                stats,
                category,
                base_result,
                tuned_result,
                winner,
            )


            # 나중에 JSONL로 저장할
            # 현재 평가 항목의 상세 결과입니다.
            results.append(
                {
                    "id": (
                        record["id"]
                    ),

                    "category": (
                        category
                    ),

                    "question": (
                        question
                    ),

                    "reference_answer": (
                        record.get(
                            "reference_answer"
                        )
                    ),

                    "must_include": (
                        record.get(
                            "must_include",
                            [],
                        )
                    ),

                    "must_not_include": (
                        record.get(
                            "must_not_include",
                            [],
                        )
                    ),

                    "base_answer": (
                        base_answer
                    ),

                    "tuned_answer": (
                        tuned_answer
                    ),

                    "base_score": (
                        base_result
                    ),

                    "tuned_score": (
                        tuned_result
                    ),

                    "score_difference": (
                        difference
                    ),

                    "winner": (
                        winner
                    ),
                }
            )


            # 현재 평가 항목의 상세 비교 결과를
            # Terminal에 바로 출력합니다.
            print_case_report(
                record,
                base_answer,
                tuned_answer,
                base_result,
                tuned_result,
                winner,
                difference,
            )


        # 모든 평가 항목의 상세 결과를 JSONL로 저장합니다.
        save_jsonl(
            RESULT_PATH,
            results,
        )


        # 전체 평가 결과 평균과 개선 정도를 계산합니다.
        summary = build_summary(
            stats,

            len(
                evaluation_records
            ),

            device,
        )


        # Summary JSON 저장
        save_summary(
            summary
        )


        # Terminal에도 Summary 출력
        print_summary(
            summary
        )


        print(
            "\n[Evaluation] Complete"
        )

        # 정상 종료
        return 0


    finally:
        # 평가 성공/실패 여부와 관계없이
        # 대형 모델 객체를 메모리에서 제거합니다.
        del tuned_model
        del base_model
        del tokenizer

        # CUDA / MPS 등의 Cache 정리
        cleanup_runtime(
            device
        )

        print(
            "[Evaluation] Runtime resources released"
        )


# 이 파일을 직접 실행했을 때만 main()을 실행합니다.
#
# python scripts/evaluate_model.py
#
# 다른 Python 코드에서 import하는 경우에는
# 평가가 자동으로 시작되지 않습니다.
if __name__ == "__main__":
    raise SystemExit(
        main()
    )