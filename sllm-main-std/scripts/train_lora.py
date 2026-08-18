"""
Qwen2.5-1.5B-Instruct 환경별 LoRA / QLoRA SFT 학습

환경별 정책:
- NVIDIA CUDA
    -> 기본: bitsandbytes 4bit NF4 QLoRA
    -> USE_QLORA=False이면 FP16 LoRA

- Apple Silicon MPS
    -> FP32 LoRA
    -> 속도보다 학습 안정성을 우선

- CPU
    -> FP32 LoRA

학습 데이터:
    dataset/training/hr_sft_train.jsonl
    dataset/training/hr_sft_validation.jsonl

출력:
    models/hr-qwen-lora/

기본 교육용 설정:
- Epochs: 3
- Max Length: 512
- Gradient Accumulation: 8
- LoRA Rank: 16
- LoRA Alpha: 32

실행:
    python scripts/train_lora.py

Mac 16GB 예:
    SFT_MAX_LENGTH=512 \
    SFT_EPOCHS=3 \
    SFT_LEARNING_RATE=1e-4 \
    SFT_GRAD_ACCUM=8 \
    python scripts/train_lora.py
"""


# 초보자를 위한 코드 읽기 가이드
#
# 이 파일은 위에서 아래까지 모든 코드가
# 차례대로 실행되는 구조가 아닙니다.
#
# 위쪽에서는 Fine-Tuning에 필요한 설정과 함수들을 먼저 정의하고,
# 파일 아래쪽의 main() 함수에서 필요한 함수들을
# 순서대로 호출합니다.
#
# 따라서 처음부터 모든 함수를 읽기보다는
# 먼저 main()을 확인하여 전체 학습 흐름을 파악한 뒤,
# main()에서 호출되는 함수의 정의를 찾아가며 확인하는 것을 권장합니다.
#
#
# 추천 코드 읽기 순서
#
# main()
#   │
#   ├─ parse_args()
#   │    → Train / Validation Dataset과 출력 경로 확인
#   │
#   ├─ detect_device_type()
#   │    → CUDA / MPS / CPU 중 어떤 장치에서 학습할지 결정
#   │
#   ├─ create_training_profile()
#   │    → 실행 장치에 맞는 LoRA / QLoRA 학습 환경 설정
#   │      Epoch, Learning Rate, Max Length,
#   │      Gradient Accumulation 등의 학습 설정도 결정
#   │
#   ├─ load_training_dataset()
#   │    → Train / Validation Dataset을 읽고 형식 검증
#   │      │
#   │      ├─ validate_jsonl_messages()
#   │      │    → JSONL 파일과 각 Record의 형식 확인
#   │      │
#   │      └─ validate_messages()
#   │           → system / user / assistant 메시지 구조 확인
#   │
#   ├─ load_tokenizer()
#   │    → Base Qwen에서 사용하는 Tokenizer 로드
#   │
#   ├─ inspect_training_sample()
#   │    → Dataset의 messages가 Qwen Chat Template을 통해
#   │      실제 어떤 학습 입력으로 변환되는지 확인
#   │
#   ├─ load_model()
#   │    → Qwen2.5-1.5B-Instruct Base Model 로드
#   │
#   ├─ create_lora_config()
#   │    → LoRA를 적용할 Layer와
#   │      Rank / Alpha / Dropout 설정
#   │
#   ├─ create_sft_config()
#   │    → Epoch / Batch Size / Learning Rate 등
#   │      실제 SFT 학습 옵션 설정
#   │
#   ├─ SFTTrainer(...)
#   │    → Model / Dataset / Tokenizer / LoRA 설정을 결합하여
#   │      Fine-Tuning을 수행할 Trainer 생성
#   │
#   ├─ trainer.train()
#   │    → 실제 Fine-Tuning 실행
#   │
#   ├─ trainer.save_model()
#   │    → 학습된 LoRA Adapter 저장
#   │
#   ├─ save_training_log()
#   │    → Step / Epoch별 Loss 등의 학습 기록 저장
#   │
#   ├─ save_training_summary()
#   │    → 최종 Loss, Best Checkpoint 등의 학습 결과 요약
#   │
#   ├─ save_training_metadata()
#   │    → Base Model, PyTorch Version, 학습 환경 등의 정보 저장
#   │
#   └─ cleanup_memory()
#        → 학습 종료 후 사용한 메모리 정리
#
#
# 핵심적으로 먼저 볼 부분
#
# 1. main()
#    → 전체 Fine-Tuning 프로그램의 실행 순서
#
# 2. create_training_profile()
#    → 실행 환경에 따라 학습 방식을 어떻게 결정하는지 확인
#
# 3. load_training_dataset()
#    → 어떤 데이터를 학습에 사용하는지 확인
#
# 4. load_model()
#    → Base Qwen을 어떤 방식으로 불러오는지 확인
#
# 5. create_lora_config()
#    → Base Model의 어디에 LoRA를 적용하는지 확인
#
# 6. create_sft_config()
#    → 실제 학습 조건을 어떻게 설정하는지 확인
#
# 7. SFTTrainer(...) / trainer.train()
#    → 설정한 Model과 Dataset으로 실제 Fine-Tuning 수행
#
#
# LoRA 학습의 핵심 흐름
#
# Base Qwen
#     ↓
# load_model()
#     ↓
# create_lora_config()
#     ↓
# SFTTrainer(...)
#     ↓
# trainer.train()
#     ↓
# LoRA Adapter
#
# 이 학습에서는 Base Qwen 전체 Parameter를 새로 학습하는 것이 아니라,
# create_lora_config()에서 지정한 Layer에 LoRA Adapter를 추가하고
# 해당 Adapter Parameter를 중심으로 학습합니다.
#
#
# 처음에는 다음 보조 함수까지 모두 이해할 필요는 없습니다.
#
# parse_bool_env()
# get_env_int()
# get_env_float()
# get_model_dtype()
# resolve_project_path()
# parse_args()
# save_training_log()
# save_training_summary()
# save_training_metadata()
# cleanup_memory()
#
# 이러한 함수들은 환경변수 처리, 경로 처리,
# 결과 저장, 메모리 정리 등 학습 프로그램을 보조하는 역할을 합니다.
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

import argparse
import gc
import json
import os
import platform

# 명령행에서 전달되는 --train, --validation 등의 옵션을 처리합니다.
import argparse

# 학습 종료 후 Python 메모리를 정리할 때 사용합니다.
import gc

# JSONL 데이터 확인 및 학습 결과 저장에 사용합니다.
import json

# 환경 변수(Environment Variable)를 읽을 때 사용합니다.
import os

# 운영체제 및 CPU 아키텍처 정보를 확인합니다.
import platform

# 학습 설정을 하나의 객체로 관리하기 위해 사용합니다.
from dataclasses import asdict, dataclass

# 운영체제에 관계없이 파일 경로를 다루기 위해 사용합니다.
from pathlib import Path

# 타입 힌트에 사용합니다.
from typing import Literal, cast

# PyTorch: 모델 학습 및 CUDA/MPS/CPU 장치 제어
import torch

# Hugging Face Dataset 로드
from datasets import DatasetDict, load_dataset

# PEFT의 LoRA 설정과 QLoRA 학습 준비
from peft import LoraConfig, prepare_model_for_kbit_training

# Hugging Face 모델 및 Tokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

# Supervised Fine-Tuning을 위한 Trainer
from trl import SFTConfig, SFTTrainer


# 1. 기본 설정

# 현재 파일(train_lora.py)의 위치를 기준으로
# 프로젝트의 최상위 디렉토리를 찾습니다.
#
# 예:
# project/
# ├── dataset/
# ├── models/
# └── scripts/
#     └── train_lora.py
#
# 현재 파일의 parent      -> scripts/
# parent.parent           -> project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Fine-Tuning에 사용할 Base Model입니다.
#
# MODEL_NAME이라는 환경 변수가 존재하면 해당 값을 사용하고,
# 없으면 Qwen2.5-1.5B-Instruct를 기본 모델로 사용합니다.
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "Qwen/Qwen2.5-1.5B-Instruct",
)


# 학습용 Dataset의 기본 위치
DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "training"
    / "hr_sft_train.jsonl"
)

# 검증용 Dataset의 기본 위치
DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "training"
    / "hr_sft_validation.jsonl"
)

# Fine-Tuning이 끝난 LoRA Adapter 저장 위치
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "hr-qwen-lora"
)

# 학습 과정의 loss, eval_loss 등의 전체 기록 저장 위치
TRAINING_LOG_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training_log.json"
)

# 최종 학습 결과 요약 정보 저장 위치
TRAINING_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "training_summary.json"
)

# 동일한 조건에서 최대한 비슷한 학습 결과를 얻기 위한 Random Seed
RANDOM_SEED = 42

# 학습 장치는 cuda, mps, cpu 중 하나만 사용하도록 타입을 제한합니다.
DeviceType = Literal[
    "cuda",
    "mps",
    "cpu",
]


# 2. 학습 설정

def parse_bool_env(
    name: str,
    default: bool,
) -> bool:
    """
    환경 변수로 전달된 문자열을 True / False로 변환합니다.

    예:
        USE_QLORA=true
        USE_QLORA=1
        USE_QLORA=yes

    위 값들은 모두 True로 처리됩니다.
    """

    # 환경 변수가 없다면 default 값을 문자열로 변환하여 사용합니다.
    value = os.getenv(
        name,
        str(default),
    )

    # 환경 변수는 문자열이므로 소문자로 변환한 뒤
    # True를 의미하는 값인지 확인합니다.
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def get_env_int(
    name: str,
    default: int,
) -> int:
    """
    환경 변수 값을 정수(int)로 읽습니다.

    예:
        SFT_MAX_LENGTH=512

    환경 변수가 없다면 default 값을 사용합니다.
    """

    try:
        return int(
            os.getenv(
                name,
                str(default),
            )
        )

    # "abc"처럼 정수로 바꿀 수 없는 값이 들어오면
    # 학습을 시작하기 전에 명확한 오류를 발생시킵니다.
    except ValueError as exc:
        raise ValueError(
            f"{name}은 정수여야 합니다."
        ) from exc


def get_env_float(
    name: str,
    default: float,
) -> float:
    """
    환경 변수 값을 실수(float)로 읽습니다.

    예:
        SFT_LEARNING_RATE=0.0001
    """

    try:
        return float(
            os.getenv(
                name,
                str(default),
            )
        )

    except ValueError as exc:
        raise ValueError(
            f"{name}은 숫자여야 합니다."
        ) from exc


@dataclass(frozen=True)
class TrainingProfile:
    """
    Fine-Tuning에 필요한 여러 설정을
    하나의 TrainingProfile 객체로 관리합니다.

    frozen=True
    -> 객체를 만든 이후 설정값이 실수로 변경되는 것을 방지합니다.
    """

    # 실행 장치: cuda / mps / cpu
    device_type: DeviceType

    # 사람이 확인하기 위한 현재 학습 방식 이름
    training_mode: str

    # Base Model의 데이터 타입
    model_dtype_name: str

    # QLoRA 사용 여부
    use_qlora: bool

    # 한 Sample에서 사용할 최대 Token 길이
    max_length: int

    # 한 번에 처리하는 학습 Sample 수
    train_batch_size: int

    # 한 번에 처리하는 Validation Sample 수
    eval_batch_size: int

    # 몇 Step의 Gradient를 모아서 한 번 업데이트할지 지정
    gradient_accumulation_steps: int

    # 메모리를 절약하기 위한 Gradient Checkpointing 사용 여부
    gradient_checkpointing: bool

    # Dataset 전체를 몇 번 반복 학습할지 지정
    epochs: float

    # 한 번의 학습에서 Parameter를 얼마나 변경할지 지정
    learning_rate: float

    # LoRA 설정
    lora_rank: int
    lora_alpha: int
    lora_dropout: float


def detect_device_type() -> DeviceType:
    """
    현재 컴퓨터에서 사용할 수 있는 학습 장치를 자동으로 확인합니다.

    우선순위:
        NVIDIA CUDA -> Apple MPS -> CPU
    """

    # NVIDIA GPU와 CUDA를 사용할 수 있는 경우
    if torch.cuda.is_available():
        return "cuda"

    # Apple Silicon의 GPU(MPS)를 사용할 수 있는 경우
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    # GPU를 사용할 수 없다면 CPU 사용
    return "cpu"


def create_training_profile(
    device_type: DeviceType,
) -> TrainingProfile:
    """
    실행 환경에 따라 실제 학습 설정을 생성합니다.

    기본 설정은 코드에 존재하지만,
    환경 변수로 원하는 값만 변경할 수 있습니다.

    예:
        SFT_MAX_LENGTH=512
        SFT_EPOCHS=3
        SFT_GRAD_ACCUM=8
    """

    # 한 학습 Sample에서 사용할 최대 Token 수
    max_length = get_env_int(
        "SFT_MAX_LENGTH",
        512,
    )

    # 전체 Dataset 반복 횟수
    epochs = get_env_float(
        "SFT_EPOCHS",
        3.0,
    )

    # 몇 번의 Batch를 모아서 Parameter를 업데이트할지 지정
    gradient_accumulation = get_env_int(
        "SFT_GRAD_ACCUM",
        8,
    )

    # LoRA의 Rank
    #
    # Rank가 커지면 학습 가능한 LoRA Parameter가 증가합니다.
    lora_rank = get_env_int(
        "SFT_LORA_RANK",
        16,
    )

    # LoRA 변화량의 반영 강도와 관련된 값
    lora_alpha = get_env_int(
        "SFT_LORA_ALPHA",
        32,
    )

    # LoRA 학습 과정에서 적용할 Dropout 비율
    lora_dropout = get_env_float(
        "SFT_LORA_DROPOUT",
        0.05,
    )

    # 너무 짧은 입력 길이를 실수로 사용하는 것을 방지합니다.
    if max_length < 128:
        raise ValueError(
            "SFT_MAX_LENGTH는 최소 128 이상이어야 합니다."
        )

    if epochs <= 0:
        raise ValueError(
            "SFT_EPOCHS는 0보다 커야 합니다."
        )

    if gradient_accumulation < 1:
        raise ValueError(
            "SFT_GRAD_ACCUM은 최소 1 이상이어야 합니다."
        )

    # NVIDIA GPU 환경
    if device_type == "cuda":

        # CUDA에서는 기본적으로 QLoRA를 사용합니다.
        #
        # USE_QLORA=False를 지정하면
        # 4bit QLoRA 대신 FP16 LoRA를 사용합니다.
        use_qlora = parse_bool_env(
            "USE_QLORA",
            True,
        )

        # QLoRA와 일반 LoRA의 기본 Learning Rate를 다르게 설정합니다.
        default_lr = (
            2e-4
            if use_qlora
            else 1e-4
        )

        return TrainingProfile(
            device_type="cuda",

            training_mode=(
                "QLoRA 4bit NF4 CUDA"
                if use_qlora
                else "FP16 LoRA CUDA"
            ),

            model_dtype_name="float16",
            use_qlora=use_qlora,

            max_length=max_length,

            train_batch_size=1,
            eval_batch_size=1,

            gradient_accumulation_steps=(
                gradient_accumulation
            ),

            gradient_checkpointing=True,

            epochs=epochs,

            learning_rate=get_env_float(
                "SFT_LEARNING_RATE",
                default_lr,
            ),

            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    # Apple Silicon GPU 환경
    if device_type == "mps":

        # MPS에서는 bitsandbytes 기반 QLoRA를 사용하지 않고
        # FP32 LoRA로 학습합니다.
        return TrainingProfile(
            device_type="mps",
            training_mode="FP32 LoRA Apple MPS",
            model_dtype_name="float32",
            use_qlora=False,

            max_length=max_length,

            train_batch_size=1,
            eval_batch_size=1,

            gradient_accumulation_steps=(
                gradient_accumulation
            ),

            gradient_checkpointing=True,

            epochs=epochs,

            learning_rate=get_env_float(
                "SFT_LEARNING_RATE",
                1e-4,
            ),

            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    # CUDA와 MPS를 사용할 수 없는 경우 CPU 사용
    #
    # CPU에서는 속도와 메모리를 고려해
    # max_length를 최대 256으로 제한합니다.
    return TrainingProfile(
        device_type="cpu",
        training_mode="FP32 LoRA CPU",
        model_dtype_name="float32",
        use_qlora=False,

        max_length=min(
            max_length,
            256,
        ),

        train_batch_size=1,
        eval_batch_size=1,

        gradient_accumulation_steps=(
            gradient_accumulation
        ),

        gradient_checkpointing=True,

        epochs=epochs,

        learning_rate=get_env_float(
            "SFT_LEARNING_RATE",
            1e-4,
        ),

        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )


def get_model_dtype(
    profile: TrainingProfile,
) -> torch.dtype:
    """
    TrainingProfile에 지정된 문자열을
    실제 PyTorch dtype으로 변환합니다.
    """

    if profile.model_dtype_name == "float16":
        return torch.float16

    return torch.float32


# 3. 데이터셋 검증

def validate_messages(
    messages: object,
    path: Path,
    line_number: int,
) -> None:
    """
    JSONL 한 줄의 messages가
    SFT 학습에 사용할 수 있는 형식인지 검사합니다.

    정상 예:

    {
        "messages": [
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답변"}
        ]
    }
    """

    # messages는 여러 개의 대화를 담아야 하므로 list여야 합니다.
    if not isinstance(messages, list):
        raise ValueError(
            f"{path.name} {line_number}번째 줄에 "
            "messages 배열이 없습니다."
        )

    # 최소 user + assistant 정도의 대화가 있어야 합니다.
    if len(messages) < 2:
        raise ValueError(
            f"{path.name} {line_number}번째 줄의 "
            "messages가 너무 짧습니다."
        )

    roles: list[str] = []

    # messages 안에 들어 있는 각각의 message를 검사합니다.
    for message in messages:

        # 하나의 message는 JSON Object(dict)여야 합니다.
        if not isinstance(message, dict):
            raise ValueError(
                f"{path.name} {line_number}번째 줄의 "
                "message 형식이 잘못되었습니다."
            )

        role = message.get("role")
        content = message.get("content")

        # 이번 Dataset에서 허용하는 역할만 사용합니다.
        if role not in {
            "system",
            "user",
            "assistant",
        }:
            raise ValueError(
                f"{path.name} {line_number}번째 줄에 "
                f"지원하지 않는 role이 있습니다: {role}"
            )

        # content가 문자열이 아니거나 비어 있으면 학습할 수 없습니다.
        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError(
                f"{path.name} {line_number}번째 줄의 "
                f"{role} content가 비어 있습니다."
            )

        roles.append(role)

    # 최소한 질문 역할인 user가 존재해야 합니다.
    if "user" not in roles:
        raise ValueError(
            f"{path.name} {line_number}번째 줄에 "
            "user 메시지가 없습니다."
        )

    # 학습 정답인 assistant 메시지가 반드시 필요합니다.
    if "assistant" not in roles:
        raise ValueError(
            f"{path.name} {line_number}번째 줄에 "
            "assistant 메시지가 없습니다."
        )

    # 마지막 assistant 메시지를 정답으로 학습시키기 위해
    # 마지막 role은 assistant여야 합니다.
    if roles[-1] != "assistant":
        raise ValueError(
            f"{path.name} {line_number}번째 줄의 "
            "마지막 메시지는 assistant여야 합니다."
        )


def validate_jsonl_messages(
    path: Path,
) -> None:
    """
    JSONL 파일 전체를 한 줄씩 읽으면서
    JSON 형식과 messages 구조를 검사합니다.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"데이터셋을 찾을 수 없습니다: {path}"
        )

    valid_count = 0

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        # enumerate(..., start=1)을 사용해
        # 오류 발생 시 실제 파일의 줄 번호를 알려줍니다.
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            # 빈 줄은 무시합니다.
            if not line:
                continue

            try:
                # JSON 문자열을 Python dictionary로 변환합니다.
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name} {line_number}번째 줄의 "
                    "JSON 형식이 잘못되었습니다."
                ) from exc

            # JSON 자체뿐 아니라 messages 내부 구조도 검사합니다.
            validate_messages(
                record.get("messages"),
                path,
                line_number,
            )

            valid_count += 1

    if valid_count == 0:
        raise ValueError(
            f"유효한 데이터가 없습니다: {path}"
        )


def load_training_dataset(
    train_path: Path,
    validation_path: Path,
) -> DatasetDict:
    """
    Train / Validation JSONL 파일을 검증한 뒤
    Hugging Face Dataset 형식으로 로드합니다.
    """

    # 문제가 있는 데이터를 모델 로딩 전에 먼저 차단합니다.
    validate_jsonl_messages(
        train_path
    )

    validate_jsonl_messages(
        validation_path
    )

    # 두 JSONL 파일을 하나의 DatasetDict로 구성합니다.
    #
    # dataset["train"]
    # dataset["validation"]
    #
    # 형태로 접근할 수 있습니다.
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(validation_path),
        },
    )

    print(
        f"[Dataset] Train records      : "
        f"{len(dataset['train'])}"
    )

    print(
        f"[Dataset] Validation records : "
        f"{len(dataset['validation'])}"
    )

    return cast(
        DatasetDict,
        dataset,
    )


# 4. Tokenizer와 실제 학습 입력 확인

def load_tokenizer() -> PreTrainedTokenizerBase:
    """
    Qwen Base Model에 맞는 Tokenizer를 로드합니다.

    Base Model과 동일한 MODEL_NAME을 사용하므로
    기존 Hugging Face 캐시가 있으면 재사용됩니다.
    """

    print(
        f"[Tokenizer] Loading: "
        f"{MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    # 일부 모델은 별도의 PAD Token이 없을 수 있습니다.
    #
    # 이 경우 EOS Token을 PAD Token으로 사용하도록 설정합니다.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # 오른쪽에 Padding Token을 추가하도록 설정합니다.
    tokenizer.padding_side = "right"

    print(
        f"[Tokenizer] EOS token : "
        f"{tokenizer.eos_token}"
    )

    print(
        f"[Tokenizer] PAD token : "
        f"{tokenizer.pad_token}"
    )

    return tokenizer


def inspect_training_sample(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> None:
    """
    첫 번째 학습 Sample이 실제 Qwen 입력 문자열로
    어떻게 변환되는지 확인합니다.

    Dataset 형식이 올바르더라도 Chat Template 적용 결과가
    예상과 다를 수 있기 때문에 학습 전에 확인합니다.
    """

    # 첫 번째 Train Record를 확인용 Sample로 사용합니다.
    sample = dataset["train"][0]

    messages = sample["messages"]

    # Qwen의 Chat Template을 적용하여
    # system/user/assistant 구조를 실제 문자열로 변환합니다.
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # 실제 학습 시 몇 개의 Token이 생성되는지 확인합니다.
    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
    )

    token_count = len(
        tokenized["input_ids"]
    )

    # 출력 확인을 위해 첫 번째 user 메시지를 찾습니다.
    user_message = next(
        (
            message["content"]
            for message in messages
            if message["role"] == "user"
        ),
        "",
    )

    # 마지막 assistant 응답을 학습 정답으로 확인합니다.
    assistant_message = next(
        (
            message["content"]
            for message in reversed(messages)
            if message["role"] == "assistant"
        ),
        "",
    )

    print("\n[Dataset] Training sample check")

    print(
        f"Rendered tokens : "
        f"{token_count}"
    )

    print(
        f"Max length      : "
        f"{max_length}"
    )

    print(
        f"Truncated       : "
        f"{token_count > max_length}"
    )

    print(
        f"User            : "
        f"{user_message[:160]}"
    )

    print(
        f"Assistant       : "
        f"{assistant_message[:240]}"
    )

    # 입력 Token 수가 max_length보다 길면
    # 학습 과정에서 일부 내용이 잘릴 수 있습니다.
    if token_count > max_length:
        print(
            "[Warning] 첫 번째 학습 샘플이 "
            "SFT_MAX_LENGTH를 초과합니다."
        )


# 5. Base Model 로드

def load_model(
    profile: TrainingProfile,
) -> PreTrainedModel:
    """
    CUDA / MPS / CPU 환경에 맞게
    Qwen Base Model을 로드합니다.
    """

    dtype = get_model_dtype(
        profile
    )

    # CUDA + QLoRA
    #
    # Base Model을 4bit로 양자화하여
    # GPU 메모리 사용량을 크게 줄입니다.
    if (
        profile.device_type == "cuda"
        and profile.use_qlora
    ):
        print(
            "[Model] Loading CUDA "
            "4bit NF4 QLoRA model"
        )

        # bitsandbytes의 4bit 양자화 방식을 설정합니다.
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,

            # NF4는 QLoRA에서 일반적으로 사용하는
            # 4bit Quantization 형식입니다.
            bnb_4bit_quant_type="nf4",

            # 실제 연산은 FP16으로 수행합니다.
            bnb_4bit_compute_dtype=torch.float16,

            # Quantization 관련 값을 다시 양자화해
            # 추가로 메모리를 절약합니다.
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,

            # 사용 가능한 GPU에 모델을 자동 배치합니다.
            device_map="auto",

            # 모델 로딩 과정에서 CPU RAM 사용량을 줄입니다.
            low_cpu_mem_usage=True,
        )

        # Quantized Model이 LoRA 학습을 수행할 수 있도록
        # k-bit Training을 위한 준비 작업을 수행합니다.
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=(
                profile.gradient_checkpointing
            ),
        )

    # CUDA + 일반 FP16 LoRA
    elif profile.device_type == "cuda":
        print(
            "[Model] Loading CUDA "
            "FP16 LoRA model"
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # 모델을 NVIDIA GPU로 이동합니다.
        model.to(
            torch.device("cuda:0")
        )

    # Apple Silicon MPS
    elif profile.device_type == "mps":
        print(
            "[Model] Loading Apple MPS "
            "FP32 LoRA model"
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,

            # 교육 환경에서 안정성을 우선하여 FP32 사용
            dtype=torch.float32,

            low_cpu_mem_usage=True,
        )

        # 모델을 Apple GPU로 이동합니다.
        model.to(
            torch.device("mps")
        )

    # CPU
    else:
        print(
            "[Model] Loading CPU "
            "FP32 LoRA model"
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        model.to(
            torch.device("cpu")
        )

    # 학습 시에는 이전 Attention 결과를 저장하는
    # KV Cache가 필요하지 않으므로 비활성화합니다.
    #
    # Gradient Checkpointing과 함께 사용할 때도
    # use_cache=False가 필요합니다.
    model.config.use_cache = False

    # Gradient Checkpointing 환경에서 입력 Tensor에
    # Gradient가 필요하도록 설정합니다.
    if (
        profile.gradient_checkpointing
        and hasattr(
            model,
            "enable_input_require_grads",
        )
    ):
        model.enable_input_require_grads()

    return model


# 6. LoRA 설정

def create_lora_config(
    profile: TrainingProfile,
) -> LoraConfig:
    """
    Qwen의 Attention과 MLP Projection Layer에
    LoRA Adapter를 추가하기 위한 설정을 만듭니다.

    Base Model의 기존 Parameter 전체를 학습하는 대신,
    아래 Layer에 작은 LoRA Parameter를 추가하여 학습합니다.
    """

    return LoraConfig(

        # LoRA의 Rank
        #
        # 값이 커질수록 학습 가능한 Parameter가 증가하지만
        # 필요한 메모리와 연산량도 증가합니다.
        r=profile.lora_rank,

        # LoRA Adapter의 변화량을 조절하는 Scaling 값
        lora_alpha=profile.lora_alpha,

        # LoRA 학습 과정의 Dropout 비율
        lora_dropout=profile.lora_dropout,

        # Base Model의 bias는 별도로 학습하지 않습니다.
        bias="none",

        # Qwen은 이전 Token을 기반으로 다음 Token을 생성하는
        # Causal Language Model입니다.
        task_type="CAUSAL_LM",

        # LoRA Adapter를 추가할 Layer를 지정합니다.
        #
        # Attention
        # q_proj    : Query
        # k_proj    : Key
        # v_proj    : Value
        # o_proj    : Attention 출력
        #
        # MLP
        # gate_proj : 정보 흐름 조절
        # up_proj   : 차원 확장
        # down_proj : 다시 원래 차원으로 축소
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


# 7. SFT Trainer 설정

def create_sft_config(
    profile: TrainingProfile,
    output_dir: Path,
) -> SFTConfig:
    """
    SFTTrainer가 사용할 학습 Hyperparameter를 설정합니다.
    """

    # 학습 결과 저장 디렉토리가 없으면 생성합니다.
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return SFTConfig(

        # Checkpoint와 Adapter가 저장될 디렉토리
        output_dir=str(output_dir),

        # Dataset 전체 반복 횟수
        num_train_epochs=(
            profile.epochs
        ),

        # 한 Step에 처리할 Train Sample 수
        per_device_train_batch_size=(
            profile.train_batch_size
        ),

        # 한 Step에 처리할 Validation Sample 수
        per_device_eval_batch_size=(
            profile.eval_batch_size
        ),

        # 여러 번의 Gradient를 누적한 뒤
        # 한 번 Parameter Update를 수행합니다.
        gradient_accumulation_steps=(
            profile.gradient_accumulation_steps
        ),

        # Parameter Update 크기
        learning_rate=(
            profile.learning_rate
        ),

        # Gradient가 지나치게 커지는 것을 방지합니다.
        max_grad_norm=1.0,

        # 초반 5% 구간에서 Learning Rate를 천천히 증가시킵니다.
        warmup_ratio=0.05,

        # 이후 Learning Rate를 Cosine 형태로 감소시킵니다.
        lr_scheduler_type="cosine",

        # 5 Step마다 학습 정보를 기록합니다.
        logging_strategy="steps",
        logging_steps=5,

        # 각 Epoch 종료 시 Validation 수행
        eval_strategy="epoch",

        # 각 Epoch 종료 시 Checkpoint 저장
        save_strategy="epoch",

        # Checkpoint는 최대 2개까지만 유지
        save_total_limit=2,

        # 학습 종료 후 Validation Loss가 가장 좋았던
        # Checkpoint를 자동으로 다시 불러옵니다.
        load_best_model_at_end=True,

        # 어떤 값을 기준으로 Best Model을 선택할지 지정
        metric_for_best_model="eval_loss",

        # Loss는 낮을수록 좋은 값입니다.
        greater_is_better=False,

        # 하나의 Sample에서 사용할 최대 Token 길이
        max_length=(
            profile.max_length
        ),

        # 여러 짧은 Sample을 한 Sequence로 묶지 않습니다.
        packing=False,

        # 중간 Activation을 모두 저장하지 않고
        # 필요할 때 다시 계산하여 메모리를 절약합니다.
        gradient_checkpointing=(
            profile.gradient_checkpointing
        ),

        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        # Conversational Dataset에서
        # assistant 응답 부분을 학습 정답으로 사용합니다.
        assistant_only_loss=True,

        # Model 자체의 dtype을 환경별 load_model()에서 관리하므로
        # Trainer의 자동 FP16/BF16 변환은 사용하지 않습니다.
        fp16=False,
        bf16=False,

        # PyTorch 기본 AdamW Optimizer 사용
        optim="adamw_torch",

        # CUDA에서만 pinned memory를 사용합니다.
        #
        # MPS 환경에서 발생할 수 있는
        # pin_memory 관련 경고도 방지합니다.
        dataloader_pin_memory=(
            profile.device_type == "cuda"
        ),

        # WandB 등의 외부 Logging Service는 사용하지 않습니다.
        report_to="none",

        # 학습 재현성을 위한 Seed
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,

        # Dataset의 messages Column을 Trainer가
        # 임의로 제거하지 않도록 합니다.
        remove_unused_columns=False,

        # Dataset 전처리 Process 수
        dataset_num_proc=1,
    )


# 8. 학습 결과 저장

def save_training_metadata(
    output_dir: Path,
    profile: TrainingProfile,
    train_path: Path,
    validation_path: Path,
    train_loss: float,
) -> None:
    """
    어떤 환경과 설정으로 학습했는지 기록합니다.

    나중에 모델 결과를 재현하거나
    서로 다른 학습 결과를 비교할 때 사용할 수 있습니다.
    """

    metadata = {
        "base_model": MODEL_NAME,

        # dataclass 객체를 일반 dictionary로 변환합니다.
        "profile": asdict(profile),

        "train_dataset": str(train_path),

        "validation_dataset": str(
            validation_path
        ),

        "final_train_loss": train_loss,

        "pytorch_version": torch.__version__,

        "platform": platform.platform(),

        "machine": platform.machine(),
    }

    metadata_path = (
        output_dir
        / "training_environment.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_training_log(
    trainer: SFTTrainer,
    output_path: Path,
) -> None:
    """
    Trainer가 학습 중 기록한 전체 Log를 JSON으로 저장합니다.

    대표적으로 다음 정보가 들어갑니다.

    - loss
    - eval_loss
    - learning_rate
    - epoch
    - step
    """

    # outputs 디렉토리가 없다면 자동 생성합니다.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # SFTTrainer가 학습 과정에서 기록한 Log 전체
    log_history = (
        trainer.state.log_history
    )

    output_path.write_text(
        json.dumps(
            log_history,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[Fine-tuning] Training log : "
        f"{output_path}"
    )


def save_training_summary(
    trainer: SFTTrainer,
    train_loss: float,
    profile: TrainingProfile,
    output_path: Path,
) -> None:
    """
    최종 학습 결과에서 중요한 값만 추려
    별도의 Summary JSON으로 저장합니다.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "base_model": MODEL_NAME,

        "device": profile.device_type,

        "training_mode": (
            profile.training_mode
        ),

        "epochs": profile.epochs,

        "max_length": (
            profile.max_length
        ),

        "learning_rate": (
            profile.learning_rate
        ),

        "gradient_accumulation_steps": (
            profile.gradient_accumulation_steps
        ),

        "lora_rank": (
            profile.lora_rank
        ),

        "lora_alpha": (
            profile.lora_alpha
        ),

        "final_train_loss": (
            round(
                train_loss,
                6,
            )
        ),

        # 전체 Optimizer Update 횟수
        "global_step": (
            trainer.state.global_step
        ),

        # Best Model 선택 기준이 된 Validation Metric
        "best_metric": (
            trainer.state.best_metric
        ),

        # Validation 결과가 가장 좋았던 Checkpoint
        "best_model_checkpoint": (
            trainer.state.best_model_checkpoint
        ),
    }

    output_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[Fine-tuning] Training summary : "
        f"{output_path}"
    )


def cleanup_memory(
    device_type: DeviceType,
) -> None:
    """
    Fine-Tuning 종료 후 Python / GPU 메모리를 정리합니다.
    """

    # Python에서 더 이상 사용하지 않는 객체를 정리합니다.
    gc.collect()

    # NVIDIA GPU Cache 정리
    if device_type == "cuda":
        torch.cuda.empty_cache()

    # Apple MPS Cache 정리
    elif device_type == "mps":
        try:
            torch.mps.synchronize()
            torch.mps.empty_cache()

        # PyTorch 버전 등에 따라 일부 함수가
        # 지원되지 않는 경우를 안전하게 무시합니다.
        except (
            AttributeError,
            RuntimeError,
        ):
            pass


# 9. 실행 옵션

def parse_args() -> argparse.Namespace:
    """
    Terminal에서 전달되는 실행 옵션을 정의합니다.

    예:

    python scripts/train_lora.py \
        --train dataset/training/train.jsonl \
        --validation dataset/training/validation.jsonl
    """

    parser = argparse.ArgumentParser(
        description=(
            "Qwen2.5-1.5B HR LoRA / QLoRA SFT 학습"
        )
    )

    # 학습 Dataset 경로
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
    )

    # Validation Dataset 경로
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )

    # LoRA Adapter 저장 경로
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def resolve_project_path(
    path: Path,
) -> Path:
    """
    상대 경로로 입력된 경로를
    PROJECT_ROOT 기준의 절대 경로로 변환합니다.

    절대 경로가 들어오면 그대로 사용합니다.
    """

    # ~/dataset 같은 경로의 ~를 실제 사용자 홈으로 변환합니다.
    path = path.expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


# 10. 실행

def main() -> int:
    """
    Fine-Tuning 전체 과정을 순서대로 실행합니다.
    """

    # Terminal 실행 옵션을 읽습니다.
    args = parse_args()

    # Dataset 및 Output 경로를 절대 경로로 변환합니다.
    train_path = resolve_project_path(
        args.train
    )

    validation_path = resolve_project_path(
        args.validation
    )

    output_dir = resolve_project_path(
        args.output
    )

    # Python / PyTorch 등의 Random Seed를 설정합니다.
    set_seed(
        RANDOM_SEED
    )

    # CUDA / MPS / CPU 중 현재 환경을 자동 감지합니다.
    device_type = detect_device_type()

    # 감지된 환경에 맞는 Training 설정을 생성합니다.
    profile = create_training_profile(
        device_type
    )

    # 실제 적용될 학습 설정을 화면에 출력합니다.
    print("[Fine-tuning] Runtime")

    print(
        f"Platform        : "
        f"{platform.platform()}"
    )

    print(
        f"PyTorch         : "
        f"{torch.__version__}"
    )

    print(
        f"Device          : "
        f"{profile.device_type}"
    )

    print(
        f"Mode            : "
        f"{profile.training_mode}"
    )

    print(
        f"Max length      : "
        f"{profile.max_length}"
    )

    print(
        f"Epochs          : "
        f"{profile.epochs}"
    )

    print(
        f"Learning rate   : "
        f"{profile.learning_rate}"
    )

    print(
        f"Gradient accum. : "
        f"{profile.gradient_accumulation_steps}"
    )

    print(
        f"LoRA rank       : "
        f"{profile.lora_rank}"
    )

    print(
        f"LoRA alpha      : "
        f"{profile.lora_alpha}"
    )

    print(
        f"LoRA dropout    : "
        f"{profile.lora_dropout}"
    )

    # Train / Validation Dataset을 검증하고 로드합니다.
    dataset = load_training_dataset(
        train_path=train_path,
        validation_path=validation_path,
    )

    # Qwen Tokenizer를 로드합니다.
    tokenizer = load_tokenizer()

    # 실제 Dataset 하나가 Qwen Chat Template을 거쳐
    # 어떤 형태로 학습되는지 미리 확인합니다.
    inspect_training_sample(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=profile.max_length,
    )

    # 현재 환경에 맞게 Qwen Base Model을 로드합니다.
    model = load_model(
        profile
    )

    # LoRA Adapter 설정을 생성합니다.
    lora_config = create_lora_config(
        profile
    )

    # SFTTrainer를 생성합니다.
    #
    # 아직 이 시점에서는 Fine-Tuning을 시작하지 않습니다.
    # 실제 학습은 아래 trainer.train()에서 시작합니다.
    trainer = SFTTrainer(
        model=model,

        args=create_sft_config(
            profile=profile,
            output_dir=output_dir,
        ),

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        # Tokenizer 및 Chat Template 처리 담당
        processing_class=tokenizer,

        # Base Model 전체가 아닌
        # LoRA Adapter를 학습하도록 PEFT 설정 전달
        peft_config=lora_config,
    )

    try:
        print(
            "\n[Fine-tuning] Training started"
        )

        # 실제 Fine-Tuning 시작
        train_result = (
            trainer.train()
        )

        # 전체 학습 과정의 최종 Train Loss
        train_loss = float(
            train_result.training_loss
        )

        print(
            f"\n[Fine-tuning] Final train loss: "
            f"{train_loss:.6f}"
        )

        # load_best_model_at_end=True이므로
        # 단순히 마지막 Epoch의 모델이 아니라
        # Validation Loss가 가장 좋았던 Adapter를 저장합니다.
        trainer.save_model(
            str(output_dir)
        )

        # 추론할 때 동일한 Tokenizer 설정을 사용할 수 있도록
        # Tokenizer 정보도 Adapter와 함께 저장합니다.
        tokenizer.save_pretrained(
            str(output_dir)
        )

        # Epoch/Step별 전체 학습 기록 저장
        save_training_log(
            trainer=trainer,
            output_path=TRAINING_LOG_PATH,
        )

        # 핵심 학습 결과 Summary 저장
        save_training_summary(
            trainer=trainer,
            train_loss=train_loss,
            profile=profile,
            output_path=TRAINING_SUMMARY_PATH,
        )

        # 모델, PyTorch 버전, 실행 환경 등의 Metadata 저장
        save_training_metadata(
            output_dir=output_dir,
            profile=profile,
            train_path=train_path,
            validation_path=validation_path,
            train_loss=train_loss,
        )

        print(
            "\n[Fine-tuning] Complete"
        )

        print(
            f"Adapter output : "
            f"{output_dir}"
        )

        # 정상 종료
        return 0

    finally:
        # 학습 성공/실패 여부와 관계없이
        # 사용한 대형 객체들을 메모리에서 제거합니다.
        del trainer
        del model
        del tokenizer

        cleanup_memory(
            profile.device_type
        )

        print(
            "[Fine-tuning] Runtime resources released"
        )


# 이 파일을 python scripts/train_lora.py처럼
# 직접 실행했을 때만 main()을 실행합니다.
#
# 다른 Python 파일에서 import할 때는 자동 실행되지 않습니다.
if __name__ == "__main__":
    raise SystemExit(
        main()
    )