import argparse
import inspect
import json
import random
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for lightweight tests
    torch = None


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
OUTPUT_DIR = THIS_DIR / "outputs"
TRAIN_FILE = DATA_DIR / "synthetic_train.jsonl"
TEST_FILE = DATA_DIR / "synthetic_test.jsonl"
ADAPTER_DIR = OUTPUT_DIR / "adapter"


@dataclass
class SimpleBitsAndBytesConfig:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: Any
    bnb_4bit_use_double_quant: bool = True


@dataclass
class SimpleLoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    task_type: str
    target_modules: list[str]


@dataclass
class SimpleTrainingArguments:
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    logging_steps: int
    save_strategy: str
    save_total_limit: int
    load_best_model_at_end: bool
    optim: str
    lr_scheduler_type: str
    warmup_ratio: float
    bf16: bool
    report_to: str
    evaluation_strategy: str | None = None
    eval_strategy: str | None = None


@dataclass
class LabConfig:
    domain: str = "suporte tecnico para notebooks"
    num_examples: int = 60
    test_ratio: float = 0.1
    seed: int = 42
    openai_model: str = "gpt-4o-mini"
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_seq_length: int = 512
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    logging_steps: int = 5


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)


def config_to_dict(config_obj: Any) -> dict[str, Any]:
    if is_dataclass(config_obj):
        return asdict(config_obj)
    if hasattr(config_obj, "to_dict"):
        return config_obj.to_dict()
    if hasattr(config_obj, "__dict__"):
        return dict(config_obj.__dict__)
    return {"value": str(config_obj)}


def build_generation_prompt(domain: str, batch_size: int) -> str:
    return (
        "Gere exemplos sinteticos em portugues para fine-tuning de um assistente.\n"
        f"Dominio: {domain}.\n"
        f"Quantidade: {batch_size}.\n"
        "Responda exclusivamente como JSON valido com uma lista de objetos contendo "
        "as chaves 'instruction' e 'response'.\n"
        "As instrucoes devem ser variadas, realistas, objetivas e prontas para treino.\n"
        "As respostas devem ser corretas, completas e sem mencionar a existencia de IA."
    )


def parse_openai_json(raw_text: str) -> list[dict[str, str]]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("A resposta da OpenAI deve ser uma lista JSON.")

    normalized = []
    for item in parsed:
        instruction = str(item["instruction"]).strip()
        response = str(item["response"]).strip()
        if instruction and response:
            normalized.append({"instruction": instruction, "response": response})
    return normalized


def generate_examples_with_openai(config: LabConfig) -> list[dict[str, str]]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Pacote 'openai' nao encontrado.") from exc

    client = OpenAI()
    examples: list[dict[str, str]] = []
    batch_size = 20

    while len(examples) < config.num_examples:
        remaining = config.num_examples - len(examples)
        prompt = build_generation_prompt(config.domain, min(batch_size, remaining))
        response = client.responses.create(
            model=config.openai_model,
            input=prompt,
            temperature=0.8,
        )
        examples.extend(parse_openai_json(response.output_text))

    return deduplicate_examples(examples)[: config.num_examples]


def build_fallback_examples(config: LabConfig) -> list[dict[str, str]]:
    intents = [
        ("bateria descarrega rapido", "reduza brilho, revise processos em segundo plano e confira a saude da bateria"),
        ("notebook nao liga", "teste carregador, mantenha o botao pressionado por 15 segundos e valide sinais de energia"),
        ("wifi cai com frequencia", "esqueca a rede, reinstale o driver e priorize a banda de 5 GHz quando disponivel"),
        ("teclado falhando", "limpe o teclado, teste outro layout e confira se o problema ocorre tambem na BIOS"),
        ("tela piscando", "atualize driver de video e verifique taxa de atualizacao e cabo do display"),
        ("ventoinha muito alta", "remova poeira, cheque uso de CPU e ajuste plano de energia"),
        ("ssd quase cheio", "remova arquivos temporarios, desinstale apps pouco usados e mova arquivos grandes"),
        ("microfone nao funciona", "valide permissoes do sistema, dispositivo padrao e driver de audio"),
        ("camera nao abre", "confirme permissao do app, atualize driver e feche programas que possam estar usando a camera"),
        ("usb nao reconhece pendrive", "troque a porta, teste em outro equipamento e verifique gerenciamento de discos"),
        ("touchpad travando", "atualize driver, desligue gestos extras e teste com mouse externo"),
        ("audio baixo", "revise mixer do sistema, realces de audio e configuracao do aplicativo"),
    ]
    contexts = [
        "durante uma reuniao",
        "apos uma atualizacao do sistema",
        "quando o equipamento sai da suspensao",
        "em uso domestico diario",
        "em um ambiente corporativo",
    ]
    user_profiles = [
        "um estudante",
        "um analista financeiro",
        "uma professora",
        "um atendente de suporte",
        "uma pessoa que trabalha em home office",
    ]

    examples = []
    for intent, action in intents:
        for context in contexts:
            for profile in user_profiles:
                instruction = (
                    f"Explique como ajudar {profile} quando o notebook apresenta o problema "
                    f"'{intent}' {context}. Forneca um passo a passo objetivo."
                )
                response = (
                    f"1. Confirme sintomas e quando o erro comeca. "
                    f"2. Oriente a acao principal: {action}. "
                    f"3. Peca um teste rapido para validar a correcao. "
                    f"4. Se o problema continuar, abra chamado com evidencias como foto, horario e mensagem de erro."
                )
                examples.append({"instruction": instruction, "response": response})

    random.Random(config.seed).shuffle(examples)
    return examples[: config.num_examples]


def deduplicate_examples(examples: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    unique = []
    for item in examples:
        key = (item["instruction"], item["response"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def split_examples(examples: list[dict[str, str]], test_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if len(examples) < 2:
        raise ValueError("Sao necessarios ao menos dois exemplos para separar treino e teste.")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, round(len(shuffled) * test_ratio))
    test_examples = shuffled[:test_size]
    train_examples = shuffled[test_size:]
    return train_examples, test_examples


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_and_save_dataset(config: LabConfig) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    ensure_directories()
    try:
        examples = generate_examples_with_openai(config)
    except Exception:
        examples = build_fallback_examples(config)

    examples = deduplicate_examples(examples)
    if len(examples) < config.num_examples:
        raise ValueError("Nao foi possivel gerar exemplos suficientes.")

    train_examples, test_examples = split_examples(
        examples[: config.num_examples],
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    write_jsonl(TRAIN_FILE, train_examples)
    write_jsonl(TEST_FILE, test_examples)
    return train_examples, test_examples


def build_quantization_config():
    compute_dtype = getattr(torch, "bfloat16", "bfloat16")
    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    except ImportError:
        return SimpleBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )


def build_lora_config():
    kwargs = {
        "r": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    try:
        from peft import LoraConfig

        return LoraConfig(**kwargs)
    except ImportError:
        return SimpleLoraConfig(**kwargs)


def build_training_arguments(config: LabConfig):
    bf16_enabled = bool(
        torch is not None
        and hasattr(torch, "cuda")
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    kwargs = {
        "output_dir": str(OUTPUT_DIR),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "logging_steps": config.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "optim": "paged_adamw_32bit",
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": bf16_enabled,
        "report_to": "none",
    }
    try:
        from transformers import TrainingArguments

        signature = inspect.signature(TrainingArguments.__init__)
        if "evaluation_strategy" in signature.parameters:
            kwargs["evaluation_strategy"] = "epoch"
        else:
            kwargs["eval_strategy"] = "epoch"
        return TrainingArguments(**kwargs)
    except Exception:
        if "eval_strategy" in kwargs and "evaluation_strategy" not in kwargs:
            kwargs["evaluation_strategy"] = kwargs["eval_strategy"]
        return SimpleTrainingArguments(**kwargs)


def format_instruction_sample(example: dict[str, str]) -> str:
    return (
        "### Instrucao:\n"
        f"{example['instruction'].strip()}\n\n"
        "### Resposta:\n"
        f"{example['response'].strip()}"
    )


def load_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_training_pipeline(config: LabConfig) -> dict[str, Any]:
    ensure_directories()
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        generate_and_save_dataset(config)

    try:
        from datasets import Dataset
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTTrainer
    except ImportError as exc:  # pragma: no cover - depends on optional packages
        raise RuntimeError(
            "Dependencias ausentes. Instale: torch transformers datasets peft trl bitsandbytes accelerate."
        ) from exc

    quantization_config = build_quantization_config()
    lora_config = build_lora_config()
    training_args = build_training_arguments(config)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=getattr(torch, "bfloat16", None),
    )
    model.config.use_cache = False
    model = get_peft_model(model, lora_config)

    train_rows = load_jsonl(TRAIN_FILE)
    test_rows = load_jsonl(TEST_FILE)
    train_dataset = Dataset.from_list([{"text": format_instruction_sample(row)} for row in train_rows])
    test_dataset = Dataset.from_list([{"text": format_instruction_sample(row)} for row in test_rows])

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
    )
    trainer.train()
    trainer.model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    return {
        "train_examples": len(train_rows),
        "test_examples": len(test_rows),
        "quantization_config": config_to_dict(build_quantization_config()),
        "lora_config": config_to_dict(build_lora_config()),
        "adapter_dir": str(ADAPTER_DIR),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline completo de dataset sintetico + QLoRA.")
    parser.add_argument(
        "command",
        choices=["generate-data", "train", "describe"],
        help="Acao desejada.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabConfig()

    if args.command == "generate-data":
        train_examples, test_examples = generate_and_save_dataset(config)
        print(json.dumps({"train_examples": len(train_examples), "test_examples": len(test_examples)}, indent=2))
        return

    if args.command == "describe":
        summary = {
            "lab_config": asdict(config),
            "quantization_config": config_to_dict(build_quantization_config()),
            "lora_config": config_to_dict(build_lora_config()),
            "training_arguments": config_to_dict(build_training_arguments(config)),
        }
        print(json.dumps(summary, indent=2, default=str))
        return

    result = run_training_pipeline(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
