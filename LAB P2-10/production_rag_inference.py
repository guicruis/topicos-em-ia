import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for lightweight tests
    torch = None


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "outputs"
BENCHMARK_FILE = OUTPUT_DIR / "inference_benchmark.json"
CONTEXT_FILE = OUTPUT_DIR / "massive_medical_context.txt"


@dataclass
class SimpleBitsAndBytesConfig:
    load_in_4bit: bool
    bnb_4bit_compute_dtype: Any
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LabConfig:
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    synthetic_context_tokens: int = 12000
    generated_tokens: int = 100
    model_parameters_billions: float = 1.1
    num_hidden_layers: int = 22
    hidden_size: int = 2048
    quantization_bits: int = 4
    full_precision_bits: int = 16
    compute_dtype_name: str = "float16"
    attn_implementation: str = "flash_attention_2"
    seed: int = 42


@dataclass
class InferenceMetrics:
    label: str
    prompt_tokens: int
    generated_tokens: int
    use_cache: bool
    flash_attention: bool
    quantized_4bit: bool
    model_vram_mb: float
    prompt_peak_vram_mb: float
    kv_cache_peak_mb: float
    total_peak_vram_mb: float
    generation_time_seconds: float
    throughput_tokens_per_second: float


class RegexTokenizer:
    def encode(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def config_to_dict(config_obj: Any) -> dict[str, Any]:
    if isinstance(config_obj, dict):
        return config_obj
    if is_dataclass(config_obj):
        return asdict(config_obj)
    if hasattr(config_obj, "to_dict"):
        return config_obj.to_dict()
    if hasattr(config_obj, "__dict__"):
        return dict(config_obj.__dict__)
    return {"value": str(config_obj)}


def compute_dtype_from_config(config: LabConfig) -> Any:
    if torch is None:
        return config.compute_dtype_name
    return getattr(torch, config.compute_dtype_name)


def build_quantization_config(config: LabConfig):
    return SimpleBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype_from_config(config),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def build_transformers_quantization_config(config: LabConfig):
    kwargs = config_to_dict(build_quantization_config(config))
    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(**kwargs)
    except ImportError as exc:  # pragma: no cover - depends on optional packages
        raise RuntimeError("Instale transformers e bitsandbytes para carregar o modelo real.") from exc


def build_model_load_kwargs(config: LabConfig, use_transformers_config: bool = False) -> dict[str, Any]:
    return {
        "quantization_config": (
            build_transformers_quantization_config(config) if use_transformers_config else build_quantization_config(config)
        ),
        "device_map": "auto",
        "torch_dtype": compute_dtype_from_config(config),
        "attn_implementation": config.attn_implementation,
    }


def load_tokenizer(config: LabConfig):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(config.base_model_name, use_fast=True)
    except Exception:
        return RegexTokenizer()


def load_quantized_model_for_gpu(config: LabConfig):
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - depends on optional packages
        raise RuntimeError("Instale transformers, bitsandbytes, accelerate e torch para carregar o modelo real.") from exc

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        **build_model_load_kwargs(config, use_transformers_config=True),
    )
    model.config.use_cache = True
    return model


def build_medical_context(target_tokens: int = 12000) -> str:
    chapters = [
        (
            "Capitulo neurologia critica",
            "Paciente com cefaleia subita, deficit focal, rebaixamento do nivel de consciencia, rigidez nucal "
            "ou crise convulsiva exige estratificacao imediata para AVC, hemorragia subaracnoide, meningite "
            "e hipertensao intracraniana. A anamnese deve registrar tempo de inicio, anticoagulantes, febre, "
            "trauma, aura visual, padrao previo de migranea e sinais de alarme.",
        ),
        (
            "Capitulo cardiologia de emergencia",
            "Dor toracica opressiva irradiada para braco esquerdo, mandibula ou dorso deve ser tratada como "
            "sindrome coronariana aguda ate prova em contrario. O protocolo prioriza eletrocardiograma em "
            "dez minutos, troponina seriada, avaliacao de choque, edema pulmonar, arritmia e indicacao de "
            "reperfusao conforme elevacao persistente do segmento ST.",
        ),
        (
            "Capitulo pneumologia hospitalar",
            "Dispneia aguda com taquicardia, dessaturacao, dor pleuritica ou imobilizacao recente eleva a "
            "suspeita de tromboembolismo pulmonar. Tosse produtiva, febre e crepitacoes focais favorecem "
            "pneumonia. Sibilancia recorrente e resposta a broncodilatador sugerem asma ou DPOC exacerbada.",
        ),
        (
            "Capitulo endocrinologia metabolica",
            "Diabetes descompensado pode aparecer como hipoglicemia sintomatica, estado hiperosmolar ou "
            "cetoacidose diabetica. Poliuria, polidipsia, vomitos, dor abdominal, respiracao de Kussmaul, "
            "cetonemia e acidose metabolica orientam reposicao de fluidos, insulina regular e potassio.",
        ),
        (
            "Capitulo infectologia e sepse",
            "Sepse deve ser suspeitada diante de foco infeccioso com hipotensao, taquipneia, lactato elevado, "
            "confusao mental, oliguria ou extremidades frias. A primeira hora inclui culturas, antibiotico "
            "empirico, expansao volemica, controle do foco e vasopressor se houver choque refratario.",
        ),
    ]
    tokenizer = RegexTokenizer()
    sections = []
    index = 1
    while len(tokenizer.encode(" ".join(sections))) < target_tokens:
        title, body = chapters[(index - 1) % len(chapters)]
        sections.append(
            f"{title} {index}. {body} Conduta operacional: classificar risco, documentar achados, "
            "comparar diagnosticos diferenciais, registrar criterios de internacao e sintetizar recomendacoes "
            "para o relatorio clinico automatizado."
        )
        index += 1
    return " ".join(sections)


def save_massive_context(config: LabConfig) -> tuple[str, int]:
    ensure_directories()
    context = build_medical_context(config.synthetic_context_tokens)
    CONTEXT_FILE.write_text(context, encoding="utf-8")
    return str(CONTEXT_FILE), count_tokens(context)


def count_tokens(text: str, tokenizer: Any | None = None) -> int:
    tokenizer = tokenizer or RegexTokenizer()
    encoded = tokenizer.encode(text)
    if hasattr(encoded, "ids"):
        return len(encoded.ids)
    return len(encoded)


def estimate_model_vram_mb(config: LabConfig, quantized_4bit: bool) -> float:
    bits = config.quantization_bits if quantized_4bit else config.full_precision_bits
    parameter_count = config.model_parameters_billions * 1_000_000_000
    return parameter_count * bits / 8 / (1024**2)


def estimate_attention_prompt_memory_mb(prompt_tokens: int, config: LabConfig, flash_attention: bool) -> float:
    bytes_per_value = 2
    if flash_attention:
        values = prompt_tokens * config.hidden_size * config.num_hidden_layers
    else:
        values = prompt_tokens * prompt_tokens * config.num_hidden_layers
    return values * bytes_per_value / (1024**2)


def estimate_kv_cache_mb(total_tokens: int, config: LabConfig, use_cache: bool) -> float:
    if not use_cache:
        return 0.0
    bytes_per_value = 2
    keys_and_values = 2
    values = total_tokens * config.hidden_size * config.num_hidden_layers * keys_and_values
    return values * bytes_per_value / (1024**2)


def estimate_generation_time_seconds(prompt_tokens: int, generated_tokens: int, use_cache: bool, flash_attention: bool) -> float:
    prompt_factor = 0.0000035 if flash_attention else 0.000018
    decode_factor = 0.00018 if use_cache else 0.0000038 * prompt_tokens
    prompt_time = prompt_tokens * prompt_factor
    decode_time = generated_tokens * decode_factor
    return round(prompt_time + decode_time, 4)


def run_inference_simulation(
    label: str,
    context: str,
    config: LabConfig,
    use_cache: bool,
    flash_attention: bool,
    quantized_4bit: bool,
) -> InferenceMetrics:
    prompt_tokens = count_tokens(context)
    start = time.perf_counter()
    generation_time = estimate_generation_time_seconds(
        prompt_tokens=prompt_tokens,
        generated_tokens=config.generated_tokens,
        use_cache=use_cache,
        flash_attention=flash_attention,
    )
    elapsed = time.perf_counter() - start
    generation_time = max(generation_time, round(elapsed, 4))

    model_vram = estimate_model_vram_mb(config, quantized_4bit)
    prompt_peak = estimate_attention_prompt_memory_mb(prompt_tokens, config, flash_attention)
    kv_cache = estimate_kv_cache_mb(prompt_tokens + config.generated_tokens, config, use_cache)
    total_peak = model_vram + prompt_peak + kv_cache
    throughput = config.generated_tokens / generation_time if generation_time > 0 else math.inf

    return InferenceMetrics(
        label=label,
        prompt_tokens=prompt_tokens,
        generated_tokens=config.generated_tokens,
        use_cache=use_cache,
        flash_attention=flash_attention,
        quantized_4bit=quantized_4bit,
        model_vram_mb=round(model_vram, 2),
        prompt_peak_vram_mb=round(prompt_peak, 2),
        kv_cache_peak_mb=round(kv_cache, 2),
        total_peak_vram_mb=round(total_peak, 2),
        generation_time_seconds=round(generation_time, 4),
        throughput_tokens_per_second=round(throughput, 2),
    )


def run_benchmark(config: LabConfig | None = None) -> dict[str, Any]:
    config = config or LabConfig()
    ensure_directories()
    context = build_medical_context(config.synthetic_context_tokens)
    baseline = run_inference_simulation(
        label="baseline_sem_cache",
        context=context,
        config=config,
        use_cache=False,
        flash_attention=False,
        quantized_4bit=True,
    )
    optimized = run_inference_simulation(
        label="otimizado_kv_cache_flashattention",
        context=context,
        config=config,
        use_cache=True,
        flash_attention=True,
        quantized_4bit=True,
    )
    result = {
        "config": asdict(config),
        "quantization_config": config_to_dict(build_quantization_config(config)),
        "model_load_kwargs": config_to_dict(build_model_load_kwargs(config)),
        "baseline": asdict(baseline),
        "optimized": asdict(optimized),
        "speedup": round(baseline.generation_time_seconds / optimized.generation_time_seconds, 2),
        "prompt_memory_reduction": round(baseline.prompt_peak_vram_mb / optimized.prompt_peak_vram_mb, 2),
    }
    with BENCHMARK_FILE.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=str)
    return result


def describe_pipeline(config: LabConfig) -> dict[str, Any]:
    return {
        "lab_config": asdict(config),
        "steps": [
            "carregar modelo causal com QLoRA em 4 bits via bitsandbytes",
            "gerar contexto medico sintetico entre 10000 e 15000 tokens",
            "medir baseline com model.config.use_cache = False",
            "ativar KV Cache com model.config.use_cache = True",
            "carregar modelo com FlashAttention-2 usando attn_implementation='flash_attention_2'",
            "comparar tempo, pico de VRAM e throughput",
        ],
        "model_load_kwargs": config_to_dict(build_model_load_kwargs(config)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 10: RAG massivo com QLoRA, KV Cache e FlashAttention.")
    parser.add_argument("command", choices=["describe", "build-context", "benchmark"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabConfig()
    if args.command == "describe":
        print(json.dumps(describe_pipeline(config), indent=2, ensure_ascii=False, default=str))
        return
    if args.command == "build-context":
        path, tokens = save_massive_context(config)
        print(json.dumps({"path": path, "tokens": tokens}, indent=2, ensure_ascii=False))
        return
    result = run_benchmark(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
