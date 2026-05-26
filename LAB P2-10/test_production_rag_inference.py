import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from production_rag_inference import (
    LabConfig,
    build_medical_context,
    build_model_load_kwargs,
    build_quantization_config,
    count_tokens,
    describe_pipeline,
    estimate_attention_prompt_memory_mb,
    estimate_model_vram_mb,
    run_benchmark,
)


class TestProductionRagInference(unittest.TestCase):
    def setUp(self):
        self.config = LabConfig(synthetic_context_tokens=10000, generated_tokens=100)

    def test_context_simulates_massive_rag_between_ten_and_fifteen_thousand_tokens(self):
        context = build_medical_context(self.config.synthetic_context_tokens)
        tokens = count_tokens(context)
        self.assertGreaterEqual(tokens, 10000)
        self.assertLessEqual(tokens, 15000)

    def test_quantization_config_uses_qlora_four_bit_loading(self):
        quantization = build_quantization_config(self.config)
        self.assertTrue(getattr(quantization, "load_in_4bit"))
        self.assertEqual(getattr(quantization, "bnb_4bit_quant_type"), "nf4")

    def test_model_load_kwargs_enable_flash_attention_two(self):
        kwargs = build_model_load_kwargs(self.config)
        self.assertEqual(kwargs["attn_implementation"], "flash_attention_2")
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertIn("quantization_config", kwargs)

    def test_four_bit_model_uses_less_vram_than_float16(self):
        qlora_vram = estimate_model_vram_mb(self.config, quantized_4bit=True)
        fp16_vram = estimate_model_vram_mb(self.config, quantized_4bit=False)
        self.assertLess(qlora_vram, fp16_vram)
        self.assertAlmostEqual(fp16_vram / qlora_vram, 4.0)

    def test_flash_attention_reduces_prompt_memory(self):
        normal = estimate_attention_prompt_memory_mb(12000, self.config, flash_attention=False)
        flash = estimate_attention_prompt_memory_mb(12000, self.config, flash_attention=True)
        self.assertLess(flash, normal)

    def test_benchmark_compares_baseline_and_optimized_pipeline(self):
        result = run_benchmark(self.config)
        self.assertFalse(result["baseline"]["use_cache"])
        self.assertTrue(result["optimized"]["use_cache"])
        self.assertTrue(result["optimized"]["flash_attention"])
        self.assertLess(
            result["optimized"]["generation_time_seconds"],
            result["baseline"]["generation_time_seconds"],
        )
        self.assertLess(
            result["optimized"]["prompt_peak_vram_mb"],
            result["baseline"]["prompt_peak_vram_mb"],
        )

    def test_describe_pipeline_mentions_required_optimizations(self):
        description = describe_pipeline(self.config)
        joined = " ".join(description["steps"]).lower()
        self.assertIn("qlora", joined)
        self.assertIn("kv cache", joined)
        self.assertIn("flashattention", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
