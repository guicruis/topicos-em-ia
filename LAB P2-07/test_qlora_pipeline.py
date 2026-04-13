import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from qlora_pipeline import (
    LabConfig,
    build_fallback_examples,
    build_lora_config,
    build_quantization_config,
    build_training_arguments,
    format_instruction_sample,
    split_examples,
)


class TestQLoRAPipeline(unittest.TestCase):
    def setUp(self):
        self.config = LabConfig()

    def test_fallback_generation_meets_minimum(self):
        examples = build_fallback_examples(self.config)
        self.assertGreaterEqual(len(examples), 50)
        self.assertIn("instruction", examples[0])
        self.assertIn("response", examples[0])

    def test_split_examples_uses_ninety_ten_ratio(self):
        examples = build_fallback_examples(self.config)
        train_rows, test_rows = split_examples(examples, test_ratio=0.1, seed=42)
        self.assertEqual(len(train_rows), 54)
        self.assertEqual(len(test_rows), 6)

    def test_quantization_config_matches_pdf(self):
        quant_config = build_quantization_config()
        self.assertTrue(quant_config.load_in_4bit)
        self.assertEqual(quant_config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(hasattr(quant_config, "bnb_4bit_compute_dtype"))

    def test_lora_config_matches_pdf(self):
        lora_config = build_lora_config()
        self.assertEqual(lora_config.r, 64)
        self.assertEqual(lora_config.lora_alpha, 16)
        self.assertAlmostEqual(lora_config.lora_dropout, 0.1)
        self.assertEqual(lora_config.task_type, "CAUSAL_LM")

    def test_training_arguments_matches_pdf(self):
        training_args = build_training_arguments(self.config)
        self.assertEqual(training_args.optim, "paged_adamw_32bit")
        self.assertEqual(str(training_args.lr_scheduler_type), "cosine")
        self.assertAlmostEqual(training_args.warmup_ratio, 0.03)

    def test_formatted_sample_contains_instruction_and_response(self):
        formatted = format_instruction_sample(
            {
                "instruction": "Explique como reiniciar o driver de rede.",
                "response": "Desative e ative o adaptador de rede.",
            }
        )
        self.assertIn("### Instrucao:", formatted)
        self.assertIn("### Resposta:", formatted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
