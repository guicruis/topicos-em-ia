import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dpo_pipeline import (
    LabConfig,
    build_preference_examples,
    build_training_arguments,
    explain_beta,
    normalize_preference_scores,
    split_examples,
)


class TestDPOPipeline(unittest.TestCase):
    def setUp(self):
        self.config = LabConfig()

    def test_preference_dataset_has_required_columns(self):
        examples = build_preference_examples(self.config)
        self.assertGreaterEqual(len(examples), 30)
        self.assertEqual(set(examples[0].keys()), {"prompt", "chosen", "rejected"})

    def test_split_examples_creates_train_and_eval_sets(self):
        examples = build_preference_examples(self.config)
        train_rows, eval_rows = split_examples(examples, eval_ratio=self.config.eval_ratio, seed=self.config.seed)
        self.assertEqual(len(train_rows), 30)
        self.assertEqual(len(eval_rows), 6)

    def test_training_arguments_match_pdf(self):
        training_args = build_training_arguments(self.config)
        self.assertEqual(training_args.optim, "paged_adamw_32bit")
        self.assertAlmostEqual(training_args.beta, 0.1)

    def test_beta_explanation_mentions_kl_role(self):
        explanation = explain_beta().lower()
        self.assertIn("kl", explanation)
        self.assertIn("temperatura", explanation)

    def test_normalized_scores_suppress_rejected_response(self):
        result = normalize_preference_scores(chosen_logprob=-0.2, rejected_logprob=-2.0)
        self.assertGreater(result["chosen_probability"], result["rejected_probability"])
        self.assertTrue(result["rejected_suppressed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
