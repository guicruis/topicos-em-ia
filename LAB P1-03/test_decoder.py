import sys
import unittest
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from decoder import (
    MockDecoder,
    build_decoder_vocab,
    build_fake_tensors,
    create_causal_mask,
    cross_attention,
    run_causal_mask_demo,
)


class TestCausalMask(unittest.TestCase):
    def test_create_causal_mask_shape_and_values(self):
        mask = create_causal_mask(5)
        self.assertEqual(mask.shape, (5, 5))
        self.assertTrue(np.allclose(np.diag(mask), 0.0))
        self.assertTrue(np.allclose(np.tril(mask), 0.0))
        self.assertTrue(np.all(np.isneginf(np.triu(mask, k=1))[np.triu_indices(5, k=1)]))

    def test_future_probabilities_are_zero(self):
        _, _, weights = run_causal_mask_demo(seq_len=5)
        future_probs = np.triu(weights, k=1)
        self.assertTrue(np.allclose(future_probs, 0.0, atol=1e-12))


class TestCrossAttention(unittest.TestCase):
    def test_cross_attention_shapes(self):
        encoder_out, decoder_state = build_fake_tensors()
        output, weights = cross_attention(
            encoder_out=encoder_out,
            decoder_state=decoder_state,
            seed=42,
            return_weights=True,
        )
        self.assertEqual(output.shape, (1, 4, 512))
        self.assertEqual(weights.shape, (1, 4, 10))
        self.assertTrue(np.allclose(np.sum(weights, axis=-1), 1.0, atol=1e-7))


class TestAutoregressiveLoop(unittest.TestCase):
    def test_generate_next_token_returns_vocab_distribution(self):
        encoder_out, _ = build_fake_tensors()
        _, vocab_tokens = build_decoder_vocab(vocab_size=10000)
        decoder = MockDecoder(vocab_tokens=vocab_tokens, d_model=512, vocab_size=10000, seed=7)
        probs = decoder.generate_next_token(["<START>", "o"], encoder_out)
        self.assertEqual(probs.shape, (10000,))
        self.assertTrue(np.allclose(np.sum(probs), 1.0, atol=1e-7))

    def test_autoregressive_decode_stops_at_eos(self):
        encoder_out, _ = build_fake_tensors()
        _, vocab_tokens = build_decoder_vocab(vocab_size=10000)
        decoder = MockDecoder(vocab_tokens=vocab_tokens, d_model=512, vocab_size=10000, seed=7)
        sequence, sentence = decoder.autoregressive_decode(encoder_out=encoder_out, max_steps=4)
        self.assertIn("<EOS>", sequence)
        self.assertLessEqual(len(sequence), 6)
        self.assertNotIn("<START>", sentence)
        self.assertNotIn("<EOS>", sentence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
