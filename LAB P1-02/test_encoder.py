import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from encoder import (
    EncoderLayer,
    FeedForwardNetwork,
    SelfAttention,
    TransformerEncoder,
    build_demo_encoder_input,
    build_vocab_dataframe,
    create_embeddings,
    layer_norm,
    prepare_input_tensor,
    sentence_to_token_ids,
    softmax,
)


class TestDataPreparation(unittest.TestCase):
    def test_vocab_dataframe_structure(self):
        vocab_df = build_vocab_dataframe(["o", "banco", "bloqueou"])
        self.assertIsInstance(vocab_df, pd.DataFrame)
        self.assertEqual(list(vocab_df.columns), ["token", "token_id"])
        self.assertEqual(vocab_df.shape, (3, 2))

    def test_sentence_to_token_ids(self):
        vocab_df = build_vocab_dataframe(["o", "banco", "bloqueou"])
        token_ids = sentence_to_token_ids("o banco bloqueou", vocab_df)
        self.assertEqual(token_ids, [0, 1, 2])

    def test_prepare_input_tensor_shape(self):
        vocab_df = build_vocab_dataframe(["o", "banco", "bloqueou"])
        embeddings = create_embeddings(vocab_size=3, d_model=64, seed=1)
        token_ids, x = prepare_input_tensor("o banco bloqueou", vocab_df, embeddings)
        self.assertEqual(token_ids, [0, 1, 2])
        self.assertEqual(x.shape, (1, 3, 64))


class TestMathBlocks(unittest.TestCase):
    def test_softmax_normalizes_last_axis(self):
        x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
        y = softmax(x, axis=-1)
        self.assertTrue(np.allclose(np.sum(y, axis=-1), np.ones(2), atol=1e-7))

    def test_layer_norm_zero_mean_unit_variance(self):
        x = np.array([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]])
        y = layer_norm(x)
        self.assertTrue(np.allclose(np.mean(y, axis=-1), 0.0, atol=1e-6))
        self.assertTrue(np.allclose(np.var(y, axis=-1), 1.0, atol=1e-5))

    def test_self_attention_output_shape(self):
        x = np.random.default_rng(7).standard_normal((2, 4, 64))
        attention = SelfAttention(d_model=64, seed=7)
        output, weights = attention.forward(x, return_attention=True)
        self.assertEqual(output.shape, (2, 4, 64))
        self.assertEqual(weights.shape, (2, 4, 4))
        self.assertTrue(np.allclose(np.sum(weights, axis=-1), 1.0, atol=1e-7))

    def test_ffn_preserves_last_dimension(self):
        x = np.random.default_rng(11).standard_normal((2, 5, 64))
        ffn = FeedForwardNetwork(d_model=64, d_ff=256, seed=11)
        output = ffn.forward(x)
        self.assertEqual(output.shape, (2, 5, 64))


class TestEncoder(unittest.TestCase):
    def test_encoder_layer_preserves_shape(self):
        x = np.random.default_rng(13).standard_normal((1, 6, 64))
        layer = EncoderLayer(d_model=64, d_ff=256, seed=13)
        output = layer.forward(x)
        self.assertEqual(output.shape, (1, 6, 64))

    def test_transformer_encoder_six_layers_preserves_shape(self):
        _, _, _, _, x = build_demo_encoder_input(d_model=64, seed=21)
        encoder = TransformerEncoder(num_layers=6, d_model=64, d_ff=256, seed=21)
        z = encoder.forward(x)
        self.assertEqual(z.shape, x.shape)

    def test_transformer_encoder_is_deterministic_with_same_seed(self):
        _, _, _, _, x = build_demo_encoder_input(d_model=64, seed=30)
        encoder_a = TransformerEncoder(num_layers=6, d_model=64, d_ff=256, seed=30)
        encoder_b = TransformerEncoder(num_layers=6, d_model=64, d_ff=256, seed=30)
        z_a = encoder_a.forward(x)
        z_b = encoder_b.forward(x)
        self.assertTrue(np.allclose(z_a, z_b, atol=1e-10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
