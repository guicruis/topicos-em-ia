import unittest
from pathlib import Path
import sys

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from attention import SelfAttention, scaled_dot_product_attention, softmax


class TesteSoftmax(unittest.TesteCase):
    def test_softmax_rows_sum_to_one(self):
        x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
        y = softmax(x, axis=-1)
        sums = np.sum(y, axis=-1)
        self.assertTrue(np.allclose(sums, np.ones_like(sums), atol=1e-7))

    def test_softmax_is_stable(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        y = softmax(x)
        self.assertFalse(np.any(np.isnan(y)))
        self.assertTrue(np.all(y > 0.0))


class TesteScaledDotProductAttention(unittest.TesteCase):
    def test_output_and_weights_shapes(self):
        q = np.random.randn(2, 3, 4)  # (batch, seq_q, d_k)
        k = np.random.randn(2, 5, 4)  # (batch, seq_k, d_k)
        v = np.random.randn(2, 5, 6)  # (batch, seq_k, d_v)

        out, w = scaled_dot_product_attention(q, k, v, return_pesos=True)

        self.assertEqual(out.shape, (2, 3, 6))
        self.assertEqual(w.shape, (2, 3, 5))
        self.assertTrue(np.allclose(np.sum(w, axis=-1), 1.0, atol=1e-7))

    def test_mask_blocks_positions(self):
        q = np.array([[[1.0, 0.0]]])          # (1, 1, 2)
        k = np.array([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
        v = np.array([[[10.0, 0.0], [0.0, 10.0]]])  # (1, 2, 2)
        mask = np.array([[[1, 0]]], dtype=bool)  

        out, w = scaled_dot_product_attention(q, k, v, mask=mask, return_pesos=True)

        self.assertTrue(np.allclose(w[0, 0, 1], 0.0, atol=1e-7))
        self.assertTrue(np.allclose(out[0, 0], np.array([10.0, 0.0]), atol=1e-6))


class TestSelfAttention(unittest.TesteCase):
    def test_forward_accepts_2d_input(self):
        x = np.random.randn(4, 8)  # (seq_len, d_model)
        attn = SelfAttention(d_model=8, seed=1)
        out, w = attn.forward(x, return_pesos=True)

        self.assertEqual(out.shape, (4, 8))
        self.assertEqual(w.shape, (4, 4))
        self.assertTrue(np.allclose(np.sum(w, axis=-1), 1.0, atol=1e-7))

    def test_forward_accepts_3d_input(self):
        x = np.random.randn(2, 4, 8)  # (batch, seq_len, d_model)
        attn = SelfAttention(d_model=8, seed=1)
        out, w = attn.forward(x, return_pesos=True)

        self.assertEqual(out.shape, (2, 4, 8))
        self.assertEqual(w.shape, (2, 4, 4))
        self.assertTrue(np.allclose(np.sum(w, axis=-1), 1.0, atol=1e-7))

    def test_dimension_mismatch_raises(self):
        x = np.random.randn(4, 7)  # d_model mismatch
        attn = SelfAttention(d_model=8, seed=1)
        with self.assertRaises(ValueError):
            attn.forward(x)

    def test_invalid_d_model_raises(self):
        with self.assertRaises(ValueError):
            SelfAttention(d_model=0)

    def test_seed_makes_initialization_deterministic(self):
        x = np.random.randn(3, 6)
        attn1 = SelfAttention(d_model=6, seed=123)
        attn2 = SelfAttention(d_model=6, seed=123)

        out1 = attn1.forward(x)
        out2 = attn2.forward(x)

        self.assertTrue(np.allclose(out1, out2, atol=1e-10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
