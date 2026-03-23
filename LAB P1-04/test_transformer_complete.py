import sys
import unittest
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from transformer_complete import (
    DecoderBlock,
    EncoderBlock,
    TransformerFromScratch,
    build_demo_vocabularies,
    create_causal_mask,
    scaled_dot_product_attention,
)


class TestAttentionUtilities(unittest.TestCase):
    def test_causal_mask_has_negative_infinity_above_diagonal(self):
        mask = create_causal_mask(4)
        self.assertEqual(mask.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(mask), 0.0))
        self.assertTrue(np.all(np.isneginf(mask[np.triu_indices(4, k=1)])))

    def test_scaled_attention_respects_causal_mask(self):
        q = np.array([[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]])
        k = q.copy()
        v = np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
        _, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=create_causal_mask(3),
            return_weights=True,
        )
        self.assertTrue(np.allclose(np.triu(weights[0], k=1), 0.0, atol=1e-12))


class TestBlocks(unittest.TestCase):
    def test_encoder_block_preserves_shape(self):
        x = np.random.default_rng(7).standard_normal((1, 3, 512))
        block = EncoderBlock(d_model=512, d_ff=2048, seed=7)
        output = block.forward(x)
        self.assertEqual(output.shape, (1, 3, 512))

    def test_decoder_block_preserves_shape(self):
        y = np.random.default_rng(9).standard_normal((1, 4, 512))
        z = np.random.default_rng(10).standard_normal((1, 2, 512))
        block = DecoderBlock(d_model=512, d_ff=2048, seed=9)
        output = block.forward(y, z)
        self.assertEqual(output.shape, (1, 4, 512))


class TestCompleteTransformer(unittest.TestCase):
    def test_encoder_decoder_pipeline_shapes(self):
        _, _, source_vocab, target_vocab = build_demo_vocabularies()
        model = TransformerFromScratch(
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            d_model=512,
            d_ff=2048,
            num_encoder_layers=2,
            num_decoder_layers=2,
            seed=42,
        )
        z = model.encode(["thinking", "machines"])
        y = model.decode(["<START>", "maquinas"], z)
        probs = model.project_to_vocab(y, ["<START>", "maquinas"])
        self.assertEqual(z.shape, (1, 2, 512))
        self.assertEqual(y.shape, (1, 2, 512))
        self.assertEqual(probs.shape, (len(target_vocab),))
        self.assertTrue(np.allclose(np.sum(probs), 1.0, atol=1e-7))

    def test_inference_stops_at_eos(self):
        _, _, source_vocab, target_vocab = build_demo_vocabularies()
        model = TransformerFromScratch(
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            d_model=512,
            d_ff=2048,
            num_encoder_layers=2,
            num_decoder_layers=2,
            seed=42,
        )
        generated_tokens, final_sentence = model.infer(["thinking", "machines"], max_steps=5)
        self.assertEqual(generated_tokens[0], "<START>")
        self.assertIn("<EOS>", generated_tokens)
        self.assertEqual(final_sentence, "maquinas pensantes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
