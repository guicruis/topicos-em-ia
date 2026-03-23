import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_transformer import TransformerSeq2Seq, collate_batch, train_one_epoch


class TestTransformerTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.pad_id = 0
        self.model = TransformerSeq2Seq(
            vocab_size=32,
            pad_id=self.pad_id,
            d_model=32,
            num_heads=4,
            d_ff=64,
            num_layers=2,
            dropout=0.0,
            max_len=32,
        )

    def test_forward_output_shape(self):
        src_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
        tgt_ids = torch.tensor([[6, 7, 8], [9, 10, 0]], dtype=torch.long)
        logits = self.model(src_ids, tgt_ids)
        self.assertEqual(tuple(logits.shape), (2, 3, 32))

    def test_target_mask_blocks_future_tokens(self):
        tgt_ids = torch.tensor([[6, 7, 8, 0]], dtype=torch.long)
        mask = self.model.create_target_mask(tgt_ids)
        self.assertTrue(mask[0, 0, 0, 1].item())
        self.assertTrue(mask[0, 0, 1, 2].item())
        self.assertFalse(mask[0, 0, 2, 1].item())

    def test_single_epoch_runs_and_returns_loss(self):
        samples = [
            {"source_text": "a", "target_text": "x", "src_ids": [1, 2, 3], "tgt_ids": [4, 5, 6, 7]},
            {"source_text": "b", "target_text": "y", "src_ids": [2, 3], "tgt_ids": [4, 8, 9, 7]},
            {"source_text": "c", "target_text": "z", "src_ids": [3, 4, 5], "tgt_ids": [4, 10, 11, 7]},
            {"source_text": "d", "target_text": "w", "src_ids": [5, 6], "tgt_ids": [4, 12, 13, 7]},
        ]
        dataloader = DataLoader(
            samples,
            batch_size=2,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, pad_id=self.pad_id),
        )
        optimizer = Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        avg_loss, avg_grad_norm = train_one_epoch(
            model=self.model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
            pad_id=self.pad_id,
        )

        self.assertGreater(avg_loss, 0.0)
        self.assertGreater(avg_grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
