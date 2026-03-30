import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bpe_wordpiece import INITIAL_VOCAB, get_stats, merge_vocab, train_bpe


class TestBPEWordPiece(unittest.TestCase):
    def test_get_stats_counts_adjacent_pairs(self):
        stats = get_stats(INITIAL_VOCAB)
        self.assertEqual(stats[("e", "s")], 9)

    def test_merge_vocab_replaces_best_pair(self):
        merged = merge_vocab(("e", "s"), INITIAL_VOCAB)
        self.assertIn("n e w es t </w>", merged)
        self.assertIn("w i d es t </w>", merged)

    def test_train_bpe_runs_five_iterations(self):
        history = train_bpe(INITIAL_VOCAB, num_merges=5)
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0]["best_pair"], ("e", "s"))
        merged_vocab_strings = " ".join(history[-1]["vocab"].keys())
        self.assertIn("est", merged_vocab_strings)


if __name__ == "__main__":
    unittest.main(verbosity=2)
