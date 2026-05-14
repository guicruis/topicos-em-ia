import sys
import unittest
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from rag_hyde_pipeline import (
    HNSWVectorIndex,
    HashEmbeddingModel,
    LabConfig,
    build_medical_fragments,
    describe_pipeline,
    local_hyde_document,
    rerank_with_cross_encoder,
    retrieve_documents,
    run_rag_pipeline,
)


class TestRagHydePipeline(unittest.TestCase):
    def setUp(self):
        self.config = LabConfig(top_k_retrieve=10, top_k_final=3)
        self.fragments = build_medical_fragments()

    def test_dataset_has_at_least_twenty_fragments(self):
        self.assertGreaterEqual(len(self.fragments), 20)
        self.assertTrue(all(fragment.doc_id.startswith("MED-") for fragment in self.fragments))

    def test_hnsw_index_exposes_required_parameters(self):
        model = HashEmbeddingModel(self.config.embedding_dim)
        vectors = model.encode([fragment.text for fragment in self.fragments])
        index = HNSWVectorIndex(vectors, self.config)
        self.assertIn(index.backend, {"hnswlib", "exact-fallback"})
        self.assertEqual(index.config.hnsw_m, 16)
        self.assertEqual(index.config.hnsw_ef_construction, 120)

    def test_hyde_expands_colloquial_query_to_technical_terms(self):
        document = local_hyde_document("dor de cabeca latejante e luz incomodando").lower()
        self.assertIn("cefaleia", document)
        self.assertIn("fotofobia", document)

    def test_retrieve_returns_top_ten_documents(self):
        model = HashEmbeddingModel(self.config.embedding_dim)
        _, retrieved, _ = retrieve_documents(
            "dor de cabeca latejante e luz incomodando",
            self.fragments,
            model,
            self.config,
        )
        self.assertEqual(len(retrieved), 10)

    def test_cross_encoder_rerank_returns_top_three_candidates(self):
        model = HashEmbeddingModel(self.config.embedding_dim)
        _, retrieved, _ = retrieve_documents("dor de cabeca e luz forte", self.fragments, model, self.config)
        reranked = rerank_with_cross_encoder("dor de cabeca e luz forte", retrieved, self.config.cross_encoder_model_name)
        self.assertEqual(len(reranked[:3]), 3)
        self.assertIsNotNone(reranked[0].rerank_score)

    def test_pipeline_result_contains_requested_outputs(self):
        result = run_rag_pipeline("dor de cabeca latejante e luz incomodando", self.config)
        self.assertEqual(len(result["top_10_retrieved"]), 10)
        self.assertEqual(len(result["top_3_final"]), 3)
        self.assertIn("M", result["hnsw_parameters"])

    def test_describe_pipeline_mentions_core_steps(self):
        description = describe_pipeline(self.config)
        joined = " ".join(description["steps"]).lower()
        self.assertIn("hyde", joined)
        self.assertIn("cross-encoder", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
