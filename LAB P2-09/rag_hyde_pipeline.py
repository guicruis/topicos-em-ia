import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
OUTPUT_DIR = THIS_DIR / "outputs"
FRAGMENTS_FILE = DATA_DIR / "medical_fragments.jsonl"
RETRIEVAL_FILE = OUTPUT_DIR / "retrieval_trace.json"


@dataclass
class LabConfig:
    domain: str = "manuais medicos privados"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    openai_model: str = "gpt-4o-mini"
    embedding_dim: int = 384
    hnsw_m: int = 16
    hnsw_ef_construction: int = 120
    hnsw_ef_search: int = 64
    top_k_retrieve: int = 10
    top_k_final: int = 3
    seed: int = 42


@dataclass
class TechnicalFragment:
    doc_id: str
    title: str
    specialty: str
    text: str


@dataclass
class RetrievedDocument:
    doc_id: str
    title: str
    specialty: str
    text: str
    retrieval_score: float
    rerank_score: float | None = None


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", normalize_text(text))


def build_medical_fragments() -> list[TechnicalFragment]:
    rows = [
        ("MED-001", "Cefaleia com aura", "neurologia", "Cefaleia pulsatil unilateral associada a fotofobia, fonofobia, nausea e escotomas cintilantes sugere migranea com aura."),
        ("MED-002", "Hipertensao grave", "cardiologia", "Pressao arterial acima de 180 por 120 mmHg com dor toracica, dispneia ou deficit neurologico indica emergencia hipertensiva."),
        ("MED-003", "Hipoglicemia", "endocrinologia", "Sudorese fria, tremores, palpitacoes e confusao mental em paciente diabetico sugerem hipoglicemia sintomatica."),
        ("MED-004", "Fotofobia e rigidez nucal", "infectologia", "Febre, cefaleia intensa, fotofobia e rigidez de nuca exigem avaliacao urgente para meningite."),
        ("MED-005", "Dor abdominal migratoria", "cirurgia", "Dor periumbilical que migra para fossa iliaca direita com anorexia e febre baixa e compativel com apendicite aguda."),
        ("MED-006", "Dispneia e sibilancia", "pneumologia", "Episodios de dispneia, tosse noturna, sibilos expiratorios e melhora com broncodilatador sugerem asma."),
        ("MED-007", "Pneumonia comunitaria", "pneumologia", "Tosse produtiva, febre, dor pleuritica e crepitacoes focais indicam pneumonia adquirida na comunidade."),
        ("MED-008", "Tromboembolismo pulmonar", "emergencia", "Dispneia subita, dor pleuritica, taquicardia e dessaturacao apos imobilizacao elevam suspeita de embolia pulmonar."),
        ("MED-009", "Acidente vascular cerebral", "neurologia", "Assimetria facial, disartria e fraqueza unilateral de inicio subito demandam protocolo de AVC."),
        ("MED-010", "Vertigem periferica", "otorrino", "Vertigem rotatoria breve desencadeada por mudanca de posicao da cabeca sugere vertigem posicional paroxistica benigna."),
        ("MED-011", "Anemia ferropriva", "hematologia", "Fadiga, palidez, pica, ferritina baixa e microcitose sao achados tipicos de anemia por deficiencia de ferro."),
        ("MED-012", "Sepse", "emergencia", "Hipotensao, taquipneia, febre ou hipotermia com foco infeccioso sugerem sepse e exigem antibiotico precoce."),
        ("MED-013", "Insuficiencia cardiaca", "cardiologia", "Ortopneia, edema maleolar, estertores bibasais e ganho ponderal rapido sugerem descompensacao de insuficiencia cardiaca."),
        ("MED-014", "Colica renal", "urologia", "Dor lombar intensa irradiada para virilha, hematuria e inquietacao sao compativeis com ureterolitiase."),
        ("MED-015", "Cetoacidose diabetica", "endocrinologia", "Poliuria, polidipsia, vomitos, respiracao de Kussmaul e cetonemia sugerem cetoacidose diabetica."),
        ("MED-016", "Crise tireotoxica", "endocrinologia", "Febre, taquiarritmia, agitacao, diarreia e hipertireoidismo conhecido sugerem tempestade tireoidiana."),
        ("MED-017", "Sinusite bacteriana", "otorrino", "Rinorreia purulenta, dor facial e sintomas por mais de dez dias favorecem rinossinusite bacteriana."),
        ("MED-018", "Glaucoma agudo", "oftalmologia", "Dor ocular intensa, halos coloridos, midriase media e nausea sugerem fechamento angular agudo."),
        ("MED-019", "Reacao anafilatica", "alergologia", "Urticaria difusa, broncoespasmo, edema de glote ou hipotensao apos exposicao a alergeno indicam anafilaxia."),
        ("MED-020", "Artrite septica", "reumatologia", "Monoartrite aguda com febre, derrame articular e dor a mobilizacao passiva requer puncao articular urgente."),
        ("MED-021", "Trombose venosa profunda", "vascular", "Edema unilateral de panturrilha, dor, calor local e antecedente de imobilizacao sugerem TVP."),
        ("MED-022", "Sindrome coronariana", "cardiologia", "Dor precordial opressiva irradiada para braco esquerdo, diaforese e nausea sugerem sindrome coronariana aguda."),
        ("MED-023", "Delirium", "geriatria", "Alteracao aguda e flutuante da atencao em idoso hospitalizado e compativel com delirium."),
        ("MED-024", "Pancreatite aguda", "gastroenterologia", "Dor epigastrica irradiada para dorso, nausea e lipase elevada sugerem pancreatite aguda."),
    ]
    return [TechnicalFragment(*row) for row in rows]


class HashEmbeddingModel:
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts: str | Iterable[str]) -> list[float] | list[list[float]]:
        if isinstance(texts, str):
            return self._encode_one(texts)
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return l2_normalize(vector)


def load_embedding_model(config: LabConfig):
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(config.embedding_model_name)
    except Exception:
        return HashEmbeddingModel(config.embedding_dim)


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class HNSWVectorIndex:
    def __init__(self, vectors: list[list[float]], config: LabConfig):
        self.vectors = vectors
        self.config = config
        self.backend = "exact-fallback"
        self.index: Any = None
        self._build_index()

    def _build_index(self) -> None:
        try:
            import hnswlib
            import numpy as np

            matrix = np.asarray(self.vectors, dtype="float32")
            index = hnswlib.Index(space="cosine", dim=matrix.shape[1])
            index.init_index(
                max_elements=matrix.shape[0],
                ef_construction=self.config.hnsw_ef_construction,
                M=self.config.hnsw_m,
                random_seed=self.config.seed,
            )
            index.add_items(matrix, list(range(matrix.shape[0])))
            index.set_ef(self.config.hnsw_ef_search)
            self.index = index
            self.backend = "hnswlib"
        except Exception:
            self.index = None

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        if self.index is not None:
            import numpy as np

            labels, distances = self.index.knn_query(np.asarray([query_vector], dtype="float32"), k=top_k)
            return [(int(label), float(1.0 - distance)) for label, distance in zip(labels[0], distances[0])]

        scored = [(idx, cosine_similarity(query_vector, vector)) for idx, vector in enumerate(self.vectors)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


def save_fragments(fragments: list[TechnicalFragment]) -> None:
    ensure_directories()
    with FRAGMENTS_FILE.open("w", encoding="utf-8") as handle:
        for fragment in fragments:
            handle.write(json.dumps(asdict(fragment), ensure_ascii=False) + "\n")


def load_fragments() -> list[TechnicalFragment]:
    if not FRAGMENTS_FILE.exists():
        fragments = build_medical_fragments()
        save_fragments(fragments)
        return fragments
    with FRAGMENTS_FILE.open("r", encoding="utf-8") as handle:
        return [TechnicalFragment(**json.loads(line)) for line in handle if line.strip()]


def build_hyde_prompt(query: str) -> str:
    return (
        "Escreva um documento medico tecnico e curto que poderia responder a consulta do paciente. "
        "Use termos clinicos, diagnosticos diferenciais e sinais de alerta. "
        f"Consulta: {query}"
    )


def generate_hypothetical_document(query: str, model_name: str) -> str:
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(model=model_name, input=build_hyde_prompt(query), temperature=0.2)
        return response.output_text.strip()
    except Exception:
        return local_hyde_document(query)


def local_hyde_document(query: str) -> str:
    query_norm = normalize_text(query)
    bridges = {
        "cabeca": "cefaleia intensa, migranea, fotofobia, fonofobia e sinais neurologicos focais",
        "luz": "fotofobia associada a cefaleia, migranea com aura ou meningite conforme febre e rigidez nucal",
        "peito": "dor toracica opressiva, sindrome coronariana aguda, dispneia e diaforese",
        "falta de ar": "dispneia, sibilancia, tromboembolismo pulmonar, pneumonia ou asma",
        "acucar": "hipoglicemia, tremores, sudorese, confusao mental e diabetes mellitus",
        "barriga": "dor abdominal, apendicite, pancreatite ou abdome agudo",
        "olho": "dor ocular, fotofobia, halos coloridos e glaucoma agudo",
    }
    matches = [value for key, value in bridges.items() if key in query_norm]
    if not matches:
        matches = ["sintomas inespecificos, sinais de alerta, diagnostico diferencial e avaliacao clinica"]
    return f"Documento hipotetico HyDE: paciente relata {query}. Termos tecnicos relevantes: {'; '.join(matches)}."


def retrieve_documents(
    query: str,
    fragments: list[TechnicalFragment],
    embedding_model: Any,
    config: LabConfig,
) -> tuple[str, list[RetrievedDocument], str]:
    hypothetical_document = generate_hypothetical_document(query, config.openai_model)
    corpus_vectors = embedding_model.encode([fragment.text for fragment in fragments])
    query_vector = embedding_model.encode(hypothetical_document)
    index = HNSWVectorIndex(corpus_vectors, config)
    retrieved = []
    for idx, score in index.search(query_vector, config.top_k_retrieve):
        fragment = fragments[idx]
        retrieved.append(
            RetrievedDocument(
                doc_id=fragment.doc_id,
                title=fragment.title,
                specialty=fragment.specialty,
                text=fragment.text,
                retrieval_score=score,
            )
        )
    return hypothetical_document, retrieved, index.backend


def lexical_cross_score(query: str, document: str) -> float:
    query_terms = set(tokenize(query))
    doc_terms = set(tokenize(document))
    if not query_terms:
        return 0.0
    overlap = len(query_terms & doc_terms) / len(query_terms)
    medical_boosts = {
        "cabeca": ["cefaleia", "migranea", "fotofobia"],
        "luz": ["fotofobia", "aura", "meningite"],
        "peito": ["precordial", "coronariana", "toracica"],
        "ar": ["dispneia", "asma", "pulmonar"],
        "olho": ["ocular", "glaucoma", "halos"],
    }
    boost = 0.0
    doc_norm = normalize_text(document)
    query_norm = normalize_text(query)
    for plain_term, technical_terms in medical_boosts.items():
        if plain_term in query_norm:
            boost += sum(0.15 for term in technical_terms if term in doc_norm)
    return overlap + boost


def rerank_with_cross_encoder(query: str, documents: list[RetrievedDocument], model_name: str) -> list[RetrievedDocument]:
    pairs = [(query, document.text) for document in documents]
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        scores = [float(score) for score in model.predict(pairs)]
    except Exception:
        scores = [lexical_cross_score(query, document.text) for document in documents]

    reranked = []
    for document, score in zip(documents, scores):
        reranked.append(
            RetrievedDocument(
                doc_id=document.doc_id,
                title=document.title,
                specialty=document.specialty,
                text=document.text,
                retrieval_score=document.retrieval_score,
                rerank_score=score,
            )
        )
    reranked.sort(key=lambda item: item.rerank_score if item.rerank_score is not None else -999.0, reverse=True)
    return reranked


def run_rag_pipeline(query: str, config: LabConfig | None = None) -> dict[str, Any]:
    config = config or LabConfig()
    ensure_directories()
    fragments = load_fragments()
    embedding_model = load_embedding_model(config)
    hypothetical_document, retrieved, backend = retrieve_documents(query, fragments, embedding_model, config)
    reranked = rerank_with_cross_encoder(query, retrieved, config.cross_encoder_model_name)
    result = {
        "query": query,
        "hypothetical_document": hypothetical_document,
        "index_backend": backend,
        "hnsw_parameters": {
            "M": config.hnsw_m,
            "ef_construction": config.hnsw_ef_construction,
            "ef_search": config.hnsw_ef_search,
        },
        "top_10_retrieved": [asdict(document) for document in retrieved],
        "top_3_final": [asdict(document) for document in reranked[: config.top_k_final]],
    }
    with RETRIEVAL_FILE.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def describe_pipeline(config: LabConfig) -> dict[str, Any]:
    return {
        "lab_config": asdict(config),
        "dataset_size": len(build_medical_fragments()),
        "steps": [
            "indexacao de fragmentos tecnicos",
            "transformacao HyDE da query coloquial",
            "busca top-10 por similaridade no indice HNSW",
            "re-ranking dos 10 candidatos por Cross-Encoder",
            "selecao final top-3 para contexto do LLM",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline RAG avancado com HNSW, HyDE e Cross-Encoder.")
    parser.add_argument("command", choices=["prepare-data", "query", "describe"])
    parser.add_argument("--query", default="dor de cabeca latejante e luz incomodando")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LabConfig()
    if args.command == "prepare-data":
        fragments = build_medical_fragments()
        save_fragments(fragments)
        print(json.dumps({"fragments": len(fragments), "path": str(FRAGMENTS_FILE)}, indent=2, ensure_ascii=False))
        return
    if args.command == "describe":
        print(json.dumps(describe_pipeline(config), indent=2, ensure_ascii=False))
        return
    result = run_rag_pipeline(args.query, config)
    print("Top-10 recuperados no funil largo:")
    for idx, document in enumerate(result["top_10_retrieved"], start=1):
        print(f"{idx:02d}. {document['doc_id']} | {document['title']} | score={document['retrieval_score']:.4f}")
    print("\nTop-3 finais apos Cross-Encoder:")
    for idx, document in enumerate(result["top_3_final"], start=1):
        print(f"{idx:02d}. {document['doc_id']} | {document['title']} | rerank={document['rerank_score']:.4f}")


if __name__ == "__main__":
    main()
