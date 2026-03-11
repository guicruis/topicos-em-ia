import numpy as np
import pandas as pd


def build_vocab_dataframe(words: list[str]) -> pd.DataFrame:
    vocab = pd.DataFrame(
        {
            "token": words,
            "token_id": np.arange(len(words), dtype=int),
        }
    )
    return vocab


def sentence_to_token_ids(sentence: str, vocab_df: pd.DataFrame) -> list[int]:
    token_to_id = dict(zip(vocab_df["token"], vocab_df["token_id"]))
    tokens = sentence.lower().split()
    missing = [token for token in tokens if token not in token_to_id]
    if missing:
        raise ValueError(f"Tokens fora do vocabulario: {missing}")
    return [token_to_id[token] for token in tokens]


def create_embeddings(vocab_size: int, d_model: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((vocab_size, d_model))


def prepare_input_tensor(
    sentence: str,
    vocab_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[list[int], np.ndarray]:
    token_ids = sentence_to_token_ids(sentence, vocab_df)
    x = embeddings[token_ids]
    x = np.expand_dims(x, axis=0)
    return token_ids, x


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class SelfAttention:
    def __init__(self, d_model: int, d_k: int | None = None, d_v: int | None = None, seed: int = 42):
        if d_model <= 0:
            raise ValueError("d_model deve ser maior que zero")

        self.d_model = d_model
        self.d_k = d_model if d_k is None else d_k
        self.d_v = d_model if d_v is None else d_v

        rng = np.random.default_rng(seed)
        self.W_q = rng.standard_normal((d_model, self.d_k)) / np.sqrt(d_model)
        self.W_k = rng.standard_normal((d_model, self.d_k)) / np.sqrt(d_model)
        self.W_v = rng.standard_normal((d_model, self.d_v)) / np.sqrt(d_model)

    def forward(self, x: np.ndarray, return_attention: bool = False):
        if x.ndim != 3:
            raise ValueError("x deve ter shape (batch_size, sequence_length, d_model)")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Ultima dimensao esperada: {self.d_model}. Recebido: {x.shape[-1]}")

        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(self.d_k)
        attention_weights = softmax(scores, axis=-1)
        output = attention_weights @ v

        if return_attention:
            return output, attention_weights
        return output


class FeedForwardNetwork:
    def __init__(self, d_model: int, d_ff: int = 256, seed: int = 42):
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model e d_ff devem ser maiores que zero")

        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_model, d_ff)) / np.sqrt(d_model)
        self.b1 = np.zeros((d_ff,))
        self.W2 = rng.standard_normal((d_ff, d_model)) / np.sqrt(d_ff)
        self.b2 = np.zeros((d_model,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = relu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class EncoderLayer:
    def __init__(self, d_model: int, d_ff: int = 256, seed: int = 42):
        self.self_attention = SelfAttention(d_model=d_model, seed=seed)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, seed=seed + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_att = self.self_attention.forward(x)
        x_norm1 = layer_norm(x + x_att)
        x_ffn = self.ffn.forward(x_norm1)
        x_out = layer_norm(x_norm1 + x_ffn)
        return x_out


class TransformerEncoder:
    def __init__(self, num_layers: int = 6, d_model: int = 64, d_ff: int = 256, seed: int = 42):
        if num_layers <= 0:
            raise ValueError("num_layers deve ser maior que zero")

        self.num_layers = num_layers
        self.d_model = d_model
        self.layers = [
            EncoderLayer(d_model=d_model, d_ff=d_ff, seed=seed + layer_idx * 10)
            for layer_idx in range(num_layers)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x
        for layer in self.layers:
            z = layer.forward(z)
        return z


def build_demo_encoder_input(
    d_model: int = 64,
    seed: int = 42,
) -> tuple[pd.DataFrame, str, list[int], np.ndarray, np.ndarray]:
    words = ["o", "banco", "bloqueou", "cartao", "do", "cliente"]
    sentence = "o banco bloqueou cartao do cliente"
    vocab_df = build_vocab_dataframe(words)
    embeddings = create_embeddings(vocab_size=len(vocab_df), d_model=d_model, seed=seed)
    token_ids, x = prepare_input_tensor(sentence, vocab_df, embeddings)
    return vocab_df, sentence, token_ids, embeddings, x


def main():
    d_model = 64
    d_ff = 256
    num_layers = 6

    vocab_df, sentence, token_ids, embeddings, x = build_demo_encoder_input(d_model=d_model, seed=42)
    encoder = TransformerEncoder(num_layers=num_layers, d_model=d_model, d_ff=d_ff, seed=42)
    z = encoder.forward(x)

    print("Frase:", sentence)
    print("Vocabulario:")
    print(vocab_df)
    print("Token IDs:", token_ids)
    print("Tabela de embeddings:", embeddings.shape)
    print("Shape de X:", x.shape)
    print("Shape de Z:", z.shape)
    print("Primeiro vetor contextualizado Z[0, 0, :5]:", np.round(z[0, 0, :5], 4))


if __name__ == "__main__":
    main()
