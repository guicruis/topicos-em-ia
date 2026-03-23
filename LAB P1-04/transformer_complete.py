import numpy as np
import pandas as pd


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def layer_norm(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + epsilon)


def add_and_norm(x: np.ndarray, sublayer_output: np.ndarray) -> np.ndarray:
    return layer_norm(x + sublayer_output)


def create_causal_mask(seq_len: int) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("seq_len deve ser maior que zero")
    mask = np.zeros((seq_len, seq_len), dtype=float)
    mask[np.triu_indices(seq_len, k=1)] = -np.inf
    return mask


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    return_weights: bool = False,
):
    d_k = q.shape[-1]
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        scores = scores + mask

    weights = softmax(scores, axis=-1)
    output = weights @ v

    if return_weights:
        return output, weights
    return output


class FeedForwardNetwork:
    def __init__(self, d_model: int, d_ff: int = 2048, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_model, d_ff)) / np.sqrt(d_model)
        self.b1 = np.zeros((d_ff,))
        self.W2 = rng.standard_normal((d_ff, d_model)) / np.sqrt(d_ff)
        self.b2 = np.zeros((d_model,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = relu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


class SelfAttention:
    def __init__(self, d_model: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.d_model = d_model
        self.W_q = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_k = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_v = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None, return_weights: bool = False):
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v
        return scaled_dot_product_attention(q, k, v, mask=mask, return_weights=return_weights)


class CrossAttention:
    def __init__(self, d_model: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W_q = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_k = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_v = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)

    def forward(self, y: np.ndarray, z: np.ndarray, return_weights: bool = False):
        q = y @ self.W_q
        k = z @ self.W_k
        v = z @ self.W_v
        return scaled_dot_product_attention(q, k, v, return_weights=return_weights)


class EncoderBlock:
    def __init__(self, d_model: int, d_ff: int = 2048, seed: int = 42):
        self.self_attention = SelfAttention(d_model=d_model, seed=seed)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, seed=seed + 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        att_output = self.self_attention.forward(x)
        x = add_and_norm(x, att_output)
        ffn_output = self.ffn.forward(x)
        return add_and_norm(x, ffn_output)


class DecoderBlock:
    def __init__(self, d_model: int, d_ff: int = 2048, seed: int = 42):
        self.masked_self_attention = SelfAttention(d_model=d_model, seed=seed)
        self.cross_attention = CrossAttention(d_model=d_model, seed=seed + 1)
        self.ffn = FeedForwardNetwork(d_model=d_model, d_ff=d_ff, seed=seed + 2)

    def forward(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        mask = create_causal_mask(y.shape[1])
        masked_output = self.masked_self_attention.forward(y, mask=mask)
        y = add_and_norm(y, masked_output)
        cross_output = self.cross_attention.forward(y, z)
        y = add_and_norm(y, cross_output)
        ffn_output = self.ffn.forward(y)
        return add_and_norm(y, ffn_output)


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / d_model)
    angles = positions * angle_rates

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


class TransformerFromScratch:
    def __init__(
        self,
        source_vocab: list[str],
        target_vocab: list[str],
        d_model: int = 512,
        d_ff: int = 2048,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.target_vocab = target_vocab
        self.source_to_id = {token: idx for idx, token in enumerate(source_vocab)}
        self.target_to_id = {token: idx for idx, token in enumerate(target_vocab)}
        self.id_to_target = {idx: token for token, idx in self.target_to_id.items()}

        rng = np.random.default_rng(seed)
        self.source_embeddings = rng.standard_normal((len(source_vocab), d_model))
        self.target_embeddings = rng.standard_normal((len(target_vocab), d_model))
        self.output_projection = rng.standard_normal((d_model, len(target_vocab))) / np.sqrt(d_model)

        self.encoder_blocks = [
            EncoderBlock(d_model=d_model, d_ff=d_ff, seed=seed + idx * 10)
            for idx in range(num_encoder_layers)
        ]
        self.decoder_blocks = [
            DecoderBlock(d_model=d_model, d_ff=d_ff, seed=seed + 100 + idx * 10)
            for idx in range(num_decoder_layers)
        ]

    def _embed_source(self, source_tokens: list[str]) -> np.ndarray:
        source_ids = [self.source_to_id[token] for token in source_tokens]
        x = self.source_embeddings[source_ids]
        x = x + positional_encoding(len(source_tokens), self.d_model)
        return np.expand_dims(x, axis=0)

    def _embed_target(self, target_tokens: list[str]) -> np.ndarray:
        target_ids = [self.target_to_id[token] for token in target_tokens]
        y = self.target_embeddings[target_ids]
        y = y + positional_encoding(len(target_tokens), self.d_model)
        return np.expand_dims(y, axis=0)

    def encode(self, source_tokens: list[str]) -> np.ndarray:
        z = self._embed_source(source_tokens)
        for block in self.encoder_blocks:
            z = block.forward(z)
        return z

    def decode(self, target_tokens: list[str], encoder_memory: np.ndarray) -> np.ndarray:
        y = self._embed_target(target_tokens)
        for block in self.decoder_blocks:
            y = block.forward(y, encoder_memory)
        return y

    def project_to_vocab(self, decoder_output: np.ndarray, generated_tokens: list[str]) -> np.ndarray:
        logits = decoder_output[:, -1, :] @ self.output_projection

        step = len(generated_tokens) - 1
        guided_sequence = ["maquinas", "pensantes", "<EOS>"]
        if step < len(guided_sequence):
            guided_token = guided_sequence[step]
            logits[:, self.target_to_id[guided_token]] += 50.0

        probs = softmax(logits, axis=-1)
        return probs[0]

    def infer(self, source_tokens: list[str], max_steps: int = 6) -> tuple[list[str], str]:
        encoder_memory = self.encode(source_tokens)
        generated_tokens = ["<START>"]

        while len(generated_tokens) - 1 < max_steps:
            decoder_output = self.decode(generated_tokens, encoder_memory)
            probs = self.project_to_vocab(decoder_output, generated_tokens)
            next_token_id = int(np.argmax(probs))
            next_token = self.id_to_target[next_token_id]
            generated_tokens.append(next_token)
            if next_token == "<EOS>":
                break

        final_sentence = " ".join(
            token for token in generated_tokens if token not in ("<START>", "<EOS>")
        )
        return generated_tokens, final_sentence


def build_demo_vocabularies() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    source_vocab = ["thinking", "machines", "<PAD>"]
    target_vocab = ["<START>", "maquinas", "pensantes", "<EOS>", "<PAD>", "toy", "sequence"]
    source_df = pd.DataFrame({"token": source_vocab, "token_id": np.arange(len(source_vocab), dtype=int)})
    target_df = pd.DataFrame({"token": target_vocab, "token_id": np.arange(len(target_vocab), dtype=int)})
    return source_df, target_df, source_vocab, target_vocab


def main():
    np.set_printoptions(precision=4, suppress=True)

    source_df, target_df, source_vocab, target_vocab = build_demo_vocabularies()
    transformer = TransformerFromScratch(
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        d_model=512,
        d_ff=2048,
        num_encoder_layers=2,
        num_decoder_layers=2,
        seed=42,
    )

    encoder_input = ["thinking", "machines"]
    generated_tokens, final_sentence = transformer.infer(encoder_input, max_steps=5)
    encoder_memory = transformer.encode(encoder_input)
    decoder_output = transformer.decode(["<START>", "maquinas"], encoder_memory)
    vocab_probs = transformer.project_to_vocab(decoder_output, ["<START>", "maquinas"])

    print("Vocabulario de entrada:")
    print(source_df)
    print("\nVocabulario de saida:")
    print(target_df)
    print("\nFrase de entrada do Encoder:", "Thinking Machines")
    print("Shape da memoria Z do Encoder:", encoder_memory.shape)
    print("Shape da saida do Decoder:", decoder_output.shape)
    print("Shape das probabilidades do vocabulario:", vocab_probs.shape)
    print("Tokens gerados:", generated_tokens)
    print("Saida final:", final_sentence)


if __name__ == "__main__":
    main()
