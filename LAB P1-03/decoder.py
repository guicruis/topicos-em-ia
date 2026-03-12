import numpy as np
import pandas as pd


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def create_causal_mask(seq_len: int) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("seq_len deve ser maior que zero")
    mask = np.full((seq_len, seq_len), -np.inf)
    return np.triu(mask, k=1) + np.tril(np.zeros((seq_len, seq_len)))


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


def run_causal_mask_demo(seq_len: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    k = q.copy()
    v = np.arange(seq_len * 4, dtype=float).reshape(seq_len, 4)
    mask = create_causal_mask(seq_len)
    _, weights = scaled_dot_product_attention(
        np.expand_dims(q, axis=0),
        np.expand_dims(k, axis=0),
        np.expand_dims(v, axis=0),
        mask=mask,
        return_weights=True,
    )
    return mask, q, weights[0]


def cross_attention(
    encoder_out: np.ndarray,
    decoder_state: np.ndarray,
    seed: int = 42,
    return_weights: bool = False,
):
    if encoder_out.ndim != 3 or decoder_state.ndim != 3:
        raise ValueError("encoder_out e decoder_state devem ter shape 3D")
    if encoder_out.shape[0] != decoder_state.shape[0]:
        raise ValueError("encoder_out e decoder_state devem ter o mesmo batch_size")
    if encoder_out.shape[-1] != decoder_state.shape[-1]:
        raise ValueError("encoder_out e decoder_state devem ter o mesmo d_model")

    d_model = encoder_out.shape[-1]
    rng = np.random.default_rng(seed)
    w_q = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
    w_k = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
    w_v = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)

    q = decoder_state @ w_q
    k = encoder_out @ w_k
    v = encoder_out @ w_v

    return scaled_dot_product_attention(q, k, v, return_weights=return_weights)


class MockDecoder:
    def __init__(self, vocab_tokens: list[str], d_model: int = 512, vocab_size: int = 10000, seed: int = 42):
        if d_model <= 0 or vocab_size <= 0:
            raise ValueError("d_model e vocab_size devem ser maiores que zero")
        if "<START>" not in vocab_tokens or "<EOS>" not in vocab_tokens:
            raise ValueError("vocab_tokens deve conter <START> e <EOS>")

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.vocab_tokens = vocab_tokens
        self.token_to_id = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        rng = np.random.default_rng(seed)
        self.token_embeddings = rng.standard_normal((vocab_size, d_model))
        self.w_q = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.w_k = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.w_v = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.w_out = rng.standard_normal((d_model, vocab_size)) / np.sqrt(d_model)

    def _tokens_to_state(self, current_sequence: list[str]) -> np.ndarray:
        token_ids = [self.token_to_id[token] for token in current_sequence]
        decoder_state = self.token_embeddings[token_ids]
        return np.expand_dims(decoder_state, axis=0)

    def masked_self_attention(self, decoder_state: np.ndarray, return_weights: bool = False):
        seq_len = decoder_state.shape[1]
        mask = create_causal_mask(seq_len)
        q = decoder_state @ self.w_q
        k = decoder_state @ self.w_k
        v = decoder_state @ self.w_v
        return scaled_dot_product_attention(q, k, v, mask=mask, return_weights=return_weights)

    def generate_next_token(
        self,
        current_sequence: list[str],
        encoder_out: np.ndarray,
        force_eos_after: int = 10,
    ) -> np.ndarray:
        decoder_state = self._tokens_to_state(current_sequence)
        masked_out = self.masked_self_attention(decoder_state)
        cross_out = cross_attention(encoder_out, masked_out, seed=123, return_weights=False)
        last_hidden = cross_out[:, -1, :]
        logits = last_hidden @ self.w_out

        banned_ids = [self.token_to_id["<START>"]]
        if len(current_sequence) < force_eos_after:
            banned_ids.append(self.token_to_id["<EOS>"])
        else:
            logits[:, self.token_to_id["<EOS>"]] = 1e9

        logits[:, banned_ids] = -1e9
        probs = softmax(logits, axis=-1)
        return probs[0]

    def autoregressive_decode(
        self,
        encoder_out: np.ndarray,
        start_token: str = "<START>",
        eos_token: str = "<EOS>",
        max_steps: int = 10,
    ) -> tuple[list[str], str]:
        sequence = [start_token]

        while len(sequence) - 1 < max_steps:
            probs = self.generate_next_token(sequence, encoder_out, force_eos_after=max_steps)
            next_token_id = int(np.argmax(probs))
            next_token = self.id_to_token.get(next_token_id, f"<TOK_{next_token_id}>")
            sequence.append(next_token)
            if next_token == eos_token:
                break

        final_sentence = " ".join(token for token in sequence if token not in (start_token, eos_token))
        return sequence, final_sentence


def build_fake_tensors(
    batch_size: int = 1,
    encoder_seq_len: int = 10,
    decoder_seq_len: int = 4,
    d_model: int = 512,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    encoder_out = rng.standard_normal((batch_size, encoder_seq_len, d_model))
    decoder_state = rng.standard_normal((batch_size, decoder_seq_len, d_model))
    return encoder_out, decoder_state


def build_decoder_vocab(vocab_size: int = 10000) -> tuple[pd.DataFrame, list[str]]:
    base_tokens = [
        "<START>",
        "<EOS>",
        "o",
        "rato",
        "roeu",
        "a",
        "roupa",
        "do",
        "rei",
        "de",
        "roma",
    ]
    filler_count = max(0, vocab_size - len(base_tokens))
    filler_tokens = [f"tok_{idx}" for idx in range(filler_count)]
    vocab_tokens = base_tokens + filler_tokens
    vocab_df = pd.DataFrame({"token": vocab_tokens, "token_id": np.arange(vocab_size, dtype=int)})
    return vocab_df, vocab_tokens


def main():
    np.set_printoptions(precision=4, suppress=True)

    print("Tarefa 1: Mascara Causal")
    mask, _, masked_weights = run_causal_mask_demo(seq_len=5)
    print("Mascara causal:")
    print(mask)
    print("Pesos de atencao com mascara causal:")
    print(masked_weights)
    print("Parte superior da matriz de probabilidades:")
    print(np.triu(masked_weights, k=1))

    print("\nTarefa 2: Cross-Attention")
    encoder_out, decoder_state = build_fake_tensors()
    cross_out, cross_weights = cross_attention(
        encoder_out=encoder_out,
        decoder_state=decoder_state,
        seed=99,
        return_weights=True,
    )
    print("Shape encoder_out:", encoder_out.shape)
    print("Shape decoder_state:", decoder_state.shape)
    print("Shape cross-attention output:", cross_out.shape)
    print("Shape cross-attention weights:", cross_weights.shape)

    print("\nTarefa 3: Loop Auto-Regressivo")
    vocab_df, vocab_tokens = build_decoder_vocab(vocab_size=10000)
    decoder = MockDecoder(vocab_tokens=vocab_tokens, d_model=512, vocab_size=10000, seed=7)
    sequence, final_sentence = decoder.autoregressive_decode(encoder_out=encoder_out, max_steps=6)
    print("Tamanho do vocabulario ficticio:", len(vocab_df))
    print("Tokens gerados:", sequence)
    print("Frase final:", final_sentence)


if __name__ == "__main__":
    main()
