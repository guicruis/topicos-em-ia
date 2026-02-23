import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
    return_pesos: bool = False,
):
    """
    Compute scaled dot-product attention.

    Shapes:
    - query: (..., seq_q, d_k)
    - key:   (..., seq_k, d_k)
    - value: (..., seq_k, d_v)
    - mask:  (..., seq_q, seq_k), com 1/True para posicoes validas e 0/False caso contrario.
    """
    d_k = query.shape[-1]
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    pesos = softmax(scores, axis=-1)
    output = np.matmul(pesos, value)

    if return_pesos:
        return output, pesos
    return output


class SelfAttention:
    """
    Single-head self-attention (NumPy implementacao).
    """

    def __init__(self, d_model: int, d_k: int | None = None, d_v: int | None = None, seed: int = 42):
        if d_model <= 0:
            raise ValueError("d_model deve ser > 0")

        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model
        self.d_v = d_v if d_v is not None else d_model

        rng = np.random.default_rng(seed)

        scale_qk = np.sqrt(self.d_model)
        scale_v = np.sqrt(self.d_model)

        self.W_q = rng.standard_normal((self.d_model, self.d_k)) / scale_qk
        self.W_k = rng.standard_normal((self.d_model, self.d_k)) / scale_qk
        self.W_v = rng.standard_normal((self.d_model, self.d_v)) / scale_v

    @staticmethod
    def _ensure_3d(x: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Aceito (seq_len, d_model) ou (batch, seq_len, d_model).
        Retorna um tensor 3D padronizado e um sinalizador indicando se a entrada era 2D.
        """
        if x.ndim == 2:
            return np.expand_dims(x, axis=0), True
        if x.ndim == 3:
            return x, False
        raise ValueError("Input x deve ter shape (seq_len, d_model) ou (batch, seq_len, d_model)")

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None, return_pesos: bool = False):
        x_3d, squeeze_output = self._ensure_3d(x)

        if x_3d.shape[-1] != self.d_model:
            raise ValueError(
                f"Dimensão esperada da última entrada {self.d_model}, obteve {x_3d.shape[-1]}"
            )

        q = np.matmul(x_3d, self.W_q)
        k = np.matmul(x_3d, self.W_k)
        v = np.matmul(x_3d, self.W_v)

        if return_pesos:
            out, pesos = scaled_dot_product_attention(q, k, v, mask=mask, return_pesos=True)
            if squeeze_output:
                return out[0], pesos[0]
            return out, pesos

        out = scaled_dot_product_attention(q, k, v, mask=mask, return_pesos=False)
        if squeeze_output:
            return out[0]
        return out


if __name__ == "__main__":
    x = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    attention = SelfAttention(d_model=4, seed=7)
    output, pesos = attention.forward(x, return_pesos=True)

    print("Output shape:", output.shape)
    print("Attention pesos shape:", pesos.shape)
