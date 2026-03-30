import re
from collections import defaultdict

from transformers import AutoTokenizer


INITIAL_VOCAB = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
}


def get_stats(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for left, right in zip(symbols, symbols[1:]):
            pairs[(left, right)] += freq
    return dict(pairs)


def merge_vocab(pair: tuple[str, str], v_in: dict[str, int]) -> dict[str, int]:
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(rf"(?<!\S){bigram}(?!\S)")
    replacement = "".join(pair)

    v_out: dict[str, int] = {}
    for word, freq in v_in.items():
        merged_word = pattern.sub(replacement, word)
        v_out[merged_word] = freq
    return v_out


def choose_best_pair(stats: dict[tuple[str, str], int]) -> tuple[str, str]:
    return sorted(stats.items(), key=lambda item: (-item[1], item[0]))[0][0]


def train_bpe(vocab: dict[str, int], num_merges: int = 5) -> list[dict[str, object]]:
    current_vocab = dict(vocab)
    history: list[dict[str, object]] = []

    for iteration in range(1, num_merges + 1):
        stats = get_stats(current_vocab)
        best_pair = choose_best_pair(stats)
        current_vocab = merge_vocab(best_pair, current_vocab)
        history.append(
            {
                "iteration": iteration,
                "best_pair": best_pair,
                "vocab": dict(current_vocab),
            }
        )
    return history


def run_wordpiece_demo(
    text: str = "Os hiper-parâmetros do transformer são inconstitucionalmente difíceis de ajustar.",
    tokenizer_name: str = "bert-base-multilingual-cased",
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer.tokenize(text)


def hash_prefix_explanation() -> str:
    return (
        "No WordPiece, o prefixo ## indica que o token e uma continuacao da subpalavra anterior, "
        "e nao o inicio de uma palavra nova. Isso permite quebrar palavras raras em partes menores "
        "ja conhecidas pelo vocabulario, reduzindo o problema de palavras desconhecidas e evitando "
        "que o modelo trave quando encontra termos fora do conjunto exato de treino."
    )


def main():
    print("Tarefa 1: Frequencias iniciais")
    stats = get_stats(INITIAL_VOCAB)
    print("Frequencia do par ('e', 's'):", stats.get(("e", "s")))

    print("\nTarefa 2: Loop de fusao BPE")
    history = train_bpe(INITIAL_VOCAB, num_merges=5)
    for step in history:
        print(f"Iteracao {step['iteration']}: melhor par = {step['best_pair']}")
        print("Vocabulario apos a fusao:")
        print(step["vocab"])

    print("\nTarefa 3: WordPiece com BERT")
    tokens = run_wordpiece_demo()
    print("Tokens WordPiece:")
    print(tokens)
    print("\nExplicacao sobre ##:")
    print(hash_prefix_explanation())


if __name__ == "__main__":
    main()

