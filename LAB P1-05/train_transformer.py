import math
import sys
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


SPECIAL_TOKENS = {"additional_special_tokens": ["<START>", "<EOS>"]}


@dataclass
class TrainingConfig:
    dataset_name: str = "bentrevett/multi30k"
    dataset_config: str | None = None
    source_lang: str = "en"
    target_lang: str = "de"
    subset_size: int = 128
    tokenizer_name: str = "bert-base-multilingual-cased"
    max_length: int = 32
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    device: str = "cpu"
    random_seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_translation_pair(example: dict, source_lang: str, target_lang: str) -> tuple[str, str]:
    if "translation" in example:
        translation = example["translation"]
        return translation[source_lang], translation[target_lang]
    if source_lang in example and target_lang in example:
        return example[source_lang], example[target_lang]
    raise KeyError(f"Campos de traducao nao encontrados para {source_lang}->{target_lang}")


def load_translation_subset(config: TrainingConfig):
    split = f"train[:{config.subset_size}]"
    if config.dataset_config:
        dataset = load_dataset(config.dataset_name, config.dataset_config, split=split)
    else:
        dataset = load_dataset(config.dataset_name, split=split)
    return dataset


def build_tokenizer(config: TrainingConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    start_id = tokenizer.convert_tokens_to_ids("<START>")
    eos_id = tokenizer.convert_tokens_to_ids("<EOS>")
    pad_id = tokenizer.pad_token_id
    return tokenizer, start_id, eos_id, pad_id


class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, config: TrainingConfig, start_id: int, eos_id: int):
        self.samples = []
        for example in dataset:
            src_text, tgt_text = extract_translation_pair(
                example,
                source_lang=config.source_lang,
                target_lang=config.target_lang,
            )
            src_ids = tokenizer(
                src_text,
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_length,
            )["input_ids"]
            tgt_ids = tokenizer(
                tgt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_length - 2,
            )["input_ids"]
            tgt_ids = [start_id] + tgt_ids + [eos_id]
            self.samples.append(
                {
                    "source_text": src_text,
                    "target_text": tgt_text,
                    "src_ids": src_ids,
                    "tgt_ids": tgt_ids,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_batch(batch, pad_id: int):
    src_tensors = [torch.tensor(item["src_ids"], dtype=torch.long) for item in batch]
    tgt_tensors = [torch.tensor(item["tgt_ids"], dtype=torch.long) for item in batch]
    src_batch = nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=pad_id)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True, padding_value=pad_id)
    return {
        "source_texts": [item["source_text"] for item in batch],
        "target_texts": [item["target_text"] for item in batch],
        "src_ids": src_batch,
        "tgt_ids": tgt_batch,
    }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build_encoding(max_len))

    def _build_encoding(self, max_len: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(max_len, self.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self.pe = self._build_encoding(x.size(1)).to(x.device)
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model deve ser divisivel por num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        output = self.out_proj(self._merge_heads(context))
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        attention_output, _ = self.self_attention(x, x, x, attention_mask=src_mask)
        x = self.norm1(x + self.dropout(attention_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        y: torch.Tensor,
        encoder_memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        masked_output, _ = self.masked_self_attention(y, y, y, attention_mask=tgt_mask)
        y = self.norm1(y + self.dropout(masked_output))
        cross_output, _ = self.cross_attention(y, encoder_memory, encoder_memory, attention_mask=src_mask)
        y = self.norm2(y + self.dropout(cross_output))
        ffn_output = self.ffn(y)
        return self.norm3(y + self.dropout(ffn_output))


class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_source_mask(self, src_ids: torch.Tensor) -> torch.Tensor:
        return (src_ids == self.pad_id).unsqueeze(1).unsqueeze(2)

    def create_target_mask(self, tgt_ids: torch.Tensor) -> torch.Tensor:
        batch_size, tgt_len = tgt_ids.size()
        pad_mask = (tgt_ids == self.pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.triu(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=tgt_ids.device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(1)
        return pad_mask | causal_mask.expand(batch_size, 1, tgt_len, tgt_len)

    def encode(self, src_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_mask = self.create_source_mask(src_ids)
        x = self.token_embedding(src_ids) * math.sqrt(self.d_model)
        x = self.dropout(self.position_encoding(x))
        for block in self.encoder_blocks:
            x = block(x, src_mask=src_mask)
        return x, src_mask

    def decode(self, tgt_ids: torch.Tensor, encoder_memory: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        tgt_mask = self.create_target_mask(tgt_ids)
        y = self.token_embedding(tgt_ids) * math.sqrt(self.d_model)
        y = self.dropout(self.position_encoding(y))
        for block in self.decoder_blocks:
            y = block(y, encoder_memory, tgt_mask=tgt_mask, src_mask=src_mask)
        return y

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        encoder_memory, src_mask = self.encode(src_ids)
        decoder_output = self.decode(tgt_ids, encoder_memory, src_mask)
        return self.output_projection(decoder_output)

    @torch.no_grad()
    def generate(
        self,
        src_ids: torch.Tensor,
        start_id: int,
        eos_id: int,
        max_steps: int = 32,
    ) -> torch.Tensor:
        self.eval()
        encoder_memory, src_mask = self.encode(src_ids)
        generated = torch.full((src_ids.size(0), 1), start_id, dtype=torch.long, device=src_ids.device)

        for _ in range(max_steps):
            decoder_output = self.decode(generated, encoder_memory, src_mask)
            logits = self.output_projection(decoder_output[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if torch.all(next_token.squeeze(-1) == eos_id):
                break
        return generated


def train_one_epoch(model, dataloader, optimizer, criterion, device, pad_id: int) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0

    for batch in dataloader:
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        decoder_input = tgt_ids[:, :-1]
        labels = tgt_ids[:, 1:]

        optimizer.zero_grad()
        logits = model(src_ids, decoder_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()

        grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_sq += float(param.grad.detach().norm(2).item() ** 2)
        total_grad_norm += math.sqrt(grad_norm_sq)

        optimizer.step()
        total_loss += float(loss.item())

    avg_loss = total_loss / max(len(dataloader), 1)
    avg_grad_norm = total_grad_norm / max(len(dataloader), 1)
    return avg_loss, avg_grad_norm


def evaluate_overfit_example(model, tokenizer, sample, start_id: int, eos_id: int, device) -> dict:
    source_ids = torch.tensor(sample["src_ids"], dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(source_ids, start_id=start_id, eos_id=eos_id, max_steps=32)
    generated_tokens = generated[0].tolist()

    decoded_tokens = []
    for token_id in generated_tokens[1:]:
        if token_id == eos_id:
            break
        decoded_tokens.append(token_id)

    generated_text = tokenizer.decode(decoded_tokens, skip_special_tokens=True).strip()
    return {
        "source_text": sample["source_text"],
        "target_text": sample["target_text"],
        "generated_text": generated_text,
        "generated_token_ids": generated_tokens,
    }


def run_training(config: TrainingConfig):
    set_seed(config.random_seed)
    device = torch.device(config.device)

    dataset = load_translation_subset(config)
    tokenizer, start_id, eos_id, pad_id = build_tokenizer(config)
    tokenized_dataset = TranslationDataset(dataset, tokenizer, config, start_id=start_id, eos_id=eos_id)

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_id=pad_id),
    )

    model = TransformerSeq2Seq(
        vocab_size=len(tokenizer),
        pad_id=pad_id,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_len=config.max_length + 8,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    loss_history = []
    for epoch in range(1, config.epochs + 1):
        avg_loss, avg_grad_norm = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            pad_id=pad_id,
        )
        loss_history.append(avg_loss)
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | grad_norm={avg_grad_norm:.4f}")

    probe = evaluate_overfit_example(
        model=model,
        tokenizer=tokenizer,
        sample=tokenized_dataset[0],
        start_id=start_id,
        eos_id=eos_id,
        device=device,
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "dataset_size": len(tokenized_dataset),
        "loss_history": loss_history,
        "overfit_probe": probe,
        "config": config,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = TrainingConfig(
        dataset_name="bentrevett/multi30k",
        dataset_config=None,
        source_lang="en",
        target_lang="de",
        subset_size=128,
        tokenizer_name="bert-base-multilingual-cased",
        max_length=32,
        d_model=128,
        num_heads=4,
        d_ff=512,
        num_layers=2,
        batch_size=8,
        epochs=10,
        learning_rate=1e-3,
        device="cpu",
        random_seed=42,
    )

    results = run_training(config)
    probe = results["overfit_probe"]

    print("\nResumo do treinamento")
    print("Dataset size:", results["dataset_size"])
    print("Loss inicial:", round(results["loss_history"][0], 4))
    print("Loss final:", round(results["loss_history"][-1], 4))
    print("\nOverfitting test")
    print("Source:", probe["source_text"])
    print("Target:", probe["target_text"])
    print("Generated:", probe["generated_text"])
    print("Generated token ids:", probe["generated_token_ids"])


if __name__ == "__main__":
    main()



