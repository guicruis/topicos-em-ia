# Topicos em IA

Repositorio para entregas da disciplina de Topicos em Inteligencia Artificial.

## Laboratorio P1-01

Implementacao de Self-Attention em NumPy.

Arquivos:
- `LAB P1-01/attention.py`: implementacao de `softmax`, `scaled_dot_product_attention` e classe `SelfAttention`.
- `LAB P1-01/test_attention.py`: suite de testes automatizados com `unittest`.

## Requisitos

- Python 3.10+ (recomendado 3.12)
- NumPy

Instalacao de dependencia:

```bash
pip install numpy
```

## Como executar

Executar exemplo simples do modulo:

```bash
python "LAB P1-01/attention.py"
```

Executar todos os testes:

```bash
python -m unittest -v "LAB P1-01/test_attention.py"
```

## Cobertura dos testes

O arquivo de testes valida:
- Propriedades do `softmax` (estabilidade e normalizacao).
- Shapes de saida e pesos no scaled dot-product attention.
- Aplicacao de mascara.
- Suporte da classe `SelfAttention` para entrada 2D e 3D.
- Tratamento de erro para dimensoes invalidas.
- Determinismo da inicializacao com `seed`.

## Laboratorio P1-02

Implementacao do forward pass de um Transformer Encoder "from scratch" usando apenas `numpy` e `pandas`.

Arquivos:
- `LAB P1-02/encoder.py`: preparacao dos dados, embeddings, self-attention, layer normalization, feed-forward network, encoder layer e stack com `N=6`.
- `LAB P1-02/test_encoder.py`: testes automatizados cobrindo preparacao de entrada, blocos matematicos e encoder completo.

Dependencias:

```bash
pip install numpy pandas
```

Executar a demonstracao do encoder:

```bash
python "LAB P1-02/encoder.py"
```

Executar os testes do P1-02:

```bash
python -m unittest -v "LAB P1-02/test_encoder.py"
```

## Laboratorio P1-03

Implementacao dos blocos centrais do Decoder: mascara causal, cross-attention e loop de inferencia auto-regressivo.

Arquivos:
- `LAB P1-03/decoder.py`: implementa `create_causal_mask`, scaled dot-product attention com mascara, `cross_attention` e `MockDecoder`.
- `LAB P1-03/test_decoder.py`: testes automatizados para mascara causal, cross-attention e geracao token a token.

Executar a demonstracao do decoder:

```bash
python "LAB P1-03/decoder.py"
```

Executar os testes do P1-03:

```bash
python -m unittest -v "LAB P1-03/test_decoder.py"
```

## Laboratorio P1-04

Implementacao do Transformer completo "from scratch", integrando Encoder, Decoder, Add & Norm, FFN, mascara causal e loop auto-regressivo de inferencia.

Arquivos:
- `LAB P1-04/transformer_complete.py`: implementa a arquitetura Encoder-Decoder completa e uma demonstracao com a entrada `Thinking Machines`.
- `LAB P1-04/test_transformer_complete.py`: testes automatizados cobrindo mascara causal, blocos Encoder/Decoder e inferencia fim-a-fim.

Executar a demonstracao do Transformer completo:

```bash
python "LAB P1-04/transformer_complete.py"
```

Executar os testes do P1-04:

```bash
python -m unittest -v "LAB P1-04/test_transformer_complete.py"
```

## Nota Sobre IA

Partes geradas/complementadas com IA, revisadas por Guilherme. A logica matematica da montagem do modelo foi revisada, ajustada e validada antes da entrega.

Ferramentas externas utilizadas nos laboratorios finais:
- `transformers` para tokenizacao com `AutoTokenizer`.
- `datasets` para carregar subconjuntos de datasets do Hugging Face.
- `torch` para treinamento, backpropagation e otimizacao do modelo do Lab 05.

## Laboratorio P1-05

Implementacao do treinamento fim-a-fim de um Transformer Encoder-Decoder com dataset real do Hugging Face, tokenizacao com `AutoTokenizer`, loop de treinamento com `CrossEntropyLoss` e `Adam`, e teste de overfitting com geracao auto-regressiva.

Arquivos:
- `LAB P1-05/train_transformer.py`: pipeline completo de dados, tokenizacao, modelo Transformer em PyTorch, treino e inferencia.
- `LAB P1-05/test_train_transformer.py`: testes sinteticos cobrindo shapes, mascara causal e uma etapa de treinamento.

Dependencias:

```bash
pip install torch datasets transformers numpy pandas
```

Executar o treinamento do Lab 05:

```bash
python "LAB P1-05/train_transformer.py"
```

Executar os testes do Lab 05:

```bash
python -m unittest -v "LAB P1-05/test_train_transformer.py"
```

## Laboratorio P1-06

Implementacao de um tokenizador BPE basico "from scratch" e exploracao pratica do WordPiece com o tokenizador multilingue do BERT.

Arquivos:
- `LAB P1-06/bpe_wordpiece.py`: implementa `get_stats`, `merge_vocab`, o loop de 5 fusoes e a demonstracao com `AutoTokenizer`.
- `LAB P1-06/test_bpe_wordpiece.py`: testes automatizados para frequencias de pares e fusao BPE.

Executar a demonstracao do Lab 06:

```bash
python "LAB P1-06/bpe_wordpiece.py"
```

Executar os testes do Lab 06:

```bash
python -m unittest -v "LAB P1-06/test_bpe_wordpiece.py"
```

## Explicacao Sobre ##

No WordPiece, o prefixo `##` indica que o token e uma continuacao da subpalavra anterior, e nao o inicio de uma palavra nova. Isso permite representar palavras raras ou desconhecidas como composicoes de partes menores ja vistas, reduzindo o tamanho do vocabulario e evitando que o modelo falhe diante de termos fora do conjunto exato de treino.

## Nota Especifica de IA no Lab 06

No Lab 06, a construcao da expressao regular usada na funcao `merge_vocab` foi assistida por IA e depois revisada, testada e ajustada manualmente para garantir que apenas pares completos de simbolos fossem fundidos.

## Laboratorio P2-07

Pipeline completo de fine-tuning eficiente com dataset sintetico em `.jsonl`, quantizacao 4-bit com `nf4`, adaptacao via LoRA e treinamento supervisionado com `SFTTrainer`.

Arquivos:
- `LAB P2-07/qlora_pipeline.py`: gera dataset sintetico no dominio escolhido, salva `train/test` em JSONL, configura QLoRA e executa o treino com `paged_adamw_32bit`.
- `LAB P2-07/test_qlora_pipeline.py`: testes automatizados para a geracao sintetica, split 90/10 e hiperparametros obrigatorios do PDF.
- `LAB P2-07/data/synthetic_train.jsonl`: conjunto de treino com 54 exemplos.
- `LAB P2-07/data/synthetic_test.jsonl`: conjunto de teste com 6 exemplos.

Dependencias:

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate openai
```

Gerar os arquivos do dataset:

```bash
python "LAB P2-07/qlora_pipeline.py" generate-data
```

Inspecionar a configuracao exigida pelo laboratorio:

```bash
python "LAB P2-07/qlora_pipeline.py" describe
```

Executar o treino QLoRA:

```bash
python "LAB P2-07/qlora_pipeline.py" train
```

Executar os testes do Lab 07:

```bash
python -m unittest -v "LAB P2-07/test_qlora_pipeline.py"
```

Observacao:
- A geracao sintetica tenta usar a API da OpenAI quando o pacote e a credencial estiverem disponiveis.
- Em ambiente offline, o script cai para uma geracao deterministica local para ainda produzir os arquivos `jsonl` exigidos para a entrega.
