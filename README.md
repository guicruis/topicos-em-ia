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
