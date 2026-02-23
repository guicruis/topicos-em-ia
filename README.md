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
