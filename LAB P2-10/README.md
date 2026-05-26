# Laboratorio 10 - Pipeline Definitivo com RAG, QLoRA e Inferencia Otimizada

Este laboratorio integra o RAG do Lab 09 com as tecnicas de eficiencia estudadas na disciplina. O objetivo e simular um fluxo de producao em que um modelo causal recebe um contexto medico massivo, carregado com QLoRA em 4 bits, e compara a geracao sem cache contra a geracao otimizada com KV Cache e FlashAttention-2.

## Arquivos

- `production_rag_inference.py`: cria contexto medico sintetico, configura quantizacao 4-bit, prepara carregamento com `attn_implementation="flash_attention_2"` e executa o benchmark.
- `test_production_rag_inference.py`: testes automatizados para contexto de 10k a 15k tokens, QLoRA, FlashAttention, KV Cache e reducao de memoria.
- `outputs/massive_medical_context.txt`: criado pelo comando `build-context`.
- `outputs/inference_benchmark.json`: criado pelo comando `benchmark` com as metricas comparativas.

## Dependencias

O laboratorio roda em modo leve sem baixar modelo. Para executar com modelo real em GPU, instale:

```bash
pip install torch transformers bitsandbytes accelerate flash-attn
```

## Como executar

Inspecionar a arquitetura e os parametros:

```bash
python "LAB P2-10/production_rag_inference.py" describe
```

Gerar o contexto massivo que simula os capitulos recuperados pelo RAG:

```bash
python "LAB P2-10/production_rag_inference.py" build-context
```

Executar o benchmark simulado de inferencia:

```bash
python "LAB P2-10/production_rag_inference.py" benchmark
```

Executar os testes:

```bash
python -m unittest -v "LAB P2-10/test_production_rag_inference.py"
```

## Metricas de Benchmark

O benchmark registra a VRAM estimada para o modelo quantizado em 4 bits, o pico de memoria da fase de prompting, o custo de KV Cache e o tempo estimado para gerar 100 tokens. A versao baseline usa `model.config.use_cache = False`, o que obriga o decoder a recalcular atencao sobre o prefixo inteiro a cada novo token. A versao otimizada usa `use_cache = True` e `attn_implementation="flash_attention_2"`, reduzindo tanto o tempo de decodificacao quanto o pico de memoria associado a matriz de atencao do prompt longo.

## Analise Arquitetural

Parte A: A combinacao de QLoRA, KV Cache e FlashAttention salva o Transformer tradicional porque ataca tres gargalos diferentes. A quantizacao em 4 bits reduz a memoria fixa dos pesos e permite carregar o modelo onde o Float16 ja consumiria VRAM demais. O KV Cache evita recomputar chaves e valores de todos os tokens anteriores a cada passo auto-regressivo, reduzindo a latencia de geracao. O FlashAttention reorganiza o calculo da atencao para nao materializar a matriz completa `N x N` na SRAM/HBM, o que reduz drasticamente o pico de memoria durante o processamento do contexto massivo.

Parte B: Se o cliente exigisse 2 milhoes de tokens em vez de 15.000, mesmo FlashAttention deixaria de resolver o problema estrutural. Ele melhora a implementacao da atencao, mas o Transformer continua tendo dependencia forte do comprimento da sequencia e precisa manter estado proporcional ao historico para inferencia eficiente. Nesse regime, a industria precisaria migrar para arquiteturas com memoria assintoticamente mais favoravel, como State Space Models e modelos do tipo Mamba, que processam sequencias longas com estado recorrente compacto e complexidade de memoria O(1) em relacao ao tamanho total do contexto.
