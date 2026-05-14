# Laboratorio 09 - RAG Avancado com HNSW, HyDE e Cross-Encoder

Este laboratorio implementa um pipeline de Retrieval-Augmented Generation para busca em fragmentos de manuais medicos privados. A consulta coloquial do usuario e transformada por HyDE em um documento tecnico hipotetico, usada para recuperar candidatos em um indice HNSW e refinada por um Cross-Encoder.

## Arquivos

- `rag_hyde_pipeline.py`: pipeline completo de dados, embeddings, indice HNSW, HyDE, recuperacao top-10 e re-ranking top-3.
- `test_rag_hyde_pipeline.py`: testes automatizados para o dataset, parametros HNSW, HyDE, recuperacao e re-ranking.
- `data/medical_fragments.jsonl`: criado pelo comando `prepare-data` com pelo menos 20 fragmentos tecnicos.
- `outputs/retrieval_trace.json`: criado pelo comando `query` com o rastro da busca.

## Dependencias

O pipeline roda em modo leve sem dependencias externas pesadas. Para usar a versao completa do laboratorio:

```bash
pip install hnswlib sentence-transformers openai
```

Se `hnswlib` nao estiver disponivel, o codigo usa uma busca exata local como fallback para manter os testes reprodutiveis. Se `sentence-transformers` nao estiver disponivel, usa embeddings deterministas por hashing. Se a API da OpenAI nao estiver configurada, o HyDE usa uma ponte semantica local.

## Como executar

Criar o dataset simulado:

```bash
python "LAB P2-09/rag_hyde_pipeline.py" prepare-data
```

Inspecionar a configuracao:

```bash
python "LAB P2-09/rag_hyde_pipeline.py" describe
```

Executar a busca solicitada no PDF:

```bash
python "LAB P2-09/rag_hyde_pipeline.py" query --query "dor de cabeca latejante e luz incomodando"
```

Executar os testes:

```bash
python -m unittest -v "LAB P2-09/test_rag_hyde_pipeline.py"
```

## Analise do HNSW

O HNSW organiza os vetores em um grafo de navegacao aproximada. O parametro `M` controla quantas conexoes cada no pode manter. Valores maiores aumentam a conectividade, melhoram o recall e reduzem a chance de o algoritmo ficar preso em vizinhos ruins, mas consomem mais memoria RAM porque cada vetor passa a guardar mais arestas.

O parametro `ef_construction` controla o tamanho da lista dinamica de candidatos durante a construcao do grafo. Valores maiores produzem um grafo mais bem conectado e melhoram a qualidade da busca, mas tornam a indexacao mais lenta e tambem podem aumentar o consumo de memoria temporaria durante a construcao.

Em comparacao com uma busca KNN exata, o HNSW troca exatidao garantida por eficiencia. O KNN exato compara a query com todos os vetores, o que cresce linearmente com o tamanho da base e exige varrer toda a matriz de embeddings. O HNSW percorre apenas uma parte pequena do grafo, entao costuma responder muito mais rapido em bases grandes. O custo e manter a estrutura de grafo em memoria: com `M` e `ef_construction` altos, a RAM cresce, mas a latencia cai e o recall melhora.
