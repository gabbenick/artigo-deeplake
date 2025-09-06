# Projeto: Artigo ERBASE 2025

Este repositório contém o código desenvolvido para o artigo publicado na **ERBASE 2025**, intitulado:  

**“Uma arquitetura de Data Lake baseado em Deep Learning para busca de imagens parasitárias de doenças socialmente determinadas”**

## Contexto

O projeto compara dois pipelines de dados para imagens médicas (SHdataset (https://zenodo.org/records/6467268)):  
- **PNG tradicional**, baseado em diretórios de arquivos.  
- **Deep Lake**, uma arquitetura moderna otimizada para *deep learning*.  

Foram realizados **benchmarks de desempenho** (armazenamento e velocidade de iteração) e um **estudo de caso com Deep Metric Learning (DML)** usando *Triplet Loss* para avaliar a qualidade dos embeddings.

## Estrutura do Repositório

- **experiment_results/**  
  - `embeddings_from_deeplake_model.npz` → Embeddings gerados pelo modelo treinado no pipeline Deep Lake.  
  - `embeddings_from_png_model.npz` → Embeddings gerados pelo modelo treinado no pipeline PNG.  
  - `final_comparison_metrics.csv` → Métricas comparativas entre os dois pipelines.  
  - `model_trained_on_deeplake.pth` → Modelo final treinado no dataset Deep Lake.  
  - `model_trained_on_pngs.pth` → Modelo final treinado no dataset PNG.  
- **deeplake_creation.py** → Script para criação do dataset no formato Deep Lake.  
- **insert_data.py** → Script para inserção dos dados e metadados no dataset.  
- **iteration_benchmark.ipynb** → Notebook para benchmarking de velocidade de iteração.  
- **size_comparison.ipynb** → Notebook para análise comparativa de espaço em disco.  
- **dml_tsne_visualization.ipynb** → Notebook de visualização dos embeddings com t-SNE.  
- **saved_model_execution.ipynb** → Notebook para execução/avaliação de modelos salvos.  
- **benchmark_resultados_agregados.csv** → Resultados agregados de benchmarks.  
- **requirements.txt** → Lista de dependências.  
- **README.md** → Documentação do projeto.  

## Como Executar

1. Crie um ambiente virtual Python 3.11.  
2. Instale as dependências:  
   ```bash
   pip install -r requirements.txt
3. Para gerar o dataset em Deep Lake:
   python deeplake_creation.py
4. Execute os notebooks em *.ipynb para reproduzir análises, visualizações e resultados.

## Principais Resultados

Deep Lake: iteração de dados mais rápida e estável, mas consumindo 59,2% mais espaço em disco.

PNG: melhor separabilidade entre classes no espaço de embeddings, refletindo em métricas levemente superiores no k-NN.

Trade-off: eficiência de gerenciamento e velocidade (Deep Lake) vs. simplicidade e menor custo de armazenamento (PNG).

## Referência

Se utilizar este código, cite o artigo conforme publicado na ERBASE 2025:

Uma arquitetura de Data Lake baseado em Deep Learning para busca de imagens parasitárias de doenças socialmente determinadas.
   
