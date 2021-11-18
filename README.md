# Genetic GNN
Gene prioritization using Graph Neural Network




## Installation
- numpy
- pandas
- Pytorch
- Pytorch Geometric
- torchmetrics
- sentencetransformers
- spacy
    - en_ner_bionlp13cg_md
    ```shell
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_bionlp13cg_md-0.3.0.tar.gz
    ```

## Workflow
### 1. Download files
see Download.md
### 2. Build Knowlege graph
```shell
snakemake -s graph/pubmed_graph_parallel.smk -p -j 32
```

### 3. Train your GNN and LinkPredictor
```shell
# GNN encoder + Linkpredictor
python train_gnn.py ...
# Link predictor only
python train_mlp.py ...
```

### 4. HBCGM output 

