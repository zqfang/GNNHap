# Genetic GNN
Gene prioritization using Graph Neural Network




## Installation
- numpy
- pandas
- Pytorch
- Pytorch Geometric
- torchmetrics
- sentencetransformers
- pubtator
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
## hidden_size 50 fits to a 24G GPU card
## hidden_size 64 fits to a 32G GPU card
python train_gnn.py --batch_size 10000 \
                    --hidden_size 64 \
                    --num_epochs 10 \
                    --mesh_embed ${WKDIR}/human_gene_unirep.embeb.csv \
                    --gene_embed ${WKDIR}/mesh.sentencetransformer.embed.csv \
                    --gene_mesh_graph ${WKDIR}/human_gene_mesh_hetero_nx.gpkl
# Link predictor only
python train_mlp.py --batch_size 100000 \
                    --hidden_size 256 \
                    --num_epochs 10 \
                    --mesh_embed ${WKDIR}/human_gene_unirep.embeb.csv \
                    --gene_embed ${WKDIR}/mesh.sentencetransformer.embed.csv \
                    --gene_mesh_graph ${WKDIR}/human_gene_mesh_hetero_nx.gpkl
```

### 4. HBCGM output 

