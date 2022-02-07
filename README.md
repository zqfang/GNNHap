# GNNHap
Graph Neural Network based Haplotype Prioritization for inbred mouse.


![GNNHap](./GNNHap.jpg)

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
This step generate the graph file: human_gene_mesh_hetero_nx.gpkl

### 3. Train your GNN and LinkPredictor
```shell
# GNN encoder + Linkpredictor
## hidden_size 50 fits to a 24G GPU card
## hidden_size 64 fits to a 32G GPU card
python GNNHap/train_gnn.py --batch_size 10000 \
                    --hidden_size 64 \
                    --num_epochs 5 \
                    --mesh_embed ${WKDIR}/human_gene_unirep.embeb.csv \
                    --gene_embed ${WKDIR}/mesh.sentencetransformer.embed.csv \
                    --gene_mesh_graph ${WKDIR}/human_gene_mesh_hetero_nx.gpkl
# Link predictor only
python GNNHap/train_mlp.py --batch_size 10000 \
                    --hidden_size 64 \
                    --num_epochs 5 \
                    --mesh_embed ${WKDIR}/human_gene_unirep.embeb.csv \
                    --gene_embed ${WKDIR}/mesh.sentencetransformer.embed.csv \
                    --gene_mesh_graph ${WKDIR}/human_gene_mesh_hetero_nx.gpkl
```

### 4. Genetic mapping using Haplomap

see the [full guide](https://github.com/zqfang/haplomap) to get Haplomap (a.k.a HBCGM) results

An snakemake pipeline in the `example` folder shows the full commands.

```shell
snakemake -s gnnhap.smk --configfile config.yaml -j 12 -p
```

### 5. Predict

Download the [GNNHap_Bundle](), which contained necessary files

**Case 1**: single result file
```python
python GNNHap/predict.py --bundle /path/to/GNNHap_Bundle  
                  --hbcgm_result_dir ${RESULTS}# parent path to *results.txt
                  --mesh_terms D018919,D009389,D043924,D003315 # separate each term with comma
                  --num_cpus 12
            
```

**Case 2**: multiple result files
**NOTE 1**: the `${HBCGM_RESULT_DIR}`  folder looks like this:
``` 
|-RESULTS
|--- MPD_000.results.txt
|--- MPD_001.results.txt
...
```

**NOTE 2**: provide a json file for `--mesh_terms` if multiple result file are predict

```python
python GNNHap/predict.py --bundle /path/to/GNNHap_Bundle  
                  --hbcgm_result_dir ${RESULTS}# parent path to *results.txt
                  --mesh_terms mpd2mesh.json # separate each term with comma
                  --num_cpus 12
            
```

### 6. DataVisualization


set the `DATA_DIR` to your GNNHap output folder in the `main.py`, then run the following command:

deployment  
```
bokeh serve --show webapp --allow-websocket-origin=peltz-app-03:5006
```