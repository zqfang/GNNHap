
import os, sys, glob, json, joblib
import numpy as np
import pandas as pd
import networkx as nx
import joblib

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm
from data import GeneMeshData
from models import HeteroGNN
from utils import add_train_args
from typing import List, Dict, Tuple, Union, AnyStr

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# argument parser
args = add_train_args()
hidden_size = args.hidden_size
batch_size = args.batch_size # 10000

BUNDLE_PATH = args.bundle
HBCGM_RESULTS = args.hbcgm_result_dir
MESH_TERMS = args.mesh_terms

os.makedirs(args.outdir, exist_ok=True)
model_weight = os.path.join(BUNDLE_PATH, "gnn_64_epoch0.pt")
genemesh_data = os.path.join(BUNDLE_PATH, "genemesh.data.pkl") # genemesh.data.pkl
mouse_human_namedict = os.path.join(BUNDLE_PATH, "mouse2human.genenames.json")
mus_tissue_exp = os.path.join(BUNDLE_PATH, "mus.compact.exprs.organs.order.txt")
if not os.path.exists(model_weight):
    raise LookupError("model weights not found")


## Load Graph nodes' metadata
mesh_nodes = joblib.load(os.path.join(BUNDLE_PATH,"mesh_nodes.pkl"))
human_gene_nodes = joblib.load(os.path.join(BUNDLE_PATH,"human_gene_nodes.pkl"))

# # read data in
print("Read graph data")

if os.path.exists(genemesh_data) and genemesh_data.endswith("data.pkl"):
    gm_data = joblib.load(genemesh_data)
elif (args.gene_mesh_graph is not None) and args.gene_mesh_graph.endswith("gpkl"):
    # Load mesh embeddings
    print("Load embeddings")
    mesh_features = pd.read_csv(args.mesh_embed, index_col=0, header=None)
    gene_features = pd.read_csv(args.gene_embed, index_col=0, header=None)
    gene_features.index = gene_features.index.astype(str)
    H = nx.read_gpickle(args.gene_mesh_graph) # note: H is undirected multigraph
    print("Build GraphData")
    # build data
    gm = GeneMeshData(H, gene_features, mesh_features)
    gm_data = gm()

else:
    raise ValueError("Need Graph Object input for prediction task !")
    sys.exit(0)

# init model
model = HeteroGNN(heterodata=gm_data, hidden_channels=args.hidden_size, num_layers=2)
model.to('cpu')

@torch.no_grad()
def predict(model, data: HeteroData, node_embed=None, device='cpu'):
    model.eval()
    data.to(device)
    ####
    y_preds = []
    if node_embed is None:
        h_dict = model(data.x_dict, data.edge_index_dict) # only need compute once for node_representations
    else:
        h_dict = node_embed
    
    edge_label_index = data['gene','genemesh','mesh'].edge_label_index
    data_loader = DataLoader(torch.arange(edge_label_index.size(1)), batch_size=batch_size)
    for i, perm in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predict', position=1, leave=True)):
        g = h_dict['gene'][edge_label_index[0, perm]] 
        m = h_dict['mesh'][edge_label_index[1, perm]]
        # NOTE: concat here
        inp = torch.cat([g, m], dim=1).to(device)
        preds = model.link_predictor(inp).view(-1)
        y_preds.append(preds.cpu())
    y_preds = torch.cat(y_preds, dim=0) # note the difference with torch.stack()
    y_preds = torch.sigmoid(y_preds).numpy()
    return y_preds, h_dict

## load model weights
ckpt = torch.load(model_weight, map_location=device)
model.load_state_dict(ckpt['state_dict'])


class HBCGM:
    def __init__(self, inputdir: AnyStr, 
                  mesh_term_ids: List, 
                  model: torch.nn.Module,
                  gm_data: HeteroData,
                  human_gene_nodes: Union[AnyStr, Dict], 
                  mouse_human_namedict: Union[AnyStr, Dict],
                  expression_headers: Union[AnyStr, List]=None):

        self.inputs = glob.glob(os.path.join(inputdir, "chr*results.txt"))
        self.mesh_terms = mesh_term_ids
        self.node2index = { 'gene2nid': {v: key for key, v in  gm_data['nid2gene'].items()}, 
                            'mesh2nid': {v:key for key, v in gm_data['nid2mesh'].items()}}
        self.gm_data = gm_data
        if not isinstance(human_gene_nodes, dict):
            human_gene_nodes = joblib.load(os.path.join(BUNDLE_PATH,"human_gene_nodes.pkl"))
        self.symbol2entrez = {}
        for entrez, value in human_gene_nodes.items():
            self.symbol2entrez[value['gene_symbol']] = entrez

        # load 
        self.mouse2human = mouse_human_namedict
        if not isinstance(mouse_human_namedict, dict):
            with open(mouse_human_namedict, 'r') as j:
                self.mouse2human = json.load(j) ## Mouse gene name to human
        self.model = model
        self.expr_header = expression_headers
        if isinstance(expression_headers, str):
            with open(expression_headers) as e:
                self.expr_header = e.read().strip().split("\n")
        

    def save(self, output:AnyStr):
        line = "MeSH_Terms_"
        for mesh_term in self.mesh_terms:
            m =  f"_{self.node2index['mesh2nid'][mesh_term]}:{mesh_term}"
            line += m
        self.headers.append(line+"\n")
       
        with open(output, 'a') as out:
            for line in self.headers:
                out.write("##"+line)
            ## Expression order
            if self.expr_header is not None:
                out.write("_".join(self.expr_header) + "\n")
            ## Table
            self.result.sort_values(['Pval','FDR','chr','blockStart']).to_csv(out, sep="\t", index=False)

    def map2human(self, result):
        case = pd.read_table(result, skiprows=3, header=None)
        if case.shape[0] < 1: return case 
        case.columns = ['GeneName','CodonFlag','Haplotype','Pval', 'EffectSize',  
                        'FDR','popPval','popFDR',
                        'chr','blockStart','blockEnd','Expression']
        case.loc[case.index, 'HumanEntrezID'] = case.GeneName.map(self.mouse2human).map(self.symbol2entrez)
        case.loc[case.index, 'NodeIDX'] = case['HumanEntrezID'].map(self.node2index['gene2nid'])
        df = case.dropna(subset=['HumanEntrezID','NodeIDX'])
        df.loc[df.index, 'NodeIDX'] = df['NodeIDX'].astype(int)
        df.loc[df.index, 'logPval'] = - np.log10(df.Pval)   
        return df
    

    def mesh_score(self, df):

        node_embed = None
        for mesh_term in self.mesh_terms:
            edge_index = torch.tensor([df.NodeIDX.to_list(), 
                                      [self.node2index['mesh2nid'][mesh_term]]*df.NodeIDX.shape[0]])
            self.gm_data['gene', 'genemesh', 'mesh'].edge_label_index = edge_index
            prediction, node_embed = predict(self.model, self.gm_data, node_embed) # takes time to run #TODO: save node embed
            MESH_SCORE = f"LiteratureScore_{mesh_term}"
            df.loc[df.index, MESH_SCORE] = prediction
        return df

    def predict(self):
        dfs = []
        for inp in self.inputs:
            df = self.map2human(inp)
            if not df.empty: dfs.append(df)
        # save last table header
        self.headers = []
        with open(inp, 'r') as r:
            for i, line in enumerate(r):
                self.headers.append(line)
                if i >= 2: break 
        # concat results
        result = pd.concat(dfs)
        result = result.reset_index(drop=True)
        ## predit score
        self.result = self.mesh_score(result)

            
### predicts
if MESH_TERMS is not None:
    MESH_TERMS = MESH_TERMS.split(",")
hbcgm = HBCGM(inputdir=HBCGM_RESULTS, 
              mesh_term_ids=MESH_TERMS,# ['D018919', 'D009389', 'D043924'], # MESH_TERMS
              model = model,
              gm_data=gm_data,
              human_gene_nodes=human_gene_nodes,
              mouse_human_namedict=mouse_human_namedict,
              expression_headers=mus_tissue_exp)

hbcgm.predict()
OUTFILE = os.path.split(HBCGM_RESULTS)[-1] + ".results.txt"
hbcgm.save(os.path.join(HBCGM_RESULTS, OUTFILE))


### run example
# python predict.py --bundle /data/bases/fangzq/Pubmed/bundle \
#                   --mesh_terms D018919,D009389,D043924
#                   --hbcgm_result_dir /data/bases/fangzq/Pubmed/test_cases/TEST/RUN_000