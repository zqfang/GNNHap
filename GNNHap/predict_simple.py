
import os, sys, glob, json, argparse
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from functools import partial


import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from tqdm.auto import tqdm
from data import GeneMeshData
from models import HeteroGNN
from typing import List, Dict, Tuple, Union, AnyStr

def add_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Output file name") 
    parser.add_argument("--output", type=str, required=True, help="Output file name") 
    parser.add_argument("--bundle", default=None, type=str, help="Path to the directory name of the required files (for predition task)")
    parser.add_argument("--species", default='human', type=str, choices=('human','mouse'), help="the orangism for input gene names. choice from {human, mouse} ")
    parser.add_argument("--gene_mesh_graph", type=str, default=None, help="genemesh.data.pkl object or genemesh.gpkl object")
    parser.add_argument("--gene_embed", default=None, type=str, help="gene_embedding file")
    parser.add_argument("--mesh_embed", default=None, type=str, help="mesh_embedding file")
    parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of GNN. Default 64")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate. Default 0.01")
    parser.add_argument("--num_epochs", default=5, type=int, help="num_epochs for training. Default 5.")
    parser.add_argument("--batch_size", default=10000, type=int, help="batch size (num_edges) for mini-batch training and testing. Default 10000")
    parser.add_argument("--num_cpus", default=6, type=int, help="Number of cpus for dataloader, Default 6.")
    args = parser.parse_args()
    return args




# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# argument parser
args = add_train_args()
hidden_size = args.hidden_size
batch_size = args.batch_size # 10000

BUNDLE_PATH = args.bundle
HBCGM_RESULTS = args.input


model_weight = os.path.join(BUNDLE_PATH, "gnn_64_epoch0.pt")
genemesh_data = os.path.join(BUNDLE_PATH, "genemesh.data.pkl") # genemesh.data.pkl

mouse_human_namedict = None
if args.species == 'mouse':
    mouse_human_namedict = os.path.join(BUNDLE_PATH, "mouse2human.genenames.json")

if not os.path.exists(model_weight):
    raise LookupError("model weights not found")


## Load Graph nodes' metadata
mesh_nodes = joblib.load(os.path.join(BUNDLE_PATH,"mesh_nodes.pkl"))
human_gene_nodes = joblib.load(os.path.join(BUNDLE_PATH,"human_gene_nodes.pkl"))
with open(os.path.join(BUNDLE_PATH,"human_gene_mesh.pmids.json"), 'r') as j:
    pubmed_json = json.load(j)
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


class Simple:
    def __init__(self, input: AnyStr, 
                  model: torch.nn.Module,
                  gm_data: HeteroData,
                  human_gene_nodes: Union[AnyStr, Dict], 
                  mesh_nodes:Union[AnyStr, Dict],
                  pubmed_ids: Union[AnyStr, Dict],
                  mouse_human_namedict: Union[AnyStr, Dict] = None,
                  ):

        self.inputs = pd.read_table(input)
        self.node2index = { 'gene2nid': {v: key for key, v in  gm_data['nid2gene'].items()}, 
                            'mesh2nid': {v:key for key, v in gm_data['nid2mesh'].items()}}
        self.gm_data = gm_data
        if not isinstance(human_gene_nodes, dict):
            human_gene_nodes = joblib.load(human_gene_nodes)
        self.symbol2entrez = {}
        for entrez, value in human_gene_nodes.items():
            self.symbol2entrez[value['gene_symbol']] = entrez

        # load 
        self.mouse2human = mouse_human_namedict # if none, means human gene list input, if dict, skip
        if isinstance(mouse_human_namedict, str):
            with open(mouse_human_namedict, 'r') as j:
                self.mouse2human = json.load(j) ## Mouse gene name to human

        self.mesh_nodes = mesh_nodes
        if not isinstance(mesh_nodes, dict):
            mesh_nodes = joblib.load(mesh_nodes)
        self.PUBMEDID = pubmed_ids
        if not isinstance(pubmed_ids, dict):
            self.PUBMEDID = json.load(pubmed_ids)
        self.model = model
        self.headers = ""
        self._node_embed = None 
    
    def get_pubmed_id(self, row):  
        entrzid = row['HumanEntrezID']
        mesh = row[self._columns[1]]
        k = f"{entrzid}__{mesh}"
        if k in self.PUBMEDID:
            return ",".join(self.PUBMEDID[k])

        ## TODO get shortest Path
        return "Indirect"
        
    def save(self, output:AnyStr):
        #mtd = self.mesh_nodes[mesh_term]['DescriptorName']
        self.result.loc[self.result.index, 'MeSH_Terms'] = self.result['MeSH'].apply(lambda m: self.mesh_nodes[m]['DescriptorName'])
        self.result.loc[self.result.index, 'PubMedID'] = self.result.apply(self.get_pubmed_id, axis=1)
        outcols = self._columns + ['HumanEntrezID','MeSH_Terms','LiteratureScore','PubMedID']
        self.result.loc[:, outcols].to_csv(output, sep="\t", index=False)

    def map2human(self, result):
        if isinstance(result, pd.DataFrame):
            case = result
        else:
            self.headers = []
            case = pd.read_table(result)

        if case.shape[0] < 1: 
            print(f"input data is empty: {result}")
            return case
        self._columns = list(case.columns) 
        assert len(self._columns) >=2, "at least two columns input needed <#GeneName, MeSH>"
        g, m = self._columns[:2] # first two column, gene, mesh

        if self.mouse2human is None:
            case.loc[case.index, 'HumanEntrezID'] = case[g].map(self.symbol2entrez)
        else: 
            case.loc[case.index, 'HumanEntrezID'] = case[g].map(self.mouse2human).map(self.symbol2entrez)
        case.loc[case.index, 'GNodeIDX'] = case['HumanEntrezID'].map(self.node2index['gene2nid'])
        case.loc[case.index, 'MNodeIDX'] = case[m].map(self.node2index['mesh2nid'])
        df = case.dropna(subset=['HumanEntrezID','GNodeIDX'])
        df.loc[df.index, 'GNodeIDX'] = df['GNodeIDX'].astype(int)
        df.loc[df.index, 'MNodeIDX'] = df['MNodeIDX'].astype(int)
        # rename column name, to compatible with webapp
        df.rename(columns = {g:'#GeneName', m: 'MeSH'}, inplace = True)
        return df
    

    def mesh_score(self, df):
        if df.shape[0] < 1: return df

        edge_index = torch.tensor([df.GNodeIDX.to_list(), df.MNodeIDX.to_list()])
        self.gm_data['gene', 'genemesh', 'mesh'].edge_label_index = edge_index
        prediction, self._node_embed = predict(self.model, self.gm_data, self._node_embed) # save node embed to reduce run time
        df.loc[df.index, 'LiteratureScore'] = prediction
        return df

    def predict(self):
        df = self.map2human(self.inputs)
        ## predit score
        self.result = self.mesh_score(df)



## START predcition
print("Inference")
hbcgm = Simple(input=HBCGM_RESULTS, 
            model = model,
            gm_data=gm_data,
            human_gene_nodes=human_gene_nodes,
            mesh_nodes = mesh_nodes,
            pubmed_ids=pubmed_json,
            mouse_human_namedict=mouse_human_namedict,)

hbcgm.predict()
hbcgm.save(args.output)
print("DONE")