
import os, sys, glob
import numpy as np
import pandas as pd
import networkx as nx
import joblib

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from sklearn.metrics import average_precision_score, roc_auc_score

from data import GeneMeshData
from models import HeteroGNN
from utils import add_train_args

# argument parser
args = add_train_args()
torch.manual_seed(seed=123456)

os.makedirs(args.outdir, exist_ok=True)

# read data in
print("Read graph data")
H = nx.read_gpickle(args.gene_mesh_nx_graph) # note: H is undirected multigraph

# Load mesh embeddings
print("Load embeddings")
mesh_features = pd.read_csv(args.mesh_embed, index_col=0)
gene_features = pd.read_csv(args.gene_embed, index_col=0, header=None)
gene_features.index = gene_features.index.astype(str)

# ## Remove isolated nodes, and nodes that don't have protein sequences 
# because it's not gonna help, if we'd like to train a neural network
isonodes = list(nx.isolates(H))
# remove isolated nodes
H.remove_nodes_from(list(nx.isolates(H)))
H.number_of_nodes()

print("Build GraphData")
# build data
gm = GeneMeshData(H, gene_features, mesh_features)
# align 
# align 
gm_data = gm()


# load test data
test_data = joblib.load("checkpoints/test.data.pkl")

# init model
model = HeteroGNN(heterodata=gm_data, hidden_channels=256, num_layers=2)
# Initialize lazy modules
with torch.no_grad():  
    out = model(test_data.x_dict, test_data.edge_index_dict)

# config
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 500, 800], gamma=0.5)
model.to('cpu')

@torch.no_grad()
def valid(data: HeteroData):
    model.eval()
    data.to('cpu')
    out = model(data.x_dict, data.edge_index_dict)
    y_score = model.distmult(out, data.edge_label_index_dict)
    y_true = data.edge_label_dict[('gene','genemesh','mesh')]
    loss = F.binary_cross_entropy_with_logits(y_score, y_true)
    y_score = torch.sigmoid(y_score).cpu().numpy()
    y_true = y_true.cpu().numpy()
    aupr = average_precision_score(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    return float(loss.item()), aupr, auroc

# release memory
del H
del gm_data
 
## Trainining
best_val_loss = np.Inf
test_data.to('cpu')



