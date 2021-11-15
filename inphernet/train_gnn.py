
import os, sys, glob, gc
import numpy as np
import pandas as pd
import networkx as nx
import joblib

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from sklearn.metrics import average_precision_score, roc_auc_score

from data import GeneMeshData
from models import HeteroGNN
from utils import add_train_args


# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# argument parser
args = add_train_args()
torch.manual_seed(seed=123456)

os.makedirs(args.outdir, exist_ok=True)
tb = SummaryWriter(log_dir = args.outdir, filename_suffix="human_genemesh_gnn")
# read data in
#H = nx.read_gpickle(args.gene_mesh_graph) # note: H is undirected multigraph

# Load mesh embeddings
print("Load embeddings")
mesh_features = pd.read_csv(args.mesh_embed, index_col=0)
gene_features = pd.read_csv(args.gene_embed, index_col=0, header=None)
gene_features.index = gene_features.index.astype(str)

# # ## Remove isolated nodes, and nodes that don't have protein sequences 
# # because it's not gonna help, if we'd like to train a neural network
# isonodes = list(nx.isolates(H))
# # remove isolated nodes
# H.remove_nodes_from(list(nx.isolates(H)))
# H.number_of_nodes()

# print("Build GraphData")
# # build data
# gm = GeneMeshData(H, gene_features, mesh_features)
# # align 
# gm_data = gm()
# # train test split
# train_data, val_data, test_data = T.RandomLinkSplit(is_undirected=True, 
#                                          add_negative_train_samples=True, 
#                                          neg_sampling_ratio=1.0,
#                                          edge_types=('gene','genemesh','mesh'), # must be tuple, not list
#                                          rev_edge_types=('mesh','rev_genemesh','gene'))(gm_data)
print("Read graph data")
# joblib.dump(train_data, filename=os.path.join(args.outdddir,"train.data.pkl"))
# joblib.dump(val_data, filename=os.path.join(args.outdir,"val.data.pkl"))
# joblib.dump(test_data, filename=os.path.join(args.outdir,"test.data.pkl"))
train_data = joblib.load("checkpoints/train.data.20211114.pkl")
val_data = joblib.load("checkpoints/val.data.20211114.pkl")
test_datat = joblib.load("checkpoints/test.data.20211114.pkl")
# init model
model = HeteroGNN(heterodata=train_data, hidden_channels=512, num_layers=2)

# config
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 500, 800], gamma=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=50, T_mult=4, eta_min=1e-4)
model.to(device)

@torch.no_grad()
def valid(data: HeteroData, device='cpu'):
    model.eval()
    data.to(device)
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
# del H
# del gm_data
# gc.collect()
## Trainining
best_val_loss = np.Inf
train_data.to(device)

print("Start training")
for epoch in range(1, 1001):
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x_dict, train_data.edge_index_dict)
    loss = model.score_loss(out, train_data.edge_label_index_dict, train_data.edge_label_dict)
    loss.backward()
    optimizer.step()
    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    train_loss = loss.item()
    val_loss, aupr, auroc = valid(val_data)  
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}, Val ROC: {auroc:.4f}, Val PR: {aupr:.3f}, lr: {lr:.7f}')
    tb.add_scalar('Train/loss', train_loss, epoch) 
    tb.add_scalar('Valid/loss', val_loss, epoch) 
    tb.add_scalar('Valid/pr', aupr, epoch) 
    tb.add_scalar('Valid/roc', auroc, epoch) 

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save({'state_dict': model.state_dict(),
                    'epoch': epoch}, 
                    os.path.join(args.outdir, 'gene_mesh_best_model.pt'))
    if epoch % 100 == 0:
        torch.save({'state_dict': model.state_dict(),
            'epoch': epoch}, 
            os.path.join(args.outdir, f'gene_mesh_epoch_{epoch}_model.pt'))
tb.close()