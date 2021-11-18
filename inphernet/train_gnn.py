
import os, sys, glob, gc
import numpy as np
import pandas as pd
import networkx as nx
import joblib

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from torchmetrics import AveragePrecision, Accuracy, AUROC
from data import GeneMeshData
from models import HeteroGNN
from utils import add_train_args
from datetime import datetime

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
criterion = torch.nn.BCEWithLogitsLoss() #
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 500, 800], gamma=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=50, T_mult=4, eta_min=1e-4)

def train(train_data: HeteroData, epoch):
    #train_data.to(device)
    model.train()
    batch_size=100000
    edge_label_index = train_data['gene','genemesh','mesh'].edge_label_index
    edge_label = train_data['gene','genemesh','mesh'].edge_label
    # minibatch training for supervision links
    data_loader = DataLoader(torch.arange(edge_label.size(0)), batch_size=batch_size, shuffle=True)
    train_loss = 0
    for i, perm in enumerate(data_loader):
        optimizer.zero_grad()
        h_dict = model(train_data.x_dict, train_data.edge_index_dict)
        g = h_dict['gene'][edge_label_index[0, perm]]
        m = h_dict['mesh'][edge_label_index[1, perm]]
        targets = edge_label[perm]
        inp = torch.cat([g, m], dim=1) # note the dim
        preds = model.link_predictor(inp)
        loss = criterion(preds.view(-1), targets) 
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + i / len(data_loader)) # so, each mini-batch will have different learning rate
        train_loss += loss.item()
        lr = scheduler.get_last_lr()[0] 
        print(f'{datetime.now()}  Epoch: {epoch:03d}, Step: {i}, Train Loss: {loss.item():.7f}, Learning rate: {lr:.7f}')  
    train_loss /= len(data_loader)
    return  train_loss


@torch.no_grad()
def valid(data: HeteroData, device='cpu'):
    model.eval()
    #data.to(device)
    batch_size=10000
    # supervision links
    edge_label_index = data['gene','genemesh','mesh'].edge_label_index
    edge_label = data['gene','genemesh','mesh'].edge_label
    # 
    val_loss = 0
    y = []
    y_preds = []
    # 
    h_dict = model(data.x_dict, data.edge_index_dict) # only need compute once for node_representations
    # minibatch testing for supervision links
    data_loader = DataLoader(torch.arange(edge_label.size(0)), batch_size=batch_size)
    for i, perm in enumerate(data_loader):
        g = h_dict['gene'][edge_label_index[0, perm]] 
        m = h_dict['mesh'][edge_label_index[1, perm]]
        targets = edge_label[perm]
        # NOTE: concat here
        inp = torch.cat([g, m], dim=1)
        preds = model.link_predictor(inp).view(-1)
        loss = criterion(preds, targets) 
        #loss.backward()
        val_loss += loss.item()
        y.append(targets)
        y_preds.append(preds)
        print(f'{datetime.now()}  Epoch: {epoch:03d}, Step: {i}, Valid Loss: {loss.item():.7f}, Learning rate: {lr:.7f}')  
    val_loss /= len(data_loader)
    y_preds = torch.cat(y_preds, dim=0) # note the difference with torch.stack()
    y = torch.cat(y, dim=0).type(torch.LongTensor)
    y_preds = torch.sigmoid(y_preds)
    ap = AveragePrecision(pos_label=1)(y_preds, y)
    auroc = AUROC(pos_label=1)(y_preds, y)
    acc = Accuracy(threshold=0.5)(y_preds, y)
    return {'val_loss': val_loss.item(), 'acc': acc.item(), 'ap': ap.item(), 'auroc': auroc.item(), 'y':y.numpy(), 'y_preds': y_preds.numpy()}

# release memory
# del H
# del gm_data
# gc.collect()
## Trainining
best_val_loss = np.Inf
train_data.to(device)
val_data.to(device)
model.to(device)


print("Start training")
for epoch in range(1, 1001):
    #model.train()
    #optimizer.zero_grad()
    #out = model(train_data.x_dict, train_data.edge_index_dict)
    #loss = model.score_loss(out, train_data.edge_label_index_dict, train_data.edge_label_dict)
    #loss.backward()
    #optimizer.step()
    #scheduler.step()
    #lr = scheduler.get_last_lr()[0]
    #train_loss = loss.item()
    train_loss = train(train_data, epoch, device)
    val_metrics = valid(val_data, device)  
    val_loss = val_metrics['val_loss']
    auroc = val_metrics['auroc']
    ap = val_metrics['auroc']
    acc = val_metrics['acc']
    print(f'{datetime.now()}  Epoch: {epoch:03d}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}, Val ROC: {auroc:.4f}, Val PR: {ap:.3f}, Val Acc: {acc:.7f}')
    tb.add_scalar('Loss', {'train': train_loss, 'valid': val_loss}, epoch) 
    tb.add_scalar('Valid', {'ap': ap, 'auroc': auroc, 'acc':acc}, epoch) 
    tb.add_pr_curve("Precision-Recall", val_metrics['y'], val_metrics['y_preds'], epoch)

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save({'state_dict': model.state_dict(),
                    'epoch': epoch}, 
                    os.path.join(args.outdir, 'human_gene_mesh_best_model.pt'))
    if epoch % 20 == 0:
        torch.save({'state_dict': model.state_dict(),
            'epoch': epoch}, 
            os.path.join(args.outdir, f'human_gene_mesh_epoch_{epoch}_model.pt'))
# finish training
tb.close()
#
print(f'{datetime.now()}: Done Training ')