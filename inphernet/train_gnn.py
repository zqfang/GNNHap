'''
The GNN for GeneMesh HeteroGraph training
'''
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
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from data import GeneMeshData
from models import HeteroGNN
from utils import add_train_args
from datetime import datetime
from tqdm.auto import tqdm

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_num_threads(60)
# argument parser
args = add_train_args()
torch.manual_seed(seed=123456)

os.makedirs(args.outdir, exist_ok=True)
tb = SummaryWriter(log_dir = args.outdir, filename_suffix=".GNN")
# # # read data in
# print("Load Gene Mesh Graph")
# H = nx.read_gpickle(args.gene_mesh_graph) # note: H is undirected multigraph

# # Load mesh embeddings
# print("Load embeddings")
# mesh_features = pd.read_csv(args.mesh_embed, index_col=0, header=None)
# gene_features = pd.read_csv(args.gene_embed, index_col=0, header=None)
# gene_features.index = gene_features.index.astype(str)

# # print("Build GraphData")
# # # build data
# gm = GeneMeshData(H, gene_features, mesh_features)
# # # align 
# gm_data = gm()
# # save data for future use
# joblib.dump(gm_data, filename=os.path.join(args.outdir,"genemesh.data.pkl"))
# # # train test split
# train_data, val_data, test_data = T.RandomLinkSplit(is_undirected=True, 
#                                          add_negative_train_samples=True, 
#                                          neg_sampling_ratio=1.0,
#                                          edge_types=('gene','genemesh','mesh'), # must be tuple, not list
#                                          rev_edge_types=('mesh','rev_genemesh','gene'))(gm_data)
# print("Save graph data")
# joblib.dump(train_data, filename=os.path.join(args.outdir,"train.data.pkl"))
# joblib.dump(val_data, filename=os.path.join(args.outdir,"val.data.pkl"))
# joblib.dump(test_data, filename=os.path.join(args.outdir,"test.data.pkl"))

print("Read graph data")
train_data = joblib.load(os.path.join(args.outdir,"train.data.pkl"))
val_data = joblib.load(os.path.join(args.outdir,"val.data.pkl"))
##test_datat = joblib.load(os.path.join(args.outdir,"test.data.pkl"))
## init model
num_epochs = args.num_epochs
hidden_size = args.hidden_size
batch_size = args.batch_size # 10000
model = HeteroGNN(heterodata=train_data, hidden_channels=hidden_size, num_layers=2)
print("Model")
print(model)
# move data to GPU
train_data.to(device)
val_data.to(device)
model.to(device)
# config
criterion = torch.nn.BCEWithLogitsLoss() #
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 500, 800], gamma=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=2, eta_min=1e-4)

def train(epoch, device='cpu'):
    #train_data.to(device)
    model.train()
    edge_label_index = train_data['gene','genemesh','mesh'].edge_label_index
    edge_label = train_data['gene','genemesh','mesh'].edge_label
    # minibatch training for supervision links
    data_loader = DataLoader(torch.arange(edge_label.size(0)), batch_size=batch_size, shuffle=True)
    train_loss = 0
    for i, perm in enumerate(tqdm(data_loader, total=len(data_loader), desc='Train', leave=True, position=1)):
        optimizer.zero_grad()
        h_dict = model(train_data.x_dict, train_data.edge_index_dict)
        g = h_dict['gene'][edge_label_index[0, perm]]
        m = h_dict['mesh'][edge_label_index[1, perm]]
        targets = edge_label[perm]
        targets = targets.to(device)
        inp = torch.cat([g, m], dim=1) # note the dim
        preds = model.link_predictor(inp)
        loss = criterion(preds.view(-1), targets) 
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + i / len(data_loader)) # so, each mini-batch will have different learning rate
        train_loss += loss.item()
        lr = scheduler.get_last_lr()[0] 
        tqdm.write(f'{datetime.now()}  Epoch: {epoch:03d}, Step: {i}, Train Loss: {loss.item():.7f}, Learning rate: {lr:.7f}')  
    train_loss /= len(data_loader)
    return  train_loss


@torch.no_grad()
def valid(epoch, device='cpu'):
    model.eval()
    #data.to(device)
    # supervision links
    edge_label_index = val_data['gene','genemesh','mesh'].edge_label_index
    edge_label = val_data['gene','genemesh','mesh'].edge_label
    # 
    val_loss = 0
    y = []
    y_preds = []
    # TODO: use MetricCollection
    # valid_ap = AveragePrecision(pos_label=1, compute_on_step=False)
    # valid_auroc = AUROC(pos_label=1, compute_on_step=False)
    # valid_acc = Accuracy(threshold=0.5, compute_on_step=False)

    h_dict = model(val_data.x_dict, val_data.edge_index_dict).to(device) # only need compute once for node_representations
    # minibatch testing for supervision links
    data_loader = DataLoader(torch.arange(edge_label.size(0)), batch_size=batch_size)
    for i, perm in enumerate(tqdm(data_loader, total=len(data_loader), desc='Valid', position=1, leave=True)):
        g = h_dict['gene'][edge_label_index[0, perm]] 
        m = h_dict['mesh'][edge_label_index[1, perm]]
        targets = edge_label[perm]
        targets = targets.type(torch.LongTensor).to(device)
        # NOTE: concat here
        inp = torch.cat([g, m], dim=1).to(device)
        preds = model.link_predictor(inp).view(-1)
        loss = criterion(preds, targets) 
        #loss.backward()
        val_loss += loss.item()
        y.append(targets.cpu())
        y_preds.append(preds.cpu())
        # valid_ap(preds, targets)
        # valid_auroc(preds, targets)
        # valid_acc(preds, targets)
    val_loss /= len(data_loader)
    y_preds = torch.cat(y_preds, dim=0) # note the difference with torch.stack()
    y = torch.cat(y, dim=0).type(torch.LongTensor).numpy()
    y_preds = torch.sigmoid(y_preds).numpy()
    # ap = valid_ap.compute()
    # auroc = valid_auroc.compute()
    # acc = valid_acc.compute()
    auroc = roc_auc_score(y, y_preds)
    acc = accuracy_score(y > 0, y_preds > 0.5)
    ap = average_precision_score(y, y_preds)
    return {'val_loss': val_loss.item(), 'acc': acc, 'ap': ap, 'auroc': auroc, 'y':y, 'y_preds': y_preds}

# release memory
# del H
# del gm_data
# gc.collect()
## Trainining
best_val_loss = np.Inf
print("Start training")
for epoch in tqdm(range(0, num_epochs), total=num_epochs, position=0, desc='Epoch'):
    train_loss = train(epoch, device)
    val_metrics = valid(epoch, device)  
    val_loss = val_metrics['val_loss']
    auroc = val_metrics['auroc']
    ap = val_metrics['auroc']
    acc = val_metrics['acc']
    tqdm.write(f'{datetime.now()}  Epoch: {epoch:03d}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}, Val ROC: {auroc:.4f}, Val PR: {ap:.3f}, Val Acc: {acc:.7f}')
    tb.add_scalar('Loss', {'train': train_loss, 'valid': val_loss}, epoch) 
    tb.add_scalar('Valid', {'ap': ap, 'auroc': auroc, 'acc':acc}, epoch) 
    tb.add_pr_curve("Precision-Recall", val_metrics['y'], val_metrics['y_preds'], epoch)

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save({'state_dict': model.state_dict(),
                    'epoch': epoch, 'hidden_size': hidden_size, 'batch_size': batch_size}, 
                    os.path.join(args.outdir, 'gnn_best_model.pt'))
    # save every epoch
    torch.save({'state_dict': model.state_dict(),
        'epoch': epoch, 'hidden_size': hidden_size, 'batch_size': batch_size}, 
        os.path.join(args.outdir, f'gnn_epoch_{epoch}.pt'))
# finish training
tb.close()
#
print(f'{datetime.now()}: Done Training ')