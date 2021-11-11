#!/usr/bin/env python
# coding: utf-8

import os, sys, glob
import numpy as np
import pandas as pd
import networkx as nx
import torch

from torch_geometric.data import HeteroData

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.utils import (negative_sampling, to_undirected, is_undirected,
                                   add_remaining_self_loops, remove_isolated_nodes, remove_self_loops)
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Tuple, List, Dict

H = nx.read_gpickle("human_gene_mesh_hetero_nx.gpkl") # note: H is undirected multigraph


# In[5]:


H.is_directed()
H.number_of_nodes() # nodes are double, because it's directed graph


# ## Remove isolated nodes, and nodes that don't have protein sequences 
# because it's not gonna help, if we'd like to train a neural network
isonodes = list(nx.isolates(H))
for inode in  isonodes:
    if inode.startswith("D"):
        print(inode)
# only genes has isolated nodes

# remove isolated nodes
H.remove_nodes_from(list(nx.isolates(H)))
H.number_of_nodes()

# Load mesh embeddings
mesh_features = pd.read_csv("mesh.sentencetransformer.embed.csv", index_col=0)
gene_features = pd.read_csv("human_gene_unirep.embeb.csv",index_col=0, header=None)

gene_features.index = gene_features.index.astype(str)


class GeneMeshData:
    """
    Convert networkX to HeteroData
    """
    def __init__(self, H: nx.MultiGraph, 
                 gene_features: pd.DataFrame = None, 
                 mesh_features: pd.DataFrame = None):
        """
        H: networkx.multiGraph, heterogenouse graph
        """
        if not isinstance(H, nx.MultiGraph):
            raise Exception("H must be a networkx.MultiGraph object")
        self.H = H.copy()
        self.data = HeteroData()
        
        gene_nodes = None
        mesh_nodes = None
        self.gene_features = gene_features
        self.mesh_features = mesh_features
        if isinstance(gene_features, pd.DataFrame):
            gene_nodes = gene_features.index.to_list()
        if isinstance(mesh_features, pd.DataFrame):
            mesh_nodes = mesh_features.index.to_list() 
            
        self._get_nodes(gene_nodes, mesh_nodes)
        self._get_triplets()
        
        if gene_features is not None:
            self.gene_features = self.gene_features.loc[self.g_nodes].values
        if mesh_features is not None:
            self.mesh_feattures = self.mesh_features.loc[self.m_nodes].values
        
        
    def __call__(self,  gene_features: pd.DataFrame = None, 
                        mesh_features: pd.DataFrame = None) -> HeteroData: 
        """
        build HeteroData Object
        """
        # build graph
        self.set_edges()

        if isinstance(gene_features, pd.DataFrame):
            self.gene_features = gene_features.loc[self.g_nodes].values
        if isinstance(mesh_features, pd.DataFrame):
            self.mesh_feattures = mesh_features.loc[self.m_nodes].values
        # set feature vectors  
        self.set_nodes('gene', self.gene_features)
        self.set_nodes('mesh', self.mesh_features)
        # add self-loop edges. 
        # each node might receive one or more (one per appropriate edge type) messages 
        # from itself during message passing
        # self.data = T.AddSelfLoops()(self.data)
        #  works like in the homogenous case, and normalizes all specified features (of all types) to sum up to one.
        self.data = T.ToUndirected()(self.data)
        #self.data = T.NormalizeFeatures()(self.data)
        return self.data
    
    def _get_nodes(self, gene_nodes: List[str]=None, mesh_nodes: List[str]=None):
        """
        Select nodes when features vector is not None. If none, keep all nodes
        """
        g_nodes = []
        m_nodes = []
        nodes_to_remove = []
        for node_idx, (node, node_dict) in enumerate(self.H.nodes(data=True)):
            if node_dict['node_label'] == 'gene':
                g = node_dict['node_entrez_id']
                if (gene_nodes is not None) and (g not in gene_nodes):
                    nodes_to_remove.append(node)
                else:
                    g_nodes.append(g)
                
            else:
                m = node_dict['node_mesh_id']
                if (mesh_nodes is not None) and (m not in mesh_nodes):
                    nodes_to_remove.append(node)
                else:
                    m_nodes.append(m)
                
        self.g_nodes = sorted(g_nodes)
        self.m_nodes = sorted(m_nodes)
        self.gene2idx = {str(v):i for i,v in enumerate(self.g_nodes)}
        self.mesh2idx = {str(v):i for i,v in enumerate(self.m_nodes)}
        self.node2idx = {**self.gene2idx, **self.mesh2idx}
        
        # remove nodes without features vectors
        if len(nodes_to_remove) > 0:
            self.H.remove_nodes_from(nodes_to_remove)
            
    def _get_triplets(self):
        """
        get edge and node
        """
        triplets = [] # head, relation, tail 
        for edge_idx, (head, tail, edge_dict) in enumerate(self.H.edges(data=True)):
            relation = edge_dict['edge_type']
            if 'weight' in edge_dict:
                weight = edge_dict['edge_weight']
            else: 
                weight = 1.0
                
            if str(tail).isdigit() and str(head).startswith('D'):
                # gene --> mesh
                head, tail = tail, head
                
            triplets.append((head, relation, tail, weight))
        self.triplets = pd.DataFrame(triplets, columns=['head','relation', 'tail', 'weight'])
        self.edge_types = sorted(self.triplets.relation.unique())
                                                               
    def set_nodes(self, node_type='gene', node_features=None):
        """
        node_type: gene, or mesh
        """
        nodes = []
        node_cat = []
        for node_idx, (node, node_dict) in enumerate(self.H.nodes(data=True)):
            if node_dict['node_type'] == node_type:
                nodes.append((node, self.node2idx[str(node)], node_dict['node_type'], node_dict['node_cat']))
            
        nodes = pd.DataFrame(nodes, columns=['node','nodeidx','metanode','node_labels'])
        nodes = nodes.sort_values('nodeidx') # this is very important step
        self.data[node_type].num_nodes = len(nodes)
        self.data[node_type].node_metatype = nodes.metanode.to_list()
        self.data[node_type].node_labels = nodes.node_labels.to_list()
        if node_features is None:
            self.data[node_type].x = torch.nn.functional.one_hot(torch.arange(0, len(nodes))).type(torch.float)
        elif isinstance(node_features, pd.DataFrame):
            self.data[node_type].x = torch.from_numpy(node_features.loc[self.m_nodes].values.astype(np.float32))
        elif isinstance(node_features, np.ndarray):
            self.data[node_type].x = torch.from_numpy(node_features.astype(np.float32))
        else:
            self.data[node_type].x = torch.tensor(node_features, dtype=torch.float)
            

    def set_edges(self, edge_features=None):
        """
        set edge index, convert to PyG compatible convert
        """
        self.edgetype2index = {t:i for i, t in enumerate(self.edge_types)}
        edge_type_tmp = torch.from_numpy(self.triplets['relation'].map(self.edgetype2index).values)
        edge_attr_full = torch.nn.functional.one_hot(edge_type_tmp, num_classes=len(self.edge_types)).type(torch.float)
        
        self.triplets['head_idx'] = self.triplets['head'].map(self.node2idx)
        self.triplets['tail_idx'] = self.triplets['tail'].map(self.node2idx)
        for edge_type in self.edge_types:
            mask = self.triplets.relation == edge_type
            triplets = self.triplets[mask]
            edge_index = torch.tensor([triplets['head_idx'].astype(int).to_list(), 
                                       triplets['tail_idx'].astype(int).to_list()])
            
            weight = torch.from_numpy(triplets['weight'].values.reshape(-1,1).astype(np.float32))
            edge_attr = edge_attr_full[torch.from_numpy(mask.values)]
            edge_attr = torch.cat([weight, edge_attr], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr]) # FIXME: how to add self-loop edge_attr for hetero ?
            
            if edge_type == 'genemesh':
                self.data['gene', edge_type, 'mesh'].edge_index = edge_index
                #self.data['gene', edge_type, 'mesh'].edge_attr = edge_attr
            elif edge_type == 'ppi':
                # to PyG compatible undirecte graph
                edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
                # add self_loops
                edge_index, edge_weight = add_remaining_self_loops(edge_index, num_nodes=len(self.g_nodes))
                self.data['gene', edge_type, 'gene'].edge_index = edge_index
                #self.data['gene', edge_type, 'gene'].edge_attr = edge_attr
            else:
                # to PyG compatible undirecte graph
                edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
                # add self_loops
                edge_index, edge_weight  = add_remaining_self_loops(edge_index, num_nodes=len(self.m_nodes))
                # FIXME: remove isolated nodes ?
                edge_index, _ , _ = remove_isolated_nodes(edge_index, num_nodes=len(self.m_nodes))
                self.data['mesh', edge_type, 'mesh'].edge_index = edge_index
                #self.data['mesh', edge_type, 'mesh'].edge_attr = edge_attr





gm = GeneMeshData(H, gene_features, mesh_features)

# align 
gm_data = gm()

train_data, val_data, test_data = T.RandomLinkSplit(is_undirected=True, 
                                         add_negative_train_samples=True, 
                                         neg_sampling_ratio=1.0,
                                         edge_type=('gene','genemesh','mesh'), # must be tuple, not list
                                         rev_edge_type=('mesh','rev_genemesh','gene'))(gm_data)



class HeteroGNN(torch.nn.Module):
    def __init__(self, heterodata: HeteroData, 
                       hidden_channels: int, num_layers: int = 2):
        super().__init__()
        _edge_types = heterodata.edge_types
        # dense layer
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in heterodata.node_types:
            self.lin_dict[node_type] = torch.nn.Sequential(Linear(-1, hidden_channels),    
                                                           torch.nn.BatchNorm1d(hidden_channels),
                                                           torch.nn.ReLU())
        # hetero convolute (gene --> mesh, mesh --> gene)
        self.gene_mesh_convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
               ('gene', 'genemesh', 'mesh'): 
                GATConv((hidden_channels, hidden_channels), hidden_channels, add_self_loops=False),
               ('mesh', 'rev_genemesh', 'gene'): 
                GATConv((hidden_channels, hidden_channels), hidden_channels, add_self_loops=False),
            }, aggr='mean')
            self.gene_mesh_convs.append(conv)
            
            
        ## gene conv and mesh conv
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # NOTE: lazy initization require define new layers every time
            # SAGEConv((-1, -1), hidden_channels)
            mconv = HeteroConv({edge_type: GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
                                for edge_type in _edge_types if edge_type[1] not in ['genemesh','rev_genemesh'] 
                               }, aggr='sum')
            #mconv = HeteroConv(hetero_tmp, aggr='sum')
            #gconv = HeteroConv({('gene', 'ppi', 'gene'): GCNConv(-1, hidden_channels, add_self_loops=False)}, aggr='mean')
            self.convs.append(mconv)
            #self.gene_convs.append(gconv)
            
        self.lin_dict2 = torch.nn.ModuleDict()
        for node_type in heterodata.node_types: 
            self.lin_dict2[node_type] = torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),    
                                        torch.nn.BatchNorm1d(hidden_channels),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_channels, hidden_channels))
            
    def forward(self, x_dict, edge_index_dict):
        # linear + relu + bn
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()} 
            
        # convolute gene mesh net
        for conv in self.gene_mesh_convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
      
        # convolute ppi and mesh net
        for mconv in self.convs:
            x_dict = mconv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        # FFN
        x_dict = {key: self.lin_dict2[key](x) for key, x in x_dict.items()} 
        
        return x_dict

    
    def distmult(self, x_dict, edge_label_index_dict):
        s = x_dict['gene'][edge_label_index_dict[('gene','genemesh','mesh')][0]]
        t = x_dict['mesh'][edge_label_index_dict[('gene','genemesh','mesh')][1]]
        score = torch.sum(s * t, dim=1)
        return score
    
    def score_loss(self,  x_dict, edge_label_index_dict, edge_label_dict):
        score = self.distmult(x_dict, edge_label_index_dict)
        target = edge_label_dict[('gene','genemesh','mesh')]
        return F.binary_cross_entropy_with_logits(score, target)


model = HeteroGNN(heterodata=gm_data, hidden_channels=256, num_layers=2)



with torch.no_grad():  # Initialize lazy modules.
    out = model(gm_data.x_dict, gm_data.edge_index_dict)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [200, 500, 800], gamma=0.5)
model.to('cpu')

def train(data: HeteroData):
    data.to('cpu')
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = model.score_loss(out, data.edge_label_index_dict, data.edge_label_dict)
    #loss = F.cross_entropy(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.item())

@torch.no_grad()
def valid(data: HeteroData):
    model.eval()
    data.to('cpu')
    #y = data.edge_label_index_dict[('gene','genemesh','mesh')]
    out = model(data.x_dict, data.edge_index_dict)
    #pred = model.dismult(out, data.edge_label_index_dict)
    #loss = model.score_loss(out, data.edge_label_index_dict, data.edge_label_dict)
    y_score = self.distmult(data.x_dict, data.edge_label_index_dict)
    y_true = data.edge_label_dict[('gene','genemesh','mesh')]
    aupr = average_precision_score(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    loss = F.binary_cross_entropy_with_logits(y_score, y_true)
    return float(loss.item()), aupr, auroc


# In[ ]:


best_val_loss = np.Inf
train_data.to('cpu')
for epoch in range(1, 1001):
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x_dict, train_data.edge_index_dict)
    loss = model.score_loss(out, train_data.edge_label_index_dict, train_data.edge_label_dict)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_loss = loss.item()
    val_loss, aupr, auroc = valid(val_data)  
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.7f}, Val Loss: {val_loss:.7f}, Val ROC: {auroc: .4f}, Val PR: {aupr:.3f}')

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save({'state_dict': model.state_dict(),
                    'best_epoch': epoch}, 
                    os.path.join('human_gene_mesh_best_model.pt'))



