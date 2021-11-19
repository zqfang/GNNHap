import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, GATConv
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops


class HeteroGNN(torch.nn.Module):
    def __init__(self, heterodata: HeteroData, 
                       hidden_channels: int, num_layers: int = 2):
        super().__init__()
        _edge_types = heterodata.edge_types
        self.node_types = heterodata.node_types
        # dense layer
        self.embed = torch.nn.ModuleDict()
        self.lin = torch.nn.ModuleDict()
        for node_type in self.node_types:
            if node_type == "mesh": 
                in_channel = 768
            elif node_type == "gene":
                in_channel = 1900
            self.embed[node_type] = torch.nn.Sequential(torch.nn.Linear(in_channel,hidden_channels),    
                                        torch.nn.BatchNorm1d(hidden_channels),
                                        torch.nn.ReLU(),)
            self.lin[node_type] = torch.nn.Linear(hidden_channels, hidden_channels)

        ## gene conv and mesh conv
        self.gconvs = torch.nn.ModuleList()
        self.mconvs = torch.nn.ModuleList()
        # self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):            
            mconv = HeteroConv({edge_type: GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
                                for edge_type in _edge_types if edge_type[1] not in ['genemesh','rev_genemesh', 'ppi'] 
                               }, aggr='mean')
            gconv = GCNConv(hidden_channels, hidden_channels, add_self_loops=False) # self_loop already added
            self.gconvs.append(gconv)
            self.mconvs.append(mconv)
            ## more simple api
            # conv = HeteroConv({edge_type: GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
            #         for edge_type in _edge_types if edge_type[1] not in ['genemesh','rev_genemesh'] 
            #         }, aggr='mean')
            # self.convs.append(conv)

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

        self.link_predictor = torch.nn.Sequential(torch.nn.Linear(hidden_channels*2, hidden_channels*2),    
                                        torch.nn.BatchNorm1d(hidden_channels*2),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_channels*2, hidden_channels),
                                        torch.nn.BatchNorm1d(hidden_channels, hidden_channels),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_channels, 1))            
        #self.loss_fn = torch.nn.BCEWithLogitsLoss()    
    def forward(self, x_dict, edge_index_dict):
        """
        edge_label_index_dict: supervision edge_index
        """
        # message
        gx= self.embed['gene'](x_dict['gene'])
        mx_dict = {'mesh': self.embed['mesh'](x_dict['mesh'])}
        # x_dict = {key: self.embed[key](x) for key, x in x_dict.items()}
        # for conv in self.convs:
        #     x_dict = conv(mx_dict, edge_index_dict)
        #     x_dict = {key: x.relu() for key, x in x_dict.items()} 
                    
        # mesh propgate, hetero conv, could be replace by RGCN
        for mconv in self.mconvs:
            mx_dict = mconv(mx_dict, edge_index_dict)
            mx_dict = {key: x.relu() for key, x in mx_dict.items()}  # bn and dropout ?
        # gene propogate, homogous conv
        for gconv in self.gconvs:
            gx = gconv(gx, edge_index_dict[('gene','ppi','gene')])
            gx = F.relu(gx) # bn and dropout ?
        # gene mesh  propogate, bipartile graph, conv carefully
        ## FIXME: makesure the partile graph is correct format
        x_dict = {'gene':gx, 'mesh': mx_dict['mesh']}
        for conv in self.gene_mesh_convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
  
        # FFN
        x_dict = {key: self.lin[key](x) for key, x in x_dict.items()} 
        # Decoder 
        return x_dict

    def distmult(self, x_dict, edge_label_index_dict):
        s = x_dict['gene'][edge_label_index_dict[('gene','genemesh','mesh')][0]]
        t = x_dict['mesh'][edge_label_index_dict[('gene','genemesh','mesh')][1]]
        score = torch.sum(s * t, dim=1)
        return score

    def score_loss(self,  x_dict, edge_label_index_dict, edge_label_dict):
        score = self.distmult(x_dict, edge_label_index_dict)
        target = edge_label_dict[('gene','genemesh','mesh')]
        #p = torch.sigmoid(score)
        #loss = self.loss_fn(score, target))
        return torch.nn.functional.binary_cross_entropy_with_logits(score, target)



    

# model 2: multiPercepton
class MLP(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP,self).__init__()
        # number of hidden nodes in each layer (512)
        # input_size = 1900 + 768
        hidden_1 = 1024
        hidden_2 = 512
        self.mlp = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_1),    
                                    torch.nn.BatchNorm1d(hidden_1),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_1, hidden_2),
                                    torch.nn.ReLU(),
                                    torch.nn.BatchNorm1d(hidden_2),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden_2, hidden_2),
                                    torch.nn.Dropout(0.2),
                                    torch.nn.Linear(hidden_2, 1)) 
    def forward(self,x):
        return self.mlp(x).view(-1) # squeeze last dim
    