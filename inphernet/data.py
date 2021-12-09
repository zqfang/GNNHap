
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data.data import Data
import torch_geometric.transforms as T
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.utils import (negative_sampling, to_undirected, is_undirected,
                                   add_remaining_self_loops, remove_isolated_nodes, remove_self_loops)

import torch_geometric.transforms as T
from torch_geometric.utils import (negative_sampling, to_undirected, is_undirected,
                                   add_remaining_self_loops, remove_isolated_nodes, remove_self_loops)
from typing import Tuple, List, Dict, Union, AnyStr

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
        self._data = None
        self.data = HeteroData()
        
        gene_nodes = None
        mesh_nodes = None
        self.gene_features = gene_features
        self.mesh_features = mesh_features
        if isinstance(gene_features, pd.DataFrame):
            gene_nodes = gene_features.index.to_list()
        if isinstance(mesh_features, pd.DataFrame):
            mesh_nodes = mesh_features.index.to_list() 
        # trim node and get edgelists
        self._get_nodes(gene_nodes, mesh_nodes)
        self._get_triplets()
        # set attributes
        if gene_features is not None:
            self.gene_features = self.gene_features.loc[self.g_nodes].values
        if mesh_features is not None:
            self.mesh_feattures = self.mesh_features.loc[self.m_nodes].values
        
        
    def __call__(self,  gene_features: pd.DataFrame = None, 
                        mesh_features: pd.DataFrame = None) -> HeteroData: 
        """
        build HeteroData Object
        """
        if self._data is not None:
            return self.data
        # build graph
        self.set_edge_index()

        if isinstance(gene_features, pd.DataFrame):
            self.gene_features = gene_features.loc[self.g_nodes].values
        if isinstance(mesh_features, pd.DataFrame):
            self.mesh_feattures = mesh_features.loc[self.m_nodes].values
        # set feature vectors  
        self.set_node_attr('gene', self.gene_features)
        self.set_node_attr('mesh', self.mesh_features)
        # add self-loop edges. 
        # each node might receive one or more (one per appropriate edge type) messages 
        # from itself during message passing
        # self.data = T.AddSelfLoops()(self.data)
        #  works like in the homogenous case, and normalizes all specified features (of all types) to sum up to one.
        # self.data = T.ToUndirected(merge=False)(self.data)
        # self.data = T.NormalizeFeatures()(self.data)
        self.data['edgetype2idx'] = self.edgetype2index
        self.data['nid2gene'] = self.idx2gene
        self.data['nid2mesh'] = self.idx2mesh
        self.H = None
        self._data = 1
        
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
        self.node2idx = {**self.gene2idx, **self.mesh2idx} # combined two gene and mesh nodes
        self.idx2gene = {i:str(v) for i,v in enumerate(self.g_nodes)}
        self.idx2mesh = {i:str(v) for i,v in enumerate(self.m_nodes)}        
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
        # to numberic values
        self.edge_types = sorted(self.triplets.relation.unique())
        self.edgetype2index = {t:i for i, t in enumerate(self.edge_types)}
        self.triplets['head_idx'] = self.triplets['head'].map(self.node2idx).astype(int)
        self.triplets['tail_idx'] = self.triplets['tail'].map(self.node2idx).astype(int)
        self.triplets['edge_type'] = self.triplets['relation'].map(self.edgetype2index).astype(int)


    def set_node_attr(self, node_type='gene', node_features=None):
        """
        node_type: gene, or mesh
        """
        nodes = []
        node_cat = []
        for node_idx, (node, node_dict) in enumerate(self.H.nodes(data=True)):
            if node_dict['node_type'] == node_type:
                nodes.append((node, self.node2idx[str(node)], node_dict['node_type'], node_dict['node_cat']))
            
        nodes = pd.DataFrame(nodes, columns=['node','nodeidx','metanode','node_labels'])
        # sorted by nodeidx is very important step (nodeidx is int)
        # This makesure your node attrbiute order is correct
        nodes = nodes.sort_values('nodeidx') 
        self.data[node_type].num_nodes = len(nodes)
        #self.data[node_type].node_metatype = nodes.metanode.to_list()
        #self.data[node_type].node_labels = nodes.node_labels.to_list()

        # if input feature data is None, then generate one-hot encoding
        if node_features is None:
            self.data[node_type].x = torch.nn.functional.one_hot(torch.arange(0, len(nodes))).type(torch.float)
        elif isinstance(node_features, pd.DataFrame):
            self.data[node_type].x = torch.from_numpy(node_features.loc[self.m_nodes].values.astype(np.float32))
        elif isinstance(node_features, np.ndarray):
            self.data[node_type].x = torch.from_numpy(node_features.astype(np.float32))
        else:
            self.data[node_type].x = torch.tensor(node_features, dtype=torch.float)
            

    def set_edge_index(self, edge_features=None):
        """
        set edge index, convert to PyG compatible convert
        """
        
        edge_type_tmp = torch.from_numpy(self.triplets.edge_type.values)
        # if not edge attr, generate one-hot encoding by edge_types
        edge_attr_full = torch.nn.functional.one_hot(edge_type_tmp, num_classes=len(self.edge_types)).type(torch.float)
        
        for edge_type in self.edge_types:
            mask = self.triplets.relation == edge_type
            triplets = self.triplets[mask]
            edge_index = torch.tensor([triplets['head_idx'].astype(int).to_list(),  # so, edge_index -> [gene, mesh]
                                       triplets['tail_idx'].astype(int).to_list()])
            
            weight = torch.from_numpy(triplets['weight'].values.reshape(-1,1).astype(np.float32))
            edge_attr = edge_attr_full[torch.from_numpy(mask.values)]
            edge_attr = torch.cat([weight, edge_attr], dim=1) # add weight to edge_attr
            edge_attr = torch.cat([edge_attr, edge_attr]) # FIXME: how to add self-loop edge_attr for hetero ?
            
            if edge_type == 'genemesh':
                self.data['gene', edge_type, 'mesh'].edge_index = edge_index
                self.data['gene', edge_type, 'mesh'].edge_attr = weight
                # rev edgetype: we sepearte the the revseve edge_type for our GNN design.
                edge_type = "rev_" + edge_type
                self.data['mesh', edge_type, 'gene'].edge_index = edge_index.flip([0])
                self.data['mesh', edge_type, 'gene'].edge_attr = weight

            elif edge_type == 'ppi':
                # to PyG compatible undirecte graph
                edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
                # remove self_loops first if there's any
                edge_index, _ = remove_self_loops(edge_index)
                # add self_loops back 
                edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=len(self.g_nodes))
                self.data['gene', edge_type, 'gene'].edge_index = edge_index
                #self.data['gene', edge_type, 'gene'].edge_attr = edge_attr
            else:
                # to PyG compatible undirecte graph
                edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=-1)
                # 
                edge_index, _ = remove_self_loops(edge_index)
                # FIXME: add self_loops for multi-relational graph as MESH ?
                # edge_index, edge_weight  = add_remaining_self_loops(edge_index, num_nodes=len(self.m_nodes))
                # FIXME: remove isolated nodes ?
                # edge_index, _ , _ = remove_isolated_nodes(edge_index, num_nodes=len(self.m_nodes))
                self.data['mesh', edge_type, 'mesh'].edge_index = edge_index
                #self.data['mesh', edge_type, 'mesh'].edge_attr = edge_attr
        # self.data['triplets'] = torch.from_numpy(self.triplets.loc[:,['head_idx','tail_idx','edge_type']].values)

class GeneMeshMLPDataset(Dataset):
    def __init__(self, edge_list, 
                    idx2gene: Dict[int, str], 
                    idx2mesh: Dict[int, str], 
                    gene_embed: Union[pd.DataFrame, AnyStr] = None, 
                    mesh_embed: Union[pd.DataFrame, AnyStr] = None, 
                    transform: AnyStr= "concat",
                    test: bool = False):
        """
        Args:
            edge_list: Tensor shape (..., 3) it's bipartile graph !!! (gene, mesh, label)
            gene_embed from Unirep output. 1900
            mesh_embed from sentencetransformer output 768
            idx2gene: edge_index to entrez
            idx2mesh: edge_index to Mesh UID
            transform (optional): Optional transform to be applied on a sample.
        """
        self._test = test
        self.transform = transform
        self.edge_list = edge_list.numpy()
        self.idx2gene = idx2gene
        self.idx2mesh = idx2mesh
        # attributes
        if isinstance(gene_embed, str):
            self.gene_embed = pd.read_csv(gene_embed, index_col=0, header=None)
        elif isinstance(gene_embed, pd.DataFrame):
            self.gene_embed = gene_embed
        else:
            self.gene_embed = torch.nn.functional.one_hot(torch.arange(0, len(idx2gene))).numpy()
        if isinstance(mesh_embed, str):
            self.mesh_embed = pd.read_csv(mesh_embed, index_col=0, header=None)
        elif isinstance(mesh_embed, pd.DataFrame):
            self.mesh_embed = mesh_embed
        else:
             self.mesh_embed = torch.nn.functional.one_hot(torch.arange(0, len(idx2mesh))).numpy()

    def __len__(self):
        return len(self.edge_list)

    def _get_embed(self, edges):
        node1, node2 = [], []
        for i, data in enumerate(edges):
            n1, n2 = data[:2]
            n1 = int(n1)
            n2 = int(n2)
            node1.append(self.idx2gene[n1])
            node2.append(self.idx2mesh[n2])
        n1embed = self.gene_embed.loc[node1].values
        n2embed = self.mesh_embed.loc[node2].values
        return n1embed, n2embed

    def _get_onehot(self, edges):
        node1, node2 = [], []
        for i, data in enumerate(edges):
            n1, n2 = data[:2]
            n1 = int(n1)
            n2 = int(n2)
            node1.append(n1)
            node2.append(n2)
        n1embed = self.gene_embed[node1]
        n2embed = self.mesh_embed[node2]
        return n1embed, n2embed  

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, int):
            index = [index]
        edges = self.edge_list[index,:]        
        if isinstance(self.gene_embed, pd.DataFrame):   
            n1embed, n2embed = self._get_embed(edges)
        else:
            n1embed, n2embed = self._get_onehot(edges)

        if self.transform == 'concat':
            embed = np.hstack([n1embed, n2embed])
        elif self.transform == "add":
            embed = n1embed + n2embed
        elif self.transform == 'hadamard':
            embed = n1embed * n2embed
        #elif self.transform == 'distance':
        #    embed = torch.cdist(node1, node2, p=2)
        else:
            raise Exception("transform input error")
        if not self._test:
            # NOTE: y should be 0,1,2,3... not one-hot encoding
            y = edges[:,2] # Float tensor for cross entropy           
            return {'embed': torch.from_numpy(embed.astype(np.float32)).squeeze(), 
                    'target': torch.from_numpy(y.astype(np.float32))}
        return {'embed': torch.from_numpy(embed.astype(np.float32)).squeeze()}