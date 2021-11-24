'''
see subsampling big graph link prediction task, so we could fit in GPUs
https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/citation2/sampler.py#L17-L41

use NeigborLoader when graph could not fit in your GPU
'''

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import NeighborSampler, NeighorLoader


class PositiveLinkNeighborSampler(NeighborSampler):
    def __init__(self, edge_index, sizes, num_nodes=None, **kwargs):
        edge_idx = torch.arange(edge_index.size(1))
        super(PositiveLinkNeighborSampler,
              self).__init__(edge_index, sizes, edge_idx, num_nodes, **kwargs)

    def sample(self, edge_idx):
        if not isinstance(edge_idx, torch.Tensor):
            edge_idx = torch.tensor(edge_idx)
        row, col, _ = self.adj_t.coo()
        batch = torch.cat([row[edge_idx], col[edge_idx]], dim=0)
        return super(PositiveLinkNeighborSampler, self).sample(batch)


class NegativeLinkNeighborSampler(NeighborSampler):
    def __init__(self, edge_index, sizes, num_nodes=None, **kwargs):
        edge_idx = torch.arange(edge_index.size(1))
        super(NegativeLinkNeighborSampler,
              self).__init__(edge_index, sizes, edge_idx, num_nodes, **kwargs)

    def sample(self, edge_idx):
        num_nodes = self.adj_t.sparse_size(0)
        batch = torch.randint(0, num_nodes, (2 * len(edge_idx), ),
                              dtype=torch.long)
        return super(NegativeLinkNeighborSampler, self).sample(batch)



