
import torch
import numpy as np
from argparse import ArgumentParser 
from sklearn.metrics import average_precision_score, roc_auc_score
from torch._C import default_generator



def apk(y_true, y_score, k):
    """
    Average precision at top-k.
    @param y_true: list of true labels.
    @param y_score: list of target scores.
    @param k: threshold
    @return: AP@k
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_score, np.ndarray):
        y_score = np.array(y_score)
    ind = np.argsort(y_score)[::-1][:k]
    score, num_hits = 0.0, 0.0
    for i, idx in enumerate(ind):
        if y_true[idx] == 1:
            num_hits += 1
            score += num_hits / (i + 1.0)
    return score / k


def rwr(A, restart_prob):
    """
    Random Walk with Restart (RWR) is used to extract features from
    similarity graph for protein nodes or HPO term nodes.


    Random Walk with Restart (RWR) on similarity network.
    :param A: n x n, similarity matrix
    :param restart_prob: probability of restart
    :return: n x n, steady-state probability
    """
    n = A.shape[0]
    A = (A + A.T) / 2
    A = A - np.diag(np.diag(A))
    A = A + np.diag(sum(A) == 0)
    P = A / sum(A)
    Q = np.linalg.inv(np.eye(n) - (1 - restart_prob) * P) @ (restart_prob * np.eye(n))
    return Q

# # calculate metrics
# aupr = average_precision_score(y_true, y_score)
# auroc = roc_auc_score(y_true, y_score)
# ap5000 = apk(y_true, y_score, k=5000)
# ap10000 = apk(y_true, y_score, k=10000)
# ap20000 = apk(y_true, y_score, k=20000)
# ap50000 = apk(y_true, y_score, k=50000)

# @torch.no_grad()
# def test():
#     model.eval()
#     z = model.encode(data.edge_index, data.edge_type)

#     valid_mrr = compute_mrr(z, data.valid_edge_index, data.valid_edge_type)
#     test_mrr = compute_mrr(z, data.test_edge_index, data.test_edge_type)

#     return valid_mrr, test_mrr


# @torch.no_grad()
# def compute_mrr(z, edge_index, edge_type):
#     """
#     z = s * r* t
#     """
#     ranks = []
#     for i in tqdm(range(edge_type.numel())):
#         (src, dst), rel = edge_index[:, i], edge_type[i]

#         # Try all nodes as tails, but delete true triplets:
#         tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
#         for (heads, tails), types in [
#             (data.train_edge_index, data.train_edge_type),
#             (data.valid_edge_index, data.valid_edge_type),
#             (data.test_edge_index, data.test_edge_type),
#         ]:
#             tail_mask[tails[(heads == src) & (types == rel)]] = False

#         tail = torch.arange(data.num_nodes)[tail_mask]
#         tail = torch.cat([torch.tensor([dst]), tail])
#         head = torch.full_like(tail, fill_value=src)
#         eval_edge_index = torch.stack([head, tail], dim=0)
#         eval_edge_type = torch.full_like(tail, fill_value=rel)

#         out = model.decode(z, eval_edge_index, eval_edge_type)
#         perm = out.argsort(descending=True)
#         rank = int((perm == 0).nonzero(as_tuple=False).view(-1)[0])
#         ranks.append(rank + 1)

#         # Try all nodes as heads, but delete true triplets:
#         head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
#         for (heads, tails), types in [
#             (data.train_edge_index, data.train_edge_type),
#             (data.valid_edge_index, data.valid_edge_type),
#             (data.test_edge_index, data.test_edge_type),
#         ]:
#             head_mask[heads[(tails == dst) & (types == rel)]] = False

#         head = torch.arange(data.num_nodes)[head_mask]
#         head = torch.cat([torch.tensor([src]), head])
#         tail = torch.full_like(head, fill_value=dst)
#         eval_edge_index = torch.stack([head, tail], dim=0)
#         eval_edge_type = torch.full_like(head, fill_value=rel)

#         out = model.decode(z, eval_edge_index, eval_edge_type)
#         perm = out.argsort(descending=True)
#         rank = int((perm == 0).nonzero(as_tuple=False).view(-1)[0])
#         ranks.append(rank + 1)

#     return (1. / torch.tensor(ranks, dtype=torch.float)).mean()




def add_train_args():
    parser = ArgumentParser()
    parser.add_argument("--gene_mesh_graph", type=str, default=None, help="genemesh.data.pkl object or genemesh.gpkl object")
    parser.add_argument("--gene_embed", default=None, type=str, help="gene_embedding file")
    parser.add_argument("--mesh_embed", default=None, type=str, help="mesh_embedding file")
    parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of GNN. Default 64")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate. Default 0.01")
    parser.add_argument("--num_epochs", default=5, type=int, help="num_epochs for training. Default 5.")
    parser.add_argument("--batch_size", default=10000, type=int, help="batch size (num_edges) for mini-batch training and testing. Default 10000")
    parser.add_argument("--num_cpus", default=6, type=int, help="Number of cpus for dataloader, Default 6.")
    parser.add_argument("--outdir", type=str, required=False, default="checkpoints", help="Output file directory") 
    parser.add_argument("--bundle", default=None, type=str, help="Path to the directory name of the required files (for predition task)")
    parser.add_argument("--hbcgm_result_dir", default=None, type=str, help="HBCGM output directory. For prediction task")
    parser.add_argument("--mesh_terms", required=False, default=None, help="MeSH term IDs. Comma sepearted if multi-mesh inputs.")
    args = parser.parse_args()
    return args