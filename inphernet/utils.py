
import numpy as np
from argparse import ArgumentParser 
from sklearn.metrics import average_precision_score, roc_auc_score

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


def add_train_args():
    parser = ArgumentParser()
    parser.add_argument("--gene_mesh_nx_graph", type=str, required=True)
    parser.add_argument("--gene_embed", default=None, type=str, help="gene_embedding file")
    parser.add_argument("--mesh_embed", default=None, type=str, help="mesh_embedding file")
    parser.add_argument("--outdir", type=str, default="checkpoints")
    args = parser.parse_args()
    return args