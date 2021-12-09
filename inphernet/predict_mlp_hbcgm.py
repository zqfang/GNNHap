'''
The LinkPredictor (MLP) for GeneMesh HeteroGraph training
'''

import os, sys, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

from data import GeneMeshMLPDataset
from models import MLP
from utils import add_train_args

pd.options.mode.chained_assignment = None
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# argument parser
args = add_train_args()

## Additional Inputs
GENE_NODES = "human_gene_nodes.pkl"
MESH_NODES = "mesh_nodes.pkl"
GENE_MESH_DATA = os.path.join(args.outdir,"genemesh.data.pkl")
##
HBCGM_OUTPUT = ""
HBCGM_MESH_TERM_ID = ""


os.makedirs(args.outdir, exist_ok=True)
# Parameters
input_size = 1900 + 768
#input_size = args.input_size 
hidden_size = args.hidden_size # 1024
num_workers = args.num_cpus # 6

# Load mesh embeddings
print("Load embeddings")
mesh_features = pd.read_csv(args.mesh_embed, index_col=0, header=None)
gene_features = pd.read_csv(args.gene_embed, index_col=0, header=None)
gene_features.index = gene_features.index.astype(str)


print("Build and Load Model")
model = MLP(input_size, hidden_size)

ckpts = os.path.join(args.outdir, "mlp_best_model.pt")
checkpoint = torch.load(ckpts, map_location=device)
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
print('Previously trained model weights state_dict loaded...')
print(model)

#### Format HBCGM output
print("Prepare HBCGM results")
human_gene_nodes = joblib.load(MESH_NODES)
mesh_nodes = joblib.load(GENE_NODES)
gm_data = joblib.load(GENE_MESH_DATA)

## Start
symbol2entrez = {}
for entrez, value in human_gene_nodes.items():
    symbol2entrez[value['gene_symbol']] = entrez


node2index = {'gene2nid': {v: key for key, v in  gm_data['nid2gene'].items()}, 
              'mesh2nid': {v:key for key, v in gm_data['nid2mesh'].items()}}

# symbol --> entrez --> node idx
def HBCGM_score(hbcgm_output, symbol2entrez, mesh_term_id):
    case = pd.read_table(hbcgm_output, skiprows=3, header=None)
    case.columns = ['GeneName','CodonFlag','Pattern','Pval',
                    'EffectSize', 'chr','blockStart','blockEnd','Expression']
    case['HumanEntrezID'] = case.GeneName.str.upper().map(symbol2entrez)
    case['NodeIDX'] = case['HumanEntrezID'].map(node2index['gene2nid'])
    case2 = case.dropna(subset=['HumanEntrezID','NodeIDX']) # drop any gene names could not mapped
    case2['NodeIDX'] = case2['NodeIDX'].astype(int)
    case2['logPval'] = - np.log2(case2.Pval)
    # build edge_label_index
    edge_index = torch.tensor([case2.NodeIDX.to_list(), [node2index['mesh2nid'][mesh_term_id]]*case2.NodeIDX.shape[0]])
    return case2, edge_index.T

@torch.no_grad()
def test(test_loader):
    # # Test the Model
    model.eval()
    y_preds = []
    for embeds in test_loader:
        inputs = embeds['embed']
        inputs = inputs.to(device)
        #targets = targets.view(-1).to(device)
        #targets = targets.view(-1).type(torch.int)
        outputs = model(inputs)
        #valid_loss += criterion(outputs, targets).item() # 
        y_pred = torch.sigmoid(outputs.data).detach().cpu().numpy()
        #targets = targets.detach().cpu().numpy()
        y_preds.append(y_pred)
    y_preds = np.concatenate(y_preds)
    return y_preds

### Start your computation

case_tmp, edge_index = HBCGM_score(hbcgm_output=HBCGM_OUTPUT,
                                   symbol2entrez=symbol2entrez,
                                   mesh_term_id=HBCGM_MESH_TERM_ID)
case_test = GeneMeshMLPDataset(edge_index, gene_features, mesh_features, gm_data['nid2gene'], gm_data['nid2mesh'], test=True)
case_loader = torch.utils.data.DataLoader(case_test, batch_size=10000,)
case_tmp['LiteratureScore'] = test(case_loader)



### Plotting
p = 1e-4
m = HBCGM_MESH_TERM_ID
tmp = case_tmp[case_tmp.Pval < p]

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(projection='3d')
majors = [0, 1, 2]
labels = ["Synonymous", "Missense", "Splicing"]
cmaps = ['Reds','Greens', 'Blues']
for i, c in enumerate(['red','green','blue']):
    flag = CodonFlag[i]
    tmp2 = tmp[tmp.CodonFlag == i]
    #ax.axhline(y=0.5, linestyle='--', color='lightgray')
    ax.scatter(xs=tmp2['logPval'].values, zs=tmp2['KGScore'].values, ys=i, c=tmp2['KGScore'], cmap=cmaps[i])
    for _, row in tmp2.iterrows():
        if row.KGScore > 0.5:
            ax.text(x= row.logPval,z=row.KGScore, y=i, s=row.GeneName, ha='left')
ax.set_zlabel(f"Literature Score ", fontweight='bold')
ax.set_xlabel("$Genetic: - \log _{10} Pval$", fontweight='bold')
ax.yaxis.set_major_locator(ticker.FixedLocator(majors, ))
ax.set_ylim([0,2])
ax.set_yticklabels(labels) # rotation='vertical')
ax.set_title(f"MeSH: {m} {mesh_nodes[m]['DescriptorName']}", fontweight='bold', loc='left')
## camera position
ax.azim = -60 # azimuth is the rotation around the z axis, 0 means "looking from +x", 90, y
ax.dist = 10 # the distance from the center visible point in data coordinates
ax.elev = 12 # elev is the angle between the eye and the xy plane
plt.show()




 
