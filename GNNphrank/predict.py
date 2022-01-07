
import os, sys, glob, json, joblib
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from functools import partial
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter
import matplotlib.pyplot as plt



import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm
from data import GeneMeshData
from models import HeteroGNN
from utils import add_train_args
from typing import List, Dict, Tuple, Union, AnyStr

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# argument parser
args = add_train_args()
hidden_size = args.hidden_size
batch_size = args.batch_size # 10000

BUNDLE_PATH = args.bundle
HBCGM_RESULTS = args.hbcgm_result_dir
MESH_TERMS = args.mesh_terms

os.makedirs(args.outdir, exist_ok=True)
model_weight = os.path.join(BUNDLE_PATH, "gnn_64_epoch0.pt")
genemesh_data = os.path.join(BUNDLE_PATH, "genemesh.data.pkl") # genemesh.data.pkl
mouse_human_namedict = os.path.join(BUNDLE_PATH, "mouse2human.genenames.json")
mus_tissue_exp = os.path.join(BUNDLE_PATH, "mus.compact.exprs.organs.order.txt")
if not os.path.exists(model_weight):
    raise LookupError("model weights not found")


## Load Graph nodes' metadata
mesh_nodes = joblib.load(os.path.join(BUNDLE_PATH,"mesh_nodes.pkl"))
human_gene_nodes = joblib.load(os.path.join(BUNDLE_PATH,"human_gene_nodes.pkl"))
with open(os.path.join(BUNDLE_PATH,"human_gene_mesh.pmids.json"), 'r') as j:
    pubmed_json = json.load(j)
# # read data in
print("Read graph data")

if os.path.exists(genemesh_data) and genemesh_data.endswith("data.pkl"):
    gm_data = joblib.load(genemesh_data)
elif (args.gene_mesh_graph is not None) and args.gene_mesh_graph.endswith("gpkl"):
    # Load mesh embeddings
    print("Load embeddings")
    mesh_features = pd.read_csv(args.mesh_embed, index_col=0, header=None)
    gene_features = pd.read_csv(args.gene_embed, index_col=0, header=None)
    gene_features.index = gene_features.index.astype(str)
    H = nx.read_gpickle(args.gene_mesh_graph) # note: H is undirected multigraph
    print("Build GraphData")
    # build data
    gm = GeneMeshData(H, gene_features, mesh_features)
    gm_data = gm()

else:
    raise ValueError("Need Graph Object input for prediction task !")
    sys.exit(0)

# init model
model = HeteroGNN(heterodata=gm_data, hidden_channels=args.hidden_size, num_layers=2)
model.to('cpu')

@torch.no_grad()
def predict(model, data: HeteroData, node_embed=None, device='cpu'):
    model.eval()
    data.to(device)
    ####
    y_preds = []
    if node_embed is None:
        h_dict = model(data.x_dict, data.edge_index_dict) # only need compute once for node_representations
    else:
        h_dict = node_embed
    
    edge_label_index = data['gene','genemesh','mesh'].edge_label_index
    data_loader = DataLoader(torch.arange(edge_label_index.size(1)), batch_size=batch_size)
    for i, perm in enumerate(tqdm(data_loader, total=len(data_loader), desc='Predict', position=1, leave=True)):
        g = h_dict['gene'][edge_label_index[0, perm]] 
        m = h_dict['mesh'][edge_label_index[1, perm]]
        # NOTE: concat here
        inp = torch.cat([g, m], dim=1).to(device)
        preds = model.link_predictor(inp).view(-1)
        y_preds.append(preds.cpu())
    y_preds = torch.cat(y_preds, dim=0) # note the difference with torch.stack()
    y_preds = torch.sigmoid(y_preds).numpy()
    return y_preds, h_dict

## load model weights
ckpt = torch.load(model_weight, map_location=device)
model.load_state_dict(ckpt['state_dict'])


class HBCGM:
    def __init__(self, inputdir: AnyStr, 
                  mesh_term_ids: Union[List,AnyStr], 
                  model: torch.nn.Module,
                  gm_data: HeteroData,
                  human_gene_nodes: Union[AnyStr, Dict], 
                  mouse_human_namedict: Union[AnyStr, Dict],
                  mesh_nodes:Union[AnyStr, Dict],
                  pubmed_ids: Union[AnyStr, Dict]
                  ):

        self.inputs = glob.glob(os.path.join(inputdir, "chr*results.txt"))
        self.mesh_terms = mesh_term_ids
        if isinstance(mesh_term_ids, str): self.mesh_terms = mesh_term_ids.split(",")
        self.node2index = { 'gene2nid': {v: key for key, v in  gm_data['nid2gene'].items()}, 
                            'mesh2nid': {v:key for key, v in gm_data['nid2mesh'].items()}}
        self.gm_data = gm_data
        if not isinstance(human_gene_nodes, dict):
            human_gene_nodes = joblib.load(human_gene_nodes)
        self.symbol2entrez = {}
        for entrez, value in human_gene_nodes.items():
            self.symbol2entrez[value['gene_symbol']] = entrez

        # load 
        self.mouse2human = mouse_human_namedict
        if not isinstance(mouse_human_namedict, dict):
            with open(mouse_human_namedict, 'r') as j:
                self.mouse2human = json.load(j) ## Mouse gene name to human
        self.mesh_nodes = mesh_nodes
        if not isinstance(mesh_nodes, dict):
            mesh_nodes = joblib.load(mesh_nodes)
        self.PUBMEDID = pubmed_ids
        if not isinstance(pubmed_ids, dict):
            self.PUBMEDID = json.load(pubmed_ids)
        self.model = model
        self.headers = ""
        self._node_embed = None 
    
    def get_pubmed_id(self, entrzid, mesh):    
        k = f"{entrzid}__{mesh}"
        if k in self.PUBMEDID:
            return ",".join(self.PUBMEDID[k])
        return "Noval"
        
    def save(self, output:AnyStr):
        line = "##MeSH_Terms"
        for mesh_term in self.mesh_terms:
            if mesh_term not in self.mesh_nodes: continue
            mtd = self.mesh_nodes[mesh_term]['DescriptorName']
            m =  f"\t{mesh_term}:{mtd}"
            c = f"PMIDs_{mesh_term}"
            get_pmid = partial(self.get_pubmed_id, mesh=mesh_term)
            self.result[c] = self.result['HumanEntrezID'].apply(get_pmid)
            line += m
        self.headers[-1] = line+"\n"
        if os.path.exists(output): os.remove(output)
        if not self.result.empty:
            self.result.drop(columns=['NodeIDX','logPval'], inplace=True)
            self.result.sort_values(['Pvalue','Chr'], inplace=True)
        with open(output, 'a') as out:
            for line in self.headers:
                out.write(line)
            ## Table
            self.result.to_csv(out, sep="\t", index=False)

    def map2human(self, result):
        if isinstance(result, pd.DataFrame):
            case = result
        else:
            self.headers = []
            case = pd.read_table(result, skiprows=5, dtype={'Haplotype': str, 'Chr': str})
            with open(result, 'r') as r:
                for i, line in enumerate(r):
                    self.headers.append(line)
                    if i >= 5: break 
        if case.shape[0] < 1: 
            print(f"input data is empty: {result}")
            return case 
        case.loc[case.index, 'HumanEntrezID'] = case['#GeneName'].map(self.mouse2human).map(self.symbol2entrez)
        case.loc[case.index, 'NodeIDX'] = case['HumanEntrezID'].map(self.node2index['gene2nid'])
        df = case.dropna(subset=['HumanEntrezID','NodeIDX'])
        df.loc[df.index, 'NodeIDX'] = df['NodeIDX'].astype(int)
        df.loc[df.index, 'logPval'] = - np.log10(df.Pvalue)   
        return df
    

    def mesh_score(self, df):
        if df.shape[0] < 1: return df
        for mesh_term in self.mesh_terms:
            if mesh_term not in self.node2index['mesh2nid']: continue
            edge_index = torch.tensor([df.NodeIDX.to_list(), 
                                      [self.node2index['mesh2nid'][mesh_term]]*df.NodeIDX.shape[0]])
            self.gm_data['gene', 'genemesh', 'mesh'].edge_label_index = edge_index
            prediction, self._node_embed = predict(self.model, self.gm_data, self._node_embed) # save node embed to reduce run time
            MESH_SCORE = f"MeSH_{mesh_term}"
            df.loc[df.index, MESH_SCORE] = prediction
        return df

    def predict(self, inputdir=None):
        dfs = []
        if inputdir is not None:
            self.inputs = glob.glob(os.path.join(inputdir, "*results.txt"))
        for inp in self.inputs:
            df = self.map2human(inp)
            if not df.empty: dfs.append(df)

        # concat results
        result = pd.concat(dfs)
        result = result.reset_index(drop=True)
        ## predit score
        self.result = self.mesh_score(result)

    def predict_agg(self, df, mesh_terms=None):
        """
        df: aggregated results from HBCGM output
        """
        if isinstance(mesh_terms, str):
            self.mesh_terms = mesh_terms.split(",")
        elif isinstance(mesh_terms, list):
            self.mesh_terms = mesh_terms

        df = self.map2human(df)
        result = df.reset_index(drop=True)
        ## predit score
        self.result = self.mesh_score(result)

    def haplotype2color(self, df=None, top=10, ax=None):

        pal = ['#1f77b4','#ff7f0e', '#2ca02c', '#d62728','#9467bd']
        paldict = {str(i):i for i, c in  enumerate(pal)}
        paldict['?'] = '#ffffff'

        if df is None:
            df = self.result.copy()
        df = df.set_index('#GeneName')
        v = df['Haplotype'].apply(lambda pattern: [paldict[p] for p in list(pattern)])
        vv = v[:top][::-1]
        gene_names = vv.index.values
        vv = vv.to_list()
        x = np.arange(-0.5, len(vv[0]), 1)  # len = numStrains
        y = np.arange(-0.5, len(vv), 1)  # len = num genes

        if ax is None: ax = plt.gca()
        # only show categorical colors
        ax.pcolormesh(x, y, vv, linewidth=2, edgecolor='white', cmap=ListedColormap(pal), vmin=-1, vmax=4)
        # Move left and bottom spines outward by 10 points
        ax.spines.left.set_position(('outward', 10))
        ax.spines.bottom.set_position(('outward', 10))
        # Hide the right and top spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.yaxis.set_ticks_position('none') # ytick off
        ax.get_xaxis().set_visible(False) # xtick and label off
        # set fixed yticklables
        ax.yaxis.set_major_locator(FixedLocator(range(len(vv))))
        ax.yaxis.set_major_formatter(FixedFormatter(gene_names))
        return ax

    def convert2newformat(self, path, expr_path=None ):
        """
        This function is used for convert old HBCGM output to new HBCGM output format
        path: the file path for old HBCGM output
        expr_path: gene_expression_compact format file.
        """
        flag = "##CodonFlag\t0:Synonymous\t1:Non-Synonymous\t2:Splicing\t3:Stop\t-1:Non-Coding\n"
        gene_expr_header = "##GeneExprMap\t"
        header="#GeneName\tCodonFlag\tHaplotype\tPvalue\tEffectSize\tChr\tChrStart\tChrEnd\tGeneExprMap\n"
        headers = []
        with open(path, 'r') as hb:
            for i, line in enumerate(hb):
                headers.append("##"+line)
                if i == 2: break
        d = pd.read_table(path, skiprows=3, header=None, dtype=str)
        if expr_path is not None:
            # expr_path = "/data/bases/shared/haplomap/PELTZ_20210609/mus.compact.exprs.txt"
            gene_expr = pd.read_table(expr_path, comment="#", header=None)
            gene_expr_dict = {row.iloc[0]: row.iloc[1] for i, row in gene_expr.iterrows()}
            with open(expr_path, 'r') as hb:
                gene_expr_header += hb.readline().strip("#")
            d.iloc[:, -1] = d.iloc[:,0].map(gene_expr_dict)
            d.iloc[:, -1] = d.iloc[:, -1].fillna("-------------")
        headers.insert(1, flag,)
        headers.insert(2, gene_expr_header )
        headers.insert(5, header)
        ofile = path.replace("txt", "updated.txt")
        if os.path.exists(ofile): os.remove(ofile)
        with open(ofile, 'a') as out:
            out.writelines(headers)
            d.to_csv(out, index=False, header=False, sep="\t")
    

            
# ### predicts
# if MESH_TERMS is not None:
#     MESH_TERMS = MESH_TERMS.split(",")
# hbcgm = HBCGM(inputdir=HBCGM_RESULTS, 
#               mesh_term_ids=MESH_TERMS,# ['D018919', 'D009389', 'D043924'], # MESH_TERMS
#               model = model,
#               gm_data=gm_data,
#               human_gene_nodes=human_gene_nodes,
#               mouse_human_namedict=mouse_human_namedict,
#               mesh_nodes = mesh_nodes)

# hbcgm.predict()
# OUTFILE = os.path.split(HBCGM_RESULTS)[-1] + ".results.mesh.txt"
# hbcgm.save(os.path.join(Path(HBCGM_RESULTS).parent, OUTFILE))


### run example
# python predict.py --bundle /data/bases/fangzq/Pubmed/bundle \
#                   --mesh_terms D018919,D009389,D043924
#                   --hbcgm_result_dir /data/bases/fangzq/Pubmed/test_cases/TEST/RUN_000


# MESH_DICT = {'10806':'D003337',
#  '10807':'D020712',
#  '10813':'D020712',
#  '26721': 'D017948',
#  '50243':'D002331',
#  '9904': 'D008076',
#  'Haloperidol': "D064420",
#  'irinotecan': 'D000077146',
#  'Cocaine':'D007858',
# }

## START HBCGM predcition
NODE_EMBED = None
MESH_DICT = {}
if MESH_TERMS is not None and MESH_TERMS.endswith("json"):
    with open(MESH_TERMS, 'r') as j:
        MESH_DICT = json.load(j) 
# path = glob.glob(os.path.join(HBCGM_RESULTS, "*results.txt"))
path = Path(HBCGM_RESULTS).glob("*results.txt")

hbcgm = HBCGM(inputdir=HBCGM_RESULTS, 
            mesh_term_ids=[],# ['D018919', 'D009389', 'D043924'], # MESH_TERMS
            model = model,
            gm_data=gm_data,
            human_gene_nodes=human_gene_nodes,
            mouse_human_namedict=mouse_human_namedict,
            mesh_nodes = mesh_nodes,
            pubmed_ids=pubmed_json)
for p in path:
    m = str(p).split("/")[-1].split(".")[0].split("_")[1] # mpd id
    m = m.split("-")[0] # strip suffix
    hbcgm.predict_agg(str(p), mesh_terms=MESH_DICT[m])
    hbcgm.save(str(p).replace("txt","mesh.txt"))