#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina' # mac")
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import joblib, glob, json


# In[3]:


import networkx as nx


# In[4]:


from pubmed import GeneMeSHGraph


# ## Convert pubmed graph to NetworkX undirected multi-relational graph
# 
# relation type
# - gene-mesh
# - gene-gene (PPI, biogrid)
# - mesh-mesh (similarites)

# In[5]:


graph = joblib.load("human_gene_mesh_dict.pkl") # if mouse data: gene_mesh_final.pkl


# In[6]:


print("Load finish")


# In[7]:


genes_all = []
mesh_all = []
for gid, gnode in graph.gene_nodes.items():
    genes_all.append(gid)
for mid, mnode in graph.mesh_nodes.items():
    mesh_all.append(mid)

genes_all = sorted(genes_all)
mesh_all = sorted(mesh_all)
gene2idx = {v:i for i, v in enumerate(genes_all)}
mesh2idx = {v:i for i, v in enumerate(mesh_all)}
node2idx2 = {'gene2idx':gene2idx, 'mesh2idx':mesh2idx}


# In[8]:


# Save id
node2idx = {v:i for i, v in enumerate(genes_all + mesh_all)}
with open("human_gene_mesh_hetero_nx.nodeidx.json", 'w') as j:
    json.dump(node2idx, j, indent=4)


# In[9]:


print("NO")


# In[219]:


# G = nx.Graph() #  # create an directed graph 

H = nx.MultiGraph()

for gid, gnode in graph.gene_nodes.items():
    if "_gene_merge" in gnode:
        del gnode['_gene_merge']
    if "_protein_merge" in gnode:
        del gnode['_protein_merge']
    H.add_node(gid, node_type='gene', node_label='gene', 
               node_cat=['gene'], 
               node_weight=len(gnode['PMIDs']),
               node_gene_symbol=gnode['gene_symbol'], 
                node_entrez_id=gid)
    
for mid, mnode in graph.mesh_nodes.items():
    H.add_node(mid, node_type='mesh',node_label='mesh', 
               node_name=mnode['DescriptorName'], 
               node_cat= np.unique([t[0] for t in mnode['TreeNums']]).tolist(),
               node_treenums = mnode['TreeNums'],
               node_ui=mnode['DescriptorUI'],
               node_mesh_id = mid,
               node_weight = len(mnode['PMIDs']),
               node_descriptor_class=mnode['DescriptorClass'], )  


# In[220]:


H.number_of_nodes()


# In[221]:


H.number_of_edges()


# In[222]:


# add gene-mesh
for edge in graph.edges:
    s =str(edge['gene_node'])
    t = str(edge['mesh_node'])
    H.add_edge(s, t,  edge_type='genemesh', edge_weight=edge['weight'])


# In[223]:


H.size()


# In[224]:


nx.write_gpickle(H, path="human_gene_mesh_hetero_nx.tmp.gpkl")


# In[ ]:


# H = nx.read_gpickle("gene_mesh_hetero_nx.tmp.gpkl")


# In[225]:


biogrid = pd.read_table("BIOGRID-ALL-4.3.195.tab2.txt")
biogrid.head()


# In[226]:


# Mus musculus (taxid:10090)
# Homo sapiens (taxid:9606)
taxid = 9606 # 10090
biogrid = biogrid[(biogrid['Organism Interactor A'] == taxid) & (biogrid['Organism Interactor B'] == taxid)]
biogrid = biogrid[['#BioGRID Interaction ID','Entrez Gene Interactor A','Entrez Gene Interactor B', 'Official Symbol Interactor A','Official Symbol Interactor B',
                  'Experimental System','Experimental System Type']]


# In[227]:


coding = sorted([int(g) for g in genes_all])
biogrid_ms = biogrid[(biogrid['Entrez Gene Interactor A'].isin(coding)) & 
                 (biogrid['Entrez Gene Interactor B'].isin(coding)) & ( biogrid['Experimental System Type'] == 'physical')]


# In[228]:


biogrid_ms.head()


# In[229]:


biogrid_ms.shape


# In[230]:


biogrid_ms['Experimental System Type'].unique()


# ## Add PPI network

# In[231]:


ppi = biogrid_ms.loc[:,['Entrez Gene Interactor A', 'Entrez Gene Interactor B', 'Experimental System Type']].astype(str)
ppi = ppi.drop_duplicates()


# In[232]:


ppi.head()


# In[233]:


ppi.shape


# In[234]:


ppi_dict = {}
for i, row in ppi.iterrows():
    s = row.loc['Entrez Gene Interactor A']
    t = row.loc['Entrez Gene Interactor B']
    key = (s,t)
    key_r = (t,s)
    if (key in ppi_dict) or (key_r in ppi_dict):
        continue
    ppi_dict[(key)] = 1


# In[235]:


print(len(ppi_dict))


# In[236]:


PPI = nx.from_pandas_edgelist(ppi, source='Entrez Gene Interactor A', target='Entrez Gene Interactor B', create_using=nx.DiGraph)


# In[237]:


PPI.number_of_edges()


# In[238]:


PPI.is_directed()


# In[239]:


PPI2 = PPI.to_undirected(reciprocal=False)


# In[240]:


PPI2.number_of_edges()


# In[241]:


ppi2 = nx.to_pandas_edgelist(PPI2)


# In[242]:


ppi2.to_csv("human_gene_ppi_undirected_edges.csv", index=False)


# In[243]:


ppi2.shape


# In[ ]:





# In[244]:


# add PPI directed 
for edge_idx, (head, tail, edge_dict) in enumerate(PPI2.edges(data=True)):
    head = str(head)
    tail = str(tail)
    H.add_edge(head, tail,  edge_type='ppi', edge_experiment_type=row['Experimental System Type'])




# ## Add Mesh Network

# In[247]:


mesh_nodes = graph.mesh_nodes


# In[248]:


len(mesh_nodes)


# Mesh Network

# In[249]:


mesh_nodes['D000018']['TreeNums'] # 'TreeNums': ['A13.869.106']


# In[250]:


# depth of MESH DAG
m = 0
tree_nums = []
treenum2mesh = {}
for mid, mnode in mesh_nodes.items():
    for treenum in mnode['TreeNums']:
        n = treenum.count(".")
        treenum2mesh[treenum] = mid
        tree_nums.append(treenum)
        m = max(m, n)


# In[251]:


tree_nums = sorted(tree_nums) # order by literal first
tree_nums = sorted(tree_nums, key=lambda x: len(x)) # then, order by length


# ## Build Mesh Graph (multi-relational, DAG)

# In[252]:


treenum2mesh['D01.029.260.110.500']


# In[253]:


# count keys
keys = set()
cats = set()
for num in tree_nums:
    ndot = num.count(".")
    cat = num[0]
    cats.add(cat)
    keys.add(f"{cat}-{ndot}")


# In[254]:


tree_nums_dict = { key: [] for key in keys}
cats_depth ={ cat: 0 for cat in cats}
for num in tree_nums:
    ndot = num.count(".")
    cat = num[0]
    cats_depth[cat] = max(cats_depth[cat], ndot)
    k = f"{cat}-{ndot}"
    tree_nums_dict[k].append(num)


# In[255]:


print(tree_nums_dict['C-0'])


# In[256]:


print(tree_nums_dict['C-9'])


# In[257]:


print(cats_depth)


# In[258]:


# build graph: upper term --> lower term
triplets = []
for cat, depth in cats_depth.items():
    while depth -1 >= 0: # include top level cat, e.g. A0,C1
        k_t = f"{cat}-{depth}"
        k_s = f"{cat}-{depth-1}"
        tree_t = tree_nums_dict[k_t]
        tree_s = tree_nums_dict[k_s]
        for s in tree_s:
            for t in tree_t:
                # if t.find(s) != -1: # hits
                if t[:-4] == s:
                    triplets.append((s, cat, t))
        depth -= 1                           


# In[259]:


triplets[:5]


# In[260]:


len(triplets)


# In[261]:


# convert to mesh id
mesh_triplets = []
for s, r, t in triplets:
    s1 = treenum2mesh[s]
    t1 = treenum2mesh[t]
    mesh_triplets.append((s1, r, t1))


# In[262]:


mesh_triplets[:5]


# In[263]:


mesh_graph = pd.DataFrame(mesh_triplets, columns=['source','relation','target'])


# In[264]:


mesh_graph.head()


# In[265]:


mesh_graph.shape


# In[266]:


mesh_graph.to_csv("mesh_edges_directed_20210901.csv", index=False)


# In[267]:


len(cats)


# In[268]:


meshnet = nx.from_pandas_edgelist(mesh_graph, source='source', target='target', edge_key='relation', create_using=nx.MultiGraph)


# In[269]:


meshnet.number_of_edges()


# In[270]:


meshnet.to_undirected().number_of_edges()


# In[271]:


# H = nx.MultiDiGraph()
for s, r, t in mesh_triplets:
    H.add_edge(s, t, edge_type=r)
    #H.add_edge(t, s, edge_type=r)


# In[272]:


H.number_of_edges()


# In[273]:


H.number_of_nodes()


# In[274]:


H.size()


# In[ ]:





# In[275]:


H.is_directed()


# In[276]:


H.is_multigraph()


# In[277]:


nx.write_gpickle(H, path="human_gene_mesh_hetero_nx.gpkl")





# In[157]:


fig, axs = plt.subplots(1,2, figsize=(12,4))
g_degree = [ len(list(H.neighbors(gene))) for gene in mesh_graph.source.values]
axs[0].loglog(sorted(g_degree, reverse=True), "b-", marker="o", label='Mesh')
axs[0].set_title("Degree rank plot")
axs[0].set_ylabel("degree")
axs[0].set_xlabel("rank")
axs[0].legend()

h = pd.Series(g_degree).plot.hist(bins=50, range=(1,100), ax=axs[1], color='b')
axs[1].set_ylabel("number")
axs[1].set_xlabel("degree")
axs[1].set_title("The distritbution of node degree")
plt.show()


# In[159]:


for mid, mnode in mesh_nodes.items():
    mnode['lables'] = set()
    for t in mnode['TreeNums']:
        mnode['lables'].add(t[0])


# In[164]:


num_cats = {k: 0 for k in cats}
for mid, mnode in mesh_nodes.items():
    mnode['lables'] = set()
    for t in mnode['TreeNums']:
        mnode['lables'].add(t[0])
        num_cats[t[0]] += 1


# In[169]:


cat2full = {'D':'Chemicals and Drugs','C':'Disease','B':'Organisms','E':'Analytical',
            'A': 'Anatomy','G': 'Phenomena and Processes', 'F':'Phychiatry and Psychology','N':'Health Care',
           'I': 'Anthropology','Z': 'Geographicals', 'H':'Disciplines and Occupations',
            'L': "Information Sciences",'M':'Named Groups','J':'Technology','V':'Publications Charateristics','K':'Humanities'}


# In[172]:


fig1, ax1 = plt.subplots(figsize=(5,4))
pie = pd.Series(num_cats)
labels = [l +": "+ cat2full[l]for l in pie.index.to_list()]
ax1.pie(pie.values, labels=pie.index.values, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.legend(labels, bbox_to_anchor=(0.9, 1))
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[207]:


## generate vector representation of nodes of a graph
from node2vec import Node2Vec
# Generate walks
n2v = Node2Vec(H, dimensions=64, #walk_length=20, p=0.25,q=4,
               num_walks=50, workers=16, weight_key='weight', seed=88)
# train node2vec model
n2w_model = n2v.fit(window=7, min_count=1, iter=3)


# In[208]:


# Save embeddings for later use
n2w_model.wv.save_word2vec_format("mesh.node2vec.embed")


