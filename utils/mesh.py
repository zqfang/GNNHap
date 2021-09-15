"""
How to build MESH DAG network. see this paper: 
Zhen-Hao Guo, Zhu-Hong You, De-Shuang Huang, Hai-Cheng Yi, Kai Zheng, Zhan-Heng Chen, Yan-Bin Wang, 
MeSHHeading2vec: a new method for representing MeSH headings as vectors based on graph embedding algorithm, 
Briefings in Bioinformatics, , bbaa037, https://doi.org/10.1093/bib/bbaa037




The MeSH consists of three parts including Main Headings, Qualifiers and Supplementary Concepts. Main Headings as the trunk of MeSH are used to describe the content or theme of the article. Qualifiers is the refinement of MeSH headings, i.e. how to be processed when it is in a specific area. Supplementary Concept is a complementary addition that is mostly related to drugs and chemistry. 

In MeSH tree structure, MeSH headings are organized as a ‘tree’ with 16 top categories in which the higher hierarchy has the broader meaning and the lower hierarchy has the specific meaning

Hence, we construct the MeSH heading relationship network from tree structure through hierarchical tree num rules.



Each MeSH heading can be described by one or more tree nums to reflect its hierarchy in the tree structure and relationships with other MeSH headings. Tree num consists of letters and numbers, the first of which is uppercase letter representing category and the rest are made up of numbers. The first two digits are fixed design following the first capital letter and can be seen the top category except capital letter.


 Each three digits represent a hierarchy in the tree structure. There are some MeSH headings such as Lung Neoplasms (C04.588.894.797.520, C08.381.540, and C08.785.520) that are described by a single type of tree num, while others such as Reflex (E01.370.376.550.650, E01.370.600.550.650, F02.830.702 and G11.561.731) can be represented by different kinds of tree num.

Whenever the last hierarchy of tree num is removed, a new tree num and corresponding MeSH heading can be generated and contacted.


 For the sake of simplicity, we treat the mode of the tree num category of MeSH heading as its label.

"""

import numpy as np
import pandas as pd
import networkx as nx
from pubmed import MeSHXMLParser


mesh = MeSHXMLParser("MeSH/desc2021.xml")
mesh_nodes = mesh.parse()


# mesh_nodes = joblib.load("mesh_nodes.pkl")
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


tree_nums = sorted(tree_nums) # order by literal first
tree_nums = sorted(tree_nums, key=lambda x: len(x)) # then, order by length


# count keys
keys = set()
cats = set()
for num in tree_nums:
    ndot = num.count(".")
    cat = num[0]
    cats.add(cat)
    keys.add(f"{cat}-{ndot}")

tree_nums_dict = { key: [] for key in keys}
cats_depth ={ cat: 0 for cat in cats}
for num in tree_nums:
    ndot = num.count(".")
    cat = num[0]
    cats_depth[cat] = max(cats_depth[cat], ndot)
    k = f"{cat}-{ndot}"
    tree_nums_dict[k].append(num)

# build graph 
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

# convert to mesh id
mesh_triplets = []
for s, r, t in triplets:
    s1 = treenum2mesh[s]
    t1 = treenum2mesh[t]
    mesh_triplets.append((s1, r, t1))

mesh_graph = pd.DataFrame(mesh_triplets, columns=['source','relation','target'])
mesh_graph.to_csv("mesh_edges_20210901.csv", index=False)


# mesh categories/labels
cat2full = {'D':'Chemicals and Drugs','C':'Disease','B':'Organisms','E':'Analytical',
            'A': 'Anatomy','G': 'Phenomena and Processes', 'F':'Phychiatry and Psychology','N':'Health Care',
           'I': 'Anthropology','Z': 'Geographicals', 'H':'Disciplines and Occupations',
            'L': "Information Sciences",'M':'Named Groups','J':'Technology','V':'Publications Charateristics','K':'Humanities'}

for mid, mnode in mesh_nodes.items():
    mnode['lables'] = set()
    for t in mnode['TreeNums']:
        mnode['lables'].add(t[0])

num_cats = {k: 0 for k in cats}
for mid, mnode in mesh_nodes.items():
    mnode['lables'] = set()
    for t in mnode['TreeNums']:
        mnode['lables'].add(t[0])
        num_cats[t[0]] += 1

fig1, ax1 = plt.subplots(figsize=(5,4))
pie = pd.Series(num_cats)
labels = [l +": "+ cat2full[l]for l in pie.index.to_list()]
ax1.pie(pie.values, labels=pie.index.values, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.legend(labels, bbox_to_anchor=(0.9, 1))
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


## generate vector representation of nodes of a graph
from node2vec import Node2Vec
# Generate walks
n2v = Node2Vec(H, dimensions=64, #walk_length=20, p=0.25,q=4,
               num_walks=50, workers=16, weight_key='weight', seed=88)
# train node2vec model
n2w_model = n2v.fit(window=7, min_count=1, iter=3)


# Save embeddings for later use
n2w_model.wv.save_word2vec_format("mesh.node2vec.embed")

embeddings = pd.read_table("mesh.node2vec.embed", skiprows=1, index_col=0, sep=" ", header=None)

from sklearn.manifold import TSNE
tsne_embed = TSNE(n_components=2).fit_transform(embeddings)

embeddings['cato'] = [ list(mesh_nodes[m]['lables'])[0] for m in embeddings.index.to_list()]

fig, ax = plt.subplots(figsize=(5,5))
for cat in pie.index:
    mask = embeddings['cato'] == cat
    ax.scatter(tsne_embed[mask,0], tsne_embed[mask,1], label=cat )
ax.legend(bbox_to_anchor=(1.1, 0.7, 0.1,0.3))
ax.set_title("Mesh categories")
plt.show()