"""
This script output mesh_nodes.pkl

How to build MESH DAG network. see this paper: https://doi.org/10.1093/bib/bbaa037


The MeSH consists of three parts including Main Headings, Qualifiers and Supplementary Concepts. 
- Main Headings as the trunk of MeSH are used to describe the content or theme of the article. 
- Qualifiers is the refinement of MeSH headings, i.e. how to be processed when it is in a specific area. 
- Supplementary Concept is a complementary addition that is mostly related to drugs and chemistry. 

In MeSH tree structure, MeSH headings are organized as a ‘tree’ with 16 top categories 
in which the higher hierarchy has the broader meaning and the lower hierarchy has the specific meaning
Hence, we construct the MeSH heading relationship network from tree structure through hierarchical tree num rules.


Each MeSH heading can be described by one or more tree nums to reflect its hierarchy in the tree structure and relationships with other MeSH headings. 
Tree num consists of letters and numbers:
- The first of which is uppercase letter representing category and the rest are made up of numbers. 
- The first two digits are fixed design following the first capital letter and can be seen the top category except capital letter.
- Each three digits represent a hierarchy in the tree structure. 

There are some MeSH headings such as Lung Neoplasms (C04.588.894.797.520, C08.381.540, and C08.785.520) 
that are described by a single type of tree num, while others such as 
Reflex (E01.370.376.550.650, E01.370.600.550.650, F02.830.702 and G11.561.731) 
can be represented by different kinds of tree num.

Whenever the last hierarchy of tree num is removed, 
a new tree num and corresponding MeSH heading can be generated and contacted.
For the sake of simplicity, we treat the mode of the tree num category of MeSH heading as its label.
However, we don't use node label in our GNN model
"""
import os, sys, joblib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pubmed import MeSHXMLParser

import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class MeshDAG:
    def __init__(self, mesh_nodes):
        self.mesh_nodes = mesh_nodes
        # mesh categories/labels
        self.cat2full = {'D':'Chemicals and Drugs',
                            'C': 'Disease',
                            'B': 'Organisms',
                            'E': 'Analytical',
                            'A': 'Anatomy',
                            'G': 'Phenomena and Processes', 
                            'F': 'Phychiatry and Psychology',
                            'N': 'Health Care',
                            'I': 'Anthropology',
                            'Z': 'Geographicals', 
                            'H': 'Disciplines and Occupations',
                            'L': "Information Sciences",
                            'M': 'Named Groups',
                            'J': 'Technology',
                            'V': 'Publications Charateristics',
                            'K': 'Humanities'
                        }
        
    def __call__(self, outfile=None):
        tree_nums, treenum2mesh = self.collect_treenums()
        tree_nums_dict, cats_depth = self.count_keys(tree_nums)
        triplets = self.build_treenum_graph(tree_nums_dict, cats_depth)
        self.mesh_graph = self.build_mesh_triplets(triplets, treenum2mesh, outfile)

    def collect_treenums(self):
        m = 0
        tree_nums = []
        treenum2mesh = {}
        for mid, mnode in self.mesh_nodes.items():
            for treenum in mnode['TreeNums']:
                n = treenum.count(".")
                treenum2mesh[treenum] = mid
                tree_nums.append(treenum)
                m = max(m, n)


        tree_nums = sorted(tree_nums) # order by literal first
        tree_nums = sorted(tree_nums, key=lambda x: len(x)) # then, order by length
        return tree_nums, treenum2mesh

    def count_keys(self, tree_nums):
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
        return tree_nums_dict, cats_depth

    def build_treenum_graph(self, tree_nums_dict, cats_depth):
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
        return triplets

    def build_mesh_triplets(self, triplets, treenum2mesh, outfile=None):
        # convert to mesh id
        mesh_triplets = []
        for s, r, t in triplets:
            s1 = treenum2mesh[s]
            t1 = treenum2mesh[t]
            mesh_triplets.append((s1, r, t1))

        mesh_graph = pd.DataFrame(mesh_triplets, columns=['source','relation','target'])
        if outfile is not None:
            mesh_graph.to_csv(outfile, index=False)
        return mesh_graph



class SequenceEncoder(object):
    def __init__(self, model_name='gsarti/scibert-nli', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        if isinstance(df, pd.Series):
            inp = df.values
        else:
            inp = df
        x = self.model.encode(inp, show_progress_bar=False, 
                              convert_to_numpy=True,
                              #convert_to_tensor=True, 
                              device=self.device)
        return x
    
    def output_dim(self, output_size):
        #Compute PCA on the train embeddings matrix
        pca = PCA(n_components=output_size)
        pca.fit(x)
        pca_comp = np.asarray(pca.components_)

        # We add a dense layer to the model, so that it will produce directly embeddings with the new size
        dense = self.models.Dense(in_features=self.model.get_sentence_embedding_dimension(), 
                                  out_features=output_size, bias=False, 
                                  activation_function=torch.nn.Identity())
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
        self.model.add_module('dense', dense)


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ## Input 
    MESH_XML = "MeSH/desc2021.xml"
    ## Output
    OUT_MESH_GRAPH = "mesh_graph.csv"
    OUT_MESH_EMBED = "mesh.sentencetransformer.embed.csv"

    # get started
    mesh = MeSHXMLParser(MESH_XML)
    mesh_nodes = mesh.parse()
    mg = MeshDAG(mesh_nodes)
    mesh_graph = mg(outfile=OUT_MESH_GRAPH)
    # embeddings
    mesh_encoder = SequenceEncoder(device=device) # model_name='sentence-transformers/allenai-specter'
    # print("Max Sequence Length:", model.max_seq_length)
    ##Change the length to 200
    #model.max_seq_length = 200

    # Note SeuqneceEncoder has max_seq_length. sentence > max_seq_leght will be trucated
    mesh_id = []
    mesh_sentences = []
    mesh_embeds = []
    for m, mnode in mesh_nodes.items():
        concepts = mnode['Concept']
        # mnode['embeds'] = []
        mesh_id.append(m)
        s = []
        for cpt in concepts:
            if 'ScopeNote' not in cpt: continue
            sentence = cpt['ScopeNote'].strip()
            s.append(sentence)
        if len(s) == 0:
            s.append(mnode['DescriptorName'])
        # max length
        mesh_sentences.append(s)
        embed = mesh_encoder(s)
        mesh_embeds.append(embed.mean(axis=0))

    mesh_embed2 = pd.DataFrame(mesh_embeds, index = mesh_id)
    mesh_embed2.to_csv(OUT_MESH_EMBED)

    # # visulization
    # # mesh categories/labels
    # cat2full = {'D':'Chemicals and Drugs',
    #             'C':'Disease', 
    #             'B':'Organisms',
    #             'E':'Analytical',
    #             'A': 'Anatomy',
    #             'G': 'Phenomena and Processes',
    #             'F':'Phychiatry and Psychology',
    #             'N':'Health Care',
    #             'I': 'Anthropology',
    #             'Z': 'Geographicals',
    #             'H':'Disciplines and Occupations',
    #             'L': "Information Sciences",
    #             'M':'Named Groups',
    #             'J':'Technology',
    #             'V':'Publications Charateristics',
    #             'K':'Humanities'}

    # for mid, mnode in mesh_nodes.items():
    #     mnode['lables'] = set()
    #     for t in mnode['TreeNums']:
    #         mnode['lables'].add(t[0])

    # num_cats = {k: 0 for k in cats}
    # for mid, mnode in mesh_nodes.items():
    #     mnode['lables'] = set()
    #     for t in mnode['TreeNums']:
    #         mnode['lables'].add(t[0])
    #         num_cats[t[0]] += 1

    # fig1, ax1 = plt.subplots(figsize=(5,4))
    # pie = pd.Series(num_cats)
    # labels = [l +": "+ cat2full[l]for l in pie.index.to_list()]
    # ax1.pie(pie.values, labels=pie.index.values, autopct='%1.1f%%', shadow=True, startangle=90)
    # ax1.legend(labels, bbox_to_anchor=(0.9, 1))
    # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.show()

