"""
This script output mesh_nodes.pkl and mesh embeddings
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
    
    # def output_dim(self, output_size):
    #     #Compute PCA on the train embeddings matrix
    #     pca = PCA(n_components=output_size)
    #     pca.fit(x)
    #     pca_comp = np.asarray(pca.components_)

    #     # We add a dense layer to the model, so that it will produce directly embeddings with the new size
    #     dense = self.models.Dense(in_features=self.model.get_sentence_embedding_dimension(), 
    #                               out_features=output_size, bias=False, 
    #                               activation_function=torch.nn.Identity())
    #     dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    #     self.model.add_module('dense', dense)


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ## Input 
    MESH_XML = sys.argv[1] # "MeSH/desc2021.xml"
    ## Output
    OUT_MESH_NODES = sys.argv[2] # "mesh_nodes.pkl"
    OUT_MESH_EMBED = sys.argv[3] # "mesh.sentencetransformer.embed.csv"

    # get started
    mesh = MeSHXMLParser(MESH_XML)
    mesh_nodes = mesh.parse()
    joblib.dump(mesh_nodes, filename= OUT_MESH_NODES)
    # mg = MeshDAG(mesh_nodes)
    # mesh_graph = mg(outfile=OUT_MESH_GRAPH)
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

