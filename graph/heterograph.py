import sys, os, joblib, json
import numpy as np
import pandas as pd
import networkx as nx

class MeshDAG:
    """
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
        return self.mesh_graph

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
            mesh_graph.to_csv(outfile, index=False, sep="\t", header=False)
        return mesh_graph




def build_ppi_network(database, gene_nodes, taxid=9606, out_edgelist=None):
    """
    Args:
        database: Biogrid database, "BIOGRID-ALL-4.3.195.tab2.txt"
        gene_nodes: gene entrzid id in the graph 
        taxid: Homo sapiens (taxid:9606), Mus musculus (taxid:10090) use Human
        out_edgelist: output file name for saving PPI edgelist. e.g. gene_ppi_undirected_edges.txt
    Return:
        nx.Graph of PPI network
    """
    ## PPI edge
    biogrid = pd.read_table(database)
    # subset 
    biogrid = biogrid[(biogrid['Organism Interactor A'] == taxid) & (biogrid['Organism Interactor B'] == taxid)]
    biogrid = biogrid[['#BioGRID Interaction ID',
                        'Entrez Gene Interactor A','Entrez Gene Interactor B', 
                        'Official Symbol Interactor A','Official Symbol Interactor B',
                        'Experimental System','Experimental System Type']]
    # select protein_coding gene and has physical interaction  
    coding = sorted([int(g) for g in gene_nodes])
    biogrid_ms = biogrid[(biogrid['Entrez Gene Interactor A'].isin(coding)) & 
                    (biogrid['Entrez Gene Interactor B'].isin(coding)) & 
                    ( biogrid['Experimental System Type'] == 'physical')]

    # build PPI graph
    ppi = biogrid_ms.loc[:,['Entrez Gene Interactor A', 'Entrez Gene Interactor B', 'Experimental System Type']].astype(str)
    ppi = ppi.drop_duplicates()

    ppi_dict = {}
    for i, row in ppi.iterrows():
        s = row.loc['Entrez Gene Interactor A']
        t = row.loc['Entrez Gene Interactor B']
        key = (s,t)
        key_r = (t,s)
        if (key in ppi_dict) or (key_r in ppi_dict):
            continue
        ppi_dict[(key)] = 1
    PPI = nx.from_pandas_edgelist(ppi, source='Entrez Gene Interactor A', target='Entrez Gene Interactor B', create_using=nx.DiGraph)
    PPI2 = PPI.to_undirected(reciprocal=False)
    ppi2 = nx.to_pandas_edgelist(PPI2)
    ppi2.to_csv(out_edgelist, index=False, header=None, sep="\t")
    return PPI2 



if __name__ == "__main__":

    ## INPUT
    IN_GENEMESH_DICT = sys.argv[1] # "human_gene_mesh_dict.pkl"
    IN_BIOGRID = sys.argv[2] # 

    ## OUTPUT
    OUT_PPI_EDGES = sys.argv[3] # "gene_ppi_undirected_edges.txt"
    OUT_MESH_GRAPH = sys.argv[4] # " mesh_edges.txt"
    OUT_GENEMESH_GPKL = sys.argv[5] # "gene_mesh_hetero_nx.gpkl"


    print("Build GENE-MESH")
    graph = joblib.load(IN_GENEMESH_DICT) # if mouse data: gene_mesh_final.pkl
    # creat a heterogouse undirected graph (in networkx, multiview graph) 
    H = nx.MultiGraph() # for networkx undirected graph, edge add only once
    ## Add nodes
    genes = []
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
        genes.append(gid)
        
    for mid, mnode in graph.mesh_nodes.items():
        H.add_node(mid, node_type='mesh',node_label='mesh', 
                node_name=mnode['DescriptorName'], 
                node_cat= np.unique([t[0] for t in mnode['TreeNums']]).tolist(),
                node_treenums = mnode['TreeNums'],
                #node_ui=mnode['DescriptorUI'],
                node_mesh_id = mid,
                node_weight = len(mnode['PMIDs']),
                node_descriptor_class=mnode['DescriptorClass'], ) 

    # add gene-mesh edge
    for edge in graph.edges:
        s =str(edge['gene_node'])
        t = str(edge['mesh_node'])
        H.add_edge(s, t,  edge_type='genemesh', edge_weight=edge['weight'], edge_PMIDs = edge['PMIDs'])

    # save tmp file
    # nx.write_gpickle(H, path="human_gene_mesh_hetero_nx.tmp.gpkl")
    # H = nx.read_gpickle("human_gene_mesh_hetero_nx.tmp.gpkl")
    print("Build PPI")
    PPI = build_ppi_network(database=IN_BIOGRID, 
                            taxid=9906, 
                            gene_nodes=genes,
                            out_edgelist=OUT_PPI_EDGES)
    print("Add PPI Edges")
    for edge_idx, (head, tail, edge_dict) in enumerate(PPI.edges(data=True)):
        head = str(head)
        tail = str(tail)
        H.add_edge(head, tail,  edge_type='ppi')
    print("Build MESH DAG")
    meshdag = MeshDAG(graph.mesh_nodes)
    mesh_graph = meshdag(outfile=OUT_MESH_GRAPH) # return a dataframe
    print("Add MESH Edges")
    for i, e in mesh_graph.iterrows():
        s, r, t = e
        H.add_edge(s, t, edge_type=r)
    # check isolated nodes, and remove them
    isonodes = list(nx.isolates(H))
    # remove isolated nodes
    print("Isolated node num: %s"%len(isonodes))
    print("Remove isolated node")
    H.remove_nodes_from(list(nx.isolates(H)))
    nx.write_gpickle(H, path=OUT_GENEMESH_GPKL)
    # remove isolated nodes that you don't need in GNN
    print("Done build heterograph")