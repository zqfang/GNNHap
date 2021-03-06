import glob
import joblib
import networkx as nx
from pubmed import GeneMeSHGraph, PubMedXMLPaser  
from tqdm import tqdm


wkdir: "/data/bases/fangzq/Pubmed"

INPUT = glob.glob("2019/*xml.gz")
OUTPUT = [xml.replace("xml.gz", "pkl") for xml in INPUT]
GRAPHS = [xml.replace("xml.gz", "human.graph.pkl") for xml in INPUT]
PUBS = [xml.split("/")[-1].replace(".xml.gz", "") for xml in INPUT]
PUBTATOR = expand("2019/pubtator/{splits}.pubtator.pkl", splits=PUBS)
PUBMETA = expand("2019/{splits}.meta.pkl", splits=PUBS)
HETERO_GRAPH = "human_gene_mesh_hetero_nx.gpkl"

###### rules ###############
rule target:
    input: HETERO_GRAPH


rule mesh_nodes:
    input: "MeSH/desc2021.xml"
    output: 
        "mesh_nodes.pkl", # metadata
        "mesh.sentencetransformer.embed.csv" # embedding
    shell:
        "python mesh.py {input} {output}"

rule gene_nodes:
    input: 
        "GRCh38_latest_protein.faa",
        "Homo_sapiens.gene_info.gz"
    output: 
        "human_gene_unirep.embeb.csv", # embedding
        "human_gene_aa.csv",
        "human_gene_nodes.pkl"
    shell:
        "python gene.py {input} {output}"

rule pubmed_parser:
    input: "2019/{splits}.xml.gz"
    output: "2019/{splits}.meta.pkl"
    run:
        pubmed = PubMedXMLPaser(input[0])
        meta = pubmed.parse()
        joblib.dump(meta, output[0])
    
    #shell:
    #    "python pubmed_xml_parse.py {input} {output}"

rule pubtator:
    input: 
        pub = "2019/{splits}.meta.pkl",
    output:
        "2019/pubtator/{splits}.pubtator.pkl"
    run:
        pubmed_meta = joblib.load(input.pub)
        gene_pubtator = GeneMeSHGraph.batch_pubtator(pubmed_meta)
        joblib.dump(gene_pubtator, filename=output[0])
        
rule graph_build:
    input: 
        pub="2019/{splits}.meta.pkl",
        pubtator = "2019/pubtator/{splits}.pubtator.pkl",
        mesh="mesh_nodes.pkl", # mesh.py output
        gene="human_gene_nodes.pkl" # gene.py output
    output:
        "2019/{splits}.human.graph.pkl"
    run:
        gene_nodes = joblib.load(input.gene)
        mesh_nodes = joblib.load(input.mesh)
        pubmed_meta = joblib.load(input.pub)
        pubtator_meta = joblib.load(input.pubtator)
        G = GeneMeSHGraph(gene_nodes=gene_nodes, mesh_nodes=mesh_nodes)
        G.mesh_node_add_pmid(pubmed_meta)
        G.gene_node_add_pmid(pubmed_meta, pubtator_meta)
        # G.edges_add()
        joblib.dump(G, filename=output[0])
                
rule graph_agg:
    input: GRAPHS
    output: "human.gene_mesh_noedge.pkl"
    run:
        G0 = joblib.load(input[0])
        for inp in tqdm(input[1:]):
            # TODO
            G = joblib.load(inp)
            for g_id, g_node in G.gene_nodes.items():
                G0.gene_nodes[g_id]['PMIDs'].update(g_node['PMIDs'])
            for m_id, m_node in G.mesh_nodes.items():
                G0.mesh_nodes[m_id]['PMIDs'].update(m_node['PMIDs'])    
        joblib.dump(G0, output[0])


rule graph_build_edge:
    input: "human.gene_mesh_noedge.pkl"
    output: 
        "human_gene_mesh_dict.pkl",
    run:
        G0 = joblib.load(input[0])
        G0.edge_add() # take a few hours to run
        joblib.dump(G0, filename=output[0])
        #G = G0.to_networkx()
        #nx.write_gpickle(G, path=output[1])

rule hetero_graph_build:
    input: 
        "human_gene_mesh_dict.pkl", 
        "BIOGRID-ALL-4.3.195.tab2.txt"
    output:
        "human_gene_ppi_undirected_edgelist.txt",
        "mesh_edgelist.txt",
        "human_gene_mesh_hetero_nx.gpkl"
    shell:
        "python heterograph.py {input} {output}"