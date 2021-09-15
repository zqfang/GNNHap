import glob
import joblib
import networkx as nx
from pubmed import GeneMeSHGraph  
from tqdm import tqdm

INPUT = glob.glob("2019/*xml.gz")
OUTPUT = [xml.replace("xml.gz", "pkl") for xml in INPUT]
GRAPHS = [xml.replace("xml.gz", "graph.pkl") for xml in INPUT]
PUBS = [xml.split("/")[-1].replace(".xml.gz", "") for xml in INPUT]
PUBTATOR = expand("2019/pubtator/{splits}.pubtator.pkl", splits=PUBS)
WHOLE_GRAPH = "gene_mesh_final.pkl"


###### rules ###############
rule target:
    input: WHOLE_GRAPH
    
    
rule pubmed_parser:
    input: "2019/{splits}.xml.gz"
    output: "2019/{splits}.pkl"
    run:
        pubmed = PubMedXMLPaser(input[0])
        meta = pubmed.parse()
        joblib.dump(meta, output[0])
    
    #shell:
    #    "python pubmed_xml_parse.py {input} {output}"


    
    
rule pubtator:
    input: 
        pub = "2019/{splits}.pkl",
    output:
        "2019/pubtator/{splits}.pubtator.pkl"
    run:
        pubmed_meta = joblib.load(input.pub)
        gene_pubtator = GeneMeSHGraph.batch_pubtator(pubmed_meta)
        joblib.dump(gene_pubtator, filename=output[0])
        
        
        
rule graph_build:
    input: 
        pub="2019/{splits}.pkl",
        pubtator = "2019/pubtator/{splits}.pubtator.pkl",
        mesh="mesh_nodes.pkl",
        gene="gene_nodes.pkl"
    output:
        "2019/{splits}.graph.pkl"
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
    output: "gene_mesh_noedge.pkl"
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
    input: "gene_mesh_noedge.pkl"
    output: "gene_mesh_final.pkl"
    run:
        G0 = joblib.load(input[0])
        G0.edge_add() # take a few hours to run
        joblib.dump(G0, filename=output[0])
        G = G0.to_networkx()
        nx.write_gpickle(G, path="gene_mesh_networkx.pkl")

        