"""
## Gene names

NCBI gene-info (also synonym information) files can be downloaded from ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO. For example, the human file is ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz and the mouse file is ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz.

Explain: https://www.ncbi.nlm.nih.gov/books/NBK3840/

Here is a R package to get all synonym (alias) names: https://github.com/oganm/geneSynonym

"""
import  pandas as pd
from pubmed import UniProtXMLParser
# 2nd way to get gene alias
genes = pd.read_table("Mus_musculus.gene_info.gz")  

# we only interested in protein-coding genes
protein_coding = genes[genes.type_of_gene == "protein-coding"]

gene_id = protein_coding.GeneID.astype(str).to_list()
gene_alias = protein_coding.Synonyms.str.split("|").to_list()

gene_nodes = {}
for g, alias, symbol in zip(gene_id, gene_alias, protein_coding.Symbol.to_list()):
    gene_nodes[g] = {'gene_symbol': symbol, 'gene_synonyms': alias }



uxp = UniProtXMLParser()
geneid2uniprot = uxp.mapping(fr='P_ENTREZGENEID', to='ID', query=gene_id)


# take a long time to run, paralle processing better
for geneid, uniprot_ids in geneid2uniprot.items():
    meta = uxp.searchall(uniprot_ids)
    #del meta['query_accession']
    protein_names = []
    gene_names = []
    accessions = []
    uids = []
    organism = []
    for uid, udata in meta.items():
        protein_names += udata['protein_names']
        gene_names += udata['gene_names']
        accessions += udata['accession']
        organism.append(udata['organism'])
        uids.append(uid)
    gene_nodes[geneid]['protein_names'] = list(set(protein_names))
    gene_nodes[geneid]['gene_names'] = list(set(gene_names))
    gene_nodes[geneid]['uniprot_accession'] = list(set(accessions))
    gene_nodes[geneid]['uniprot_id'] = list(set(uids))
    gene_nodes[geneid]['organism'] = list(set(organism))  

for gid, data in gene_nodes.items():
    if '-' in data['gene_synonyms']:
        data['gene_synonyms'].pop(data['gene_synonyms'].index("-"))

joblib.dump(gene_nodes, filename="gene_nodes.pkl")