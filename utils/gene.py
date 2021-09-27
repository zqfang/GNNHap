"""
This script output gene_nodes.pkl

## Gene names

NCBI gene-info (also synonym information) files can be downloaded from ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO. 

For example, the human file is ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz 
and the mouse file is ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz.
or https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/Homo_sapiens.gene_info.gz

Explain: https://www.ncbi.nlm.nih.gov/books/NBK3840/

Here is a R package to get all synonym (alias) names: https://github.com/oganm/geneSynonym

"""
import pandas as pd
from Bio import SeqIO
from jax_unirep import get_reps
from pubmed import UniProtXMLParser

# protein sequences
proteins = SeqIO.parse("GRCh38_latest_protein.faa", "fasta")
# 2nd way to get gene alias
genes = pd.read_table("Homo_sapiens.gene_info.gzz")  

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



# get embedding
from gseapy.parser import Biomart
bm = Biomart()
attrs = bm.get_attributes(dataset='hsapiens_gene_ensembl') 
filters = bm.get_filters(dataset='hsapiens_gene_ensembl')

queries = protein_coding.GeneID.astype(str).to_list()
results = bm.query(dataset='hsapiens_gene_ensembl', 
                    attributes=['entrezgene_id', 'refseq_peptide'],
                    filters={'entrezgene_id': queries}
                    )         

results.dropna()['entrezgene_id'].nunique()


prot_df = []
for seq_record in SeqIO.parse("GRCh38_latest_protein.faa", "fasta"):
    prot_df.append((seq_record.id, str(seq_record.seq)))
prot_df = pd.DataFrame(prot_df, columns=['prot_id','peptide'])
results = results.drop_duplicates().dropna()
prot_df.index = prot_df.prot_id.str.split(".").str[0]
res = results.merge(prot_df, left_on='refseq_peptide', right_index=True, how='left')
res = res.dropna()

# only select AA with max sequence length
AA = res.groupby(['entrezgene_id'])['peptide'].agg(lambda x: x.loc[x.str.len().idxmax()])


h_avg, h_final, c_final= get_reps(AA.to_list(), mlstm_size=1900)
prot_embed = pd.DataFrame(h_avg, index=AA.index)
prot_embed.to_csv("human_gene_unirep.embeb.csv")
