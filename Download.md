# File Downloads


## 1. Gene_info and protein sequences
NCBI gene-info (also synonym information) files can be downloaded [here](ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO). 

[Explain](https://www.ncbi.nlm.nih.gov/books/NBK3840/)

- GENE_INFO:
    - Human:
        - ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz 
    - Mouse: 
        - ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz.
        - https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/Homo_sapiens.gene_info.gz

- Protein Sequences:
    - Human:
        - ftp://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers/GRCh38_latest_protein.faa.gz
    - Mouse:
        - ftp://ftp.ncbi.nlm.nih.gov/refseq/M_musculus/annotation_releases/108/GCF_000001635.26_GRCm38.p6/GCF_000001635.26_GRCm38.p6_protein.faa.gz  




Here is a R package to get all synonym (alias) names: https://github.com/oganm/geneSynonym


### 2. MeSH Database

Download bulk data (xml format) from [here](https://www.nlm.nih.gov/databases/download/mesh.html)

Note:
    - extract descriptor and its uid (this is what we need)
    - extract qualifier and its uid
    - extract ScopeNote to get sentence embeddings 

MeSH tree: hierachical tree. Treeview [here](https://meshb.nlm.nih.gov/treeView)  



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

### 3. Pubmed Titles and Abstracts

Just bulk download                  
```shell                             
# connect to ncbi ftp server         
ftp -i ftp.ncbi.nlm.nih.gov          
                                  
## login anonymous                   
 ## username: anonymous               
## password: <blank> (hit enter)     
## set your local directory to save files. default: current
#ftp> lcd /home/user/yourdirectory   
ftp> mget pubmed/baseline/*xml.gz   
ftp> mget pubmed/updatefiles/*xml.gz
```