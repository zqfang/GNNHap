import os, json
import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path

from bokeh.palettes import Category10

## AAAS colors
dict_color = {'0':'#3B4992',
              '1':'#EE0000',
              '2':'#008B45',
              '3':'#631879',
              '4':'#9467bd', 
              '5':'#008280',
              '6':'#BB0021',
              '7':'#5F559B',
              '8':'#A20056',
              '9':'#808180',
              '?':'#ffffff',#'#1B1919',}
             }
expr_color = {'P':'#D13917', 'A': '#4C4A4B', 'M':'#ffffff', '-':'#ffffff'}
codon_flag = {'0':'Synonymous','1':'Non-Synonymous','2':'Splicing', '3':'Stop', '-1':'Non-Coding'}
codon_color_dict = {str(i) : Category10[10][i+1] for i in range(-1, 8)}
gene_expr_order = []
mesh_terms = {}

with open("/data/bases/fangzq/Pubmed/mouse_gene2entrezid.json", 'r') as j:
    GENE2ENTREZ = json.load(j) 

def get_pubmed_link(pmids):
    if pmids in ["Indirect", "Unknown_Gene"]:
        return pmids
    html = []
    for pid in pmids.split(","):
        #s =  f'<a href="https://pubmed.ncbi.nlm.nih.gov/{pid}" target="_blank">{pid}</a>'
        # s = f'https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids={pid}&concepts={gene}'
        s = f'<a href="https://www.ncbi.nlm.nih.gov/research/pubtator/index.html?view=docsum&query={pid}" target="_blank">{pid}</a>'
        html.append(s)
    return ",".join(html)


def get_html_links(genename):
    if genename in GENE2ENTREZ:
        entrzid = GENE2ENTREZ[genename] 
        html_string = f'<a href="https://www.ncbi.nlm.nih.gov/gene/{entrzid}" target="_blank">{genename}</a>'
        return html_string
    return genename

def get_html_string(pattern):    
    html_string = ""  
    for r in list(pattern):
        c = dict_color.get(r)
        s = f'<span style="color:{c};font-size:12pt;text-shadow: 1px 1px 2px #000000;">&#9612;</span>'
        html_string += s     
    return html_string

@lru_cache(maxsize=32)
def load_ghmap(dataset):
    fname = os.path.join(dataset)
    df = pd.read_table(fname, skiprows=6, dtype={'Haplotype': str, 'Chr': str})
    headers = []
    with open(fname, 'r') as d:
        for i, line in enumerate(d):
            headers.append(line.strip("\n#").split("\t"))
            if i == 6: break
            
    df.columns = headers[-1]
    df['Pattern'] = df['Haplotype'].astype(str)
    df['Haplotype'] = df.Haplotype.apply(get_html_string)
    df['GeneName'] = df.GeneName.apply(get_html_links)
    
    # dataset_name, codon_flag, gene_expr_order, strains, traits, mesh_terms = headers[:6]
    headers[2] = headers[2][-1].split(";")
 
    cf = [s.split(":") for s in headers[1][1:]]
    headers[1] = {k:v for k, v in cf}

    mesh_terms = [s.split(":") for s in headers[5][1:]]
    mesh_terms = {v:k for k, v in mesh_terms}
    headers[5] = mesh_terms

    df['Impact'] = df['CodonFlag'].astype(str).map(headers[1])
    df['logPvalue'] = -np.log10(df['Pvalue'])
    df['logPvalue'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['logPvalue'].fillna(df['logPvalue'].max(), inplace=True)
    df['CodonColor'] = df['CodonFlag'].astype(str).map(codon_color_dict)
     
    #mesh_columns = [m for m in headers[-1] if m.startswith("MeSH") ]
    
    return df, headers 


def get_color(pattern):
    """
    pattern: haplotype pattern
    """
    colors = []
    for r in list(pattern):
        c = dict_color.get(r)
        colors.append(c)
    return colors

def get_expr(pattern, gene_expr_order):
    """
    pattern: expr pattern, eg PPAAPPP
    """
    ep, ep2 = "", ""
    for r, g in zip(list(pattern), gene_expr_order):
        c = expr_color.get(r)
        s = f'<span style="color:{c};font-size:10pt;text-shadow: 1px 1px 2px #000000;">&#9612;</span>'
        s2 = f'<span style="color:{c};font-size:10pt;">&#9632; {g}</span><br>'
        ep += s 
        if r == 'P': ep2 += s2 # only show expressed tissues
    return ep, ep2


def get_datasets(data_dir):
    data = []
    path = Path(data_dir).glob("*.results.mesh.txt")
    for p in path:
        d = p.stem.split(".")[0]
        data.append(d)
    return sorted(data)