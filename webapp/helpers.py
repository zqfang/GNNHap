from curses import keyname
from email.policy import default
import glob, os, json, itertools
import numpy as np
import pandas as pd
import networkx as nx
from functools import lru_cache
from pathlib import Path
from bokeh.palettes import Category10
from bokeh.plotting import from_networkx
from flask import request
## HELPERS
STRAINS = {'129P2': '129P2/OlaHsd',
        '129S1': '129S1/SvImJ',
        '129S5': '129S5/SvEvBrd',
        'A_J': 'A/J',
        'AKR': 'AKR/J',
        'B10': 'B10.D2-Hc<0> H2<d> H2-T18<c>/oSnJ',
        'BALB': 'BALB/cJ',
        'BPL': 'BPL/1J',
        'BPN': 'BPN/3J',
        'BTBR': 'BTBR T<+> Itpr3<tf>/J',
        'BUB': 'BUB/BnJ',
        'C3H': 'C3H/HeJ',
        'C57BL10J': 'C57BL/10J',
        'C57BL/6J': 'C57BL/6J',
        'C57BL6NJ': 'C57BL/6NJ',
        'C57BRcd': 'C57BR/cdJ',
        'C57LJ': 'C57L/J',
        'C58': 'C58/J',
        'CAST': 'CAST/EiJ',
        'CBA': 'CBA/J',
        'CEJ': 'CE/J',
        'DBA1J': 'DBA/1J',
        'DBA': 'DBA/2J',
        'FVB': 'FVB/NJ',
        'ILNJ': 'I/LnJ',
        'KK': 'KK/HlJ',
        'LGJ': 'LG/J',
        'LPJ': 'LP/J',
        'MAMy': 'MA/MyJ',
        'MOLF': 'MOLF/EiJ',
        'MRL': 'MRL/MpJ',
        'NOD': 'NOD/ShiLtJ',
        'NON': 'NON/ShiLtJ',
        'NOR': 'NOR/LtJ',
        'NUJ': 'NU/J',
        'NZB': 'NZB/BlNJ',
        'NZO': 'NZO/HlLtJ',
        'NZW': 'NZW/LacJ',
        'PJ': 'P/J',
        'PLJ': 'PL/J',
        'PWD': 'PWD/PhJ',
        'PWK': 'PWK/PhJ',
        'RBF': 'RBF/DnJ',
        'RFJ': 'RF/J',
        'RHJ': 'RHJ/LeJ',
        'RIIIS': 'RIIIS/J',
        'SEA': 'SEA/GnJ',
        'SJL': 'SJL/J',
        'SMJ': 'SM/J',
        'SPRET': 'SPRET/EiJ',
        'ST': 'ST/bJ',
        'SWR': 'SWR/J',
        'TALLYHO': 'TALLYHO/JngJ',
        'WSB': 'WSB/EiJ'}

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

### functions ###


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

def get_hblock_links(row, filename):
    chrom = row['Chr']
    bstart = row['BlockStart']
    bsize = row['BlockSize']
    pos = chrom + ":" + str(row['ChrStart']) +"-"+ str(row['ChrEnd'])
    pattern = row['Pattern']
    html_string = f'<a href="{request.url_root}haploblock/{filename}_{pos}_{bstart}_{bsize}_{pattern}" target="_blank">{pos}</a>'
    return html_string

def get_html_string(pattern):    
    html_string = ""  
    for r in list(pattern):
        c = dict_color.get(r)
        s = f'<span style="color:{c};font-size:12pt;text-shadow: 1px 1px 2px #000000;">&#9612;</span>'
        html_string += s     
    return html_string

@lru_cache(maxsize=5)
def load_ghmap(dataset):
    print(dataset)
    fname = os.path.join(dataset)
    headers = []
    with open(fname, 'r') as d:
        for i, line in enumerate(d):
            if line.startswith("#"):
                headers.append(line.strip("\n#").split("\t"))
            if i == 6: break
    df = pd.read_table(fname, comment="#", header=None, names =headers[-1],  dtype={'Haplotype': str, 'Chr': str})
    df['Pattern'] = df['Haplotype'].astype(str)
    df['Haplotype'] = df.Haplotype.apply(get_html_string)
    df['GeneName'] = df.GeneName.apply(get_html_links)
    df['Position'] = df.apply(get_hblock_links, args=(os.path.basename(dataset),),axis=1)
    # dataset_name, codon_flag, gene_expr_order, strains, traits, mesh_terms = headers[:6]
    headers[2] = headers[2][-1].split(";")
 
    cf = [s.split(":") for s in headers[1][1:]]
    headers[1] = {k:v for k, v in cf}
    if len(headers) == 7: # dataset with MeSH_Terms
        mesh_terms = [s.split(":") for s in headers[5][1:]]
        mesh_terms = {v:k for k, v in mesh_terms}
        headers[5] = mesh_terms
    else:
        headers.insert(5, dict())

    df['Impact'] = df['CodonFlag'].astype(str).map(headers[1])
    df['logPvalue'] = -np.log10(df['Pvalue'])
    df['logPvalue'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['logPvalue'].fillna(df['logPvalue'].max(), inplace=True)
    df['CodonColor'] = df['CodonFlag'].astype(str).map(codon_color_dict)
     
    mesh_columns = [m for m in headers[-1] if m.startswith("MeSH") ]
    mx = "EffectSize"
    if len(mesh_columns) > 0: 
        mx = mesh_columns[0]    
        df['PubMed'] = df.loc[:, mx.replace("MeSH", "PMIDs")]
    df['LitScore'] = df[mx]
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
    path = list(Path(data_dir).glob("**/*.results.mesh.txt")) #+ list(Path(data_dir).glob("**/*.results.txt"))
    return sorted([str(p).split("/")[-1] for p in path])
    # for p in path:
    #     d = p.stem.split(".")[0]
    #     data.append(d)
    # return sorted(list(set(data)))

def get_data(dataset):
    df, headers = load_ghmap(dataset)
    df = df[df.CodonFlag>=0]
    if df.empty:
        print("No significant values loaded")
        return  
    columns = ['GeneName', 'CodonFlag','Haplotype','EffectSize', 'Pvalue', 'FDR',
                'PopPvalue', 'PopFDR', 'Position', 'LitScore','PubMed'] 
    if df.columns.str.startswith("Pop").sum() == 0: 
        # kick out 'FDR', PopPvalue', 'PopFDR',
        _columns = [ columns[i] for i in range(len(columns)) if  i not in [5, 6, 7] ]   
    else:
        _columns = columns
    # update mesh, bar, scatter
    mesh_columns = [m for m in df.columns if m.startswith("MeSH_") ]
    if len(mesh_columns) == 0:
        message = f"<p> Warning: <br> Dataset {dataset} not load Mesh Score !</p>"
        #_columns.pop(_columns.index("PubMed")) # kick out PubMed
        #_columns.pop(_columns.index('LitScore'))
        _columns.pop(-1)
        _columns.pop(-1)

    # self.myTable.columns = _columns    
    dataset_name, codon_flag, gene_expr_order, strains, traits, mesh_terms = headers[:6]
    # if (dataset_name[0].lower().find("indel") != -1) or (dataset_name[0].lower().find("_sv") != -1):
    #     codon_flag = {'0':'Low','1':'Moderate','2':'High', '-1':'Modifier'}
    if not mesh_terms: #  if empty 
        mesh_terms = {'EffectSize':'EffectSize'}
    new_data = {'dataset_name': dataset_name, 
                'codon_flag': codon_flag, 
                'gene_expr_order': gene_expr_order,
                'strains': strains, 
                'traits': traits, 
                'mesh_terms': mesh_terms,  
                'columns': _columns, 
                'mesh_columns':mesh_columns,
                'datasource': df.to_dict(orient='list'),}
    return new_data


def read_trait(filename):
    # out = []
    suf = filename.lower().split(".")[-1]
    if suf == "txt":
        out = pd.read_table(filename, comment="#", index_col=0)
    elif suf == "csv":
        out = pd.read_csv(filename, comment="#", index_col=0)
    elif suf in ['xls','xlsx']:
        out = pd.read_excel(filename, comment="#", index_col=0)
    return out 

def get_datasets(data_dir):
    data = []
    path = list(Path(data_dir).glob("*.results.mesh.txt")) + list(Path(data_dir).glob("*.results.txt"))
    for p in path:
        d = p.stem.split(".")[0]
        data.append(d)
    return sorted(list(set(data)))


def symbol_check(symbols):
    """
    a simple checker if inputs are mouse genes or human
    """
    human = sum([s.isupper() for s in symbols])
    if human / (len(symbols)) > 0.5:
        return True
    return False


def get_common_neigbhor_subgraph(H, entrezid, meshid):
    neighbors = list(nx.common_neighbors(H, entrezid, meshid )) + [entrezid, meshid]
    sg = nx.subgraph(H, neighbors).copy()
    pmids = ['Indirect']
    if sg.has_edge(entrezid, meshid):
        pmids = sg.get_edge_data(entrezid, meshid, key=0)['edge_PMIDs']
    # pos = nx.layout.spring_layout(sx, k=5)
    degrees = dict(nx.degree(sg))
    nx.set_node_attributes(sg, name='node_degree', values=degrees)
    adjusted_node_size = dict([(node, int(np.clip(degree, a_max=100, a_min=10))) for node, degree in degrees.items()])
    nx.set_node_attributes(sg, name='node_size_adjust', values=adjusted_node_size)
    #Create a network graph object
    # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
    network_graph = from_networkx(sg, nx.spring_layout, k=6, scale=4, center=(0, 0))
    node_data = network_graph.node_renderer.data_source.data
    xs = []
    ys = []
    graph_layout = {}
    for n in node_data['index']:
        x, y = network_graph.layout_provider.graph_layout[n]
        xs.append(float(x))
        ys.append(float(y))
        graph_layout[n] = [xs[-1], ys[-1]]

    node_data['x'] = xs
    node_data['y'] = ys
    return {
            'node_data': dict(node_data) ,
            'edge_data': dict(network_graph.edge_renderer.data_source.data),
            'graph_layout': graph_layout,
            'pmid': pmids
            }



def snp_view(data_path, pattern=None, chrom=19, bstart=36449, bsize=5, strains=None, haplo_strains=None):
    """
    show snp table view
    Args:
        data_path: haplotype file output of (eblocks -p)
        pattern: haplotype block pattern 
        chrom, bstart, bsize: haplotype block cooridnates
        strains: strain names coresponse to pattern 
 
    Return: tuple of  
        table: html table of haplotypeblock
        bed:  red file of haplotyepblock
    """
    # Retrieve the desired SNPs from the database file.
    if data_path is None:
        data_path = "/data/bases/shared/haplomap/HBCGM_DATA/MPD_24412-f/chr19.indel.haplotypes.txt"
    # haplob = pd.read_table(data_path, header=None, usecols=range(6))
    #print(haplob)
    start = int(bstart)
    end = start + int(bsize)
    table = """
    <table cellspacing=3>
    <thead>
    <TR halign="center">
    <TH valign="bottom">ID</TH><TH valign="bottom">Chr</TH><TH valign="bottom">Position</TH>
    """

    ## Read file block
    assert start < end
    # add headings for each strain abbrev
    HEADER_STRAINS = strains
    HAPLO_STRAINS = haplo_strains
    if strains is None:
        HEADER_STRAINS = range(len(pattern))

    if haplo_strains is None:
        HAPLO_STRAINS = range(len(pattern))
    ## strains num must be equal
    assert len(strains) == len(haplo_strains)

    for p, strain in zip(pattern, HEADER_STRAINS):
        c = dict_color[p]
        table +=f"<TH class=\"verticalTableHeader\" halign=\"left\" height=\"50\" valign=\"bottom\" style=\"background-color:{c}\"><span>{STRAINS[strain]}</span></TH>"
    table += "<TH valign=\"bottom\">Gene</TH><TH valign=\"bottom\">Annotation</TH></TR></thead><tbody><TR>"
    # read file block
    fh = open(data_path, 'r')
    snp_block = itertools.islice(fh, start, end)
    bed = [] # Chr, ChrStart, ChrEnd, Name
    for line in snp_block:
        row = line.strip().split()
        snpChr, snpPos, snpID, snpAlleles = row[:4]
        snpID = snpID.rstrip("_")
        table = table + f"<TD halign=\"center\" >{snpID}</TD>" +\
                        f"<TD halign=\"center\" >{snpChr}</TD>" +\
                        f"<TD halign=\"center\" >{snpPos}</TD>"
        # write snp allelle
        for s in HEADER_STRAINS:
            a = snpAlleles[HAPLO_STRAINS.index(s)] ## note: match the order of alleles to header 
            c = '#ffffff' # {'P':'#D13917', 'A': '#4C4A4B', 'M':'#ffffff', '-':'#ffffff'}
            if a == '0':
                c = '#dddcdd'
            elif a == '1':
                c = '#E77918'
            table += f"<TD halign=\"center\" style=\"background-color:{c}\">{a}</TD>"
        # print annotations
        snpGene, snpGeneAnno = "", ""
        if len(row) == 6:
            snpGene, snpGeneAnno = row[4:6]
        elif len(row) > 6:
            snpGene = "<br>".join(row[slice(4, len(row), 2)])
            snpGeneAnno = "<br>".join(row[slice(5, len(row), 2)])
        table += f"<TD halign=\"center\" >{snpGene}</TD><TD halign=\"center\" >{snpGeneAnno}</TD></TR>\n"
        tmp = snpID.split("_") # var, chrom, start, size, type
        snpEnd = int(snpPos) - 1
        snpStart = snpEnd -1 
        if (not snpID.startswith("SNP")) and (len(tmp) > 3):
            snpStart = int(tmp[2]) - 1 # for bed file
            snpEnd = snpStart + int(tmp[3])
        _bed = f"{snpChr}\t{snpStart}\t{snpEnd}\t"
        if len(snpGene) > 0:
            _bed += f"{snpGene};{snpGeneAnno}"
        _bed += "\n"
        bed.append(_bed)
    table += "</tbody></table>\n"
    fh.close()
    return table, bed