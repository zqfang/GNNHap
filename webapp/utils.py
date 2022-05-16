import glob, os, subprocess, tempfile
from re import sub
import pandas as pd
from pathlib import Path

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

def read_trait(filename):
    # out = []
    suf = filename.lower().split(".")[-1]
    # with open(filename, 'r') as inp:
    #     for line in inp:
    #         if line.startswith("#"): continue
    #         line = line.strip().split()
    #         out.append(out)
    if suf == "txt":
        out = pd.read_table(filename, comment="#")
    elif suf == "csv":
        out = pd.read_csv(filename, comment="#")
    elif suf in ['xls','xlsx']:
        out = pd.read_excel(filename, comment="#")
    return out 

def get_datasets(data_dir):
    data = []
    path = list(Path(data_dir).glob("*.results.mesh.txt")) + list(Path(data_dir).glob("*.results.txt"))
    for p in path:
        d = p.stem.split(".")[0]
        data.append(d)
    return sorted(list(set(data)))


# save suprocess output in a file
# with open('output.txt', 'w') as f:
#     out = subprocess.run('ping 127.0.0.1', shell=True, stdout=f, text=True)
def run_gnnhap_predict(gene_symbol, mesh_terms, dataset=None):
    genes = gene_symbol.strip().split()
    meshs = mesh_terms.strip().split()
    tf = tempfile.NamedTemporaryFile(dir=os.path.dirname(dataset))
    tf.name = "simple.txt" if dataset is None else os.path.basename(dataset)+".txt"
    print(genes)
    print(meshs)
    edgelist = tf.name
    print(edgelist)
    print(dataset)
    with open(tf.name, 'w') as f:
        f.write("#GeneName\tMeSH\n")
        for m in meshs:
            for g in genes:
                f.write(f"{g}\t{m}\n")
        f.seek(0)
    cmd = "/home/fangzq/miniconda/envs/fastai/bin/python " +\
        "/home/fangzq/github/GNNHap/GNNHap/predict_simple.py " +\
        "--bundle /data/bases/fangzq/Pubmed/bundle " +\
        f"--input {edgelist} " +\
        f"--output {dataset}.gnn.txt " +\
        "--species mouse"
    # 如果传递单个字符串，shell必须为True
    # 要获取命令执行的结果或者信息，在调用run()方法的时候，请指定stdout=subprocess.PIPE。则返回值会包含stddout 属性
    # try:
    #     ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    # except subprocess.CalledProcessError as e:
    #     print(e.output)
      
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ret, cmd
