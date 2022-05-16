
import os, uuid, subprocess, re, json
from wsgiref.util import request_uri
import numpy as np
import pandas as pd
from functools import partial
from genericpath import exists

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from bokeh.embed import components
from bokeh.resources import INLINE
from bkgnnhap import GNNHapResults, GNNHapGraph
from utils import STRAINS, read_trait, get_datasets, run_gnnhap_predict
from helpers import gene_expr_order, codon_flag, mesh_terms, load_ghmap
app = Flask(__name__)

app.config.from_file("config.json", load=json.load)


# app.config['HBCGM_DIR'] = "/home/fangzq/github/GNNHap/webapp/HBCGM_DATA"
# app.config['GRAPH_DIR'] = "/home/fangzq/github/GNNHap/webapp/GRAPH_DATA"
# app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506' # os.urandom(24).hex()

TRAIT_DATA = {}
MESH_TERM = ""
GENE_SYMBOL = ""


@app.route('/', methods=['GET', 'POST'])
def index(): 
    global TRAIT_DATA
    global MESH_TERM
    global GENE_SYMBOL
    if request.method == 'POST':
        TRAIT_DATA = {}
        #mesh_terms = request.form.get('mesh_term')
        #mesh_terms = MESH_TERM.strip().split()
        # run HBCGM by uploaded file
        request
        MESH_TERM = request.form.get('mesh_term', "")
        GENE_SYMBOL = request.form.get('gene_symbol', "")
        print(request.form)
        if 'file' in request.files: # check if the post request has the file part
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['HBCGM_FOLDER'], filename)
            file.save(filepath)
            d = read_trait(filepath)
            #d = read_trait(filename)
            TRAIT_DATA = {}
            for i, row in d.iterrows():
                if row.iloc[0] in STRAINS:
                    TRAIT_DATA[row.iloc[0]] = row.iloc[-1]
        else:   
            for abbr, full in STRAINS.items():
                data = request.form.get(abbr)
                if (data is not None) and data != "": 
                    TRAIT_DATA[abbr] = data
        if (len(TRAIT_DATA) < 1) and (len(GENE_SYMBOL) < 1) and len(MESH_TERM) < 1:
            flash('No input given')
            return redirect(request.url) 
        ## This is for debugging
        return redirect(url_for('run', trait_data=TRAIT_DATA, 
                                    strains=STRAINS, 
                                    gene_symbols=GENE_SYMBOL,
                                    mesh_terms=MESH_TERM))

    html = render_template('index.html',strains=STRAINS,).encode(encoding='UTF-8')
    return html

@app.route("/run")
def run():
    html = render_template('run.html', 
    trait_data=TRAIT_DATA, 
    mesh_terms=MESH_TERM,
    gene_symbols=GENE_SYMBOL,
    strains=STRAINS,
    ).encode(encoding='UTF-8')
    return html


@app.route('/results')
def results():
    gnnhap = GNNHapResults(data_dir = app.config['HBCGM_DIR'], dataset=None)
    layout = gnnhap.build()
    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    # render template
    script, div = components(layout)
    html = render_template(
        'result.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    ).encode(encoding='UTF-8')
    return html

@app.route('/results/<uid>')
def results_uid(uid):
    #if not uid.startswith("MPD_"): uid2 = "MPD_"+uid
    # if not uid.endswith("")
    dataset = None
    data_dir = os.path.join(app.config['HBCGM_DIR'], uid)
    pat = re.compile("MPD_(.+)_([indel|snp]).results.mesh.txt")
    ## must have HBCGM_DATA directory
    if uid.startswith("MPD") and uid.endswith(".results.mesh.txt"):
        # if pat.search(uid):
        #     _uid = pat.search(uid).groups()[0]
        #     print(_uid)
        #     data_dir = os.path.join(app.config['HBCGM_DIR'], _uid)
        _uid = uid.split("_")[1]
        dataset = uid
        data_dir = os.path.join(app.config['HBCGM_DIR'], _uid)
    print(data_dir)
    print(dataset)
    gnnhap = GNNHapResults(data_dir = data_dir, dataset=dataset)
    layout = gnnhap.build()
    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    # render template
    script, div = components(layout)
    html = render_template(
        'result.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    ).encode(encoding='UTF-8')
    return html




@app.route('/graph')
def graph():
    g = GNNHapGraph(data_dir=app.config['GRAPH_DIR'], dataset=None)
    #g = GNNHapGraph(data_dir="/home/fangzq/github/GNNHap/webapp", dataset="simple.txt.gnn.txt")
    layout = g.build_graph()
    #render_template('index.html',strains=STRAINS,)
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    # render template
    script, div = components(layout)
    html = render_template(
        'result.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    ).encode(encoding='UTF-8')
    return html

@app.route('/graph/<uid>')
def graph_uid(uid):
    data_dir = app.config['GRAPH_DIR']
    # if uid.endswith(".gnn.txt"):
    #     _uid = uid.split(".")[0]
    #     data_dir = os.path.join(data_dir, _uid)
    g = GNNHapGraph(data_dir=data_dir, dataset=uid)
    #g = GNNHapGraph(data_dir="/home/fangzq/github/GNNHap/webapp", dataset="simple.txt.gnn.txt")
    layout = g.build_graph()
    #render_template('index.html',strains=STRAINS,)
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    # render template
    script, div = components(layout)
    html = render_template(
        'result.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    ).encode(encoding='UTF-8')
    return html

#background process happening without any refreshing
@app.route('/background_process')
def background_process():
    uid = str(uuid.uuid4())
    strain=""
    pheno=""
    if (len(GENE_SYMBOL) > 1) and (len(MESH_TERM) > 1) and (len(TRAIT_DATA) < 1):
        ret, cmd = run_gnnhap_predict(gene_symbol=GENE_SYMBOL, 
                                      mesh_terms=MESH_TERM, 
                                      dataset=os.path.join(app.config['GRAPH_DIR'], uid))
        print(ret.stdout)
        return jsonify({"uuid": uid, 
                         "cmd": cmd, 
                         "status": ret.stdout, 
                         "url": request.url_root + f"graph/{uid}"
                         })
    
    for k, v in TRAIT_DATA.items():
        strain += k +","
        pheno += v+","
    wkdir= os.path.join(app.config['HBCGM_DIR'], uid)
    cmd = f"/home/fangzq/miniconda/bin/snakemake -s {os.path.dirname(__file__)}/haplomap.smk " +\
        "--rerun-incomplete -p " +\
        f"--configfile {os.path.dirname(__file__)}/config.yaml " +\
        f"-j 32 --config WKDIR={wkdir} UUID={uid} STRAIN={strain} PHENO={pheno}"
    mesh_term = MESH_TERM.strip().split()

    if len(mesh_term) >=1:
        cmd += f" MESH_TERMS={','.join(mesh_term)}" 
        cmd = cmd.replace("haplomap.smk", "gnnhap.smk")
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True )
    # print(ret.stdout)
    return jsonify({"uuid": uid, 
                        "cmd":cmd, 
                        "status": ret.stdout, 
                        "url": request.url_root + f"results/{uid}"
                        })

@app.route('/GRAPH_DATA/<uid>')
def data_uid(uid):
    ## must have data directory
    if not uid.endswith(".gnn.txt"): uid = uid + ".gnn.txt"
    df = pd.read_table(os.path.join(app.config['GRAPH_DIR'], uid))

    return jsonify({'url': f"/GRAPH_DATA/{uid}",
                    'data': df.to_dict(orient='list')})


@app.route('/HBCGM_DATA/<uid>')
def hbcgm_uid(uid):
    # pat = re.compile("MPD_(.+)_([indel|snp]).results.mesh.txt")
    # uid2 = "results.mesh.txt"
    # ## must have HBCGM_DATA directory
    # if uid.startswith("MPD_") and uid.endswith(".results.mesh.txt"):
    #     if pat.search(uid):
    #         _uid = pat.search(uid).groups()[0]
    #         uid2 = f"{_uid}/{uid}"
    
    # DATASET1 = os.path.join(app.config['HBCGM_DIR'], uid)
    # DATASET2 = os.path.join(app.config['HBCGM_DIR'], uid2)
    # if os.path.exists(DATASET1): ## uid must endswith "results.mesh.txt"
    #     DATASET = DATASET1
    # elif os.path.exists(DATASET2):
    #     DATASET = DATASET2
    DATASET = os.path.join(app.config['HBCGM_DIR'], uid)
    
    df, headers = load_ghmap(DATASET)
    df = df[df.CodonFlag>=0]
    if df.empty:
        print("No significant values loaded")
        return  
    columns = ['GeneName', 'CodonFlag','Haplotype','EffectSize', 'Pvalue', 'FDR',
                'PopPvalue', 'PopFDR', 'Chr', 'ChrStart', 'ChrEnd', 'LitScore','PubMed'] 
    if df.columns.str.startswith("Pop").sum() == 0: 
        # kick out 'FDR', PopPvalue', 'PopFDR',
        _columns = [ columns[i] for i in range(len(columns)) if  i not in [5, 6, 7] ]   
    else:
        _columns = columns
    # update mesh, bar, scatter
    mesh_columns = [m for m in df.columns if m.startswith("MeSH_") ]
    if len(mesh_columns) == 0:
        message = f"<p> Warning: <br> Dataset {uid} not load Mesh Score !</p>"
        #_columns.pop(_columns.index("PubMed")) # kick out PubMed
        #_columns.pop(_columns.index('LitScore'))
        _columns.pop(-1)
        _columns.pop(-1)

    # self.myTable.columns = _columns    
    dataset_name, codon_flag, gene_expr_order, strains, traits, mesh_terms = headers[:6]
    # if (dataset_name[0].lower().find("indel") != -1) or (dataset_name[0].lower().find("_sv") != -1):
    #     codon_flag = {'0':'Low','1':'Moderate','2':'High', '-1':'Modifier'}
    x_range = list(range(0, len(strains)))
    if not mesh_terms: #  if empty 
        mesh_terms = {'EffectSize':'EffectSize'}
    new_data = {'url': f"HBCGM_DATA/{uid}",
                'dataset_name': dataset_name, 
                'codon_flag': codon_flag, 
                'gene_expr_order': gene_expr_order,
                'strains': strains, 
                'traits': traits, 
                'mesh_terms': mesh_terms,  
                'columns': _columns, 
                'mesh_columns':mesh_columns,
                'datasource': df.to_dict(orient='list'),}
    return jsonify(new_data)




if __name__ == '__main__':
    app.run(debug=True,host="peltz-app-03", port=5006)