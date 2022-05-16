
import os, uuid, subprocess, re, json, tempfile
import pandas as pd

from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from bokeh.embed import components
from bokeh.resources import INLINE
from bkgnnhap import GNNHapResults, GNNHapGraph
from helpers import STRAINS, read_trait, get_data, symbol_check

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
    """
    home page
    """
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
            filepath = os.path.join(app.config['HBCGM_DATA'], filename)
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

@app.route("/about")
def about():
    """
    about page
    """
    return render_template('about.html').encode(encoding='UTF-8')


@app.route("/run")
def run():
    """
    run page
    """
    html = render_template('run.html', 
    trait_data=TRAIT_DATA, 
    mesh_terms=MESH_TERM,
    gene_symbols=GENE_SYMBOL,
    strains=STRAINS,
    ).encode(encoding='UTF-8')
    return html

@app.route('/background_process')
def background_process():
    """
    background process happening without any refreshing
    this function is called whtn you running GNNHap pipeline 
    """
    uid = str(uuid.uuid4())
    ## comd for prediction only with gene symbols and mesh terms
    if (len(GENE_SYMBOL) > 1) and (len(MESH_TERM) > 1) and (len(TRAIT_DATA) < 1):
        ret, cmd = run_gnnhap_predict(gene_symbol=GENE_SYMBOL, 
                                      mesh_terms=MESH_TERM, 
                                      dataset=os.path.join(app.config['GRAPH_DIR'], uid))
        print(ret.stdout)
        return jsonify({"uuid": uid, 
                         "cmd": cmd, 
                         "status": ret.stdout, 
                         #"returncode": ret.returncode,
                         "url": request.url_root + f"graph/{uid}"
                        })
    
    ## format cmd for GNNHap
    strain =  []
    trait =  []
    for k, v in TRAIT_DATA.items():
        strain.append(k)
        trait.append(v)
    strain = ",".join(strain)
    pheno = ",".join(trait)

    wkdir= os.path.join(app.config['HBCGM_DIR'], uid)
    cmd = f"""{app.config["SNAKEMAKE"]} -s {app.config["GNNHAP"]}/webapp/gnnhap.smk \
              --rerun-incomplete -p -j 32 \
              --configfile {app.config["GNNHAP"]}/webapp/config.yaml \
              --config WKDIR={wkdir} UUID={uid} STRAIN={strain} PHENO={pheno}"""
    
    mesh_term = MESH_TERM.strip().split()
    if len(mesh_term) >=1:
        cmd += f" MESH_TERMS={','.join(mesh_term)}" 
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True )
    # print(ret.stdout)
    return jsonify({"uuid": uid, 
                    "cmd":cmd, 
                    "status": ret.stdout, 
                    #"returncode": ret.returncode,
                    "url": request.url_root + f"results/{uid}"
                    })

def run_gnnhap_predict(gene_symbol, mesh_terms, dataset=None):
    """
    helper function for runing prediction only
    """
    genes = gene_symbol.strip().split()
    meshs = mesh_terms.strip().split()
    ## check whether human gene symbol or mouse symbol

    tf = tempfile.NamedTemporaryFile(dir=os.path.dirname(dataset))
    tf.name = "simple.txt" if dataset is None else os.path.basename(dataset)+".txt"
    edgelist = tf.name
    with open(tf.name, 'w') as f:
        f.write("#GeneName\tMeSH\n") # header is needed
        for m in meshs:
            for g in genes:
                f.write(f"{g}\t{m}\n")
        f.seek(0)
    species = "mouse"
    if symbol_check(genes): species = "human"
    cmd = f"""{app.config["PYTHON"]} {app.config["GNNHAP"]}/GNNHap/predict_simple.py \
                --bundle {app.config["GNNHAP_BUNDLE"]} \
                --input {edgelist} \
                --output {dataset}.gnn.txt \
                --species {species}"""      
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ret, cmd


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
    """
    render /results/<uid> page.

    if uid is named with MPD_***_results.mesh.txt, then render page too.
    """
    #if not uid.startswith("MPD_"): uid2 = "MPD_"+uid
    # if not uid.endswith("")
    dataset = None
    data_dir = os.path.join(app.config['HBCGM_DIR'], uid)
    # pat = re.compile("MPD_(.+)_([indel|snp]).results.mesh.txt")
    ## must have HBCGM_DATA directory
    if uid.startswith("MPD") and uid.endswith(".results.mesh.txt"):
        # if pat.search(uid):
        #     _uid = pat.search(uid).groups()[0]
        #     print(_uid)
        #     data_dir = os.path.join(app.config['HBCGM_DIR'], _uid)
        _uid = uid.split("_")[1]
        dataset = uid
        data_dir = os.path.join(app.config['HBCGM_DIR'], _uid)
        """
        return json file for same page render
        """
        return jsonify(get_data(os.path.join(data_dir, dataset)))

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


@app.route('/HBCGM_DATA/<uid>')
def hbcgm_uid(uid):
    """
    background process url.
    this function is called when selected datasets in the "/results" page
    """
    DATASET = os.path.join(app.config['HBCGM_DIR'], uid)
    return jsonify(get_data(DATASET))

@app.route('/graph')
def graph():
    """
    read dataset and render page when open url "/graph". This page could select all datasets that avaible to show
    """
    g = GNNHapGraph(data_dir=app.config['GRAPH_DIR'], dataset=None)
    layout = g.build_graph()
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
    """
    read dataset and render page when open url "/graph/<uid>"
    """
    data_dir = app.config['GRAPH_DIR']
    if uid.endswith(".gnn.txt"): # dirty trick if select new dataset when in page "/graph/<uid>"
        _uid = uid.split(".")[0]
        # data_dir = os.path.join(data_dir, _uid)
        #TODO: read results in a subfolder
        df = pd.read_table(os.path.join(app.config['GRAPH_DIR'], uid))
        return jsonify({'url': f"/GRAPH_DATA/{uid}",
                        'data': df.to_dict(orient='list')})
    # now read data
    g = GNNHapGraph(data_dir=data_dir, dataset=uid)
    layout = g.build_graph()
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


@app.route('/GRAPH_DATA/<uid>')
def data_uid(uid):
    """
    background process: get json data when select datasets in "/graph" page 
    """
    ## must have data directory
    if not uid.endswith(".gnn.txt"): uid = uid + ".gnn.txt"
    df = pd.read_table(os.path.join(app.config['GRAPH_DIR'], uid))

    return jsonify({'url': f"/GRAPH_DATA/{uid}",
                    'data': df.to_dict(orient='list')})



if __name__ == '__main__':
    app.run(debug=True, 
            host="0.0.0.0", 
            port=5006)