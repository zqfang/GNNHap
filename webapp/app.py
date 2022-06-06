
from ast import Sub
import os, uuid, subprocess, re, json, tempfile
import numpy as np
import pandas as pd
import networkx as nx
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
from bokeh.embed import components
from bokeh.resources import INLINE

from bkgnnhap import GNNHapResults, GNNHapGraph, SubGraph
from helpers import STRAINS, read_trait, get_data, symbol_check, get_common_neigbhor_subgraph, get_html_links, snp_view
# from bokeh.palettes import Spectral8

app = Flask(__name__)

app.config.from_file("config.json", load=json.load)
# app.config['HBCGM_DATA'] = "/home/fangzq/github/GNNHap/webapp/HBCGM_DATA"
# app.config['GRAPH_DATA'] = "/home/fangzq/github/GNNHap/webapp/GRAPH_DATA"
# app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506' # os.urandom(24).hex()
TRAIT_DATA = {}
MESH_TERM = ""
GENE_SYMBOL = ""
#GENE_MESH_GRAPH = nx.read_gpickle(app.config['GENE_MESH_GRAPH'])
GENE_MESH_GRAPH = ""

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
                                      dataset=os.path.join(app.config['GRAPH_DATA'], uid))
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

    wkdir= os.path.join(app.config['HBCGM_DATA'], uid)
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
    graph_data_dict = get_common_neigbhor_subgraph(GENE_MESH_GRAPH, "23411", "D006311")
    gnnhap = GNNHapResults(data_dir = app.config['HBCGM_DATA'], dataset=None, graph_data_dict=graph_data_dict)
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
    data_dir = os.path.join(app.config['HBCGM_DATA'], uid)
    # pat = re.compile("MPD_(.+)_([indel|snp]).results.mesh.txt")
    ## must have HBCGM_DATA directory
    if uid.startswith("MPD") and uid.endswith(".results.mesh.txt"):
        # if pat.search(uid):
        #     _uid = pat.search(uid).groups()[0]
        #     print(_uid)
        #     data_dir = os.path.join(app.config['HBCGM_DATA'], _uid)
        _uid = uid.split("_")[1]
        dataset = uid
        data_dir = os.path.join(app.config['HBCGM_DATA'], _uid)
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
    DATASET = os.path.join(app.config['HBCGM_DATA'], uid)
    return jsonify(get_data(DATASET))

@app.route('/graph')
def graph():
    """
    read dataset and render page when open url "/graph". This page could select all datasets that avaible to show
    """
    graph_data_dict = get_common_neigbhor_subgraph(GENE_MESH_GRAPH, "23411", "D006311")
    g = GNNHapGraph(data_dir=app.config['GRAPH_DATA'], dataset=None, graph_data_dict=graph_data_dict)
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


@app.route('/search')
def search():
    """
    read dataset and render page when open url "/search". This page could select all datasets that avaible to show
    """
    graph_data_dict = get_common_neigbhor_subgraph(GENE_MESH_GRAPH, "23411", "D006311")
    g = SubGraph(GENE_MESH_GRAPH, graph_data_dict=graph_data_dict)
    layout = g.build_graph()
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()
    # render template
    script, div = components(layout)
    html = render_template(
        'search.html',
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
    data_dir = app.config['GRAPH_DATA']
    if uid.endswith(".gnn.txt"): # dirty trick if select new dataset when in page "/graph/<uid>"
        _uid = uid.split(".")[0]
        # data_dir = os.path.join(data_dir, _uid)
        #TODO: read results in a subfolder
        df = pd.read_table(os.path.join(app.config['GRAPH_DATA'], uid))
        df['GeneName'] = df['#GeneName'].apply(get_html_links)
        return jsonify({'url': f"/GRAPH_DATA/{uid}",
                        'data': df.to_dict(orient='list')})
    # now read data
    g = GNNHapGraph(data_dir=data_dir, dataset=uid, graph_data_dict=None)
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
    df = pd.read_table(os.path.join(app.config['GRAPH_DATA'], uid))

    return jsonify({'url': f"/GRAPH_DATA/{uid}",
                    'data': df.to_dict(orient='list')})



@app.route('/graph_process/<gene>_<meshid>')
def graph_process(gene, meshid):
    """
    update graph data using networkx
    """
    return jsonify(get_common_neigbhor_subgraph(GENE_MESH_GRAPH, gene, meshid))
    # # ## testing
    # N = 8
    # node_indices = list(range(N))
    # # generate ellipses based on the ``node_indices`` list
    # circ = [i*2*np.pi/8 for i in node_indices]

    # # create lists of x- and y-coordinates
    # x = [np.cos(i)*np.random.randint(low=1, high=10) for i in circ]
    # y = [np.sin(i)*np.random.randint(low=1, high=10) for i in circ]
    # node_indices = ['C'+str(i) for i in node_indices]
    # # # assign a palette to ``fill_color`` and add it to the data source
    # node_data = dict(
    #         index=node_indices,
    #         node_type_color=Spectral8,
    #         node_size_adjust = [10]*N,
    #         #node_type_color=['black']*N,
    #         node_marker=['circle']*N, 
    #         node_name = [str(i) for i in node_indices],
    #         x=x, 
    #         y=y)

    #     # add the rest of the assigned values to the data source
    # edge_data = dict(
    #         start=['C0']*N,
    #         end=node_indices,
    #         edge_weight_adjust=[2]*N)


    # # convert the ``x`` and ``y`` lists into a dictionary of 2D-coordinates
    # # and assign each entry to a node on the ``node_indices`` list
    # graph_layout = dict(zip(node_indices, zip(x, y)))


    # return jsonify({'node_data': node_data,
    #                 'edge_data': edge_data,
    #                 'graph_layout': graph_layout })

@app.route('/haplotype')
def haplotype():
    table = "<p>This is a demo Page. To get SNP view, please redirect to Results page</p>"
    roi = url_for('static', filename='roi.bed')
    html = render_template('haplotype.html', 
                           snp_view=table, 
                           position="19:43800045-43806998", 
                           roi=roi).encode(encoding='UTF-8')
    return html


@app.route('/haploblock/<dataset>_<position>_<blockStart>_<blockSize>_<pattern>')
def haploblock(dataset, position, blockStart, blockSize, pattern):
    pat = re.compile("MPD_(.+)_(\w+).results.mesh.txt")
    chrom = position.split(":")[0] # position pattern: 19:43800045-43806998
    if not chrom.lower().startswith("chr"):
        chrom = "chr"+chrom
    if pat.search(dataset):
        uid, vartype = pat.search(dataset).groups()
        data_path =  os.path.join(app.config['HBCGM_DATA'], dataset)
        haplo_path = os.path.join(app.config['HBCGM_DATA'], "MPD_"+uid, f"{chrom}.{vartype}.haplotypes.txt")
        if not os.path.exists(haplo_path): # walk to next level if not exist
            haplo_path = os.path.join(app.config['HBCGM_DATA'], uid, "MPD_"+uid, f"{chrom}.{vartype}.haplotypes.txt")
            data_path =  os.path.join(app.config['HBCGM_DATA'], uid, dataset)

        headers = []
        with open(data_path, 'r') as d:
            for i, line in enumerate(d):
                if line.startswith("#"):
                    headers.append(line.strip("\n#").split("\t"))
                if i == 6: break
        table, bed = snp_view(data_path=haplo_path,
                         pattern=pattern, 
                         chrom=chrom, 
                         bstart=blockStart, 
                         bsize=blockSize, 
                         strains=headers[3])
        # write this temp file. only used for highlight haplotype blocks in the genome view
        roi = url_for('static', filename='roi.bed') # critical to use url_for here to enable http get access the file
        with open(os.path.join(request.script_root, "static/roi.bed"), 'w') as _roi:
            _roi.writelines(bed)
        html = render_template('haplotype.html', snp_view=table, position=position, roi=roi).encode(encoding='UTF-8')
        return html


if __name__ == '__main__':
    app.run(debug=True, 
            host="peltz-app-03",
            #host="0.0.0.0", 
            port=5006)