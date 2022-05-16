
from genericpath import exists
from operator import indexOf
import os, glob
import numpy as np
import pandas as pd
from functools import partial
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, TableColumn, DateFormatter, DataTable, HTMLTemplateFormatter, CellFormatter
from bokeh.models import ColorBar, LinearColorMapper, LabelSet, Legend
from bokeh.models import FixedTicker, RangeSlider, CDSView, BooleanFilter, GroupFilter, CustomJS
from bokeh.models.glyphs import Patches
from bokeh.models.widgets import Select, TextInput, Dropdown, AutocompleteInput, Div, Button
from bokeh.layouts import column, row

from bokeh.palettes import Category10
from bokeh.transform import factor_cmap, linear_cmap, factor_mark
from bokeh.core.enums import MarkerType
from helpers import gene_expr_order, codon_flag, mesh_terms
from helpers import load_ghmap, get_color, get_expr, get_datasets, get_pubmed_link
from utils import STRAINS

# dataset id
class GNNHapResults:
    def __init__(self, data_dir, dataset=None):
        ### setup data and plots
        ## data
        DATASETS = glob.glob(data_dir+"**/*.results.mesh.txt")
        DATASETS2 = [os.path.basename(d) for d in DATASETS]
        if dataset is None:
            dataset = DATASETS2[0]

        self.DATA_DIR = data_dir
        global gene_expr_order
        global mesh_terms
        global codon_flag
        print(dataset)
        # update new data
        df, headers = load_ghmap(os.path.join(self.DATA_DIR, dataset))
        df = df[df.CodonFlag>=0]
        if df.empty:
            self.message.text = f"<p> Error: <br> Dataset is a Empty Table !!! <br> Please Input a new dataset name.</p>"
            return      
        dataset_name, codon_flag, gene_expr_order, strains, traits, mesh_terms = headers[:6]
        mesh_terms_ = {k:[v] for k, v in mesh_terms.items()}
        codon_flag_ = {k:[v] for k, v in codon_flag.items() }
        self.source_codon = ColumnDataSource(data=codon_flag_)
        self.source_meshs = ColumnDataSource(data=mesh_terms_)
        self.source = ColumnDataSource(df)
        self.source_bar = ColumnDataSource(data=dict(strains=strains,traits=traits, colors=['#A19B9B']*len(strains)))   

        self.dataset = AutocompleteInput(completions=DATASETS2, #value="MPD_26711-f_Indel",width=550,
                                title="Dataset:", value=dataset, width=550,)
        # mesh terms options
        self.meshid = Select(title="Select MeSH:", value=list(mesh_terms.keys())[0], options=list(mesh_terms.keys()), width=300,) # need to update dynamically
        # groupfilter options
        self.impact = Select(title="Select Group:", value=list(codon_flag.values())[0], options=list(codon_flag.values()), width=200,) # need to update dynamically
        # others
        self.symbol = TextInput(value = '', title = "Gene Name:", width=300,)
        self.pval = TextInput(value = '', title = "P-value:",width=300,)
        self.litscore = TextInput(value = '', title = "MeSH Score:",width=300,)
        self.codon = TextInput(value = '', title = "Impact (Codon Flag)",width=300,)
        # message box
        self.message = Div(text="""<h3> Gene Expression: </h3>""", width=300, height=150)
        # gene expression pattern
        self.exprs = Div(text="""<h3> Gene Expressed in: </h3>""", width=300, height=800)

        self.slider = RangeSlider(title="LitScore Range", start=0.0, end=1.0, value=(0.5, 1.0), step=0.01)
        # data view
        self.bool_filter = BooleanFilter()
        self.group_filter = GroupFilter()
        self.view = CDSView(source=self.source, filters=[self.bool_filter, self.group_filter ])

        ## Datatable
        columns = ['GeneName', 'CodonFlag','Haplotype','EffectSize', 'Pvalue', 'FDR',
                'PopPvalue', 'PopFDR', 'Chr', 'ChrStart', 'ChrEnd', 'LitScore','PubMed'] 
        columns = [ TableColumn(field=c, title=c, formatter=HTMLTemplateFormatter() 
                                if c in ['Haplotype','GeneName', 'PubMed'] else CellFormatter()) for c in columns ] # skip index  
        self.columns = columns                     
        self.myTable = DataTable(source=self.source, columns=columns, width =1200, height = 600, index_position=0,
                            editable = False, view=self.view, name="DataTable",sizing_mode="stretch_width") # autosize_mode="fit_viewport"

        # download
        self.button = Button(label="Download Table", button_type="success")
        self.barplot()
        self.scatterplot()
        
        # self.data_update(dataset)
    ## bar plot
    def barplot(self,):
        bar = figure(plot_width=550, plot_height=500, # x_range=strains, 
                title="Dataset", 
                toolbar_location="below",
                x_range=self.source_bar.data['strains'],
                tools='pan,reset,lasso_select,save', output_backend="svg", name="Bar")
        bar.toolbar.logo = None
        bar.vbar(x='strains', top='traits', source=self.source_bar, line_width=0, fill_color='colors', width=0.7)
        #bar.vbar(x='x_range', top='traits', source=self.source_bar, line_width=0, fill_color='colors', width=0.7)
        bar.xgrid.grid_line_color = None
        #p.y_range.start = 0
        #bar.xaxis.axis_label = ""#"Strains"
        #bar.xaxis.ticker = FixedTicker(ticks=self.source_bar.data['x_range'])
        #bar.xaxis.major_label_overrides = {k: str(v) for k, v in zip(self.source_bar.data['x_range'], self.source_bar.data['strains'])}
        bar.title.text = "Dataset: "
        bar.yaxis.axis_label = "Values"
        bar.xaxis.major_label_orientation = np.pi/4
        # or alternatively:
        #p.xaxis.major_label_orientation = "vertical"
        self.bar = bar


    def scatterplot(self,):
        # hover tooltips
        TOOLTIPS = [
            ("logPval, LitScore", "($x, $y)"),
            ("GeneName", "@GeneName{safe}"), # use the {safe} format after the column name to disable the escaping of HTML
            ("Impact", "@Impact")]

        sca = figure(plot_width=550, plot_height=500, 
                    tools="pan,box_zoom,reset,lasso_select,save",
                    output_backend="svg",  
                    #active_drag="lasso_select",
                    toolbar_location="below",
                    tooltips=TOOLTIPS, name='Scatter',
                    x_axis_label= "Genetic:  - log10 Pvalue",
                    y_axis_label= "Literature Score")

        sca.add_layout(Legend(), 'above') # put legend outside
        sca.legend.orientation = "horizontal"
        sca.toolbar.logo = None

        sca.scatter('logPvalue', 'LitScore', 
                    size=8, 
                    source=self.source, 
                    #fill_color=fcmap,# mapper,
                    #marker=fmark,
                    color='CodonColor',
                    legend_field="Impact",
                    line_color=None, 
                    selection_color="#2F0303", #alpha=0.6, 
                    view=self.view,
                    nonselection_alpha=0.1, )#selection_alpha=0.4)
        sca.title.text =  "MeSH Terms"
        #sca.legend.location = "bottom_right"
        self.sca = sca
            
    def data_update(self, uid=None):

        # if (dataset_name[0].lower().find("indel") != -1) or (dataset_name[0].lower().find("_sv") != -1):
        #     codon_flag = {'0':'Low','1':'Moderate','2':'High', '-1':'Modifier'}
        self.dataset.js_on_change('value', CustomJS(args=dict(source=self.source, 
                           bool_filt=self.bool_filter,
                           group_filt = self.group_filter, 
                           bar = self.bar,
                           sca = self.sca,
                           meshid = self.meshid,
                           impact = self.impact,
                           source_meshs = self.source_meshs,
                           source_bar = self.source_bar,
                           source_codon = self.source_codon,
                           ), 
                code="""
                    const uid = cb_obj.value;
                    bool_filt.booleans = null;
                    console.log("The URL of this page is: " + window.location.href); // get current url 
                    var current_url =  window.location.href.split("/");
                    var url = uid;
                    if (current_url[current_url.length -1] == "results")
                    {

                        url = "HBCGM_DATA/" + uid;
                    }
                    console.log(url);
                    var xmlhttp = new XMLHttpRequest(); // read file on the server side 
                    // Define a callback function
                    xmlhttp.onload = function() {
                        if (xmlhttp.readyState == 4 && xmlhttp.status == 200) 
                        {
                            // Here you can use the Data
                            var new_data = JSON.parse(xmlhttp.responseText);
                    
                            // updates
                            url = new_data['url'];
                            meshid.options =  Object.keys(new_data['mesh_terms']);
                            impact.value = new_data['codon_flag']['1'];
                            impact.options =  Object.values(new_data['codon_flag']);
                            group_filt.group = impact.value;
                            bar.x_range.factors = new_data['strains'];
                            source_bar.data = { 
                                               'strains': new_data['strains'], 
                                               'traits': new_data['traits'], 
                                               'colors': Array(new_data['strains'].length).fill('#A19B9B')
                                               }
                            bar.title.text = "Dataset: "+new_data['dataset_name'][0];
                            if (new_data['mesh_columns'].length > 0)
                            {
                                sca.title.text = new_data['mesh_columns'][0];
                                var tmp = new_data['mesh_columns'][0].split("_");
                                var pubc = "PMIDs_" + tmp[tmp.length-1];
                                new_data['datasource']['PubMed']= new_data['datasource'][pubc];
                            }
                            source.data = new_data['datasource'];
                            var new_mesh = {};
                            for (let k in new_data['mesh_terms'])
                            {
                                new_mesh[k] = [new_data['mesh_terms'][k]];
                                meshid.value = k;
                            }
                            source_meshs.data = new_mesh;

                            var new_codon = {};
                            for (let k in new_data['codon_flag'])
                            {
                                new_codon[k] = [new_data['codon_flag'][k]];
                            }
                            source_codon.data = new_codon;
                        }        
                    }
                    // Send a request
                    xmlhttp.open("GET", url, true);
                    xmlhttp.send();

                    bar.change.emit();
                    sca.change.emit();
                    source.change.emit();
                    source_meshs.change.emit();
                    source_codon.change.emit();
                """))

        self.impact_update()
        self.mesh_update()

    def slider_update(self): # attr, old, new
        self.slider.js_on_change('value', 
        CustomJS(args=dict(source=self.source, 
                           filt=self.bool_filter ), 
                code="""
                const nrows = source.get_length();
                const data = source.data;
                const low = cb_obj.value[0];
                const high = cb_obj.value[1];
                const boo = [];
                const lit = data["LitScore"];
                for (let i = 0; i < nrows; i++)
                { 
                    const lt = lit[i];
                    if ((lt <= high) && (lt >= low))
                    {
                        boo.push(true);
                    } 
                    else
                    {
                        boo.push(false);
                    }  
                }
                filt.booleans = boo;
                source.change.emit(); """))

    def impact_update(self): # # attr, old, new
        self.impact.js_on_change('value', CustomJS(args=dict(source=self.source, 
                                mytable=self.myTable, 
                                impact=self.impact, 
                                filt=self.group_filter), 
                                code="""
                                const imp = cb_obj.value;
                                mytable.disabled = true;
                                filt.column_name = "Impact";
                                filt.group = imp;
                                mytable.disabled = false;
                                source.change.emit();
                                """))
        

    ### setup callbacks
    def gene_update(self): # attr, old, new
        self.source.selected.js_on_change('indices', CustomJS(args=dict(source=self.source, 
                                                              bar=self.source_bar,
                                                              symbol=self.symbol, 
                                                              pval=self.pval, 
                                                              litscore=self.litscore,
                                                              message = self.message,
                                                              mesh_terms= self.source_meshs,
                                                              meshid=self.meshid,
                                                              expr = self.exprs,
                                                              codon=self.codon,
                                                              source_codon=self.source_codon), 
                                                              code="""
            var columns = source.columns(); 
            const nrows = source.get_length();
            const selected_index = source.selected.indices[0];
            const s = source.data["GeneName"][selected_index];
            const start = s.indexOf(">");
            const end = s.indexOf("</");
            symbol.value = s.slice(start+1, end);
            pval.value = source.data["Pvalue"][selected_index].toString();
            litscore.value = source.data["LitScore"][selected_index].toString();
            codon.value = source.data["Impact"][selected_index].toString();
            //console.log(mesh_terms.data);
            //console.log(meshid.value);
            var pid = "PMIDs_" + mesh_terms.data[meshid.value][0];
            //console.log(pid);
            var papers = source.data[pid][selected_index]; // string
            //console.log(papers);
            var pid_html = [];
            if ((papers != "Indirect") || (papers != "Unknown_Gene"))
            {
                //console.log(pid);
                const pax = papers.split(",");
                for (var j=0; j < pax.length; j++)
                {
                    var pa = pax[j];
                    if (pa == "Indirect") continue;
                    var ps = `<a href="https://www.ncbi.nlm.nih.gov/research/pubtator/index.html?view=docsum&query=${pa}" target="_blank">${pa}</a>`;
                    pid_html.push(ps);
                }
                message.text = "<h3>PubMedIDs:</h3><p>" + pid_html.join(",") ; // + "</p><h3>Gene Expression:</h3>expr"
            }    
            const pattern = source.data["Pattern"][selected_index];
            const pat_colors = [];

            var dict_color = {"0": "#3B4992",
                        "1": "#EE0000",
                        "2": "#008B45",
                        "3": "#631879",
                        "4": "#9467bd", 
                        "5": "#008280",
                        "6": "#BB0021",
                        "7": "#5F559B",
                        "8": "#A20056",
                        "9": "#808180",
                        "?": "#ffffff" }
            for (var i = 0; i < pattern.length; i++) {
                const p = pattern.charAt(i);
                pat_colors.push(dict_color[p]);
            }
            bar.data["colors"] = pat_colors;
            source.change.emit();
            //symbol.change.emit();
            //litscore.change.emit();
            //pval.change.emit();
            //codon.change.emit();
            bar.change.emit();
        """))

    def mesh_update(self):
        self.meshid.js_on_change('value', CustomJS(args=dict(source=self.source, 
                                            sca=self.sca, 
                                            meshid=self.meshid,
                                            message=self.message, 
                                            mesh_terms = mesh_terms),
                                            code="""
            const mesh = meshid.value;
            var columns = source.columns(); 
            const nrows = source.get_length();
            if (mesh == "EffectSize")
            {
                sca.title.text = mesh;
                source.data["LitScore"] = source.data[mesh];
                // sca.yaxis.axis_label = "Effect Size"

            } 
            else
            {
                sca.title.text = mesh; // mesh_terms[mesh] + " : " + mesh; 
                var mesh1 = "MeSH_" + mesh_terms[mesh];
                const pubmed = "PMIDs_" + mesh_terms[mesh];
                const mesh_columns =[];
                for (let i = 0; i < columns.length; i ++)
                {
                    if (columns[i].startsWith("MeSH"))
                    {
                        mesh_columns.push(columns[i]);
                    }
                }
                if (mesh_columns.length < 1)
                {
                    return;
                }
                if (!mesh_columns.includes(mesh1))
                {
                    mesh1 = mesh_columns[0];
                    message.text = "<p>Sorry, input MeSH not found! <br> Selected: " + mesh1 + "</p>";
                    return;
                }
                source.data["LitScore"] = source.data[mesh1]; 
                source.data["PubMed"] = source.data[pubmed];
                //sca.yaxis.axis_label = "Literature Score";
            }
            source.change.emit();
            sca.change.emit();"""))

    def build(self,):    
        # datatable
        self.data_update()
        self.gene_update()
        self.impact_update()
        self.mesh_update()
        self.slider_update()
        # for single_control in self.controls.values():
        #     single_control.js_on_change('value', self.callback)
        self.button.js_on_click(CustomJS(args=dict(source=self.myTable.source, dataset=self.dataset.value),
                                    code=open(os.path.join(os.path.dirname(__file__), "static/js/download.js")).read()))

        ## set up layout
        inputs = row(self.dataset, self.meshid, self.impact, )
        # curdoc.add_root(inputs)
        figs = row(self.bar, self.sca)
        #figs = row(robj1, robj2)
        figs2 = column(figs, self.myTable)
        # curdoc.add_root(figs2)
        robj4 = column(self.button, self.symbol, self.pval, self.codon, self.slider, self.message, self.exprs)
        o = row(figs2, robj4)
        layout = column(inputs, o)

        return layout

# dataset id
class GNNHapGraph:
    def __init__(self, data_dir, dataset=None):
        ### setup data and plots
        ## data
        DATASETS = glob.glob(data_dir+"**/*.gnn.txt")
        DATASETS2 = [os.path.basename(d) for d in DATASETS]
        if dataset is None:
            dataset = DATASETS2[0]
        if not dataset.endswith(".gnn.txt"):
            dataset = dataset + ".gnn.txt"
        self.DATA_DIR = data_dir
        df = pd.read_table(os.path.join(data_dir, dataset))
        self.source = ColumnDataSource(df)
        self.dataset = AutocompleteInput(completions=DATASETS2, #value="MPD_26711-f_Indel",width=550,
                                title="Dataset:", value=dataset, width=550,)
        # mesh terms options
        self.meshid = Select(title="Select MeSH:", value="", 
                             options=df['MeSH_Terms'].unique().tolist(), width=300,) # need to update dynamically
        # download
        self.button = Button(label="Download Table", button_type="success",  width=250)
        # groupfilter options
        # message box
        self.message = Div(text="""<h3> Message: <br> </h3>""", width=300, height=150)
        self.slider = RangeSlider(title="LitScore Range", start=0.0, end=1.0, value=(0.5, 1.0), step=0.01)
        # data view
        self.bool_filter = BooleanFilter()
        self.group_filter = GroupFilter()
        self.view = CDSView(source=self.source, filters=[self.bool_filter, self.group_filter ])

        ## Datatable
        self._columns = ["#GeneName",	"MeSH", "HumanEntrezID", "MeSH_Terms", "LiteratureScore", "PubMedID"] 
        columns = [ TableColumn(field=c, title=c, formatter=HTMLTemplateFormatter()) for c in self._columns ] # skip index                      
        self.myTable = DataTable(source=self.source, columns=columns, 
                            width =800, height = 600, index_position=0,
                            editable = True, view=self.view, name="DataTable",
                            sizing_mode="stretch_width") # autosize_mode="fit_viewport"
    def data_update(self):
        self.dataset.js_on_change('value',  CustomJS(args=dict(source=self.source, 
                                bool_filt=self.bool_filter,
                                meshid = self.meshid,
                                group_filt = self.group_filter), 
                                code="""
                                const uid = cb_obj.value;
                                console.log("The URL of this page is: " + window.location.href);
                                var current_url =  window.location.href.split("/");
                                var url = uid;
                                if (current_url[current_url.length -1] == "graph")
                                {
                                    url = "GRAPH_DATA/" + uid;
                                }
                                bool_filt.booleans = null;
                                group_filt.group = null;
                                var xmlhttp = new XMLHttpRequest(); // read file on the server side 
                                // Define a callback function
                                xmlhttp.onload = function() {
                                    if (xmlhttp.readyState == 4 && xmlhttp.status == 200) 
                                    {
                                        // Here you can use the Data
                                        var new_data = JSON.parse(xmlhttp.responseText);
                                        // console.log(new_data);
                                        source.data = new_data['data'];
                                        var new_set = new Set( new_data['data']['MeSH_Terms'] );
                                        var new_arr = Array.from(new_set);   
                                        meshid.options = new_arr;
                                        meshid.value = new_arr[0];
                                        console.log(new_arr);        
                                        url = new_data['url'];
                                    }        
                                }
                                // Send a request
                                xmlhttp.open("GET", url, true);
                                xmlhttp.send();
                                source.change.emit();"""))

    def slider_update(self): # attr, old, new
        self.slider.js_on_change('value', 
        CustomJS(args=dict(source=self.source, 
                           filt=self.bool_filter ), 
                code="""
                const nrows = source.get_length();
                const data = source.data;
                const low = cb_obj.value[0];
                const high = cb_obj.value[1];
                const boo = [];
                const lit = data["LiteratureScore"];
                for (let i = 0; i < nrows; i++)
                { 
                    const lt = lit[i];
                    if ((lt <= high) && (lt >= low))
                    {
                        boo.push(true);
                    } 
                    else
                    {
                        boo.push(false);
                    }  
                }
                filt.booleans = boo;
                source.change.emit(); """))


    def mesh_update(self):
        self.meshid.js_on_change('value', CustomJS(args=dict(source=self.source, 
                                mytable=self.myTable, 
                                filt=self.group_filter), 
                                code="""
                                const imp = cb_obj.value;
                                filt.column_name = "MeSH_Terms";
                                filt.group = imp;
                                source.change.emit();
                                """))
    def build_graph(self):
        # datatable
        self.data_update()
        self.mesh_update()
        self.slider_update()
        self.button.js_on_click(CustomJS(args=dict(source=self.myTable.source, 
                                                   dataset=self.dataset.value),
                                    code=open(os.path.join(os.path.dirname(__file__), "static/js/download.js")).read()))

        ## set up layout
        o = row(self.meshid, self.button)
        inputs = column(self.dataset, o, self.slider)
        widgt = row(inputs, self.message)
        # widgt = row(inputs, download)
        #figs = row(robj1, robj2)
        layout = column(widgt, self.myTable)
        return layout

    


