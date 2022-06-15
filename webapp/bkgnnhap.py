import os, glob
import numpy as np
import pandas as pd
import networkx as nx

from bokeh.plotting import figure, from_networkx
from bokeh.models import ColumnDataSource, TableColumn, DataTable, HTMLTemplateFormatter, CellFormatter
from bokeh.models import RangeSlider, CDSView, BooleanFilter, GroupFilter, CustomJS, Legend
from bokeh.models.widgets import Select, TextInput, AutocompleteInput, Div, Button
from bokeh.layouts import column, row

from bokeh.models import Range1d, Scatter, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges, LabelSet
from bokeh.models import GraphRenderer, StaticLayoutProvider
from bokeh.palettes import Spectral3, Spectral8
# from bokeh.transform import linear_cmap

from helpers import gene_expr_order, codon_flag, mesh_terms
from helpers import load_ghmap, get_html_links


class Graph:
    """
    graph visualization using bokeh
    """
    def __init__(self, graph_data_dict=None):
        self._graph() # build graph

        ## init graph
        if graph_data_dict is None:
            self._graph_init()
        else:
            self.graph.node_renderer.data_source.data =  graph_data_dict['node_data']
            self.graph.edge_renderer.data_source.data =  graph_data_dict['edge_data']
            self.graph.layout_provider =  StaticLayoutProvider(graph_layout=graph_data_dict['graph_layout'])
    def _graph(self):
        """
        build graph components
        """
        #Choose colors for node and edge highlighting
        node_highlight_color = 'white'
        edge_highlight_color = 'black'

        #Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
        size_by_this_attribute = 'node_size_adjust'
        color_by_this_attribute = 'node_type_color'
        #Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [
            ("Name", "@node_name"),
            ("NodeType", "@node_type"),
            ("Degree", "@node_degree"),
                #("NodeColor", "$color[swatch]:node_type_color"),
        ]
        #Create a plot — set dimensions, toolbar, and title
        plot = figure(tooltips = HOVER_TOOLTIPS, frame_width=550, frame_height=450,
                    output_backend="svg", #"webgl",  
                    tools="pan,wheel_zoom,save,reset", 
                    active_scroll='wheel_zoom',
                    x_range=Range1d(-10.1, 10.1),
                    y_range=Range1d(-10.1, 10.1),
                    x_axis_location=None, 
                    y_axis_location=None,
                    toolbar_location="above",
                    title="Gene-MeSH 1-hop Subgraph")
        plot.toolbar.logo = None
        plot.grid.visible = False 
        ## or remove grid by
        # plot.xgrid.grid_line_color = None
        # plot.ygrid.grid_line_color = None
        graph = GraphRenderer()
        #Set node sizes and colors according to node degree (color as category from attribute)
        graph.node_renderer.glyph = Scatter(size=size_by_this_attribute, 
                                            fill_color=color_by_this_attribute, 
                                            marker="node_marker")
        #Set node highlight colors
        graph.node_renderer.hover_glyph = Scatter(size=size_by_this_attribute, 
                                                  fill_color=node_highlight_color, 
                                                  line_width=2)
        graph.node_renderer.selection_glyph = Scatter(size=size_by_this_attribute, 
                                                      fill_color=node_highlight_color, 
                                                      line_width=2)
        #Set edge opacity and width
        graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, #line_color="edge_color",
                                              line_width="edge_weight_adjust")
        #Set edge highlight colors
        graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, 
                                                        line_width=2)
        graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color,  
                                                    line_width=2)

        #Highlight nodes and edges
        graph.selection_policy = NodesAndLinkedEdges()
        graph.inspection_policy = NodesAndLinkedEdges() # EdgesAndLinkedNodes()# 
        plot.renderers.append(graph)
        # ##Add Node labels
        # node x, y positions
        labels = LabelSet(x='x', y='y', text='node_name', 
                          source=graph.node_renderer.data_source, 
                          background_fill_color='white', 
                          text_font_size='12px', 
                          background_fill_alpha=.7)
        plot.renderers.append(labels)
        self.graph = graph 
        self.graphplot = plot

    def _graph_init(self):
        """
        only used for inititiation when no input graph data are given
        """
        N = 8
        node_indices = list(range(N))
        # generate ellipses based on the ``node_indices`` list
        circ = [i*2*np.pi/8 for i in node_indices]

        # create lists of x- and y-coordinates
        x = [np.cos(i) for i in circ]
        y = [np.sin(i) for i in circ]
        # assign a palette to ``fill_color`` and add it to the data source
        node_indices = ['C'+str(i) for i in node_indices]
        self.graph.node_renderer.data_source.data = dict(
            index=node_indices,
            node_type_color=Spectral8,
            node_size_adjust = [10]*N,
            #node_type_color=['black']*N,
            node_marker=['circle']*N, 
            node_name = [str(i) for i in node_indices],
            x=x, 
            y=y)

        # add the rest of the assigned values to the data source
        self.graph.edge_renderer.data_source.data = dict(
            start=['C0']*N,
            end=node_indices,
            edge_weight_adjust=[2]*N)
        # convert the ``x`` and ``y`` lists into a dictionary of 2D-coordinates
        # and assign each entry to a node on the ``node_indices`` list
        graph_layout = dict(zip(node_indices, zip(x, y)))
        # use the provider model to supply coourdinates to the graph
        self.graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)




class GNNHapResults(Graph):
    """
    visualitzation of GNNHap results
    """
    def __init__(self, data_dir, dataset=None, graph_data_dict=None):
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
        self.haploblock = Div(text="""<h3> Selected Haplotypeblock: </h3>""", width=300, height=800)

        self.slider = RangeSlider(title="LitScore Range", start=0.0, end=1.0, value=(0.5, 1.0), step=0.01)
        # data view
        self.bool_filter = BooleanFilter()
        self.group_filter = GroupFilter()
        self.view = CDSView(source=self.source, filters=[self.bool_filter, self.group_filter ])

        ## Datatable
        columns = ['GeneName', 'CodonFlag','Haplotype','EffectSize', 'Pvalue', 'FDR',
                'PopPvalue', 'PopFDR', 'Position', 'LitScore','PubMed'] # 'Chr', 'ChrStart', 'ChrEnd'
        columns = [ TableColumn(field=c, title=c, formatter=HTMLTemplateFormatter() 
                                if c in ['Haplotype','GeneName', 'PubMed', 'Position'] else CellFormatter()) for c in columns ] # skip index  
        self.columns = columns                     
        self.myTable = DataTable(source=self.source, columns=columns, width =1200, height = 400, index_position=0,
                            editable = True, view=self.view, name="DataTable",sizing_mode="stretch_width") # autosize_mode="fit_viewport"
        

        # download
        self.button = Button(label="Download Table", button_type="success")
        self.barplot()
        self.scatterplot()
        super(GNNHapResults, self).__init__(graph_data_dict=graph_data_dict)
        # self.data_update(dataset)

    def _haploblock(self,):
        _header = [f"<th>{s}</th>" for s in self.source_bar.data['strain']]
        _header = "<thead><tr>" + "".join(_header) + "</tr></thead>"
        
        _message = """
          <tbody>
            {% for k, v in trait_data.items() %}
              <tr>
                  <td>{{ k }}</td>
                  <td>{{ strains[k] }}</td>
                  <td> {{ v }}</td>
              </tr>
              {% endfor %}
          </tbody>
        </table>
        """
        table = "<table>" + _header + _message +"</table"

    ## bar plot
    def barplot(self,):
        bar = figure(plot_width=600, plot_height=300, # x_range=strains, 
                title="Dataset", 
                toolbar_location="above",
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

        sca = figure(frame_width=550, frame_height=450,  ## note: frame_* is the pixls of axes (not including axis, legend) region
                    tools="pan,box_zoom,reset,lasso_select,save",
                    output_backend="svg",  
                    #active_drag="lasso_select",
                    toolbar_location="above",
                    tooltips=TOOLTIPS, name='Scatter',
                    x_axis_label= "Genetic:  - log10 Pvalue",
                    y_axis_label= "Literature Score")

        sca.add_layout(Legend(), 'right') # put legend outside
        #sca.legend.orientation = "horizontal"
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
                        url = "HBCGM_DATA/" + uid; // update url to get json data
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
                                console.log(pubc);
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
                                                              haploblock = self.haploblock,
                                                              codon=self.codon,
                                                              source_codon=self.source_codon,
                                                              graph=self.graph,
                                                              genemeshplot=self.graphplot), 
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
                message.text = "<div><h3>PubMed Links:</h3><pre>" + pid_html.join(",") +"</pre></div>" ;
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

            var gene_id = source.data["HumanEntrezID"][selected_index];
            var mesh_id = mesh_terms.data[meshid.value][0];
            // get json data
            $.getJSON(`/graph_process/${gene_id}_${mesh_id}`,
              function(data) {
                // update graph data here
                graph.node_renderer.data_source.data = data['node_data'];
                graph.edge_renderer.data_source.data = data['edge_data'];
                graph.layout_provider.graph_layout = data['graph_layout'];
              });
            graph.change.emit();
            genemeshplot.change.emit();
            source.change.emit();
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
        figs = row(self.graphplot, self.sca)
        #figs = row(robj1, robj2)
        # curdoc.add_root(figs2)
        info = column(self.button, self.symbol, self.pval, self.codon, self.slider) # self.exprs
        
        layout = column(inputs, row(self.bar, info, self.message), figs, self.myTable)

        return layout

class GNNHapGraph(Graph):
    def __init__(self, data_dir, dataset=None, graph_data_dict=None):
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
        df['GeneName'] = df['#GeneName'].apply(get_html_links)
        self.source = ColumnDataSource(df)
        self.dataset = AutocompleteInput(completions=DATASETS2, #value="MPD_26711-f_Indel",width=550,
                                title="Dataset:", value=dataset, width=550,)
        # self.gene = AutocompleteInput(completions=DATASETS2, #value="MPD_26711-f_Indel",width=550,
        #                         title="Dataset:", value=dataset, width=550,)
        # mesh terms options
        self.meshid = Select(title="Select MeSH:", value="", 
                             options=df['MeSH_Terms'].unique().tolist(), width=300,) # need to update dynamically
        # download
        self.button = Button(label="Download Table", button_type="success",  width=250)
        # groupfilter options
        # message box
        self.message = Div(text="""<h3> Message: <br> </h3>""", width=300, height=150)
        self.slider = RangeSlider(title="Literature Score Range", start=0.0, end=1.0, value=(0.5, 1.0), step=0.01)
        # data view
        self.bool_filter = BooleanFilter()
        self.group_filter = GroupFilter()
        self.view = CDSView(source=self.source, filters=[self.bool_filter, self.group_filter ])

        ## Datatable
        self._columns = ["GeneName", "HumanEntrezID", "MeSH_Terms", "LiteratureScore", "PubMedID", "IND"] 
        columns = [ TableColumn(field=c, title=c, formatter=HTMLTemplateFormatter()) for c in self._columns ] # skip index                      
        self.myTable = DataTable(source=self.source, columns=columns, 
                            width =600, height = 500, index_position=0,
                            editable = True, view=self.view, name="DataTable",
                            sizing_mode="stretch_width") # autosize_mode="fit_viewport"
        ## init gene mesh graph
        super(GNNHapGraph,self).__init__(graph_data_dict)
        

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
                                    url = "GRAPH_DATA/" + uid; // update url when select dataset with different location
                                }
                                bool_filt.booleans = null;
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
                                        group_filt.group = meshid.value;   
                                        source.data['GeneName'] = source.data['#GeneName'];

                                        for (var j=0; j < source.data['GeneName'].length; j++)
                                        {
                                            var gn = source.data['GeneName'][j];
                                            var ge = source.data['HumanEntrezID'][j];
                                            var ps = `<a href="https://www.ncbi.nlm.nih.gov/gene/${ge}" target="_blank">${gn}</a>`;
                                            source.data['GeneName'][j] = ps;
                                        }
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

    ### setup callbacks
    def table_gene_update(self): # attr, old, new
        self.source.selected.js_on_change('indices', CustomJS(args=dict(source=self.source, 
                                                              message = self.message,
                                                              graph=self.graph,
                                                              genemeshplot=self.graphplot), 
                                                              code="""
            const selected_index = source.selected.indices[0];
            var papers = source.data["PubMedID"][selected_index]; 
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
                message.text = "<pre><h3>PubMed Links:</h3><p>" + pid_html.join(",") +"</p></pre>"; 
            }        
            var gene_id = source.data["HumanEntrezID"][selected_index];
            var mesh_id = source.data["MeSH"][selected_index];
            // get json data
            $.getJSON(`/graph_process/${gene_id}_${mesh_id}`,
              function(data) {
                // update graph data here
                graph.node_renderer.data_source.data = data['node_data'];
                graph.edge_renderer.data_source.data = data['edge_data'];
                graph.layout_provider.graph_layout = data['graph_layout'];
              });
            source.change.emit();
            graph.change.emit();
            genemeshplot.change.emit();
        """))

    def build_graph(self):
        # datatable
        self.data_update()
        self.mesh_update()
        self.slider_update()
        self.table_gene_update()
        self.button.js_on_click(CustomJS(args=dict(source=self.myTable.source, 
                                                   dataset=self.dataset.value),
                                    code=open(os.path.join(os.path.dirname(__file__), "static/js/download.js")).read()))

        ## set up layout
        o = row(self.meshid, self.button)
        inputs = column(self.dataset, o, self.slider)
        widgt = row(inputs, self.message)
        # widgt = row(inputs, download)
        #figs = row(robj1, robj2)
        tab = row(self.myTable, self.graphplot)
        layout = column(widgt, tab)
        return layout

    

class SubGraph(Graph):
    def __init__(self, gene_mesh_graph, graph_data_dict=None):
        """
        gene_mesh_graph: networkx graph 
        """
        ### setup data and plots
        ## data
        self.gene_mesh_graph = gene_mesh_graph
        self.nid2name = {}
        self.nname2id = {}
        genes = []
        mesh = []
        for node,node_attr in gene_mesh_graph.nodes(data=True):
            self.nname2id[node_attr['node_name']] = [node]
            if node.startswith("D"):
                mesh.append(node_attr['node_name'])
            else:
                genes.append(node_attr['node_name'])
        self.source_name2id = ColumnDataSource(data=self.nname2id)
        self.gene = AutocompleteInput(completions=sorted(genes), #value="MPD_26711-f_Indel",width=550,
                                title="Human Gene Symbol:", value="SIRT1", width=300,)
        self.meshid = AutocompleteInput(completions=sorted(mesh), #value="MPD_26711-f_Indel",width=550,
                                title="MeSH Terms:", value="Hearing Disorders", width=300,)
        # message box
        self.message = Div(text="""<h3> PMIDs: <br> </h3>""", width=300, height=200)
        ## init gene mesh graph
        super(SubGraph,self).__init__(graph_data_dict)        
    def mesh_update(self):
        self.meshid.js_on_change('value', CustomJS(args=dict(gene=self.gene, 
                                graph=self.graph,
                                name2id = self.source_name2id,
                                genemeshplot=self.graphplot,
                                message = self.message,
                                ), 
                                code="""
            const mesh_name = cb_obj.value;
            const mesh_id = name2id.data[mesh_name][0];
            var gene_name = gene.value;
            var gene_id = name2id.data[gene_name][0];
            // get json data
            $.getJSON(`/graph_process/${gene_id}_${mesh_id}`,
              function(data) {
                // update graph data here
                graph.node_renderer.data_source.data = data['node_data'];
                graph.edge_renderer.data_source.data = data['edge_data'];
                graph.layout_provider.graph_layout = data['graph_layout'];

                const papers = data['pmid'];
                //console.log(pid);
                var pid_html = [];
                for (var j=0; j < papers.length; j++)
                {
                    var pa = papers[j];
                    if (pa == "Indirect") continue;
                    var ps = `<a href="https://www.ncbi.nlm.nih.gov/research/pubtator/index.html?view=docsum&query=${pa}" target="_blank">${pa}</a>`;
                    pid_html.push(ps);
                }
                message.text = "<div><h3>PubMed Links:</h3><pre>" + pid_html.join(",") +"</pre></div>" ;
              });
            graph.change.emit();
            genemeshplot.change.emit();
                                """))
    def gene_update(self):
        self.gene.js_on_change('value', CustomJS(args=dict(mesh=self.meshid, 
                                graph=self.graph,
                                name2id = self.source_name2id,
                                genemeshplot=self.graphplot,
                                message = self.message,
                                ), 
                                code="""
            const gene_name = cb_obj.value;
            const mesh_name = mesh.value;
            console.log(mesh_name);
            const mesh_id = name2id.data[mesh_name][0];
            var gene_id = name2id.data[gene_name][0];
            // get json data
            $.getJSON(`/graph_process/${gene_id}_${mesh_id}`,
              function(data) {
                // update graph data here
                graph.node_renderer.data_source.data = data['node_data'];
                graph.edge_renderer.data_source.data = data['edge_data'];
                graph.layout_provider.graph_layout = data['graph_layout'];
                const papers = data['pmid'];
                //console.log(pid);
                var pid_html = [];
                for (var j=0; j < papers.length; j++)
                {
                    var pa = papers[j];
                    if (pa == "Indirect") continue;
                    var ps = `<a href="https://www.ncbi.nlm.nih.gov/research/pubtator/index.html?view=docsum&query=${pa}" target="_blank">${pa}</a>`;
                    pid_html.push(ps);
                }
                message.text = "<div><h3>PubMed Links:</h3><pre>" + pid_html.join(",") +"</pre></div>" ;
              });
            graph.change.emit();
            genemeshplot.change.emit();
                                """))

    def build_graph(self):
        # datatable
        self.mesh_update()
        self.gene_update()
        ## set up layout
        o = row(self.meshid, self.gene)
        o1 = row(self.graphplot, self.message)
        inputs = column(o, o1)
        return inputs
