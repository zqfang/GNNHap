# Bokeh app for Data Visualization of GNNHap output

This is a bokeh application for exploring GNNHap/HBCGM results interactively and download results.


## Installation

- flask
- bokeh
- numpy
- pandas

## Run 



**NOTE**: This app will only search files endswith "results.txt" or "results.mesh.txt".  
Modify the file path pattern if you'd like to use your own data.

Example result files could be found in the `example/PeltzData` folder.


debug  
```shell
## --dev autoreload files 
python app.py
```

view at: peltz-app-03:5006


## A Snapshot 
![GNNHap](static/images/GNNHap.png)


## Dev
see guide [here](https://docs.bokeh.org/en/2.4.1/docs/user_guide/server.html)