# Bokeh app for Data Visualization of HBCGM output

This is a bokeh application for exploring HBCGM results interactively and download results.

## Run 

set the `DATA_DIR` to your GNNHap output folder in the `main.py`, then run the following command:


debug  
```shell
## --dev autoreload files 
bokeh serve --show webapp --allow-websocket-origin=peltz-app-03:5006 --dev webapp/*.py --log-level=debug
```

deployment  
```
bokeh serve --show webapp --allow-websocket-origin=peltz-app-03:5006
```

view at: peltz-app-03:5006

**Note**:
HBCGM results must be prioritzation by MeSH terms (GNNHap) first, then run this app


![GNNHap](static/images/GNNHap.png)


## Dev
see guide [here](https://docs.bokeh.org/en/2.4.1/docs/user_guide/server.html)