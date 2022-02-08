# Bokeh app for Data Visualization of GNNHap output

This is a bokeh application for exploring GNNHap/HBCGM results interactively and download results.



## Run 

Before you start, you need to edit the `DATA_DIR` path in the `main.py`.

**NOTE**: This app will only search files endswith "results.txt" or "results.mesh.txt".  
Modify the file path pattern if you'd like to use your own data.

Example result files could be found in the `example/PeltzData` folder.


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


## A Snapshot 
![GNNHap](static/images/GNNHap.png)


## Dev
see guide [here](https://docs.bokeh.org/en/2.4.1/docs/user_guide/server.html)