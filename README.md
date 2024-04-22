This is a general-purpose time-series data cleaning user interface.
It's main purpose is to perform the data cleaning task of the preparation of ML training data.

A video guide is available at: https://youtu.be/vf5UN73QOW4

The functions of the UI include:
1) Calculation of periodicities within each column of a dataset
2) Missing value imputation
3) Data transformation into trend, seasonal, and residual components
4) Data outlier detection for each of these components
5) Data anomaly rectification
6) Various visualization methods interspersed along the way

The possible outputs obtained from the UI:
1) Missing value imputation results
2) Decomposition into trend, seasonal, and residual results
3) Outlier detection results
4) Anomaly rectification results

All of these outputs should be sufficient input for the training of ML models

Running the UI can be done through VS-code, where:
1) The python and shiny for python extensions need to be installed to vscode
2) A correct interpreter has to be selected, one which includes all the necessary packages listed on the first cell of the app.py file
3) After this, the app.py file can be started, and can be opened in the browser via vscode
