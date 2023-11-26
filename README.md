# Regression on the tabular data

### Setup environment

1. Create an environment with python3 (tested with python 3.11.5) using conda, venv, etc.
2. Install requirements using the following command: ```pip install -r requirements.txt```

### EDA

To look at EDA you should run the following command ```jupyter-notebook``` from the project root and then access
EDA.ipynb file from the web interface.

### Train

Train on your own data is possible using __train.py__ script. Your data should be organized in the same way as 
_data/train.csv_. 

```python train.py --data-path data/train.csv```

After some time trained pipeline will be saved to the "trained_pipeline.pkl" file. 
If you would like to change the default path where to save the trained pipeline you can pass __--trained-pipe-path__ 
to the train script like the following:

```python train.py --data-path data/train.csv --trained-pipe-path my_custom_path.pkl```


### Predict

To predict with a trained model you can use __predict.py__ script. Usage example:

```python predict.py --pipe-path trained_pipeline.pkl --data-path data/hidden_test.csv --result-path result.csv```

Results will be saved to result.csv via pandas.

If you want to run prediction on your own data it should be structured as _data/hidden_test.csv_
