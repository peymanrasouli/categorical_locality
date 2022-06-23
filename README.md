# Categorical Locality

This repository contains the implementation source code of the following paper:

Interpreting Categorical Data Classifiers using Explanation-based Locality

# Setup
1- Clone the repository using HTTP/SSH:
```
git clone https://github.com/peymanrasouli/categorical_locality
```
2- Create a conda virtual environment:
```
conda create -n categorical_locality python=3.6
```
3- Activate the conda environment: 
```
conda activate categorical_locality
```
4- Standing in categorical_locality directory, install the requirements:
```
pip install -r requirements.txt
```
# Reproducing the results
To reproduce the explanation results of categorical_locality method vs. baselines with:

1- Linear Regression as interpretable model run:
```
python local_explanation_lr.py
```
2- Decision Tree as interpretable model run:
```
python local_explanation_dt.py
```
