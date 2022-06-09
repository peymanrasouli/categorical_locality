# Categorical Locality

This repository contains the implementation source code of the following paper:

Explanation-based Locality for Explaining Categorical Data Classifiers

<!-- [Meaningful Data Sampling for a Faithful Local Explanation Method](https://link.springer.com/chapter/10.1007/978-3-030-33607-3_4)

BibTeX:

    @inproceedings{rasouli2019meaningful,
                   title={Meaningful Data Sampling for a Faithful Local Explanation Method},
                   author={Rasouli, Peyman and Yu, Ingrid Chieh},
                   booktitle={International Conference on Intelligent Data Engineering and Automated Learning},
                   pages={28--38},
                   year={2019},
                   organization={Springer}
    } -->

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
