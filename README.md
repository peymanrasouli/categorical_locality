# Categorical Locality

This repository contains the implementation source code of the following paper:

[Interpreting Categorical Data Classifiers using Explanation-based Locality](https://ieeexplore.ieee.org/document/10031039)

BibTeX:
    
    @inproceedings{rasouli2022interpreting,
                  title={Interpreting Categorical Data Classifiers using Explanation-based Locality},
                  author={Rasouli, Peyman and Yu, Ingrid Chieh and Jim{\'e}nez-Ruiz, Ernesto},
                  booktitle={2022 IEEE International Conference on Data Mining Workshops (ICDMW)},
                  pages={163--170},
                  year={2022},
                  organization={IEEE}
    }

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
