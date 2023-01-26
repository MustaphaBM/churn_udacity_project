# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

# Project Description
The main goal of this project is to predict the customers that are most likely to Churn. To do the predictions we used data that have many features such as credit card information and 

# Files and data description
The tree below shows how the files and directory are organized.
```
    |── data
    │   └── bank_data.csv
    ├── images
    │   ├── eda
    │   │   ├── churn_distribution.png
    │   │   ├── customer_age_distribution.png
    │   │   ├── heatmap_distribution.png
    │   │   ├── marital_status_distribution.png
    │   │   └── total_trans_ct_distribution.png
    │   └── results
    │       ├── feature_importance.png
    │       ├── logistic_results.png
    │       ├── rf_results.png
    │       └── roc_curves.png
    ├── logs
    │   └── churn_library.log
    ├── models
    │   ├── logistic_model.pkl
    │   └── rfc_model.pkl
    ├── churn_library.py
    ├── churn_notebook.ipynb
    ├── churn_script_logging_and_tests.py
    ├── Guide.ipynb
    ├── README.md
    ├── requirements_py3.6.txt
    └── requirements_py3.8.txt
```
There are 2 main files that contain all the code 
- churn_library.py: contains all the code of the pipeline
- churn_script_logging_and_tests.py: contains logging and unit test for all the code written in ```churn_library.py```

# Running Files
### Step 1 : create the environment 
    conda create --name udacity_churn_env python=3.8 
    conda activate udacity_churn_env

### step 2 : install the requirements
    conda install --file requirements.txt

### step 3 : run the pipeline
    python churn_library.py

### step 4 : [Optional] run Tests and logs
    python churn_script_logging_and_tests.py
