"""This is churn test module

This module implements testing and logging for churn
module functions

Author: Mustapha Benhaj Miniaoui
Data: January 19th, 2023
"""

import logging
from pathlib import Path

import joblib

import churn_library as cls

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        data = cls.import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err
    return data


def test_eda(data):
    """
    test perform eda function
    """
    cls.perform_eda(data)
    eda_directory = Path("images") / "eda"
    generated_files = list(eda_directory.iterdir())
    generated_files_names = [plot_file.name for plot_file in generated_files]
    expected_files = ['marital_status_distribution.png',
                      'churn_distribution.png',
                      'customer_age_distribution.png',
                      'total_trans_ct_distribution.png',
                      'heatmap_distribution.png']
    for plot_file in expected_files:
        if plot_file in generated_files_names:
            logging.info("SUCCESS - %s was generated successfully", plot_file)
        else:
            logging.error("File %s not found", plot_file)
            raise FileNotFoundError

    assert len(generated_files) == 5


def test_encoder_helper(data):
    """
    test encoder helper
    """

    categorical_columns = ["Gender",
                           "Education_Level",
                           "Marital_Status",
                           "Income_Category",
                           "Card_Category"]
    response = "Churn"
    try:
        encoded_data = cls.encoder_helper(
            data, categorical_columns, response)
        logging.info("SUCCESS - dataframe encoded successfully")
    except KeyError as err:
        logging.error("ERROR - Column doesn't exist in dataframe columns")
        raise err
    expected_added_columns = ["Gender_Churn",
                              "Education_Level_Churn",
                              "Marital_Status_Churn",
                              "Income_Category_Churn",
                              "Card_Category_Churn"]
    try:
        for column in expected_added_columns:
            assert column in encoded_data.columns
    except AssertionError as err:
        logging.error("There is missing columns in result dataframe")
        raise err


def test_perform_feature_engineering(data):
    """
        test perform_feature_engineering
    """
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    try:
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            data, keep_cols)
        logging.info("SUCCESS - Features were generated successfully")
    except KeyError as err:
        logging.error("ERROR - Not all passed columns exist in the dataframe")
        raise err
    try:
        assert x_train.shape[1] == x_test.shape[1] == len(
            keep_cols)
        logging.info("SUCCESS - the engineered data has the right shape")
    except AssertionError as err:
        logging.error("ERROR - Engineered data does not have the right shape")
        raise err
    return x_train, x_test, y_train, y_test


def test_train_models(x_train, x_test, y_train, y_test):
    """
    test train_models
    """
    cls.train_models(x_train, x_test, y_train, y_test)

    try:
        joblib.load("./models/logistic_model.pkl")
        joblib.load("./models/rfc_model.pkl")
        logging.info("SUCCESS - Models were trained and save successfully")
    except FileNotFoundError:
        logging.error("ERROR - Models were not trained and saved")
    expected_files = ['roc_curves.png',
                      'feature_importance.png',
                      "rf_results.png",
                      'logistic_results.png']
    results_directory = Path("images/results")
    generated_files = list(results_directory.iterdir())
    generated_files_names = [plot_file.name for plot_file in generated_files]
    for file_name in expected_files:
        if file_name in generated_files_names:
            logging.info(
                "SUCCESS - %s file was generated successfully" %
                file_name)
        else:
            logging.error("ERROR - Missing file")
            raise FileNotFoundError


if __name__ == "__main__":
    dataframe = test_import()
    test_eda(dataframe)
    test_encoder_helper(dataframe)
    x_train_data, x_test_data, y_train_data, y_test_data = test_perform_feature_engineering(
        dataframe)
    test_train_models(x_train_data, x_test_data, y_train_data, y_test_data)
