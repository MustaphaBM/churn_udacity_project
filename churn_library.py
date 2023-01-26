"""Churn library module
This module implements all needed function for predicting churn
customers.

Author: Mustapha Benhaj Miniaoui
Date: January 19th, 2023
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

sns.set()
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    """

    dataframe = pd.read_csv(pth)
    return dataframe


def create_plot_and_save(dataframe, column, figure_size=(20, 10)):
    """Creates a plot for a column
    input:
        df: pandas dataframe
        column: (str) the name of the column
        figure_size: (tuple) representing (width, height)
    output:
        None
    """
    plt.figure(figsize=figure_size)
    file_path = "images/eda/%s_distribution.png" % column.lower()
    if column in ("Churn", "Customer_Age"):
        dataframe[column].hist()
    elif column == "Marital_Status":
        dataframe[column].value_counts('normalize').plot(kind="bar")
    elif column == 'Total_Trans_Ct':
        sns.histplot(dataframe[column], stat='density', kde=True)
    elif column == "Heatmap":
        sns.heatmap(
            dataframe.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        file_path.replace("distribution", "")
    plt.savefig(file_path)
    plt.close()


def perform_eda(dataframe):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    eda_columns = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        'Total_Trans_Ct',
        "Heatmap"]
    for column in eda_columns:
        create_plot_and_save(dataframe, column)


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        category_groups = dataframe.groupby(category).mean()['Churn']
        new_column_name = "%s_%s" % (category, response)
        dataframe[new_column_name] = [category_groups.loc[val]
                                      for val in dataframe[category]]
    return dataframe


def perform_feature_engineering(dataframe, keep_cols):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    features = pd.DataFrame()
    target = dataframe['Churn']
    features[keep_cols] = dataframe[keep_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    classification_report_data = {
        "Random Forest":
        {
            "Train": [y_train, y_train_preds_rf],
            "Test": [y_test, y_test_preds_rf],
            "plot_name": "rf_results.png",
        },
        "Logistic Regression":
        {
            "Train": [y_train, y_train_preds_lr],
            "Test": [y_test, y_test_preds_lr],
            "plot_name": "logistic_results.png",
        }
    }

    for plot_title, plot_data in classification_report_data.items():
        plt.rc('figure', figsize=(8, 8))
        plt.text(0.01, 1.0, str("%s %s" % (plot_title, "Train")), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01,
                 0.05,
                 str(classification_report(*plot_data["Train"])),
                 {'fontsize': 10},
                 fontproperties='monospace')
        plt.text(0.01, 0.6, str("%s %s" % (plot_title, "Test")), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01,
                 0.7,
                 str(classification_report(*plot_data["Test"])),
                 {'fontsize': 10},
                 fontproperties='monospace')
        plt.axis('off')
        plt.savefig("./images/results/%s" % plot_data["plot_name"])
        plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    plt.title("Feature Importance")
    plt.ylabel('Importance')

    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def prepare_roc_plots(lrc_model, cv_rfc_model, x_test, y_test):
    """Prepares roc curve plot

    input:
        lrc_model: model object
        cv_rfc_model: model object containing feature_importances_
        x_test: (_type_): _description_
        y_test (_type_): _description_
    output :
        None
    """
    lrc_plot = plot_roc_curve(lrc_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    _ = plot_roc_curve(
        cv_rfc_model.best_estimator_,
        x_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/roc_curves.png")
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Create estimators
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Train models
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    # Save models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Predict
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Models interpretation
    prepare_roc_plots(lrc, cv_rfc, x_test, y_test)
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    feature_importance_plot(
        cv_rfc,
        x_train,
        "images/results/feature_importance.png")


if __name__ == "__main__":
    data = import_data("data/bank_data.csv")
    perform_eda(data)
    encoded_data = encoder_helper(data,
                                  ["Gender",
                                   "Education_Level",
                                   "Marital_Status",
                                   "Income_Category",
                                   "Card_Category"],
                                  "Churn")
    KEEP_COLS = [
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

    X_train_data, X_test_data, y_train_data, y_test_data = perform_feature_engineering(
        encoded_data, KEEP_COLS)
    train_models(X_train_data, X_test_data, y_train_data, y_test_data)
