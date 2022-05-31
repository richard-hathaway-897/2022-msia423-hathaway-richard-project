import logging

import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.base
import sklearn.exceptions
import sklearn

import src.read_write_s3
import src.data_preprocessing

logger = logging.getLogger(__name__)

def evaluate_model(
        test: pd.DataFrame,
        predictions: pd.DataFrame,
        response_column: str) -> dict:
    """This function takes in test data (true values) as well as predicted classes and class probabilities. It computes
    metrics such as AUC and Accuracy as well as the confusion matrix and a classification report.

    Args:
        test (pd.DataFrame): The input test data as a dataframe
        ypred_proba_test (pd.Series): The predicted class probabilities
        ypred_bin_test (pd.Series): The predicted classes
        response_column (str): The name of the column in the test data that contains the response variable.
        cm_columns (typing.List): List of column labels for the confusion matrix
        cm_index (typing.List): List of index labels for the confusion matrix

    Returns:
        auc_accuracy, confusion_matrix, classification_report_df
            (typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):
            A tuple of dataframes, the first of which is the AUC/accuracy, the second which is the confusion matrix,
            and the third of which is the classification report.
    """
    # Set initial values of the return

    # Read in the test data and compute auc, accuracy, confusion matrix, and classification report
    r2 = np.nan
    mse = np.nan
    metrics_dict = {}
    try:
        true_value = test[response_column]
    except KeyError:
        # This error will occur if the response column does not exist in the test dataframe
        logger.error("Failed to get the true values, the column '%s' does not exist in the dataframe."
                     "Returning metrics R^2 and MSE of NaN.",
                     response_column)
    else:
        # A Value Error can occur on these operations in several cases, including if there is
        # an invalid data type as one of the entries of the csv, or if there are unequal number of entries
        # for the predicted values vs true values.
        try:
            r2 = sklearn.metrics.r2_score(true_value, predictions)
        except ValueError as val_error:
            logger.error("Encountered a value error. Returning NaN for R^2: %s", val_error)

        try:
            mse = sklearn.metrics.mean_squared_error(true_value, predictions)
        except ValueError as val_error:
            logger.error("Encountered a value error. Returning NaN for MSE: %s", val_error)

        logger.info("Test R^2 is %f, test MSE is %f.", r2, mse)
        metrics_dict = {'R^2': r2, 'MSE': mse}

    return metrics_dict
