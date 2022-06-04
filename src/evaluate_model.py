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
    """
    """
    # Set initial values of the return

    # Read in the test data and compute auc, accuracy, confusion matrix, and classification report

    try:
        true_value = test[response_column]
    except KeyError:
        # This error will occur if the response column does not exist in the test dataframe
        logger.error("Failed to get the true values, the column '%s' does not exist in the dataframe."
                     "Returning metrics R^2 and MSE of NaN.",
                     response_column)
        raise KeyError
    else:
        # A Value Error can occur on these operations in several cases, including if there is
        # an invalid data type as one of the entries of the csv, or if there are unequal number of entries
        # for the predicted values vs true values.
        try:
            r2 = sklearn.metrics.r2_score(true_value, predictions)
        except ValueError as val_error:
            logger.error("Encountered a value error. Returning NaN for R^2: %s", val_error)
            raise val_error

        try:
            mse = sklearn.metrics.mean_squared_error(true_value, predictions)
        except ValueError as val_error:
            logger.error("Encountered a value error. Returning NaN for MSE: %s", val_error)
            raise val_error

        logger.info("Test R^2 is %f, test MSE is %f.", r2, mse)
        metrics_dict = {'R^2': r2, 'MSE': mse}

    return metrics_dict
