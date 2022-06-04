import logging

import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.base
import sklearn.exceptions
import sklearn

logger = logging.getLogger(__name__)


def evaluate_model(
        test: pd.DataFrame,
        predictions: pd.DataFrame,
        response_column: str) -> dict:
    """This function takes in test data and predictions and computes metrics such as r2 and mean-squared-error based on
     how well the predictions match the test data.

    Args:
        test (pd.DataFrame): The dataframe of test data.
        predictions (pd.DataFrame): The dataframe of predictions.
        response_column (str): The name of the column containing the response in the test data.

    Returns:
        metrics_dict (dict): This is a dictionary with two keys: R^2 and MSE, each of which contains the value of the
            metric computed.

    Raises:
        KeyError: This function raises a KeyError if the response column does not exist in the test data.
        ValueError: This function raises a ValueError if the function fails to compute R^2 or MSE.

    """
    try:
        true_value = test[response_column]
    except KeyError as key_error:
        # This error will occur if the response column does not exist in the test dataframe
        logger.error("Failed to get the true values, the column '%s' does not exist in the dataframe."
                     "Returning metrics R^2 and MSE of NaN.",
                     response_column)
        raise key_error
    else:
        # A Value Error can occur on these operations in several cases, including if there is
        # an invalid data type as one of the entries of the csv, or if there are unequal number of entries
        # for the predicted values vs true values.
        try:
            r_squared = sklearn.metrics.r2_score(true_value, predictions)
        except ValueError as val_error:
            logger.error("Encountered a value error. Returning NaN for R^2: %s", val_error)
            raise val_error

        try:
            mse = sklearn.metrics.mean_squared_error(true_value, predictions)
        except ValueError as val_error:
            logger.error("Encountered a value error. Returning NaN for MSE: %s", val_error)
            raise val_error

        logger.info("Test R^2 is %f, test MSE is %f.", r_squared, mse)
        metrics_dict = {"R^2": r_squared, "MSE": mse}

    return metrics_dict
