import logging

import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.base
import sklearn.exceptions

logger = logging.getLogger(__name__)


def make_predictions(new_data: pd.DataFrame,
                     model: sklearn.base.BaseEstimator,
                     response_column: str,
                     is_test_data: bool) -> pd.Series:
    """This function reads in a pandas dataframe and makes predictions using a trained model object from sklearn.

    Args:
        new_data (pd.DataFrame): The input data as a dataframe to predict.
        model (sklearn.base.BaseEstimator): The trained model object to use for predictions.
        response_column (str): The name of the response column.
        is_test_data (bool): A boolean indicating whether or not the data is test data in the model pipeline or is
            new data from the web application. The test data would have the true response value in it,
            which must be removed.

    Returns:
        predictions_series (pd.Series): The predictions as a pandas series.

    Raises:
        KeyError: This function raises a key error if a column that the model was trained on does not exist in the
            input data.
        ValueError: This function raises a value error if there is an error in the model prediction process, such
            as the model object not being fit or the hyperparameters being invalid.

    """

    predictions = None

    # If the input data is test data in the model pipeline, remove the response column
    if is_test_data:
        try:
            predictors = new_data.drop([response_column], axis=1)
        except KeyError as key_error:
            logger.error("The response column was not present in the dataframe. Returning an empty series. %s",
                         key_error)
            raise key_error

    # Otherwise, the input data does not contain the response column
    else:
        predictors = new_data

    # Try to make the predictions. Keep track of if there was an error making the predictions.
    successful_prediction = True
    try:
        predictions = model.predict(predictors)
    except sklearn.exceptions.NotFittedError as model_not_fitted:
        # This error will occur if the sklearn model object was not successfully fit in model training.
        logger.error("Prediction failed. Attempting to predict using an unfit model. %s", model_not_fitted)
        successful_prediction = False
    except KeyError as key_error:
        # This error can occur if the predictor columns are not in the test set.
        logger.error("Prediction failed. The required columns were not present in the dataset. %s", key_error)
        raise key_error
    except ValueError as val_error:
        # This error can occur if the model object has invalid hyperparameters.
        logger.error("Prediction failed. The model object contained invalid hyperparameters. %s", val_error)
        successful_prediction = False
    except TypeError as type_error:
        # This error can occur if the model object has hyperparameters of an invalid data type.
        logger.error("Prediction failed. The model object contained hyperparameters with an invalid datatype."
                     " %s", type_error)
        successful_prediction = False

    # If the prediction is successful, convert the results to pandas Series and return them
    if successful_prediction:
        predictions_series = pd.Series(predictions)
        logger.info("Successfully predicted estimates.")

    # If prediction fails, raise a ValueError
    else:
        logger.warning("Prediction failed. No output files saved. Returning empty an dataframe.")
        raise ValueError("Failed to generate predictions due to an invalid model object.")

    return predictions_series


def classify_traffic(traffic_prediction: float) -> str:
    """This function classifies a traffic volume into either "light", "moderate", or "heavy".

    Args:
        traffic_prediction (float): The input prediction for the traffic volume.

    Returns (str): A string indicating if the traffic volume is "light", "moderate", or "heavy".

    Raises:
        ValueError: This function raises a ValueError if the prediction is negative.

    """
    if traffic_prediction < 0:
        raise ValueError("Received a negative prediction.")
    if traffic_prediction < 1500:
        return "light"
    if traffic_prediction < 3000:
        return "moderate"

    return "heavy"
