import logging

import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.base
import sklearn.exceptions

import src.read_write_s3
import src.data_preprocessing

logger = logging.getLogger(__name__)


def make_predictions(new_data: pd.DataFrame, model: sklearn.base.BaseEstimator, response_column: str, is_test_data: bool) -> pd.Series:

    predictions_series = pd.Series(dtype='float64')
    predictions = None
    if is_test_data:
        try:
            predictors = new_data.drop([response_column], axis=1)
        except KeyError as key_error:
            logger.error("The response column was not present in the dataframe. Returning an empty series. %s", key_error)
            return predictions_series
    else:
        predictors = new_data

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
        successful_prediction = False
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

    # If prediction fails, print a warning.
    else:
        logger.warning("Prediction failed. No output files saved. Returning empty an dataframe.")

    return predictions_series


def classify_traffic(traffic_prediction: float) -> str:
    if traffic_prediction < 2000:
        return "light"
    elif traffic_prediction < 3000:
        return "moderate"
    else:
        return "heavy"
