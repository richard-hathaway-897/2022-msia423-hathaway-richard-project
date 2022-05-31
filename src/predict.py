import logging

import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.base
import sklearn.exceptions

import src.read_write_s3
import src.data_preprocessing

logger = logging.getLogger(__name__)

# predictors = {'temp': 288.3,
#               'clouds_all': 75,
#               'holiday': "None",
#               'rain_1h': 0.01,
#               'weather_main': 'Drizzle',
#               'month': 9,
#               'day_of_week': "Tuesday",
#               'hour': 13
#             }


def predict_preprocess(predictors: dict, preprocess_params: dict) -> pd.DataFrame:
    # Call to validate()
    predictors_dict_for_df = {}
    for column_name, value in predictors.items():
        predictors_dict_for_df[column_name] = [value]
    prediction_df = pd.DataFrame(predictors_dict_for_df)

    prediction_df = src.data_preprocessing.collapse_weather_categories(prediction_df, preprocess_params)
    prediction_df = src.data_preprocessing.binarize_column(prediction_df, preprocess_params)
    prediction_df = src.data_preprocessing.log_transform(prediction_df, preprocess_params)
    prediction_df["temp"] = src.data_preprocessing.fahrenheit_to_kelvin(prediction_df["temp"]) # TODO: Is temp hardcoded?

    prediction_df = prediction_df.drop(list(preprocess_params["log_transform_columns"]) +
                                        list(preprocess_params["binarize_columns"]), axis=1)
    logger.info("Dropped the following columns from the dataset: %s", str(list(preprocess_params[
                                                                                   "log_transform_columns"]) +
                                                                          list(preprocess_params["binarize_columns"])))
    return prediction_df


def ohe_new_predict():
    pass
    # one_hot_encode_columns = ["weather_main", "month", "hour", "day_of_week"] # TODO USE YAML
    # # if s3_bool:
    # #     src.s3_actions.s3_read_from_file(model_object_path, "./trained_model_object_s3.joblib") #TODO Fix hardcoding
    # #     src.s3_actions.s3_read_from_file(ohe_object_path, "./ohe_object.joblib")
    # logger.info("Attempt to load model objects")
    # try:
    #     model_object = joblib.load(model_object_path)
    # except Exception as e:
    #     logger.error(e)
    # logger.info("Loaded Trained Model Object")
    # try:
    #     one_hot_encoder = joblib.load(ohe_object_path)
    # except Exception as e:
    #     logger.error(e)
    # logger.info("Loaded One Hot Encoder")
    #
    # #new_data = pd.DataFrame(predictors)
    # one_hot_array = one_hot_encoder.transform(new_data[["weather_main", "month", "hour", "day_of_week"]]) # TODO use yaml config herer
    # #print(ohe_new_data)
    #
    # one_hot_column_names = one_hot_encoder.get_feature_names_out()
    # one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_column_names)
    # data_one_hot_encoded = new_data.join(one_hot_df).drop(one_hot_encode_columns, axis=1)
    # print(data_one_hot_encoded.head())
    # logger.info("One Hot Encoded the new data")


def make_predictions(new_data: pd.DataFrame, model: sklearn.base.BaseEstimator, response_column: str) -> pd.Series:

    predictions_series = pd.Series(dtype='float64')
    predictions = None
    try:
        predictors = new_data.drop([response_column], axis=1)
    except KeyError as key_error:
        logger.error("The response column was not present in the dataframe. Returning an empty series.", key_error)
        return predictions_series

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

# with open("/Users/richard/Documents/school_work/SpringQuarter/AVC/2022-msia423-hathaway-richard-project/config/model_config.yaml", "r", encoding="utf-8") as preprocess_yaml:
#     preprocess_parameters = yaml.load(preprocess_yaml, Loader=yaml.FullLoader)
# df = predict_preprocess(predictors, preprocess_parameters["preprocess_data"])
# predict(df)
