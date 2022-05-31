import logging
import typing

import pandas as pd
import sklearn.base

import src.data_preprocessing
import src.remove_outliers
import sklearn.preprocessing

logger = logging.getLogger(__name__)

def predict_preprocess(predictors: dict,
                       collapse_weather_categories_params: dict,
                       binarize_column_params: dict,
                       log_transform_params: dict,
                       remove_outlier_params: dict,
                       one_hot_encoding_params: dict,
                       one_hot_encoder: sklearn.preprocessing.OneHotEncoder) -> pd.DataFrame:
    print(binarize_column_params)
    print(log_transform_params)

    try:
        predictors = validate_app_input(predictors)
    except ValueError:
        logger.error("An input data type was not valid")
        raise ValueError


    predictors_dict_as_single_item_lists = {}
    for column_name, value in predictors.items():
        predictors_dict_as_single_item_lists[column_name] = [value]
    prediction_df = pd.DataFrame(predictors_dict_as_single_item_lists)

    prediction_df = src.data_preprocessing.collapse_weather_categories(prediction_df, collapse_weather_categories_params)
    prediction_df = src.data_preprocessing.binarize_column(prediction_df, **binarize_column_params)
    prediction_df = src.data_preprocessing.log_transform(prediction_df, **log_transform_params)
    prediction_df["temp"] = src.data_preprocessing.fahrenheit_to_kelvin(prediction_df["temp"]) # TODO: Is temp hardcoded?

    cols_drop = list(log_transform_params["log_transform_column_names"]) + \
                list(binarize_column_params["binarize_column_names"])
    prediction_df = src.data_preprocessing.columns_drop(prediction_df, columns=cols_drop)
    prediction_df = src.remove_outliers.remove_outliers(prediction_df,
                                               **remove_outlier_params["feature_columns"],
                                               **remove_outlier_params["valid_values"],
                                               include_response=False)
    prediction_df = prediction_df.reset_index(drop=True)

    one_hot_encode_columns = one_hot_encoding_params["one_hot_encode_columns"]  # TODO USE YAML
    logger.info(str(one_hot_encode_columns))

    one_hot_array = one_hot_encoder.transform(prediction_df[one_hot_encode_columns])  # TODO use yaml config herer
    one_hot_column_names = one_hot_encoder.get_feature_names_out()
    one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_column_names)
    print(one_hot_df)
    data_one_hot_encoded = prediction_df.join(one_hot_df).drop(one_hot_encode_columns, axis=1)
    print(data_one_hot_encoded.head())
    logger.info("One Hot Encoded the new data")

    if data_one_hot_encoded.empty:
        raise ValueError

    return data_one_hot_encoded

def validate_app_input(input_dict: dict):
    valid_input = True
    if not isinstance(input_dict, dict):
        valid_input = False
    elif not len(input_dict) > 0:
        valid_input = False
    else:
        try:
            input_dict = validate_app_input_dtype(input_dict)
        except ValueError:
            valid_input = False
    if not valid_input:
        raise ValueError
    return input_dict

def validate_app_input_dtype(input_dict: dict):
    new_query_params = {}
    valid_input = True
    try:
        new_query_params["temp"] = float(input_dict["temp"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["clouds_all"] = float(input_dict["clouds_all"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["weather_main"] = str(input_dict["weather_main"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["month"] = int(input_dict["month"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["hour"] = int(input_dict["hour"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["day_of_week"] = str(input_dict["day_of_week"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["holiday"] = str(input_dict["holiday"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["rain_1h"] = float(input_dict["rain_1h"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    if not valid_input:
        raise ValueError

    return new_query_params