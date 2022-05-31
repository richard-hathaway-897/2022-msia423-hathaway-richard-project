import logging
import typing

import pandas as pd

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

    predictors_dict_as_single_item_lists = {}
    for column_name, value in predictors.items():
        predictors_dict_as_single_item_lists[column_name] = [value]
    prediction_df = pd.DataFrame(predictors_dict_as_single_item_lists)

    prediction_df = app_input_transformations(prediction_df=prediction_df,
                                              collapse_weather_categories_params=collapse_weather_categories_params,
                                              log_transform_params=log_transform_params,
                                              binarize_column_params=binarize_column_params,
                                              remove_outlier_params=remove_outlier_params)
    data_one_hot_encoded = app_input_one_hot_encode(prediction_df=prediction_df,
                                                    one_hot_encoder=one_hot_encoder,
                                                    one_hot_encode_columns=one_hot_encoding_params["one_hot_encode_columns"])

    if data_one_hot_encoded.empty:
        raise ValueError

    return data_one_hot_encoded


def app_input_transformations(prediction_df,
                              collapse_weather_categories_params,
                              binarize_column_params,
                              log_transform_params,
                              remove_outlier_params):
    prediction_df = src.data_preprocessing.collapse_weather_categories(prediction_df,
                                                                       collapse_weather_categories_params)
    prediction_df = src.data_preprocessing.binarize_column(prediction_df, **binarize_column_params)
    prediction_df = src.data_preprocessing.log_transform(prediction_df, **log_transform_params)
    prediction_df["temp"] = src.data_preprocessing.fahrenheit_to_kelvin(
        prediction_df["temp"])  # TODO: Is temp hardcoded?

    cols_drop = list(log_transform_params["log_transform_column_names"]) + \
                list(binarize_column_params["binarize_column_names"])
    prediction_df = src.data_preprocessing.columns_drop(prediction_df, columns=cols_drop)
    prediction_df = src.remove_outliers.remove_outliers(prediction_df,
                                                        **remove_outlier_params["feature_columns"],
                                                        **remove_outlier_params["valid_values"],
                                                        include_response=False)
    prediction_df = prediction_df.reset_index(drop=True)
    return prediction_df


def app_input_one_hot_encode(prediction_df, one_hot_encoder, one_hot_encode_columns):

    one_hot_array = one_hot_encoder.transform(prediction_df[one_hot_encode_columns])
    one_hot_column_names = one_hot_encoder.get_feature_names_out()
    one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_column_names)
    data_one_hot_encoded = prediction_df.join(one_hot_df).drop(one_hot_encode_columns, axis=1)
    logger.info("One Hot Encoded the new data")

    return data_one_hot_encoded


def validate_app_input(input_dict: dict, validate_user_input_params):
    valid_input = True
    if not isinstance(input_dict, dict):
        valid_input = False
    elif not len(input_dict) > 0:
        valid_input = False
    else:
        try:
            input_dict = validate_app_input_dtype(input_dict, **validate_user_input_params)
        except ValueError:
            valid_input = False
    if not valid_input:
        raise ValueError
    return input_dict


def validate_app_input_dtype(input_dict: dict,
                            column_names,
                            float_columns):

    new_query_params = {}
    valid_input = True
    for col in column_names:
        if col in float_columns:
            try:
                new_query_params[col] = float(input_dict[col])
            except ValueError:
                logger.error("A float was not entered for %s.", col)
                valid_input = False
            except KeyError:
                logger.error("%s was not a field found in the input data.", col)
                valid_input = False
        else:
            try:
                new_query_params[col] = str(input_dict[col])
            except ValueError:
                logger.error("A string was not entered for %s.", col)
                valid_input = False
            except KeyError:
                logger.error("%s was not a field found in the input data.", col)
                valid_input = False

    if not valid_input:
        raise ValueError

    return new_query_params

