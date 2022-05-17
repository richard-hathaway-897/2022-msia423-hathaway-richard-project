import logging

import numpy as np
import pandas as pd

import src.s3_actions

logger = logging.getLogger(__name__)


def clean_data(data_source: str, clean_data_path: str, delimiter: str = ","):
    try:
        traffic = src.s3_actions.s3_read(s3_source=data_source, delimiter=delimiter)
    except ValueError as value_error:
        logger.error("Failed to read data in preprocessing.")
        raise ValueError(value_error)

    traffic_before_shape = traffic.shape
    logger.info("%d records and %d columns in raw data.", traffic_before_shape[0], traffic_before_shape[1])
    traffic = traffic.drop_duplicates(keep='first')
    traffic_after_shape = traffic.shape

    logger.info("%d duplicate records dropped from the data. Clean data has %d records and %d columns.",
                traffic_before_shape[0]-traffic_after_shape[0], traffic_after_shape[0], traffic_after_shape[1])
    traffic = traffic.reset_index(drop=True)
    src.s3_actions.s3_write(data_source=traffic, s3_destination=clean_data_path, delimiter=delimiter)


def generate_features(data_source: str, clean_data_path: str, delimiter = ",", **preprocess_params: dict) -> None:

    try:
        traffic = src.s3_actions.s3_read(s3_source=data_source, delimiter=delimiter)
    except ValueError as value_error:
        logger.error("Failed to read data in preprocessing.")
        raise ValueError(value_error)

    traffic = create_datetime_features(traffic)

    traffic = remove_outliers(traffic, **preprocess_params["remove_outliers"])
#    traffic = select_training_features(traffic, **preprocess_params["select_training_features"])

    for collapse_key, collapse_value in preprocess_params["collapse_weather_categories"]:
        traffic.loc[traffic["weather_main"] == collapse_value["original_category"], "weather_main"] = collapse_value[
            "to_category"]

    for column_binarize in preprocess_params["binarize_columns"]:
        traffic[column_binarize + "_binary"] = traffic[column_binarize]\
            .apply(func=binarize, args=[preprocess_params["binarize_zero_value"]])

    for column_log in preprocess_params["log_transform_columns"]:
        traffic["log_" + column_log] = np.log1p(traffic[column_log])

    traffic = traffic.drop([preprocess_params["drop_columns"] +
                            preprocess_params["log_transform_columns"] +
                            preprocess_params["binarize_columns"]], axis=1)

    traffic = traffic.reset_index(drop=True)

    src.s3_actions.s3_write(data_source=traffic, s3_destination=clean_data_path, delimiter=delimiter)


# def select_training_features(data: pd.DataFrame, **select_training_features_params: dict) -> pd.DataFrame:
#
#     for column_binarize in select_training_features_params["binarize_columns"]:
#         data[column_binarize + "_binary"] = data[column_binarize]\
#             .apply(func=binarize, args=[select_training_features_params["binarize_zero_value"]])
#
#     for column_log in select_training_features_params["log_transform_columns"]:
#         data["log_" + column_log] = np.log1p(data[column_log])
#
#     data = data.drop([select_training_features_params["drop_columns"] +
#                       select_training_features_params["log_transform_columns"] +
#                       select_training_features_params["binarize_columns"]], axis=1)
#
#     return data
#

def remove_outliers(data: pd.DataFrame, **remove_outlier_params: dict) -> pd.DataFrame:
    # TODO: Check to make sure these columns exist in the dataframe.
    data = data[data["temp"] >= remove_outlier_params["temp_min"]]
    data = data[data["temp"] <= remove_outlier_params["temp_max"]]

    data = data[data["rain_1h"] >= remove_outlier_params["rain_min"]]
    data = data[data["rain_1h"] <= remove_outlier_params["rain_max"]]

    data = data[data["clouds_all"] >= remove_outlier_params["clouds_min"]]
    data = data[data["clouds_all"] <= remove_outlier_params["clouds_max"]]

    data = data[data["traffic_volume"] >= remove_outlier_params["traffic_min"]]
    data = data[data["traffic_volume"] <= remove_outlier_params["traffic_min"]]

    return data

    # Check to make sure this is not infinity or - infinity


def create_datetime_features(data: pd.DataFrame) -> pd.DataFrame:
    data["date_time"] = pd.to_datetime(data["date_time"])
    data["year"] = pd.to_datetime(data["date_time"]).dt.year
    data["month"] = pd.to_datetime(data["date_time"]).dt.month
    data["hour"] = pd.to_datetime(data["date_time"]).dt.hour
    data["day_of_week"] = pd.to_datetime(data["date_time"]).dt.day_name()

    return data

    # TODO: Can these "year", "month", etc. stay hardcoded?


def binarize(value: str, zero_value: str) -> int:
    """
    Makes a column into a binary 0-1 variable
    """
    if value == zero_value:
        return 0
    else:
        return 1
    # TODO: Is this hardcoding?


# Check for duplicates
# Write out as an artifact what the performance metrics are