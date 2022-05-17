import logging
import typing

import numpy as np
import pandas as pd
import sklearn.preprocessing

import src.s3_actions

logger = logging.getLogger(__name__)


def clean_data(data_source: str, clean_data_path: str, delimiter: str = ","):
    try:
        traffic = src.s3_actions.s3_read(s3_source=data_source, delimiter=delimiter)
    except ValueError as value_error:
        logger.error("Failed to read data in raw data for cleaning.")
        raise ValueError(value_error)

    traffic_before_shape = traffic.shape
    logger.info("%d records and %d columns in raw data.", traffic_before_shape[0], traffic_before_shape[1])
    traffic = traffic.drop_duplicates(keep='first')
    traffic_after_shape = traffic.shape

    logger.info("%d duplicate records dropped from the data. Clean data has %d records and %d columns.",
                traffic_before_shape[0]-traffic_after_shape[0], traffic_after_shape[0], traffic_after_shape[1])
    traffic = traffic.reset_index(drop=True)
    src.s3_actions.s3_write(data_source=traffic, s3_destination=clean_data_path, delimiter=delimiter)


def generate_features(data_source: str, features_path: str, preprocess_params: dict, delimiter=",") -> None:

    try:
        traffic = src.s3_actions.s3_read(s3_source=data_source, delimiter=delimiter)
    except ValueError as value_error:
        logger.error("Failed to read data in preprocessing.")
        raise ValueError(value_error)

    traffic_original_shape = traffic.shape
    logger.info("Data prior to generating features has %d records and %d columns.", traffic_original_shape[0], traffic_original_shape[1])
    traffic = create_datetime_features(traffic)
    traffic_datetime_shape = traffic.shape
    logger.info("After generating datetime features, the data has %d records and %d columns.", traffic_datetime_shape[0],
                traffic_datetime_shape[1])


    traffic = remove_outliers(traffic, preprocess_params["remove_outliers"])
    traffic_outlier_shape = traffic.shape
    logger.info("After removing outliers, the data has %d records and %d columns. %d records were removed.",
                traffic_outlier_shape[0],
                traffic_outlier_shape[1],
                traffic_datetime_shape[0] - traffic_outlier_shape[0])

    collapse_dict = preprocess_params["collapse_weather_categories"]
    for collapse_key in collapse_dict.keys():
        records_to_collapse = traffic.loc[traffic["weather_main"] == collapse_dict[collapse_key]["original_category"]].shape[0]
        traffic.loc[traffic["weather_main"] == collapse_dict[collapse_key]["original_category"], "weather_main"] = collapse_dict[collapse_key]["to_category"]
        logger.info("Reassigned %d records with 'weather_main' category of '%s' to '%s'", records_to_collapse, collapse_dict[collapse_key]["original_category"], collapse_dict[collapse_key]["to_category"])

    for column_binarize in preprocess_params["binarize_columns"]:
        traffic[column_binarize + "_binary"] = traffic[column_binarize]\
            .apply(func=binarize, args=[preprocess_params["binarize_zero_value"]])
        logger.info("Binarized column %s. Added column '%s_binarize' to the dataset.", column_binarize, column_binarize)

    for column_log in preprocess_params["log_transform_columns"]:
        traffic["log_" + column_log] = np.log1p(traffic[column_log])
        logger.info("Log transformed column %s. Added column 'log_%s' to the dataset.", column_log, column_log)

    traffic = traffic.drop(list(preprocess_params["drop_columns"]) +
                           list(preprocess_params["log_transform_columns"]) +
                           list(preprocess_params["binarize_columns"]), axis=1)
    logger.info("Dropped the following columns from the dataset: %s", str(list(preprocess_params["drop_columns"]) +
                                                                          list(preprocess_params["log_transform_columns"]) +
                                                                          list(preprocess_params["binarize_columns"])))

    traffic = one_hot_encoding(traffic, preprocess_params["one_hot_encode_columns"])
    traffic = traffic.reset_index(drop=True)
    traffic_final_shape = traffic.shape
    logger.info("Finished generating features. Final dataset contains %d records and %d columns", traffic_final_shape[0], traffic_final_shape[1])

    src.s3_actions.s3_write(data_source=traffic, s3_destination=features_path, delimiter=delimiter)


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

def remove_outliers(data: pd.DataFrame, remove_outlier_params: dict) -> pd.DataFrame:
    # TODO: Check to make sure these columns exist in the dataframe.
    data = data[data["temp"] >= remove_outlier_params["temp_min"]]
    data = data[data["temp"] <= remove_outlier_params["temp_max"]]

    data = data[data["rain_1h"] >= remove_outlier_params["rain_mm_min"]]
    data = data[data["rain_1h"] <= remove_outlier_params["rain_mm_max"]]

    data = data[data["clouds_all"] >= remove_outlier_params["clouds_min"]]
    data = data[data["clouds_all"] <= remove_outlier_params["clouds_max"]]

    data = data[data["traffic_volume"] >= remove_outlier_params["traffic_min"]]
    data = data[data["traffic_volume"] <= remove_outlier_params["traffic_max"]]

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


def one_hot_encoding(data: pd.DataFrame, one_hot_encode_columns: typing.List) -> pd.DataFrame:

    logger.info("Prior to One Hot Encoding, data has %d columns.", data.shape[1])
    one_hot_encoder = sklearn.preprocessing.OneHotEncoder(drop='first', sparse=False)
    one_hot_array = one_hot_encoder.fit_transform(data[one_hot_encode_columns])
    one_hot_column_names = one_hot_encoder.get_feature_names_out()
    one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_column_names)

    data_one_hot_encoded = data.join(one_hot_df).drop(one_hot_encode_columns, axis=1)
    logger.info("One Hot Encoded the following columns: %s", str(one_hot_encode_columns))
    logger.info("After One Hot Encoding, data has %d columns.", data_one_hot_encoded.shape[1])

    return data_one_hot_encoded

# Check for duplicates
# Write out as an artifact what the performance metrics are