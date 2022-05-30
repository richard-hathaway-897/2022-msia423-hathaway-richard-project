import logging
import typing
import dateutil

import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection

import src.read_write_s3
import src.remove_outliers

logger = logging.getLogger(__name__)

def generate_features(data: pd.DataFrame,
                      create_datetime_features_params: dict,
                      collapse_weather_categories_params: dict,
                      log_transform_params: dict,
                      binarize_column_params: dict,
                      remove_outlier_params: dict,
                      one_hot_encoding_params: dict,
                      train_test_split_params: dict,
                      drop_columns: typing.List
                      ) -> typing.Tuple[pd.DataFrame, pd.DataFrame, sklearn.preprocessing.OneHotEncoder]:

    data_original_shape = data.shape
    logger.info("Data prior to generating features has %d records and %d columns.",
                data_original_shape[0], data_original_shape[1])
    data = create_datetime_features(data, **create_datetime_features_params)
    data = binarize_column(data, **binarize_column_params)
    data = collapse_weather_categories(data, collapse_weather_categories_params)
    data = log_transform(data, **log_transform_params)

    cols_drop = drop_columns + \
                list(log_transform_params["log_transform_column_names"]) + \
                list(binarize_column_params["binarize_column_names"])
    data = columns_drop(data, columns=cols_drop)
    data = src.remove_outliers.remove_outliers(data,
                                               **remove_outlier_params["feature_columns"],
                                               **remove_outlier_params["valid_values"])
    data = data.reset_index(drop=True)
    data, one_hot_encoder = one_hot_encoding(data, **one_hot_encoding_params)
    train, test = create_train_test_split(data, **train_test_split_params)
    logger.info("Finished generating features.")
    logger.info("Train contains %d records and %d columns", train.shape[0], train.shape[1])
    logger.info("Test contains %d records and %d columns", test.shape[0], test.shape[1])

    return train, test, one_hot_encoder

def columns_drop(data, columns):
    try:
        data = data.drop(columns, axis=1)
    except KeyError as key_error:
        logger.error("At least one column that was attempted to drop does not exist. %s", key_error)
    else:
        logger.info("Dropped the following columns from the dataset: %s",
                    str(columns))
    return data

def collapse_weather_categories(data: pd.DataFrame, collapse_dict: dict):
    for collapse_key in collapse_dict.keys():
        records_to_collapse = \
            data.loc[data["weather_main"] == collapse_dict[collapse_key]["original_category"]].shape[0]
        data.loc[data["weather_main"] == collapse_dict[collapse_key]["original_category"], "weather_main"] = \
            collapse_dict[collapse_key]["to_category"]
        logger.info("Reassigned %d records with 'weather_main' category of '%s' to '%s'", records_to_collapse,
                    collapse_dict[collapse_key]["original_category"], collapse_dict[collapse_key]["to_category"])

    return data


def log_transform(data: pd.DataFrame, log_transform_column_names: typing.List, log_transform_new_column_prefix: str) -> pd.DataFrame:
    for column_log in log_transform_column_names:
        try:
            data[log_transform_new_column_prefix + column_log] = np.log1p(data[column_log])
        except TypeError:
            logger.error("Could not log transform the column. Data type cannot be log transformed.")
        except KeyError:
            logger.error("Could not log transform the column. The specified column %s is not in the dataframe.", column_log)
        else:
            logger.info("Log transformed column %s. Added column '%s' to the dataset.",
                        column_log, log_transform_new_column_prefix + column_log)

    return data


def binarize_column(data: pd.DataFrame, binarize_column_names, binarize_new_column_prefix, binarize_zero_value):
    for column_binarize in binarize_column_names:
        try:
            data[binarize_new_column_prefix + column_binarize] = data[column_binarize] \
                .apply(func=binarize, args=[binarize_zero_value])
        except KeyError:
            logger.error("Could not binarize the column. The specified column does not exist in the dataframe.")
        else:
            logger.info("Binarized column %s. Added column '%s' to the dataset.",
                        column_binarize, binarize_new_column_prefix + column_binarize)

    return data

def create_datetime_features(data: pd.DataFrame,
                             original_datetime_column,
                             month_column,
                             hour_column,
                             day_of_week_column) -> pd.DataFrame:

    try:
        data[original_datetime_column] = data[original_datetime_column] \
                .apply(func=validate_date_time)
    except KeyError:
        logger.error("Could not generate datetime features. "
                     "The specified date time column %s does not exist in the dataframe.",
                     original_datetime_column)
    else:

        data.dropna(axis=0, subset=[original_datetime_column], inplace=True)

        data[month_column] = data[original_datetime_column].dt.month
        data[hour_column] = data[original_datetime_column].dt.hour
        data[day_of_week_column] = data[original_datetime_column].dt.day_name()

        data_shape = data.shape
        logger.info("After generating columns '%s', '%s', and '%s', the data has %d records and %d columns.",
                    month_column,
                    hour_column,
                    day_of_week_column,
                    data_shape[0],
                    data_shape[1])

    return data


def validate_date_time(date_time_string: str):
    try:
        date_time = pd.to_datetime(date_time_string)
    except dateutil.parser._parser.ParseError as invalid_date:
        logger.error("Invalid date found, removing record. ", invalid_date)
        return None
    else:
        return date_time



def binarize(value: str, zero_value: str) -> int:
    """
    Makes a column into a binary 0-1 variable
    """
    if value == zero_value:
        return 0
    else:
        return 1


def fahrenheit_to_kelvin(temp_deg_f: float) -> float:
    kelvin = (temp_deg_f - 32) * (5/9) + 273.15
    return kelvin


def one_hot_encoding(data: pd.DataFrame,
                     one_hot_encode_columns: typing.List,
                     sparse: bool = True,
                     drop: str = 'first'
                     ) -> typing.Tuple[pd.DataFrame, sklearn.preprocessing.OneHotEncoder]:

    logger.info("Prior to One Hot Encoding, data has %d columns.", data.shape[1])
    data_one_hot_encoded = pd.DataFrame()
    one_hot_encoder = sklearn.preprocessing.OneHotEncoder(drop=drop, sparse=sparse)
    try:
        one_hot_array = one_hot_encoder.fit_transform(data[one_hot_encode_columns])
    except KeyError as key_error:
        logger.error("At least one column did not exist in the dataframe. %s", key_error)
    else:
        one_hot_column_names = one_hot_encoder.get_feature_names_out()
        one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_column_names)

        data_one_hot_encoded = data.join(one_hot_df).drop(one_hot_encode_columns, axis=1)
        logger.info("One Hot Encoded the following columns: %s", str(one_hot_encode_columns))
        logger.info("After One Hot Encoding, data has %d columns.", data_one_hot_encoded.shape[1])


        logger.info("Number of NA values: %d", data_one_hot_encoded.isna().sum().sum())

    if data_one_hot_encoded.empty:
        logger.warning("One Hot Encoding Failed. Returning empty dataframe.")
    return data_one_hot_encoded, one_hot_encoder

def create_train_test_split(
        data: pd.DataFrame,
        test_size: float = 0.4,
        random_state: int = 24,
        shuffle: bool = True) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    """This function creates a train/test split from an input dataframe.

    Args:
        data (pd.DataFrame): the input dataframe.
        test_size (float): A float greater than 0 and less than 1 specifying the
            proportion of the data that should be held out in the test set.
        random_state (int): An integer specifying the random state.
        shuffle (bool): A boolean specifying whether the data should be shuffled before splitting into train and test
            data.

    Returns:
        train, test (Tuple[pd.DataFrame, pd.DataFrame]): A tuple of dataframes, the first of which is the train dataset
            and the second of which is the test dataset. If the train test split fails,
            two empty dataframes are returned.

    """
    try:
        train, test = sklearn.model_selection.train_test_split(
            data, test_size=test_size, random_state=random_state, shuffle=shuffle)
    except TypeError as type_error:
        # This error can occur if the input is not a dataframe
        logger.error(
            "Invalid input type. Check that the input is a dataframe. %s", type_error
        )
        logger.warning("Returning two empty dataframes.")
        train, test = (pd.DataFrame(), pd.DataFrame())
    except ValueError as val_error:
        # This error can occur if invalid parameters are passed to train_test_split (for example, a test size of 1.2)
        logger.error("Invalid parameters passed to the train_test_split function. %s", val_error)
        train, test = (pd.DataFrame(), pd.DataFrame())
    else:
        logger.info(
            "Created train/test split of the data using test size of %f, shuffle set to '%r', "
            "and random state = %d.", test_size, shuffle, random_state)
    return train, test
