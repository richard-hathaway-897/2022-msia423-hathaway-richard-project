import logging
import typing
import datetime
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
                      log_transform_params: dict,
                      binarize_column_params: dict,
                      remove_outlier_params: dict,
                      one_hot_encoding_params: dict,
                      train_test_split_params: dict,
                      drop_columns: typing.List
                      ) -> typing.Tuple[pd.DataFrame, pd.DataFrame, sklearn.preprocessing.OneHotEncoder]:
    """This function operates as an orchestration function that organizes the feature transformation function calls
        for the model pipeline. It issues function calls to do the following tasks: create datetime features,
        binarize any columns, log transform any columns, drop un-needed columns, remove outliers,
        one-hot-encode the data, and create the train-test-split.


    Args:
        data (pd.DataFrame): Input pandas dataframe.
        create_datetime_features_params (dict): Dictionary of parameters needed to create datetime features.
        log_transform_params (dict): Dictionary of parameters needed to log transform any columns required.
        binarize_column_params (dict): Dictionary of parameters needed to binarize any columns required.
        remove_outlier_params (dict): Dictionary of parameters needed to remove outliers.
        one_hot_encoding_params (dict): Dictionary of parameters needed to one-hot-encode the data.
        train_test_split_params (dict): Dictionary of parameters needed to create the train-test-split.
        drop_columns (typing.List): List of columns that are not neede that can be dropped.

    Returns:
        train, test, one_hot_encoder (typing.Tuple[pd.DataFrame, pd.DataFrame, sklearn.preprocessing.OneHotEncoder]):
            This function returns a tuple of 3 elements. The first is the training data, the second is the testing
            data, and the last one is the one_hot_encoder sklearn object.

    Raises:
        KeyError: This function raises a key error if one of the required columns for the feature transformations
            does not exist in the dataframe.
        TypeError: This function raises a type error if one of the columns contains an unexpected datatype.
        ValueError: This function raises a value error if invalid parameters are passed to create the train-test-split.
    """

    logger.info("Data prior to generating features has %d records and %d columns.",
                data.shape[0], data.shape[1])
    # Create datetime features month, hour, and day of week.
    try:
        data = create_datetime_features(data, **create_datetime_features_params)
    except KeyError as create_datetime_error:
        # This error can occur if the field with the original date-time information is not in the dataframe.
        logger.error("Failed to create datetime features.")
        raise create_datetime_error

    # Convert specified columns to 0-1 binary variables.
    try:
        data = binarize_column(data, **binarize_column_params)
    except KeyError as binarize_error:
        # This error occurs if the columns to binarize are not in the dataframe.
        logger.error("Failed to binarize columns.")
        raise binarize_error
    try:
        data = log_transform(data, **log_transform_params)
    except (KeyError, TypeError) as log_transform_error:
        # This error occurs if either the columns to log transform are not in the dataframe or the column datatype
        # is not numeric. Individual exceptions are caught in the log_transform function, but here the program should
        # handle the error in the same way, so catch both of them in one except block.
        logger.error("Failed to log transform columns.")
        raise log_transform_error

    # Try to drop the specified columns.
    data = columns_drop(data, columns=drop_columns + \
                                        list(log_transform_params["log_transform_column_names"]) + \
                                        list(binarize_column_params["binarize_column_names"]))
    # Try to remove outliers.
    try:
        data = src.remove_outliers.remove_outliers(data,
                                                   **remove_outlier_params["feature_columns"],
                                                   **remove_outlier_params["valid_values"])
    # Handle any error in outlier removal the same way, so the program can catch both errors in one except statement.
    except (KeyError, TypeError) as remove_outlier_error:
        logger.error("Failed to remove outliers.")
        raise remove_outlier_error
    data = data.reset_index(drop=True)

    # Try to one hot encode the data
    try:
        data, one_hot_encoder = one_hot_encoding(data, **one_hot_encoding_params)
    except KeyError as one_hot_encode_error:
        # This error can occur if the requested columns to one-hot-encode do not exist in the dataframe.
        logger.error("Failed to one hot encode the data.")
        raise one_hot_encode_error

    # Try to create the train-test-split
    try:
        train, test = create_train_test_split(data, **train_test_split_params)

    # This error can occur if invalid parameters are fed to the train test split function or if the input is not a
    # dataframe.
    except (TypeError, ValueError) as create_train_test_split_error:
        logger.error("Failed to split the data into train and test.")
        raise create_train_test_split_error

    logger.info("Finished generating features."
                "Train contains %d records and %d columns. "
                "Test contains %d records and %d columns",
                train.shape[0], train.shape[1], test.shape[0], test.shape[1])

    return train, test, one_hot_encoder


def columns_drop(data: pd.DataFrame, columns: typing.List) -> pd.DataFrame:
    """This function drops any specified columns from the dataframe.

    Args:
        data (pd.DataFrame): Input pandas dataframe.
        columns (typing.List): A list of columns to drop from the dataframe.

    Returns:
        data (pd.DataFrame): This function returns the data without the dropped columns. If the operation fails,
        the original dataframe will be returned.

    """
    try:
        data = data.drop(columns, axis=1)
    except KeyError as key_error:
        logger.error("At least one column that was attempted to drop does not exist. %s", key_error)
    else:
        logger.info("Dropped the following columns from the dataset: %s",
                    str(columns))
    return data


def log_transform(data: pd.DataFrame,
                  log_transform_column_names: typing.List,
                  log_transform_new_column_prefix: str) -> pd.DataFrame:
    """This function log transforms a list of input columns on a dataframe. This function uses np.log1p in order to
        avoid taking the log of zero.

    Args:
        data (pd.DataFrame): The input dataframe
        log_transform_column_names (typing.List): The list of column names for which each column in the list
            should be log-transformed.
        log_transform_new_column_prefix (str): The prefix to append to create the new column name. For example,
            if log_transform_new_column_prefix = "log_", then the new column name will be "log_" + column_name.

    Returns:
        data (pd.DataFrame): The dataframe containing the new log-transformed column.

    Raises:
        TypeError: This function raises a TypeError if the data type is not numeric
        KeyError: This function raises a KeyError if the columns to log-transform do not exist in the dataframe.

    """
    for column_log in log_transform_column_names:
        try:
            data[log_transform_new_column_prefix + column_log] = np.log1p(data[column_log])
        except TypeError as type_error:
            # For example, a string cannot be log-transformed.
            logger.error("Could not log transform the column. Data type cannot be log transformed.")
            raise type_error
        except KeyError as key_error:
            logger.error("Could not log transform the column. The specified column %s is not in the dataframe.",
                         column_log)
            raise key_error
        else:
            logger.info("Log transformed column %s. Added column '%s' to the dataset.",
                        column_log, log_transform_new_column_prefix + column_log)

    return data


def binarize_column(data: pd.DataFrame,
                    binarize_column_names: typing.List,
                    binarize_new_column_prefix: str,
                    binarize_zero_value: str) -> pd.DataFrame:
    """This function transforms a list of columns into a columns of 0-1 variables. One value is specified to be 0, and
    the remainder of values are turned into a 1.

    Args:
        data (pd.DataFrame): The input dataframe.
        binarize_column_names (typing.List): A list of columns to binarize.
        binarize_new_column_prefix (str): The prefix to append to the new column name. For example, if
            binarize_new_column_prefix is equal to "binarize", the new column name will be "binarize_" + column_name.
        binarize_zero_value (str): The value that should be assigned to zero. All other values will be assigned to one.

    Returns:
        data (pd.DataFrame): The dataframe with the binarized columns.

    Raises:
        KeyError: This function raises a KeyError if the columns to binarize do not exist in the dataframe.
    """
    # Loop through the list of columns to binarize.
    for column_binarize in binarize_column_names:
        try:
            data[binarize_new_column_prefix + column_binarize] = data[column_binarize] \
                .apply(func=binarize, args=[binarize_zero_value])
        except KeyError as key_error:
            logger.error("Could not binarize the column. The specified column does not exist in the dataframe.")
            raise key_error
        else:
            logger.info("Binarized column %s. Added column '%s' to the dataset.",
                        column_binarize, binarize_new_column_prefix + column_binarize)

    return data


def binarize(value: str, zero_value: str) -> int:
    """This function is a helper function that converts an input string value to a zero-one binary variable
    depending on the "zero_value"

    Args:
        value (str): Value to convert to 0-1 variable
        zero_value (str): Value to convert to zero. All values not matching the zero will be converted to one.

    Returns:
        return_value (int): Returns either a 0 or 1 depending on the input data.

    Raises: This function does not raise any errors.

    """
    # If the zero_value is an invalid datatype (not a string), return 1.
    return_value = 1
    if not isinstance(zero_value, str):
        return return_value
    if value == zero_value:
        return_value = 0

    return return_value


def create_datetime_features(data: pd.DataFrame,
                             original_datetime_column: str,
                             month_column: str,
                             hour_column: str,
                             day_of_week_column) -> pd.DataFrame:
    """This function extracts the month, hour, and day of the week from a string date_time value and adds each
        of them as new columns in the input pandas dataframe.

    Args:
        data (pd.DataFrame): The input dataframe with which to generate the datetime features
        original_datetime_column (str): The name of the column holding the date_time as a string.
        month_column (str): The name of the output column for the month variable.
        hour_column (str): The name of the output column for the hour variable.
        day_of_week_column (str): The name of the output column for the day of the week variable.

    Returns:
        data (pd.DataFrame): Returns the pandas dataframe that includes the newly generated columns.

    Raises:
        KeyError: This function raises a KeyError if the original column specified as date_time does not exist in the
            dataframe.

    """
    # Try to convert the datetime string to a pandas datetime object for each row. If it fails, "None" will be returned
    # for the specific row.
    try:
        data[original_datetime_column] = data[original_datetime_column] \
                .apply(func=validate_date_time)
    except KeyError as key_error:
        # This error will occur if the column specified as the original datetime column does not exist in the dataframe.
        logger.error("Could not generate datetime features. "
                     "The specified date time column %s does not exist in the dataframe.",
                     original_datetime_column)
        raise key_error
    else:
        # If any strings could not be converted to datetime, drop them.
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


def validate_date_time(date_time_string: str) -> typing.Union[datetime.datetime, None]:
    """This function converts a date_time string in a pandas dataframe to a datetime.datetime object

    Args:
        date_time_string (str): The input datetime string

    Returns:
        date_time(typing.Union[datetime.datetime, None]): If the string is successfully converted, a datetime.datetime
        object is returned. If the string is not able to be converted to a datetime object, then None is returned.
    """
    date_time = None
    try:
        date_time = pd.to_datetime(date_time_string)
    except dateutil.parser.ParserError as invalid_date:
        logger.error("Invalid date found, removing record. %s", invalid_date)
        return date_time
    else:
        return date_time


def fahrenheit_to_kelvin(temp_deg_f: float) -> float:
    """This function converts from degrees fahrenheit to degrees kelvin using the standard formula for conversion
    between these two temperature units.

    Args:
        temp_deg_f (float): The input temperature in degrees fahrenheit.

    Returns:
        kelvin (float): The temperature converted to kelvin.

    Raises:
        TypeError: This function raises a type error if the input data type cannot be converted to Kelvin.

    """
    try:
        kelvin = (temp_deg_f - 32) * (5/9) + 273.15
    except TypeError as type_error:
        logger.error("The value passed could not be converted to Kelvin. %s", type_error)
        raise type_error
    return kelvin


def one_hot_encoding(data: pd.DataFrame,
                     one_hot_encode_columns: typing.List,
                     sparse: bool = True,
                     drop: str = "first"
                     ) -> typing.Tuple[pd.DataFrame, sklearn.preprocessing.OneHotEncoder]:
    """This function creates a one-hot-encoder object and uses it to one-hot-encode an input dataframe.
        For more details see https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html

    Args:
        data (pd.DataFrame): The input pandas dataframe.
        one_hot_encode_columns (typing.List): A list of columns in the dataframe to one-hot-encode.
        sparse (bool): Whether to return a sparse matrix of an array. For more details see:
            https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
            Default is True.
        drop (str): Parameter specifying whether/how to drop one of the categories from the one-hot-encoded columns.
            For more details see:
            https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
            Default is 'first'.

    Returns:
        data_one_hot_encoded, one_hot_encoder (typing.Tuple[pd.DataFrame, sklearn.preprocessing.OneHotEncoder]):
            The function returns a tuple where the first element is the dataframe that is one-hot-encoded, and the
            second element is the sklearn OneHotEncoder object.

    Raises:
        KeyError: This function raises a KeyError if the columns specified to one-hot-encode do not exist in the
            dataframe.

    """

    logger.info("Prior to One Hot Encoding, data has %d columns.", data.shape[1])

    # Try to fit the one-hot-encoder
    one_hot_encoder = sklearn.preprocessing.OneHotEncoder(drop=drop, sparse=sparse)
    try:
        one_hot_array = one_hot_encoder.fit_transform(data[one_hot_encode_columns])
    except KeyError as key_error:
        # Error will occur if any of the columns in one_hot_encode_columns do not exist in the dataframe.
        logger.error("One hot encoding failed. At least one column did not exist in the dataframe. %s", key_error)
        raise key_error
    else:

        # Join the one_hot_encoded columns back to the original dataframe and drop the original columns.
        one_hot_column_names = one_hot_encoder.get_feature_names_out()
        one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_column_names)
        data_one_hot_encoded = data.join(one_hot_df).drop(one_hot_encode_columns, axis=1)

        logger.info("One Hot Encoded the following columns: %s", str(one_hot_encode_columns))
        logger.info("After One Hot Encoding, data has %d columns.", data_one_hot_encoded.shape[1])
        logger.info("Number of NA values: %d", data_one_hot_encoded.isna().sum().sum())

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

    Raises:
        TypeError: This function raises a TypeError if the input argument is not a dataframe.
        ValueError: This function raises a ValueError if there are invalid parameters passed to the train_test_split
            function.

    """
    try:
        train, test = sklearn.model_selection.train_test_split(
            data, test_size=test_size, random_state=random_state, shuffle=shuffle)
    except TypeError as type_error:
        # This error can occur if the input is not a dataframe
        logger.error("Invalid input type. Check that the input is a dataframe. %s", type_error)
        raise type_error
    except ValueError as val_error:
        # This error can occur if invalid parameters are passed to train_test_split (for example, a test size of 1.2)
        logger.error("Invalid parameters passed to the train_test_split function. %s", val_error)
        raise val_error
    else:
        logger.info(
            "Created train/test split of the data using test size of %f, shuffle set to '%r', "
            "and random state = %d.", test_size, shuffle, random_state)
    return train, test
