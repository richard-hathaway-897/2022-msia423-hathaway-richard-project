import logging
import typing

import pandas as pd
import sklearn.preprocessing

import src.data_preprocessing
import src.remove_outliers

logger = logging.getLogger(__name__)


def predict_preprocess(predictors: dict,
                       binarize_column_params: dict,
                       log_transform_params: dict,
                       remove_outlier_params: dict,
                       temperature_column: str,
                       one_hot_encoding_params: dict,
                       one_hot_encoder: sklearn.preprocessing.OneHotEncoder) -> pd.DataFrame:
    """This is an orchestration function that calls the necessary functions to perform feature transformations on the
    user's input from the web application. It calls functions to binarize columns, log-transform columns,
    remove outliers, and to one-hot-encode the user input with a fit one-hot-encoder.

    Args:
        predictors (dict): The user input as a dictionary.
        binarize_column_params (dict): The parameters needed to binarize the columns.
        log_transform_params (dict): The parameters needed to log-transform the columns.
        remove_outlier_params (dict): The parameters needed to remove outliers.
        temperature_column (str): The name of the column containing the temperature. Used for converting from
            fahrenheit to kelvin.
        one_hot_encoding_params (dict): The parameters needed to one-hot-encode the user input.
        one_hot_encoder (sklearn.preprocessing.OneHotEncoder): A pre-trained one-hot-encoder sklearn model.

    Returns:
        data_one_hot_encoded (pd.DataFrame): A pandas dataframe with the user input after feature transformation and
            one-hot-encoding.

    Raises:
        KeyError: This function raises a key error if one of the required columns for the feature transformations
            does not exist in the dataframe.
        TypeError: This function raises a type error if one of the columns contains an unexpected datatype.

    """

    # First, make each value of the dictionary into a list of 1 value so it can be used to create a dataframe of 1 row
    predictors_dict_as_single_item_lists = {}
    for column_name, value in predictors.items():
        predictors_dict_as_single_item_lists[column_name] = [value]
    prediction_df = pd.DataFrame(predictors_dict_as_single_item_lists)

    # Call the app_input_transformations function
    try:
        prediction_df = app_input_transformations(prediction_df=prediction_df,
                                                  log_transform_params=log_transform_params,
                                                  binarize_column_params=binarize_column_params,
                                                  remove_outlier_params=remove_outlier_params,
                                                  temperature_column=temperature_column)
    # Catch both TypeError and KeyError in one block. Handle the errors in the same way.
    except (TypeError, KeyError) as input_transformations_error:
        logger.error("Failed to complete input transformations for the user input.")
        raise input_transformations_error

    # Call the function to one_hot_encode the user input with the trained one-hot-encoder.
    try:
        data_one_hot_encoded = \
            app_input_one_hot_encode(prediction_df=prediction_df,
                                     one_hot_encoder=one_hot_encoder,
                                     one_hot_encode_columns=one_hot_encoding_params["one_hot_encode_columns"])
    except KeyError as key_error:
        # This exception will occur if the columns expected by the one-hot-encoder do not exist in the user input
        logger.info("Failed to one-hot-encoded the user input.")
        raise key_error

    return data_one_hot_encoded


def app_input_transformations(prediction_df: pd.DataFrame,
                              binarize_column_params: dict,
                              log_transform_params: dict,
                              remove_outlier_params: dict,
                              temperature_column: str) -> pd.DataFrame:
    """This function is another orchestration function that organizes calls to functions for binarizing columns,
    log-transforming columns, transforming fahrenheit to kelvin, and removing outliers
    (which is also data validation for the user-input)

    Args:
        prediction_df (pd.DataFrame): The input dataframe with the user input.
        binarize_column_params (dict): The parameters needed to binarize the columns.
        log_transform_params (dict): The parameters needed to log-transform the columns.
        remove_outlier_params (dict): The parameters needed to remove outliers.
        temperature_column (str): The name of the column containing the temperature.

    Returns:
        prediction_df (pd.DataFrame): The function returns the data after the 2 transformations plus outlier removal
        have been performed.

    Raises:
        KeyError: This function raises a key error if one of the required columns for the feature transformations
            does not exist in the dataframe.
        TypeError: This function raises a type error if one of the columns contains an unexpected datatype.

    """
    # Binarize columns
    try:
        prediction_df = src.data_preprocessing.binarize_column(prediction_df, **binarize_column_params)
    except KeyError as binarize_error:
        # This error will occur if columns to binarize do not exist in the user input
        logger.error("Failed to binarize columns.")
        raise binarize_error

    # Log transform columns
    try:
        prediction_df = src.data_preprocessing.log_transform(prediction_df, **log_transform_params)
    except (TypeError, KeyError) as log_transform_error:
        # Handle all errors from log-transformation in the same manner.
        logger.error("Failed to log transform the specified columns.")
        raise log_transform_error

    # Convert fahrenheit to kelvin
    try:
        prediction_df[temperature_column] = src.data_preprocessing.fahrenheit_to_kelvin(
            prediction_df[temperature_column])
    except TypeError as temp_conversion_error:
        # This error will occur if input temperature is not a float or an integer
        logger.error("Failed to convert from fahrenheit to kelvin.")
        raise temp_conversion_error

    # Drop columns not needed after transformations
    cols_drop = list(log_transform_params["log_transform_column_names"]) + \
        list(binarize_column_params["binarize_column_names"])
    prediction_df = src.data_preprocessing.columns_drop(prediction_df, columns=cols_drop)

    # Perform data validation on the user input by calling invalid input an "outlier" and removing it.
    try:
        prediction_df = src.remove_outliers.remove_outliers(prediction_df,
                                                            **remove_outlier_params["feature_columns"],
                                                            **remove_outlier_params["valid_values"],
                                                            include_response=False)
    except (KeyError, TypeError) as remove_outlier_error:
        # Handle any error from removing outliers in the same way. It means that the user input cannot be validated if
        # an error is raised here.
        logger.error("Failed to remove outliers.")
        raise remove_outlier_error

    prediction_df = prediction_df.reset_index(drop=True)
    return prediction_df


def app_input_one_hot_encode(prediction_df: pd.DataFrame,
                             one_hot_encoder: sklearn.preprocessing.OneHotEncoder,
                             one_hot_encode_columns: typing.List) -> pd.DataFrame:
    """This function one-hot-encodes the input data using a pre-trained one-hot-encoder object

    Args:
        prediction_df (pd.DataFrame): The input dataframe to be one-hot-encoded.
        one_hot_encoder (sklearn.preprocessing.OneHotEncoder): The one-hot-encoder object to use to transform the data.
        one_hot_encode_columns (typing.List): The list of columns to one-hot-encode.

    Returns:
        data_one_hot_encoded (pd.DataFrame): The dataframe with the user input that is one-hot-encoded.

    Raises:
        KeyError: This function raises a KeyError if the columns expected by the one hot encoder are not present in the
        dataframe.

    """

    # Attempt to one hot encode the new data
    try:
        one_hot_array = one_hot_encoder.transform(prediction_df[one_hot_encode_columns])
    except KeyError as key_error:
        logger.error("Could not one-hot-encode the user input. "
                     "The one-hot-encode columns specified do not exist in the data. %s", key_error)
        raise key_error

    # Join the one hot encoded data back with the original data and drop the columns not needed after one-hot-encoding.
    one_hot_column_names = one_hot_encoder.get_feature_names_out()
    one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_column_names)
    data_one_hot_encoded = prediction_df.join(one_hot_df).drop(one_hot_encode_columns, axis=1)
    logger.info("One Hot Encoded the new data")

    return data_one_hot_encoded


def validate_app_input(input_dict: dict, validate_user_input_params: dict) -> dict:
    """This function validates user input into the app. It validates the following things:
    1. That the input is in dictionary format.
    2. That the dictionary is not empty.
    3. That all the keys that are expected to be in the dictionary are actually in the dictionary.
    4. That all the keys have the expected datatype for the associated values.

    Args:
        input_dict (dict): The input dictionary with the user input from the web application.
        validate_user_input_params (dict): The parameters for checking if the column names and data types are valid.
            It has two keys. One is 'column_names' and should have a list of the column names expected, and the other
            is 'float_columns', which should list the columns that are expected to be numeric.

    Returns
        input_dict (dict): This function returns the input dictionary of user input and ensures that the values are
            all of the correct data type.

    Raises:
        ValueError: This function raises a ValueError if the input user parameters cannot be validated.

    """
    valid_input = True

    # Check that the input argument is a dictionary
    if not isinstance(input_dict, dict):
        logger.error("Invalid data type. The input data is not in the form of dictionary.")
        valid_input = False

    # Check that the dictionary is not empty.
    elif not len(input_dict) > 0:
        logger.error("The input data is empty.")
        valid_input = False

    # Validate the input data
    else:
        try:
            # Validate the data type and that the columns exist
            input_dict = validate_app_input_dtype(input_dict, **validate_user_input_params)
        except ValueError:
            valid_input = False
            logger.error("The input data's data types could not be validated.")

    # If data validation fails, raise a ValueError
    if not valid_input:
        raise ValueError("The input data could not be validated.")
    return input_dict


def validate_app_input_dtype(input_dict: dict,
                             column_names: typing.List,
                             float_columns: typing.List) -> dict:
    """This function validates that all of the columns that are expected to be in the user input are actually in the
    user input and that they are of the proper data type. It does this by trying to cast all values to the expected
    data type.

    Args:
        input_dict (dict): The dictionary of user input
        column_names (typing.List): The list of keys that are expected to be in the user input
        float_columns (typing.List): The list of keys that are expected to have numeric values

    Returns:
        new_query_params (dict): A new dictionary of the user input with all values in the expected data type.

    Raises:
        ValueError: This function raises a ValueError if any of the expected columns do not exist or if any of the
            expected values cannot be transformed to the expected data type.

    """
    new_query_params = {}
    valid_input = True

    # For every column
    for col in column_names:
        # If the column is expected to be a float, try to cast it to a float.
        if col in float_columns:
            try:
                new_query_params[col] = float(input_dict[col])
            except ValueError:
                logger.error("A float was not entered for %s.", col)
                valid_input = False
            except KeyError:
                logger.error("%s was not a field found in the input data.", col)
                valid_input = False

        # Else, try to cast it to a string if it is not expected to be numeric.
        else:
            try:
                new_query_params[col] = str(input_dict[col])
            except ValueError:
                logger.error("A string was not entered for %s.", col)
                valid_input = False
            except KeyError:
                logger.error("%s was not a field found in the input data.", col)
                valid_input = False

    # If the user input cannot be validated, raise a value error.
    if not valid_input:
        raise ValueError("The input data types were not valid.")

    return new_query_params
