import logging
import typing

import pandas as pd

logger = logging.getLogger(__name__)


def remove_outliers(data: pd.DataFrame,
                    weather_column: str,
                    day_of_week_column: str,
                    temperature_column: str,
                    clouds_column: str,
                    rain_column: str,
                    hour_column: str,
                    month_column: str,
                    temp_min: float,
                    temp_max: float,
                    log_rain_mm_min: float,
                    log_rain_mm_max: float,
                    clouds_min: float,
                    clouds_max: float,
                    hours_min: int,
                    hours_max: int,
                    month_min: int,
                    month_max: int,
                    valid_week_days: typing.List,
                    valid_weather: typing.List,
                    response_min: float = 0,
                    response_max: float = 10000,
                    response_column: str = "traffic_volume",
                    include_response: bool = True) -> pd.DataFrame:
    """This function removes outliers for each column in the weather dataset based on the min/max allowable values
    or the valid categories.

    Args:
        data (pd.DataFrame): The input dataframe.
        weather_column (str): The name of the column containing the weather.
        day_of_week_column (str): The name of the column containing the day of the week.
        temperature_column (str): The name of the column containing the temperature.
        clouds_column (str): The name of the column containing the clouds.
        rain_column (str): The name of the column containing the rain last hour information.
        hour_column (str): The name of the column containing the hour.
        month_column (str): The name of the column containing the month.
        temp_min (float): The minimum allowable temperature.
        temp_max (float): The maximum allowable temperature.
        log_rain_mm_min (float): The minimum allowable log_rain
        log_rain_mm_max (float): The maximum allowable log_rain
        clouds_min (float): The minimum allowable cloud percentage
        clouds_max (float): The maximum allowable cloud percentage
        hours_min (int): The minimum allowable hour
        hours_max (int): The maximum allowable hour
        month_min (int): The minimum allowable month
        month_max (int): The maximum allowable month
        valid_week_days (typing.List): A list of valid week days
        valid_weather (typing.List): A list of valid weather categories
        response_min (float): The minimum allowable value for the response column.
        response_max (float): The maximum allowable value for the response column.
        response_column (str): The name of the response column.
        include_response (bool): Boolean indicating whether the response column is included in the input data.

    Returns:
        data (pd.DataFrame): The dataframe with the outliers removed.

    Raises:
        KeyError: This function raises a KeyError if the column specified does not exist in the dataframe.
        TypeError: This function raises a TypeError if one of the minimum/maximum values does not match the datatype
            it is comparing to.
    """
    data_shape = data.shape

    # I wrap all 8 calls to filter_data into one try block because it is un-necessary to catch exceptions from each
    # separate call individually. The exception handling and logging in filter_data alerts users to the specific error
    # that occurs, and this try block just catches and re-raises the individual errors from filter_data.
    # So if I am handling exceptions from filter_data in the same way for all calls to filter data,
    # I can just catch and handle it once rather than with 8 separate try-except blocks.
    try:
        data = filter_data(data, column_name=temperature_column, min_value=temp_min, max_value=temp_max)
        data = filter_data(data, column_name=rain_column, min_value=log_rain_mm_min, max_value=log_rain_mm_max)
        data = filter_data(data, column_name=clouds_column, min_value=clouds_min, max_value=clouds_max)
        data = filter_data(data, column_name=hour_column, min_value=hours_min, max_value=hours_max)
        data = filter_data(data, column_name=month_column, min_value=month_min, max_value=month_max)
        data = filter_data(data, column_name=weather_column, valid_categories=valid_weather, categorical=True)
        data = filter_data(data, column_name=day_of_week_column, valid_categories=valid_week_days, categorical=True)
        if include_response:
            data = filter_data(data, column_name=response_column, min_value=response_min, max_value=response_max)
    except KeyError as key_error:
        # This error can occur if the specified columns do not exist in the dataframe.
        logger.error("Failed to remove outliers. One of the columns specified does not exist in the dataframe.")
        raise key_error
    except TypeError as type_error:
        # This error can occur if the datatype of the min/max values do not match the datatype of the column.
        logger.error("Failed to remove outliers. One of the input parameters is of the wrong type.")
        raise type_error

    data_outlier_shape = data.shape

    if data.empty:
        logger.error("Failed to remove outliers. Returning an empty dataframe.")

    else:
        logger.info("After removing outliers, the data has %d records and %d columns. %d records were removed.",
                    data_outlier_shape[0],
                    data_outlier_shape[1],
                    data_shape[0] - data_outlier_shape[0])

    return data


def filter_data(data: pd.DataFrame,
                column_name: str,
                min_value: float = 0,
                max_value: float = 0,
                valid_categories: typing.List = None,
                categorical: bool = False) -> pd.DataFrame:
    """This function filters an input dataframe for values of the specified column that are in between the min and max
        values specified or are in the list of valid categories.

    Args:
        data (pd.DataFrame): An input pandas dataframe
        column_name (str): The name of the column to filter.
        min_value (float): The minimum allowable value to filter for. Only required for numeric columns.
        max_value (float):  The maximum allowable value to filter for. Only required for numeric columns.
        valid_categories (typing.List): The list of valid categories. Only used for categorical variables.
        categorical (bool): A boolean indicating if the column is categorical. Default is False.

    Returns:
        data (pd.DataFrame): The dataframe with the invalid rows of the column removed.

    Raises:
        KeyError: This function raises a KeyError if the column specified is not found in the dataframe.
        TypeError: This function raises a TypeError if the minimum/maximum value specified is different from the data
            type in the column.

    """
    initial_shape = data.shape

    # Filter numeric columns based on min/max values.
    if not categorical:
        # Key Errors occur if the column does not exist in the dataframe.
        # Type Errors occur if the data type of the column does not match the min/max values passed.
        try:
            data = data[data[column_name] >= min_value]
        except KeyError as key_error:
            logger.error("The column '%s' does not exist in the dataframe.", column_name)
            raise key_error
        except TypeError as type_error:
            logger.error("The column '%s' could not be filtered because the data type of the column did "
                         "not match the data type of the min value.", column_name)
            raise type_error
        try:
            data = data[data[column_name] <= max_value]
        except KeyError as key_error:
            logger.error("The column '%s' does not exist in the dataframe.", column_name)
            raise key_error
        except TypeError as type_error:
            logger.error("The column '%s' could not be filtered because the data type of the column did "
                         "not match the data type of the max value.", column_name)
            raise type_error

    # Filter categorical column using the list of valid categories
    else:
        try:
            data = data[data[column_name].isin(valid_categories)]
        except KeyError as key_error:
            logger.error("The column '%s' does not exist in the dataframe.", column_name)
            raise key_error

    logger.debug("Removed %d records that matched the outlier criteria for '%s'.",
                 initial_shape[0] - data.shape[0],
                 column_name)

    return data
