import logging

import pandas as pd

logger = logging.getLogger(__name__)


def remove_outliers(data: pd.DataFrame,
                    weather_column,
                    day_of_week_column,
                    temperature_column,
                    clouds_column,
                    rain_column,
                    hour_column,
                    month_column,
                    temp_min,
                    temp_max,
                    log_rain_mm_min,
                    log_rain_mm_max,
                    clouds_min,
                    clouds_max,
                    hours_min,
                    hours_max,
                    month_min,
                    month_max,
                    valid_week_days,
                    valid_weather,
                    response_min=0,
                    response_max=10000,
                    response_column="traffic_volume",
                    include_response=True) -> pd.DataFrame:
    # Note, filter data checks whether or not the column exists and if the data types match.
    data_shape = data.shape
    data = filter_data(data, column_name=temperature_column, min_value=temp_min, max_value=temp_max)
    data = filter_data(data, column_name=rain_column, min_value=log_rain_mm_min, max_value=log_rain_mm_max)
    data = filter_data(data, column_name=clouds_column, min_value=clouds_min, max_value=clouds_max)
    data = filter_data(data, column_name=hour_column, min_value=hours_min, max_value=hours_max)
    data = filter_data(data, column_name=month_column, min_value=month_min, max_value=month_max)
    data = filter_data(data, column_name=weather_column, valid_categories=valid_weather, categorical=True)
    data = filter_data(data, column_name=day_of_week_column, valid_categories=valid_week_days, categorical=True)
    if include_response:
        data = filter_data(data, column_name=response_column, min_value=response_min, max_value=response_max)

    data_outlier_shape = data.shape

    if data.empty:
        logger.error("Failed to remove outliers. Returning an empty dataframe.")

    else:
        logger.info("After removing outliers, the data has %d records and %d columns. %d records were removed.",
                    data_outlier_shape[0],
                    data_outlier_shape[1],
                    data_shape[0] - data_outlier_shape[0])

    return data


def filter_data(data, column_name, min_value = 0, max_value = 0, valid_categories=[], categorical = False):
    error = False
    if not categorical:
        try:
            data = data[data[column_name] >= min_value]
        except KeyError:
            logger.error("The column '%s' does not exist in the dataframe.", column_name)
            error = True
        except TypeError:
            logger.error("The column '%s' could not be filtered because the data type of the column did not match the data"
                         " type of the min value.", column_name)
            error = True
        try:
            data = data[data[column_name] <= max_value]
        except KeyError:
            logger.error("The column '%s' does not exist in the dataframe.", column_name)
            error = True
        except TypeError:
            logger.error("The column '%s' could not be filtered because the data type of the column did not match the data"
                         " type of the max value.", column_name)
            error = True
    else:
        try:
            data = data[data[column_name].isin(valid_categories)]
        except KeyError:
            logger.error("The column '%s' does not exist in the dataframe.", column_name)
            error = True
    if error:
        data = pd.DataFrame()

    return data
