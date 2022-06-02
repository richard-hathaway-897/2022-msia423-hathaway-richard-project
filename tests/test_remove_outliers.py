import numpy as np
import pandas as pd
import sklearn
import pytest

import src.remove_outliers


def test_remove_outliers():
    input_test = [
                    [288.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0],
                    [-29.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0]
                ]
    df_input_test = pd.DataFrame(data=input_test, columns = ["temp", "clouds_all", "weather_main", "traffic_volume",
                                                                        "month", "hour", "day_of_week", "binarize_holiday",
                                                                        "log_rain_1h"])
    expected_output = [
        [288.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0]
    ]
    df_expected_output = pd.DataFrame(data=expected_output, columns = ["temp", "clouds_all", "weather_main", "traffic_volume",
                                                                  "month", "hour", "day_of_week", "binarize_holiday",
                                                                  "log_rain_1h"])

    df_test_output = src.remove_outliers.remove_outliers(data=df_input_test,
                                                            weather_column="weather_main",
                                                            day_of_week_column="day_of_week",
                                                            temperature_column="temp",
                                                            clouds_column="clouds_all",
                                                            rain_column="log_rain_1h",
                                                            hour_column="hour",
                                                            month_column="month",
                                                            temp_min=233.1,
                                                            temp_max= 319.3,
                                                            log_rain_mm_min= 0,
                                                            log_rain_mm_max= 5.7,
                                                            clouds_min= 0,
                                                            clouds_max= 100,
                                                            hours_min=0,
                                                            hours_max=23,
                                                            month_min=1,
                                                            month_max=12,
                                                            response_min=100,
                                                            response_max=10000,
                                                            valid_weather=["Clouds"],
                                                            valid_week_days=["Tuesday"])
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)


def test_remove_outliers_invalid_column():
    input_test = [
                    [288.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0],
                    [-29.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0]
                ]
    df_input_test = pd.DataFrame(data=input_test, columns = ["INVALID_COLUMN", "clouds_all", "weather_main", "traffic_volume",
                                                                        "month", "hour", "day_of_week", "binarize_holiday",
                                                                        "log_rain_1h"])

    df_expected_output = pd.DataFrame()

    df_test_output = src.remove_outliers.remove_outliers(data=df_input_test,
                                                            weather_column="weather_main",
                                                            day_of_week_column="day_of_week",
                                                            temperature_column="temp",
                                                            clouds_column="clouds_all",
                                                            rain_column="log_rain_1h",
                                                            hour_column="hour",
                                                            month_column="month",
                                                            temp_min=233.1,
                                                            temp_max= 319.3,
                                                            log_rain_mm_min= 0,
                                                            log_rain_mm_max= 5.7,
                                                            clouds_min= 0,
                                                            clouds_max= 100,
                                                            hours_min=0,
                                                            hours_max=23,
                                                            month_min=1,
                                                            month_max=12,
                                                            response_min=100,
                                                            response_max=10000,
                                                            valid_weather=["Clouds"],
                                                            valid_week_days=["Tuesday"])
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)

def test_filter_data():
    input_test = [
        [288.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0],
        [-29.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["temp", "clouds_all", "weather_main", "traffic_volume",
                                                           "month", "hour", "day_of_week", "binarize_holiday",
                                                           "log_rain_1h"])
    expected_output = [
        [288.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0]
    ]
    df_expected_output = pd.DataFrame(data=expected_output,
                                      columns=["temp", "clouds_all", "weather_main", "traffic_volume",
                                               "month", "hour", "day_of_week", "binarize_holiday",
                                               "log_rain_1h"])
    df_test_output = src.remove_outliers.filter_data(data=df_input_test,
                                                     column_name="temp",
                                                     min_value=233.1,
                                                     max_value=319.3,
                                                     categorical=False)
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)


def test_filter_data_invalid_column():
    input_test = [
        [288.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0],
        [-29.28, 40, "Clouds", 5545, 10, 9, "Tuesday", 0, 0.0]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["INVALID_COLUMN", "clouds_all", "weather_main", "traffic_volume",
                                                           "month", "hour", "day_of_week", "binarize_holiday",
                                                           "log_rain_1h"])

    df_expected_output = pd.DataFrame()
    df_test_output = src.remove_outliers.filter_data(data=df_input_test,
                                                     column_name="temp",
                                                     min_value=233.1,
                                                     max_value=319.3,
                                                     categorical=False)
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)

