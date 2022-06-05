import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

import src.preprocess_app_input


# First, define several dictionary parameters that will be used throughout the unit tests of this module.
log_transform_params = {"log_transform_column_names": ["rain_1h"], "log_transform_new_column_prefix": "log_"}

binarize_column_params = {"binarize_column_names": ["holiday"],
                          "binarize_new_column_prefix": "binarize_",
                          "binarize_zero_value": "None"}

remove_outlier_params = {
    "feature_columns": {
        "response_column": "traffic_volume",
        "month_column": "month",
        "hour_column": "hour",
        "day_of_week_column": "day_of_week",
        "temperature_column": "temp",
        "clouds_column": "clouds_all",
        "weather_column": "weather_main",
        "rain_column": "log_rain_1h",
    },
    "valid_values": {
        "temp_min": 233.1,
        "temp_max": 319.3,
        "log_rain_mm_min": 0,
        "log_rain_mm_max": 5.7,
        "clouds_min": 0,
        "clouds_max": 100,
        "hours_min": 0,
        "hours_max": 23,
        "month_min": 1,
        "month_max": 12,
        "response_min": 100,
        "response_max": 10000,
        "valid_weather": [
            "Clouds",
            "Clear",
            "Mist",
            "Rain",
            "Snow",
            "Drizzle",
            "Haze",
            "Thunderstorm",
            "Fog",
            "Smoke",
            "Squall"],
        "valid_week_days": [
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday"]
    }

}


def test_app_input_transformations():
    """This function tests the successful execution of the app_input_transformations function.
    """

    # Define test input
    input_test = [
        [32, 40, "Clouds", 10, 9, "Tuesday", "None", 0.0]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["temp", "clouds_all", "weather_main",
                                                           "month", "hour", "day_of_week", "holiday",
                                                           "rain_1h"])
    # Define the expected output
    expected_output = [
        [273.15, 40, "Clouds", 10, 9, "Tuesday", 0, 0.0]
    ]
    df_expected_output = pd.DataFrame(data=expected_output,
                                      columns=["temp", "clouds_all", "weather_main",
                                               "month", "hour", "day_of_week", "binarize_holiday",
                                               "log_rain_1h"])

    df_test_output = src.preprocess_app_input.app_input_transformations(prediction_df=df_input_test,
                                                                        log_transform_params=log_transform_params,
                                                                        binarize_column_params=binarize_column_params,
                                                                        remove_outlier_params=remove_outlier_params,
                                                                        temperature_column="temp")
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)


def test_app_input_transformations_invalid_user_input():
    """This function tests the execution of the app_input_transformations function when the user input is not valid,
    such as entering a string for the temperature. It should raise a TypeError.
    """
    input_test = [
        ["Not a temp", 40, "Clouds", 10, 9, "Tuesday", "None", 0.0]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["temp", "clouds_all", "weather_main",
                                                           "month", "hour", "day_of_week", "holiday",
                                                           "rain_1h"])

    with pytest.raises(TypeError):
        src.preprocess_app_input.app_input_transformations(prediction_df=df_input_test,
                                                           log_transform_params=log_transform_params,
                                                           binarize_column_params=binarize_column_params,
                                                           remove_outlier_params=remove_outlier_params,
                                                           temperature_column="temp")


def test_app_input_one_hot_encode():
    """This function tests the successful execution of the one_hot_encode function. It should one-hot-encode the input.
    """

    # First define and fit a one-hot-encoder to use in the unit test.
    train_one_hot_encoder_input = [
       [1.0, "A"],
       [1.0, "B"]
    ]
    df_train_one_hot_encoder = pd.DataFrame(data=train_one_hot_encoder_input, columns=["column1", "column2"])

    one_hot_encoder = OneHotEncoder(drop="first", sparse=False)
    one_hot_encoder = one_hot_encoder.fit(df_train_one_hot_encoder[["column2"]])

    # define the test input and the expected output
    test_input = [
        [1.0, "A"],
        [1.0, "B"]
    ]
    df_test_input = pd.DataFrame(data=test_input, columns=["column1", "column2"])

    expected_output = [
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    df_expected_output = pd.DataFrame(data=expected_output, columns=["column1", "column2_B"])

    df_test_output = src.preprocess_app_input.app_input_one_hot_encode(prediction_df=df_test_input,
                                                                       one_hot_encoder=one_hot_encoder,
                                                                       one_hot_encode_columns=["column2"])

    pd.testing.assert_frame_equal(df_expected_output, df_test_output)


def test_app_input_one_hot_encode_invalid_column():
    """This function tests the execution of the one_hot_encode function when the function tries to one-hot-encode
    a column name that the original one-hot-encoder was not trained on. It should raise a KeyError.
    """
    # First define and fit a one-hot-encoder to use in the unit test.
    train_one_hot_encoder_input = [
       [1.0, "A"],
       [1.0, "B"]
    ]
    df_train_one_hot_encoder = pd.DataFrame(data=train_one_hot_encoder_input, columns=["column1", "column2"])

    one_hot_encoder = OneHotEncoder(drop="first", sparse=False)
    one_hot_encoder = one_hot_encoder.fit(df_train_one_hot_encoder[["column2"]])

    # Define the test input.
    test_input = [
        [1.0, "A"],
        [1.0, "B"]
    ]
    df_test_input = pd.DataFrame(data=test_input, columns=["column1", "INVALID_COLUMN_NAME"])

    with pytest.raises(KeyError):
        src.preprocess_app_input.app_input_one_hot_encode(prediction_df=df_test_input,
                                                          one_hot_encoder=one_hot_encoder,
                                                          one_hot_encode_columns=["column2"])


def test_validate_app_input():
    """This function tests the successful execution of the validate_app_input function. It should return the
    original data cast to the correct data type.
    """
    test_input = {"temp": "32", "day_of_week": "Tuesday"}

    expected_output = {"temp": 32, "day_of_week": "Tuesday"}

    validate_user_input_params = {"column_names": ["temp", "day_of_week"], "float_columns": ["temp"]}
    test_output = src.preprocess_app_input.validate_app_input(input_dict=test_input,
                                                              validate_user_input_params=validate_user_input_params)

    assert expected_output == test_output


def test_validate_app_input_invalid_datatype():
    """This unit test tests the execution of the validate_app_input function when a field is of an invalid data type,
    such as temp not being a numeric column.
    """
    test_input = {"temp": "abc", "day_of_week": "Tuesday"}

    validate_user_input_params = {"column_names": ["temp", "day_of_week"], "float_columns": ["temp"]}
    with pytest.raises(ValueError):
        src.preprocess_app_input.validate_app_input(input_dict=test_input,
                                                    validate_user_input_params=validate_user_input_params)


def test_validate_app_input_dtype():
    """This function tests the successful execution of the validate_app_input_dtype function. It should return the
    original data cast to the correct data type.
    """
    test_input = {"temp": "14", "day_of_week": "Wednesday"}
    expected_output = {"temp": 14, "day_of_week": "Wednesday"}

    column_names = ["temp", "day_of_week"]
    float_columns = ["temp"]
    test_output = src.preprocess_app_input.validate_app_input_dtype(input_dict=test_input,
                                                                    column_names=column_names,
                                                                    float_columns=float_columns)

    assert expected_output == test_output


def test_validate_app_input_dtype_invalid_user_input():
    """This unit test tests the execution of the validate_app_input_dtype function when a field is of an
    invalid data type, such as temp not being a numeric column.
    """
    test_input = {"temp": "NOT A FLOAT", "day_of_week": "Wednesday"}

    column_names = ["temp", "day_of_week"]
    float_columns = ["temp"]
    with pytest.raises(ValueError):
        src.preprocess_app_input.validate_app_input_dtype(input_dict=test_input,
                                                          column_names=column_names,
                                                          float_columns=float_columns)
