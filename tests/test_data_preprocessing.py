import numpy as np
import pandas as pd
import sklearn
import pytest

import src.data_preprocessing

# def test_generate_features():
#     input_test = [
#             [1.0, 2.0, "None", "2022-05-31 08:00:00"],
#             [1.0, 2.0, "other_value", "2022-05-31 08:00:00"]
#         ]
#     df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2", "column3", "date_time"])
#     expected_output_train = [
#             [1.0, 5, 8, "Tuesday", 0.0, np.log1p(2.0)]
#         ]
#     df_expected_output_train = pd.DataFrame(data=expected_output_train,
#                                             columns=["column1", "month", "hour", "day_of_week", "binarize_column3", "log_column2"])
#     expected_output_test = [
#             [1.0, 5, 8, "Tuesday", 0.0, np.log1p(2.0)]
#         ]
#     df_expected_output_test = pd.DataFrame(data=expected_output_train,
#                                            columns=["column1", "month", "hour", "day_of_week", "binarize_column3",
#                                                      "log_column2"])
#
#     log_transform_params = {'log_transform_column_names': 'column2', 'log_transform_new_column_prefix': 'log_'}
#
#     binarize_column_params = {'binarize_column_names': "holiday",
#                               'binarize_new_column_prefix': 'binarize_',
#                               'binarize_zero_value': "None"}
#
#     one_hot_encoding_params = {"one_hot_encode_columns": []}
#
#     drop_columns = ["column2", "column3", "date_time"]
#
#     create_datetime_features_params = {"original_datetime_column": "date_time",
#                                        "month_column": "month",
#                                        "hour_column": "hour",
#                                        "day_of_week_column": "day_of_week"}
#
#     train_test_split_params = {"test_size": 0.5, "random_state": 24, "shuffle": True}
#
#     df_output_train, df_output_test, output_one_hot_encoder = src.data_preprocessing.generate_features()

def test_columns_drop():

    input_test_drop_cols = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ]
    df_input_test_drop_cols = pd.DataFrame(data=input_test_drop_cols, columns=["column1", "column2", "column3"])

    expected_output_drop_cols = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0]
    ]

    df_expected_output_drop_cols = pd.DataFrame(data=expected_output_drop_cols, columns=["column1", "column2"])
    df_test_output_drop_cols = src.data_preprocessing.columns_drop(data=df_input_test_drop_cols, columns=["column3"])
    pd.testing.assert_frame_equal(df_expected_output_drop_cols, df_test_output_drop_cols)


def test_columns_drop_invalid_column():

    input_test_drop_cols = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]
        ]
    df_input_test_drop_cols = pd.DataFrame(data=input_test_drop_cols, columns=["column1", "column2", "column3"])

    expected_output_drop_cols = [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0]
    ]

    df_expected_output_drop_cols = pd.DataFrame(data=expected_output_drop_cols, columns=["column1", "column2", "column3"])
    df_test_output_drop_cols = src.data_preprocessing.columns_drop(data=df_input_test_drop_cols, columns=["column4"])
    pd.testing.assert_frame_equal(df_expected_output_drop_cols, df_test_output_drop_cols)


def test_log_transform():

    input_test = [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0]
        ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    expected_output = [
        [1.0, 2.0, np.log1p(2.0)],
        [1.0, 2.0, np.log1p(2.0)],
        [1.0, 2.0, np.log1p(2.0)]
    ]

    df_expected_output = pd.DataFrame(data=expected_output, columns=["column1", "column2", "log_column2"])
    df_test_output = src.data_preprocessing.log_transform(data=df_input_test,
                                                          log_transform_column_names=["column2"],
                                                          log_transform_new_column_prefix="log_")
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)

def test_log_transform_invalid_column():

    input_test = [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0]
        ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    expected_output = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0]
    ]

    df_expected_output = pd.DataFrame(data=expected_output, columns=["column1", "column2"])
    df_test_output = src.data_preprocessing.log_transform(data=df_input_test,
                                                          log_transform_column_names=["column3"],
                                                          log_transform_new_column_prefix="log_")
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)


def test_binarize_column():

    input_test = [
            [1.0, "None"],
            [1.0, "Other Value A"],
            [1.0, "Other Value B"]
        ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    expected_output = [
        [1.0, "None", 0],
        [1.0, "Other Value A", 1],
        [1.0, "Other Value B", 1]
    ]

    df_expected_output = pd.DataFrame(data=expected_output, columns=["column1", "column2", "binarize_column2"])
    df_test_output = src.data_preprocessing.binarize_column(data=df_input_test,
                                                            binarize_column_names=["column2"],
                                                            binarize_new_column_prefix="binarize_",
                                                            binarize_zero_value="None")

    pd.testing.assert_frame_equal(df_expected_output, df_test_output)


def test_binarize_column_invalid_column():
    input_test = [
        [1.0, "None"],
        [1.0, "Other Value A"],
        [1.0, "Other Value B"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    expected_output = [
        [1.0, "None"],
        [1.0, "Other Value A"],
        [1.0, "Other Value B"]
    ]

    df_expected_output = pd.DataFrame(data=expected_output, columns=["column1", "column2"])
    df_test_output = src.data_preprocessing.binarize_column(data=df_input_test,
                                                            binarize_column_names=["column3"],
                                                            binarize_new_column_prefix="binarize_",
                                                            binarize_zero_value="None")

    pd.testing.assert_frame_equal(df_expected_output, df_test_output)

def test_binarize():
    input_test = "None"
    expected_output = 0
    test_output = src.data_preprocessing.binarize(value = input_test, zero_value="None")
    assert test_output == expected_output

def test_binarize_invalid_zero_value_type():
    input_test = "None"
    expected_output = 1
    test_output = src.data_preprocessing.binarize(value=input_test, zero_value=pd.DataFrame())
    assert test_output == expected_output


def test_create_datetime_features():
    input_test = [
        [1.0, "2022-05-31 08:00:00"],
        [1.0, "2022-06-01 23:13:45"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "date_time"])

    expected_output = [
        [1.0, pd.to_datetime("2022-05-31 08:00:00"), 5, 8, "Tuesday"],
        [1.0, pd.to_datetime("2022-06-01 23:13:45"), 6, 23, "Wednesday"]
    ]
    df_expected_output = pd.DataFrame(data=expected_output, columns=["column1", "date_time", "month", "hour", "day_of_week"])
    df_test_output = src.data_preprocessing.create_datetime_features(data=df_input_test,
                                                                  original_datetime_column="date_time",
                                                                  month_column="month",
                                                                  hour_column="hour",
                                                                  day_of_week_column="day_of_week")
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)

def test_create_datetime_features_invalid_date():
    input_test = [
        [1.0, "2022-06-01 23:13:45"],
        [1.0, "2022-05-32 08:00:00"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "date_time"])

    expected_output = [
        [1.0, pd.to_datetime("2022-06-01 23:13:45"), 6, 23, "Wednesday"]
    ]
    df_expected_output = pd.DataFrame(data=expected_output, columns=["column1", "date_time", "month", "hour", "day_of_week"])
    df_test_output = src.data_preprocessing.create_datetime_features(data=df_input_test,
                                                                     original_datetime_column="date_time",
                                                                     month_column="month",
                                                                     hour_column="hour",
                                                                     day_of_week_column="day_of_week")
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)

def test_validate_date_time():
    assert src.data_preprocessing.validate_date_time("2022-06-01 23:13:45"), pd.to_datetime("2022-06-01 23:13:45")

def test_validate_date_time_invalid_date():
    assert src.data_preprocessing.validate_date_time("2022-06-32 23:13:45") is None

def test_fahrenheit_to_kelvin():
    input_value = 32
    expected_output = 273.15
    test_output = src.data_preprocessing.fahrenheit_to_kelvin(32)
    assert expected_output == test_output


def test_fahrenheit_to_kelvin_invalid_datatype():
    with pytest.raises(TypeError):
        src.data_preprocessing.fahrenheit_to_kelvin(temp_deg_f="abc")


def test_one_hot_encoding():
    input_test = [
        [1.0, "A"],
        [1.0, "B"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    expected_output = [
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    df_expected_output = pd.DataFrame(data=expected_output,
                                      columns=["column1", "column2_B"])

    expected_one_hot_encoder = sklearn.preprocessing.OneHotEncoder(drop='first', sparse=False)
    expected_one_hot_encoder.fit(df_input_test[["column2"]])

    df_test_output, test_output_one_hot_encoder = src.data_preprocessing.one_hot_encoding(data=df_input_test,
                                                                                          one_hot_encode_columns=["column2"],
                                                                                          sparse=False,
                                                                                          drop="first")

    pd.testing.assert_frame_equal(df_expected_output, df_test_output)
    assert expected_one_hot_encoder.get_params() == test_output_one_hot_encoder.get_params()

def test_one_hot_encoding():
    input_test = [
        [1.0, "A"],
        [1.0, "B"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    expected_output = [
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    df_expected_output = pd.DataFrame(data=expected_output,
                                      columns=["column1", "column2_B"])

    expected_one_hot_encoder = sklearn.preprocessing.OneHotEncoder(drop='first', sparse=False)
    expected_one_hot_encoder.fit(df_input_test[["column2"]])

    df_test_output, test_output_one_hot_encoder = src.data_preprocessing.one_hot_encoding(data=df_input_test,
                                                                                          one_hot_encode_columns=["column2"],
                                                                                          sparse=False,
                                                                                          drop="first")

    pd.testing.assert_frame_equal(df_expected_output, df_test_output)
    assert expected_one_hot_encoder.get_params() == test_output_one_hot_encoder.get_params()

def test_one_hot_encoding_invalid_columns():
    input_test = [
        [1.0, "A"],
        [1.0, "B"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    df_expected_output = pd.DataFrame()
    expected_one_hot_encoder = sklearn.preprocessing.OneHotEncoder(drop='first', sparse=False)

    df_test_output, test_output_one_hot_encoder = src.data_preprocessing.one_hot_encoding(data=df_input_test,
                                                                                          one_hot_encode_columns=["INVALID_COLUMN"],
                                                                                          sparse=False,
                                                                                          drop="first")

    pd.testing.assert_frame_equal(df_expected_output, df_test_output)
    assert expected_one_hot_encoder.get_params() == test_output_one_hot_encoder.get_params()

def test_create_train_test_split():
    input_test = [
        [1.0, 2.0],
        [1.0, 2.0]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    expected_output_train = [
        [1.0, 2.0]
    ]
    df_expected_output_train = pd.DataFrame(data=expected_output_train, columns=["column1", "column2"])

    expected_output_test = [
        [1.0, 2.0]
    ]
    df_expected_output_test = pd.DataFrame(data=expected_output_test, columns=["column1", "column2"], index = [1])

    df_output_train, df_output_test = src.data_preprocessing.create_train_test_split(data = df_input_test,
                                                                                     test_size = 0.5,
                                                                                     random_state=24,
                                                                                     shuffle=True)

    pd.testing.assert_frame_equal(df_expected_output_train, df_output_train)
    pd.testing.assert_frame_equal(df_expected_output_test, df_output_test)


def test_create_train_test_split_invalid_parameters():
    input_test = [
        [1.0, 2.0],
        [1.0, 2.0]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    df_expected_output_train = pd.DataFrame()
    df_expected_output_test = pd.DataFrame()

    df_output_train, df_output_test = src.data_preprocessing.create_train_test_split(data = df_input_test,
                                                                                     test_size = 1.5,
                                                                                     random_state=24,
                                                                                     shuffle=True)

    pd.testing.assert_frame_equal(df_expected_output_train, df_output_train)
    pd.testing.assert_frame_equal(df_expected_output_test, df_output_test)