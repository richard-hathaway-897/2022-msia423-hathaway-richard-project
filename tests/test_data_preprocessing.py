import numpy as np
import pandas as pd
import pytest

import src.data_preprocessing


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
