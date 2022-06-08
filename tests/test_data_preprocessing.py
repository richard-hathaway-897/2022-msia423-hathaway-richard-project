import numpy as np
import pandas as pd
import sklearn
import pytest

import src.data_preprocessing


def test_columns_drop() -> None:
    """This unit test tests the successful execution of the columns_drop function, which should drop the specified
    columns.
    """

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


def test_columns_drop_invalid_column() -> None:
    """This unit test tests the "unhappy path" of the columns_drop function. It tests if a column that does not exist
    in the dataframe is passed as the column to drop. The function should gracefully recover and just returned the
    original dataframe that was passed in.
    """

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

    df_expected_output_drop_cols = pd.DataFrame(data=expected_output_drop_cols,
                                                columns=["column1", "column2", "column3"])
    df_test_output_drop_cols = src.data_preprocessing.columns_drop(data=df_input_test_drop_cols, columns=["column4"])
    pd.testing.assert_frame_equal(df_expected_output_drop_cols, df_test_output_drop_cols)


def test_log_transform() -> None:
    """This unit test tests the successful execution of the log transform function.

    """

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


def test_log_transform_invalid_column() -> None:
    """This unit test tests the execution of the log_transform function when an column name that is not in the dataframe
    is passed to the function. It should raise a KeyError.
    """

    input_test = [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0]
        ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    with pytest.raises(KeyError):
        src.data_preprocessing.log_transform(data=df_input_test,
                                             log_transform_column_names=["column3"],
                                             log_transform_new_column_prefix="log_")


def test_binarize_column() -> None:
    """This unit test tests the successful execution of the binarize_column function. It should transform a specfied
    column into a 0-1 binary variable.

    """

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


def test_binarize_column_invalid_column() -> None:
    """This unit test tests the execution of the binarize_column function when a column name that is not in the
    dataframe is passed to the function. It should raise a KeyError.
    """
    input_test = [
        [1.0, "None"],
        [1.0, "Other Value A"],
        [1.0, "Other Value B"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    with pytest.raises(KeyError):
        src.data_preprocessing.binarize_column(data=df_input_test,
                                               binarize_column_names=["column3"],
                                               binarize_new_column_prefix="binarize_",
                                               binarize_zero_value="None")


def test_binarize() -> None:
    """This unit test tests the successful execution of the binarize function. It should transform an input value into
    a 0 or a 1.

    """
    input_test = "None"
    expected_output = 0
    test_output = src.data_preprocessing.binarize(value = input_test, zero_value="None")
    assert test_output == expected_output


def test_binarize_invalid_zero_value_type() -> None:
    """This unit test tests the execution of the binarize function when an unexpected value is passed as the zero_value
    (for example, a pandas dataframe instead of the required string). The function should be able to
    recover from this error and return the value of 1.

    """
    input_test = "None"
    expected_output = 1
    test_output = src.data_preprocessing.binarize(value=input_test, zero_value=pd.DataFrame())
    assert test_output == expected_output


def test_create_datetime_features() -> None:
    """This unit test tests the successful execution of the create_datetime_features function. It should transform
    extract datetime features such as month, hour, and day of week from the input date-time column.

    """
    input_test = [
        [1.0, "2022-05-31 08:00:00"],
        [1.0, "2022-06-01 23:13:45"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "date_time"])

    expected_output = [
        [1.0, pd.to_datetime("2022-05-31 08:00:00"), 5, 8, "Tuesday"],
        [1.0, pd.to_datetime("2022-06-01 23:13:45"), 6, 23, "Wednesday"]
    ]
    df_expected_output = pd.DataFrame(data=expected_output,
                                      columns=["column1", "date_time", "month", "hour", "day_of_week"])
    df_test_output = src.data_preprocessing.create_datetime_features(data=df_input_test,
                                                                     original_datetime_column="date_time",
                                                                     month_column="month",
                                                                     hour_column="hour",
                                                                     day_of_week_column="day_of_week")
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)


def test_create_datetime_features_invalid_column() -> None:
    """This unit test tests the execution of the create_datetime_features function when a column name that is not in the
    dataframe is passed to the function as the column that contains the date-time features. It should raise a KeyError.
    """
    input_test = [
        [1.0, "2022-06-01 23:13:45"],
        [1.0, "2022-05-32 08:00:00"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "INVALID_COLUMN"])

    with pytest.raises(KeyError):
        src.data_preprocessing.create_datetime_features(data=df_input_test,
                                                        original_datetime_column="date_time",
                                                        month_column="month",
                                                        hour_column="hour",
                                                        day_of_week_column="day_of_week")


def test_validate_date_time() -> None:
    """This unit test tests the successful execution of the validate_date_time function. It converts a date time string
    to a datetime object.
    """

    assert src.data_preprocessing.validate_date_time("2022-06-01 23:13:45"), pd.to_datetime("2022-06-01 23:13:45")


def test_validate_date_time_invalid_date() -> None:
    """This unit test tests the execution of the validate_date_time function when the input string cannot be parsed
    as a date (for example, June 32). It should return None.
    """
    assert src.data_preprocessing.validate_date_time("2022-06-32 23:13:45") is None


def test_fahrenheit_to_kelvin() -> None:
    """This unit test tests the successful execution of the fahrenheit_to_kelvin to function.
    """
    expected_output = 273.15
    test_output = src.data_preprocessing.fahrenheit_to_kelvin(32)
    assert expected_output == test_output


def test_fahrenheit_to_kelvin_invalid_datatype() -> None:
    """This unit test tests the execution of the fahrenheit_to_kelvin to function when the input is not a numeric
    value. It should raise a TypeError.
    """
    with pytest.raises(TypeError):
        src.data_preprocessing.fahrenheit_to_kelvin(temp_deg_f="abc")


def test_one_hot_encoding() -> None:
    """This unit test tests the successful execution of the one_hot_encoding function. It should one-hot-encode
    the input data and return also return one-hot-encoded object.
    """
    # Define input test data
    input_test = [
        [1.0, "A"],
        [1.0, "B"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    # define expected output dataframe
    expected_output = [
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    df_expected_output = pd.DataFrame(data=expected_output,
                                      columns=["column1", "column2_B"])

    # Define the expected output one-hot-encoder.
    expected_one_hot_encoder = sklearn.preprocessing.OneHotEncoder(drop="first", sparse=False)
    expected_one_hot_encoder.fit(df_input_test[["column2"]])

    df_test_output, test_output_one_hot_encoder = \
        src.data_preprocessing.one_hot_encoding(data=df_input_test,
                                                one_hot_encode_columns=["column2"],
                                                sparse=False,
                                                drop="first")
    # Assert the dataframes are equal and that the one-hot-encoders have the same parameters.
    pd.testing.assert_frame_equal(df_expected_output, df_test_output)
    assert expected_one_hot_encoder.get_params() == test_output_one_hot_encoder.get_params()


def test_one_hot_encoding_invalid_columns() -> None:
    """This unit test tests the execution of the one_hot_encoding function when a column name that is not in the
    dataframe is passed to the function as a column to one-hot-encode. It should raise a KeyError.
    """
    input_test = [
        [1.0, "A"],
        [1.0, "B"]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    with pytest.raises(KeyError):
        src.data_preprocessing.one_hot_encoding(data=df_input_test,
                                                one_hot_encode_columns=["INVALID_COLUMN"],
                                                sparse=False,
                                                drop="first")


def test_create_train_test_split() -> None:
    """This function tests the successful execution of the create_train_test_split function.
    """
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

    # asser that both the output and expected training data are equal and the output and expected test data are equal.
    pd.testing.assert_frame_equal(df_expected_output_train, df_output_train)
    pd.testing.assert_frame_equal(df_expected_output_test, df_output_test)


def test_create_train_test_split_invalid_parameters() -> None:
    """This unit test tests the execution of the create_train_test_split function when invalid parameters for creating
    a train test split are passed, such as a test size of 1.5. It should raise a ValueError.
    """
    input_test = [
        [1.0, 2.0],
        [1.0, 2.0]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    with pytest.raises(ValueError):
        src.data_preprocessing.create_train_test_split(data=df_input_test,
                                                       test_size=1.5,
                                                       random_state=24,
                                                       shuffle=True)
