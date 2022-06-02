"""
Unit tests for the module generate_features.py
It both 'happy' and 'unhappy' tests the following functions:
    1. log_transform
    2. column_multiplication
    3. create_range
    4. create_norm_range
"""

import numpy as np
import pandas as pd
import pytest

import src.clean_data

# Define the test dataframe values to pass to each function inside of each unit test.

# Define the test dataframe indices and columns
# df_in_index_test_generate_features = [0, 1, 2]
#
# df_in_columns_test_generate_features = [
#     "visible_mean",
#     "visible_max",
#     "visible_min",
#     "visible_mean_distribution",
#     "visible_contrast",
#     "visible_entropy",
#     "visible_second_angular_momentum",
#     "IR_mean",
#     "IR_max",
#     "IR_min",
#     "class",
# ]

# Create the test dataframe



def test_clean_data() -> None:
    """



    """
    input_test_clean_data = [
        [1.0, 2.0, np.nan],
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0]
    ]
    df_input_test_clean_data = pd.DataFrame(data=input_test_clean_data)

    expected_output_clean_data = [
        [1.0, 2.0, 3.0]
    ]
    df_expected_output_clean_data = pd.DataFrame(data=expected_output_clean_data)

    df_test_output_clean_data = src.clean_data.clean_data(data=df_input_test_clean_data, duplicated_method='first')
    pd.testing.assert_frame_equal(df_expected_output_clean_data, df_test_output_clean_data)


def test_clean_data_wrong_input_type() -> None:
    """

    """
    df_input_test_clean_data = "This object is not a pandas dataframe."

    with pytest.raises(TypeError):
        df_test_output_clean_data = src.clean_data.clean_data(data=df_input_test_clean_data, duplicated_method='first')
