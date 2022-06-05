import numpy as np
import pandas as pd
import pytest

import src.clean_data


def test_clean_data() -> None:
    """This unit test tests the successful execution of the clean_data function. If successful, it should return
    the dataframe cleaned without duplicate rows or rows with NaN values.
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

    df_test_output_clean_data = src.clean_data.clean_data(data=df_input_test_clean_data, duplicated_method="first")
    pd.testing.assert_frame_equal(df_expected_output_clean_data, df_test_output_clean_data)


def test_clean_data_wrong_input_type() -> None:
    """This unit test tests the execution of the clean_data function when an object that is not a pandas dataframe is
    passed to the function. It should raise a TypeError.

    """
    df_input_test_clean_data = "This object is not a pandas dataframe."

    with pytest.raises(TypeError):
        src.clean_data.clean_data(data=df_input_test_clean_data, duplicated_method="first")
