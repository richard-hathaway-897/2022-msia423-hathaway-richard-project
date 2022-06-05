import pandas as pd
import pytest

import src.validate


def test_validate_dataframe():
    """This unit test tests the successful execution of the validate_dataframe function. It should return True if
    the data is able to be validated.
    """
    input_test = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    true_output = src.validate.validate_dataframe(df_input_test, duplicated_method='first')

    assert true_output


def test_validate_dataframe_invalid_duplicated_input():
    """This unit test tests the execution of the validate_dataframe function when an invalid value is passed as the
    duplicated_method, such as an integer. It should raise a ValueError.
    """
    df_input_test = pd.DataFrame()

    with pytest.raises(ValueError):
        src.validate.validate_dataframe(df_input_test, duplicated_method=34)

