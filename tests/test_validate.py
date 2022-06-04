import pandas as pd
import pytest

import src.validate


def test_validate_dataframe():
    input_test = [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0]
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["column1", "column2"])

    true_output = src.validate.validate_dataframe(df_input_test, duplicated_method='first')

    assert true_output

def test_validate_dataframe_empty():

    df_input_test = pd.DataFrame()

    with pytest.raises(ValueError):
        src.validate.validate_dataframe(df_input_test, duplicated_method='first')

