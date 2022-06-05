import pandas as pd
import pytest
import sklearn

import src.evaluate_model


def test_evaluate_model():
    """This unit test tests the successful evaluation of the evaluate_model function. It should return a dictionary with
    metrics for R^2 and MSE.
    """
    # Define the input predictions
    input_test_prediction = [
        [1.0],
        [3.0],
        [3.0]
    ]
    df_input_test_prediction = pd.DataFrame(data=input_test_prediction)

    # Define the input test data
    input_test_true_value = [
        [2.0, 1.0],
        [2.0, 1.0],
        [2.0, 1.0]
    ]
    df_input_test_true_value = pd.DataFrame(data=input_test_true_value, columns=["response", "other_col"])
    expected_output = {"R^2": sklearn.metrics.r2_score(df_input_test_true_value["response"],
                                                       df_input_test_prediction),
                       "MSE": sklearn.metrics.mean_squared_error(df_input_test_true_value["response"],
                                                                 df_input_test_prediction)}
    test_output = src.evaluate_model.evaluate_model(test=df_input_test_true_value, predictions=df_input_test_prediction,
                                                    response_column="response")
    assert test_output == expected_output


def test_evaluate_model_too_few_predictions():
    """This unit test tests the execution of the evaluate_model function when the input predictions do not have the
    same shape as the input true values. It should raise a ValueError."""
    input_test_prediction = [
        [1.0],
        [3.0]
    ]
    df_input_test_prediction = pd.DataFrame(data=input_test_prediction)

    input_test_true_value = [
        [2.0, 1.0],
        [2.0, 1.0],
        [2.0, 1.0]
    ]
    df_input_test_true_value = pd.DataFrame(data=input_test_true_value, columns=["response", "other_col"])

    with pytest.raises(ValueError):
        src.evaluate_model.evaluate_model(test=df_input_test_true_value,
                                          predictions=df_input_test_prediction,
                                          response_column="response")
