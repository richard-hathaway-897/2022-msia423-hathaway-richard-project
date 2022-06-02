import pandas as pd
import numpy as np
import sklearn

import src.evaluate_model


def test_evaluate_model():
    input_test_prediction = [
        [1.0],
        [3.0],
        [3.0]
    ]
    df_input_test_prediction = pd.DataFrame(data=input_test_prediction)

    input_test_true_value = [
        [2.0, 1.0],
        [2.0, 1.0],
        [2.0, 1.0]
    ]
    df_input_test_true_value = pd.DataFrame(data=input_test_true_value, columns = ["response", "other_col"])
    expected_output = {'R^2': sklearn.metrics.r2_score(df_input_test_true_value["response"], df_input_test_prediction),
                       'MSE': sklearn.metrics.mean_squared_error(df_input_test_true_value["response"], df_input_test_prediction)}
    test_output = src.evaluate_model.evaluate_model(test=df_input_test_true_value, predictions=df_input_test_prediction,
                                                    response_column="response")
    assert test_output==expected_output

def test_evaluate_model_too_few_predictions():
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
    df_input_test_true_value = pd.DataFrame(data=input_test_true_value, columns = ["response", "other_col"])
    expected_output = {'R^2': np.nan,
                       'MSE': np.nan}
    test_output = src.evaluate_model.evaluate_model(test=df_input_test_true_value, predictions=df_input_test_prediction,
                                                    response_column="response")
    assert test_output==expected_output