import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pytest

import src.predict


def test_make_predictions():
    """This unit test tests the execution of the make_predictions function. It should output a pandas Series with the
    correct predictions."""

    # First, fit a random forest model to use to make predictions for this unit test.
    model_training = [
        [1.0, 9.0, 1.0],
        [3.0, 5.0, 2.0],
        [3.0, 6.0, 3.0],
        [8.0, 4.0, 4.0],
        [3.0, 5.0, 5.0],
        [7.0, 10.0, 4.0],
        [1.0, 4.0, 3.0],
        [23.0, 5.0, 2.0],
        [3.0, 2.0, 1.0],
        [19.0, 4.0, 0.0],
        [7.0, 5.0, 10.0],
        [3.0, 2.0, 7.0]
    ]
    df_model_training = pd.DataFrame(data=model_training, columns = ["response", "column1", "column2"])
    rf_test = RandomForestRegressor(n_estimators=10,
                                 criterion="squared_error",
                                 min_samples_split=2,
                                 max_features=2,
                                 random_state=24)
    rf_test.fit(X=df_model_training[["column1", "column2"]], y=df_model_training["response"])

    # Now, define the input test data and the expected output and use the random forest model to make the prediction
    # on the input test data.
    input_test = [
        [2.0, 1.0, 3.0],
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["response", "column1", "column2"])

    expected_output = pd.Series(rf_test.predict(X=df_input_test[["column1", "column2"]]))

    test_output = src.predict.make_predictions(new_data=df_input_test,
                                               model=rf_test,
                                               response_column="response",
                                               is_test_data=True)
    pd.testing.assert_series_equal(expected_output, test_output)


def test_make_predictions_model_not_fit():
    """This unit test tests the execution of the make_predictions function when the model object is not fit.
    It should raise a ValueError."""

    rf_test = RandomForestRegressor(n_estimators=10,
                                    criterion="squared_error",
                                    min_samples_split=2,
                                    max_features=2,
                                    random_state=24)

    input_test = [
        [2.0, 1.0, 3.0],
    ]
    df_input_test = pd.DataFrame(data=input_test, columns=["response", "column1", "column2"])

    with pytest.raises(ValueError):
        src.predict.make_predictions(new_data=df_input_test,
                                     model=rf_test,
                                     response_column="response",
                                     is_test_data=True)


def test_classify_traffic():
    """This unit test tests the successful execution of the classify traffic function.
    """
    assert "light" == src.predict.classify_traffic(traffic_prediction=1000)


def test_classify_traffic_negative_value():
    """This unit test tests the execution of the classify traffic function when the input predicton is negative.
    It should raise a ValueError
    """
    with pytest.raises(ValueError):
        src.predict.classify_traffic(traffic_prediction=-100)
