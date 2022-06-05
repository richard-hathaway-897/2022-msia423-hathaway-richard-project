import pandas as pd
import pytest

import src.train_model


def test_train_model():
    """This unit test tests the successful execution of the train_model function. It should train a
    RandomForestRegressor sklearn model. It tests if the parameters of the expected random forest model are the same
    as the output random forest model."""

    # Define data for model training
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
    df_model_training = pd.DataFrame(data=model_training, columns=["response", "column1", "column2"])

    # Define the expected parameters of the random forest model
    expected_output_params = {
        "bootstrap": True, "ccp_alpha": 0.0, "criterion": "squared_error", "max_depth": None, "max_features": 2,
        "max_leaf_nodes": None, "max_samples": None, "min_impurity_decrease": 0.0, "min_samples_leaf": 1,
        "min_samples_split": 2, "min_weight_fraction_leaf": 0.0, "n_estimators": 10, "n_jobs": -1, "oob_score": True,
        "random_state": 24, "verbose": 0, "warm_start": False
    }

    true_output_model = src.train_model.train_model(train_data=df_model_training,
                                                    response_column="response",
                                                    n_estimators=10,
                                                    criterion="squared_error",
                                                    min_samples_split=2,
                                                    max_features=2,
                                                    oob_score=True,
                                                    n_jobs=-1,
                                                    random_state=24)

    true_output_attrs = true_output_model.get_params()

    assert expected_output_params == true_output_attrs


def test_train_model_no_response_column():
    """This unit test tests the execution of the train_model function when the wrong name for the response column
    is passed to the function, and that column does not exist in the training dataframe. It should raise a KeyError.
    """

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
    df_model_training = pd.DataFrame(data=model_training, columns=["response", "column1", "column2"])

    with pytest.raises(KeyError):
        src.train_model.train_model(train_data=df_model_training,
                                    response_column="INVALID_RESPONSE_COLUMN",
                                    n_estimators=10,
                                    criterion="squared_error",
                                    min_samples_split=2,
                                    max_features=2,
                                    oob_score=True,
                                    n_jobs=-1,
                                    random_state=24)
