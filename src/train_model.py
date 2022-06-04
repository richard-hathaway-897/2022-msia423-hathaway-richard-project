import logging

import pandas as pd
import sklearn.ensemble
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def train_model(train_data: pd.DataFrame,
                response_column: str,
                n_estimators: int,
                criterion: str,
                min_samples_split: int,
                max_features: int,
                oob_score: bool,
                n_jobs: int,
                random_state: int) -> sklearn.ensemble.RandomForestRegressor:
    """This function trains a Random Forest Regressor model from sklearn using input training data and specified
        parameters. See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        for more information.

    Args:
        train_data (pd.DataFrame): The training data as an input training data.
        response_column (str): The name of the response column.
        n_estimators (int): The number of trees to grow in the random forest.
        criterion (str): The error criterion by which to split nodes in the tree.
        min_samples_split (int): The minimum number of samples required in a node to split the tree.
        max_features (int): The number of variables to randomly select at each split in the random forest.
        oob_score (bool): Whether to calculate the training out-of-bag score.
        n_jobs (int): The number of parallel jobs to run. For more information, see:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        random_state (int): The random state for the random forest to ensure model reproducability.

    Returns:
        rf_model (sklearn.ensemble.RandomForestRegressor): The trained random forest regressor model.

    Raises:
        KeyError: A KeyError will be raised if the response column is not found in the training data.
        ValueError: A ValueError will be raised if fitting the Random Forest model fails because invalid parameters
            were passed to the model object.

    """
    # Separate the predictors and response.
    try:
        predictors = train_data.drop([response_column], axis=1)
    except KeyError as key_error:
        # This error will occur if the training data does not contain the response column.
        logger.error("The specified response column does not exist in the training data. %s", key_error)
        raise key_error
    response = train_data[response_column]

    # Define the random forest model and attempt to fit it.
    rf_model = RandomForestRegressor(n_estimators=n_estimators,
                                     criterion=criterion,
                                     min_samples_split=min_samples_split,
                                     max_features=max_features,
                                     oob_score=oob_score,
                                     n_jobs=n_jobs,
                                     random_state=random_state)
    try:
        rf_model.fit(X=predictors, y=response)
    except ValueError as val_error:
        # This error can occur if the parameters of the Random Forest Model are in valid.
        logger.error("Failed to fit model. %s", val_error)
        raise val_error
    else:
        logger.info("Random Forest training OOB Score was: %f", rf_model.oob_score_)

    return rf_model
