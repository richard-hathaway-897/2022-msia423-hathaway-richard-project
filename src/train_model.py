import logging

import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection

import src.s3_actions

logger = logging.getLogger(__name__)


def train_model(model_data_source: str, delimiter: str = ","):

    # TODO: Data Validation when reading in. CHECK FOR NULLS!
    # TODO: CHECK that traffic_volume exists. Better yet, put it in config
    # TODO: use yaml file

    try:
        data = src.s3_actions.s3_read(s3_source=model_data_source, delimiter=delimiter)
    except ValueError as value_error:
        logger.error("Failed to read data in preprocessing.")
        raise ValueError(value_error)

    try:
        predictors = data.drop(["traffic_volume"], axis=1)
    except KeyError as key_error:
        logger.error(key_error)
        raise key_error

    response = data["traffic_volume"]

    train_predictors, test_predictors, train_response, test_response = \
        sklearn.model_selection.train_test_split(predictors, response, test_size=0.2, random_state=123, shuffle=True)

    rf = RandomForestRegressor(n_estimators=200, criterion="squared_error",
                               min_samples_split=5, max_features="sqrt",
                               oob_score=True, n_jobs=-1)

    try:
        rf.fit(X=train_predictors, y=train_response)
    except ValueError as val_error:
        logger.error("Failed to fit model. %s", val_error)
    else:
        logger.info("Random Forest OOB Score: %f", rf.oob_score_)
        test_r2 = rf.score(X = test_predictors, y = test_response)
        logger.info("Random Forest R^2 on validation set: %f", test_r2)




