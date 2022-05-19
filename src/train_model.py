import logging

import sklearn.linear_model
import sklearn.preprocessing
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection
import joblib

import src.s3_actions

logger = logging.getLogger(__name__)


def train_model(model_data_source: str, model_training_params: dict, model_output_local_path: str, model_output_s3_path: str, delimiter: str = ","):

    # TODO: Data Validation when reading in. CHECK FOR NULLS!
    try:
        data = src.s3_actions.s3_read(s3_source=model_data_source, delimiter=delimiter)
    except ValueError as value_error:
        logger.error("Failed to read data in preprocessing.")
        raise ValueError(value_error)

    response_col = model_training_params["response_column"]
    try:
        predictors = data.drop([response_col], axis=1)
    except KeyError as key_error:
        logger.error(key_error)
        raise key_error

    response = data[response_col]
    print(predictors)
    train_test_params = model_training_params["train_test_split"]
    train_predictors, test_predictors, train_response, test_response = \
        sklearn.model_selection.train_test_split(predictors,
                                                 response,
                                                 test_size=train_test_params["test_size"],
                                                 random_state=train_test_params["random_state"],
                                                 shuffle=train_test_params["shuffle"])

    random_forest_params = model_training_params["random_forest"]
    rf_model = RandomForestRegressor(n_estimators=random_forest_params["n_estimators"],
                                     criterion=random_forest_params["criterion"],
                                     min_samples_split=random_forest_params["min_samples_split"],
                                     max_features=random_forest_params["max_features"],
                                     oob_score=random_forest_params["oob_score"],
                                     n_jobs=random_forest_params["n_jobs"])

    try:
        rf_model.fit(X=train_predictors, y=train_response)
    except ValueError as val_error:
        logger.error("Failed to fit model. %s", val_error)
    else:
        logger.info("Random Forest OOB Score: %f", rf_model.oob_score_)
        test_r2 = rf_model.score(X = test_predictors, y = test_response)
        logger.info("Random Forest R^2 on validation set: %f", test_r2)

        # TODO Add option to not upload to S3 and just save locally.
        joblib.dump(rf_model, model_output_local_path)

        if model_output_s3_path is not "None":
            src.s3_actions.s3_write_from_file(model_output_local_path, model_output_s3_path)




