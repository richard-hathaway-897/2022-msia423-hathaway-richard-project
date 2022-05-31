import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def train_model(train_data: pd.DataFrame,
                response_column,
                n_estimators,
                criterion,
                min_samples_split,
                max_features,
                oob_score,
                n_jobs,
                random_state):

    # TODO: Data Validation when reading in. CHECK FOR NULLS!
    try:
        predictors = train_data.drop([response_column], axis=1)
    except KeyError as key_error:
        logger.error(key_error)
        raise key_error

    response = train_data[response_column]
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
        logger.error("Failed to fit model. %s", val_error)
    else:
        logger.info("Random Forest training OOB Score was: %f", rf_model.oob_score_)

    return rf_model

        # test_r2 = rf_model.score(X = test_predictors, y = test_response)
        # logger.info("Random Forest R^2 on validation set: %f", test_r2)
        #
        # # TODO Add option to not upload to S3 and just save locally.
        # joblib.dump(rf_model, model_output_local_path)
        #
        # if model_output_s3_path is not "None":
        #     src.s3_actions.s3_write_from_file(model_output_local_path, model_output_s3_path)




