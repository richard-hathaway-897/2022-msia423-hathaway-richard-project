import logging

import pandas as pd
import sqlite3
import sqlalchemy.exc

import src.read_write_functions
import src.preprocess_app_input
import src.predict
from src.create_tables_rds import QueryManager, ActivePrediction

logger = logging.getLogger(__name__)

def run_app_prediction(new_query_params: dict, model_object_path: str, one_hot_encoder_path: str, config_dict: dict):

    logger.info("Making predictions")
    one_hot_encoder = src.read_write_functions.load_model_object(one_hot_encoder_path)

    if one_hot_encoder is None:
        logger.error("Could not load the one hot encoder.")
        raise FileNotFoundError
    model = src.read_write_functions.load_model_object(model_object_path)

    if model is None:
        logger.error("Could not load the trained model object.")
        raise FileNotFoundError

    try:
        predictors = src.preprocess_app_input.validate_app_input(new_query_params, config_dict["process_user_input"]["validate_user_input"])
    except ValueError as val_error:
        logger.error("An input data type was not valid.")
        raise val_error

    try:
        prediction_df = src.preprocess_app_input.predict_preprocess(predictors=predictors,
                                                                    one_hot_encoder=one_hot_encoder,
                                                                    remove_outlier_params=config_dict["remove_outliers"],
                                                                    **config_dict["generate_features"]["pipeline_and_app"])
    except (ValueError, KeyError) as preprocess_error:
        logger.error("Failed to complete data preprocessing steps of user input.")
        raise preprocess_error

    prediction = src.predict.make_predictions(prediction_df,
                                              model=model,
                                              is_test_data = False,
                                              **config_dict["predict"])
    if prediction.empty:
        logger.error("No prediction was made.")
        raise ValueError("The prediction could not be made.")
    else:
        traffic_volume = src.predict.classify_traffic(prediction[0])

    logger.info("Prediction: %f", prediction[0])

    return prediction, traffic_volume


def run_update_historical_queries(query_manager:QueryManager,
                                  new_query_params: dict,
                                  database_uri_string: str,
                                  prediction: float):
    try:
        query_count = query_manager.search_for_query_count(query_params=new_query_params)
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Error page returned. Not able to locate query in the database: %s. Error: %s ",
                     database_uri_string, database_exception)
        raise database_exception

    if query_count == 0:

        try:
            query_manager.add_new_query(query_params=new_query_params,
                                        query_prediction=prediction)

        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
            logger.error("Error page returned. Not able to add query to the database: %s. Error: %s ",
                         database_uri_string, database_exception)
            raise database_exception
        else:
            logger.info("Successfully added query to the historical queries table.")

    else:
        try:
            query_manager.increment_query_count(query_params=new_query_params)
        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
            logger.error("Error page returned. Not able to increment query count in the database: %s. Error: %s ",
                         database_uri_string, database_exception)
            raise database_exception
        else:
            logger.info("Successfully incremented the query count for the passed query.")


def run_update_active_prediction(query_manager: QueryManager,
                                 database_uri_string: str,
                                 prediction: float,
                                 traffic_volume: str):

    try:
        row_count = query_manager.session.query(ActivePrediction).count()
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Error page returned. Not able to query active prediction table in the database: %s. Error: %s ",
                     database_uri_string, database_exception)
        raise database_exception

    if row_count == 0:
        try:
            query_manager.create_most_recent_query()
        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
            logger.error("Error page returned. Not able to create an empty row in the ActivePredictions table in the "
                         "database: %s. Error: %s ", database_uri_string, database_exception)
            raise database_exception
        else:
            logger.info("Created an empty row in ActivePredictions table.")

    try:
        query_manager.update_active_prediction(new_prediction_value=prediction, new_volume=traffic_volume)
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Error page returned. Not able to update the active prediction in the database: %s. Error: %s ",
                      database_uri_string, database_exception)
        raise database_exception
    else:
        logger.info("Updated the active prediction.")

