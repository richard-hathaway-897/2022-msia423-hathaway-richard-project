import logging
import typing

import sqlite3
import sqlalchemy.exc
import pandas as pd

import src.read_write_functions
import src.preprocess_app_input
import src.predict
from src.create_tables_rds import QueryManager, ActivePrediction

logger = logging.getLogger(__name__)


def run_app_prediction(new_query_params: dict,
                       model_object_path: str,
                       one_hot_encoder_path: str,
                       config_dict: dict) -> typing.Tuple[pd.Series, str]:
    """This function is an orchestration function that runs the necessary steps to generate a prediction based on
        user input from the web app. It calls functions to read the model objects, validate the user input, transform
        the user input, and generate the predictions.

    Args:
        new_query_params (dict): Input query from the web app as a dictionary, with each input being a key-value pair.
        model_object_path (str): The path to the trained model object.
        one_hot_encoder_path (str): The path to the fit one-hot-encoder.
        config_dict (dict): The configuration YAML file loaded in as a dictionary.

    Returns:
        prediction (pd.Series): The prediction as a pandas series. For the web app, because the user can only enter
            in one query at a time, the prediction will be a pandas series with one entry.
        traffic_volume (str): A string of either 'light', 'medium', or 'heavy' classifying the traffic prediciton.
    """

    # Try to read the one-hot-encoder
    try:
        one_hot_encoder = src.read_write_functions.load_model_object(one_hot_encoder_path)
    except ValueError as val_error:
        # This error will occur if the one-hot-encoder cannot be read.
        logger.error("Failed to load in the one-hot-encoder object.")
        raise val_error

    # Try to read the model object
    try:
        model = src.read_write_functions.load_model_object(model_object_path)
    except ValueError as val_error:
        # This error will occur if the model object cannot be read.
        logger.error("Failed to load in the trained model object.")
        raise val_error

    # Try to validate the user input.
    try:
        predictors = \
            src.preprocess_app_input.validate_app_input(new_query_params,
                                                        config_dict["process_user_input"]["validate_user_input"])
    except ValueError as val_error:
        # This error will occur if the user input fails the validation step.
        logger.error("An input data type was not valid. %s", val_error)
        raise val_error
    logger.info("Successfully validated user input.")

    # Try to run prediction pre-processing steps such as data transformations and one-hot-encoding on the input
    # user data.
    try:
        prediction_df = src.preprocess_app_input.\
            predict_preprocess(predictors=predictors,
                               one_hot_encoder=one_hot_encoder,
                               remove_outlier_params=config_dict["remove_outliers"],
                               **config_dict["process_user_input"]["app_input_transformations"],
                               **config_dict["generate_features"]["pipeline_and_app"])
    # Catch both TypeError and KeyError in one except block. More detailed exception handling occurs in the module,
    # where there are custom logging messages for each individual exception. Here, because I want to handle these
    # errors in the same way, it is more succinct to catch them in one except block.
    except (TypeError, KeyError) as preprocess_error:
        logger.error("Failed to complete data preprocessing steps of user input. %s", preprocess_error)
        raise preprocess_error
    logger.info("Successfully preprocessed user input.")

    # Create the prediction
    try:
        prediction = src.predict.make_predictions(prediction_df, model=model, is_test_data = False,
                                                  **config_dict["predict"])
    # Catch both ValueError and KeyError in one except block. More detailed exception handling occurs in the module,
    # where there are custom logging messages for each individual exception. Here, because I want to handle these
    # errors in the same way, it is more succinct to catch them in one except block.
    except (KeyError, ValueError) as prediction_error:
        logger.error("Failed to make prediction. %s", prediction_error)
        raise prediction_error
    logger.info("Successfully made prediction.")
    # Classify the prediction into light, medium, or heavy traffic.
    try:
        traffic_volume = src.predict.classify_traffic(prediction[0])
    except ValueError as val_error:
        # This error can occur if the input prediction is negative. A logger message is printed inside the
        # classify_traffic function.
        raise val_error

    return prediction, traffic_volume


def run_update_historical_queries(query_manager: QueryManager,
                                  new_query_params: dict,
                                  prediction: float) -> None:
    """This function is an orchestration function that runs functions for updating the HistoricalQueries table
    after a user makes a query by either adding the query to the table, or, if the query already exists in the database,
    incrementing the count for the number of times the app has seen this particular query.

    Args:
        query_manager (src.create_tables_rds.QueryManager): The QueryManager object that creates the sqlalchemy
            connection to the database.
        new_query_params (dict): The user input as a dictionary.
        prediction (float): The value of the model's prediction.

    Returns:
        This function does not return any objects.

    Raises:
        sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
            and fails.
        sqlalchemy.exc.OperationalError: This exception will be raised if the app tries to acces the AWS RDS instance
            and fails.
    """
    # Throughout this function, sqlite3.OperationalError and sqlalchemy.exc.OperationalError are caught in one except
    # block because these errors need to be handled in the same way. It does not matter which database failed to
    # connect, just that the database connection failed.

    # Search the database for the user's input query to see if it already exists in the database. Get the row count.
    try:
        query_count = query_manager.search_for_query_count(query_params=new_query_params)
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Error page returned. Not able to locate query in the database. %s ", database_exception)
        raise database_exception

    # If the query does not exist in the database (the row count will be zero):
    if query_count == 0:

        # Try to add the query to the database.
        try:
            query_manager.add_new_query(query_params=new_query_params,
                                        query_prediction=prediction)

        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
            logger.error("Error page returned. Not able to add query to the database. %s ", database_exception)
            raise database_exception
        else:
            logger.info("Successfully added query to the historical queries table.")

    # Else, if the query does already exist, increment the count for the particular query.
    else:
        try:
            query_manager.increment_query_count(query_params=new_query_params)
        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
            logger.error("Error page returned. Not able to increment query count in the database. %s ",
                         database_exception)
            raise database_exception
        else:
            logger.info("Successfully incremented the query count for the passed query.")


def run_update_active_prediction(query_manager: QueryManager,
                                 prediction: float,
                                 traffic_volume: str) -> None:
    """This function is an orchestration function that runs functions for updating the ActivePrediction table
    after a user makes a query. If the ActivePrediction table is empty, than the active prediction row is initialized,
    otherwise, the row is overwritten with the new most recent ("Active") prediction.

    Args:
        query_manager (src.create_tables_rds.QueryManager): The QueryManager object that creates the sqlalchemy
            connection to the database.
        prediction (float): The value of the model's prediction.
        traffic_volume (str): A string of either 'light', 'medium', or 'heavy' that indicates the traffic level.

    Returns:
        This function does not return any objects.

    Raises:
        sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
            and fails.
        sqlalchemy.exc.OperationalError: This exception will be raised if the app tries to access a the AWS RDS instance
            and fails.
    """
    # Throughout this function, sqlite3.OperationalError and sqlalchemy.exc.OperationalError are caught in one except
    # block because these errors need to be handled in the same way. It does not matter which database failed to
    # connect, just that the database connection failed.

    # Determine if the table is empty or not
    try:
        row_count = query_manager.session.query(ActivePrediction).count()
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Error page returned. Not able to query active prediction table in the database. %s ",
                     database_exception)
        raise database_exception

    # If the table is empty (The row count will be zero)
    if row_count == 0:
        # Initialize the active prediction row in the table.
        try:
            query_manager.create_most_recent_query()
        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
            logger.error("Error page returned. Not able to create an empty row in the ActivePredictions table in the "
                         "database. %s ", database_exception)
            raise database_exception
        else:
            logger.info("Created an empty row in ActivePredictions table.")

    # Overwrite the row in the ActivePredictions table with the most recent prediction.
    try:
        query_manager.update_active_prediction(new_prediction_value=prediction, new_volume=traffic_volume)
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Error page returned. Not able to update the active prediction in the database. %s ",
                     database_exception)
        raise database_exception
    else:
        logger.info("Updated the active prediction.")
