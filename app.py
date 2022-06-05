import logging.config
import typing

import sqlite3
import sqlalchemy.exc
from flask import Flask, render_template, request, redirect, url_for, Response
import numpy as np

from src.create_tables_rds import (
    QueryManager,
    HistoricalQueries,
    AppMetrics,
    ActivePrediction,
)
import src.predict
import src.validate
import src.read_write_functions
import src.preprocess_app_input
import src.app_module

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile("config/flaskconfig.py")

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])
logger.debug(
    "Web app should be viewable at %s:%s if docker run command maps local "
    "port to the same port as configured for the Docker container "
    "in config/flaskconfig.py (e.g. `-p 5000:5000`). Otherwise, go to the "
    "port defined on the left side of the port mapping "
    "(`i.e. -p THISPORT:5000`). If you are running from a Windows machine, "
    "go to 127.0.0.1 instead of 0.0.0.0.",
    app.config["HOST"],
    app.config["PORT"],
)

# Initialize the database session
try:
    query_manager = QueryManager(app)
except ValueError:
    logger.error("The Flask App failed to be initialized. Invalid Flask App.")
    render_template("database_error.html")
except sqlalchemy.exc.OperationalError as database_error:
    logger.error("The Flask App failed to be initialized. Failed to connect to the database.")
    render_template("database_error.html")


@app.route("/")
def index() -> str:
    """This function queries the database for information needed to render the index page of web application.
    Index.html will display the most recent prediction, the top 5 most popular predictions, and the number of likes
    and dislikes the application has received.
    Index.html is stored in app/templates/index.html template.

    Returns:
        (str): This function returns the rendered index.html page or the rendered database_error.html page if an error
         is encountered.

    Raises:
        This function does not raise any exceptions.

    """
    # For this function, catch both sqlite3.OperationalError and sqlalchemy.exc.OperationalError in one except statement
    # because both exceptions need to be handled in the same way, which is returning the database_error.html page.

    # Search the HistoricalQueries table for the top 5 most popular queries.
    try:
        user_query = (
            query_manager.session.query(HistoricalQueries)
            .order_by(HistoricalQueries.query_count.desc())
            .limit(app.config["MAX_ROWS_SHOW"])
            .all())
        logger.debug("Retrieve top 5 historical queries.")
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Not able to query the database: %s.",
                     database_exception)
        return render_template("database_error.html")

    # Query the ActivePrediction table to get the most recent prediction.
    try:
        most_recent_prediction = query_manager.session.query(ActivePrediction).first()
        logger.debug("Retrieve most recent prediction.")
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Not able to retrieve the active prediction from the database. %s ",
                     database_exception)
        return render_template("database_error.html")

    # Query the AppMetrics table to get the count of likes and dislikes of the web app.
    try:
        like_dislike_count = query_manager.session.query(AppMetrics).all()
        logger.info("Retrieve likes and dislikes.")
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Not able to retrieve the likes and dislikes from the database. %s ",
                     database_exception)
        return render_template("database_error.html")

    logger.debug("Navigate to Index page.")
    return render_template(
        "index.html",
        user_query=user_query,
        like_dislike_count=like_dislike_count,
        prediction=most_recent_prediction)


@app.route("/add", methods=["POST"])
def enter_query_parameters() -> typing.Union[Response, str]:
    """This is an orchestration function that retrieves the user input from the web application form and then calls
    functions that generate a prediction using that user input. It also updates the necessary database tables after
    the prediction is completed.

    Returns:
        typing.Union[Response, str]: This function returns a redirect for the index.html page or renders the template
        of error.html or database_error.html.

    Raises:
        This function does not raise any exceptions.
    """

    # Retrieve the user input from the html form and store in a dictionary.
    new_query_params = {}
    new_query_params["temp"] = request.form["temperature"]
    new_query_params["clouds_all"] = request.form["cloud_percentage"]
    new_query_params["weather_main"] = request.form["weather_description"]
    new_query_params["month"] = request.form["month"]
    new_query_params["hour"] = request.form["hour"]
    new_query_params["day_of_week"] = request.form["day_of_week"]
    new_query_params["holiday"] = request.form["holiday"]
    new_query_params["rain_1h"] = request.form["rainfall_hour"]

    # Try to load in the YAML configuration file.
    try:
        config_dict = src.read_write_functions.read_yaml(app.config["MODEL_CONFIG_PATH"])
    except FileNotFoundError:
        logger.error("Could not load in the YAML configuration file.")
        return render_template("error.html")

    # Create the prediction.
    try:
        prediction, traffic_volume = src.app_module.run_app_prediction(
            new_query_params,
            model_object_path=app.config["PATH_TRAINED_MODEL_OBJECT"],
            one_hot_encoder_path=app.config["PATH_TRAINED_ONE_HOT_ENCODER"],
            config_dict=config_dict)
    # Catch both ValueError and KeyError in one except statement because the exceptions are handled in the same way
    # for both errors.
    except (ValueError, KeyError):
        logger.error("Error: Prediction could not be made due to missing model object, invalid input, "
                     "or invalid configurations.")
        return render_template("error.html")

    # Try to update the HistoricalQueries Table.
    try:
        src.app_module.run_update_historical_queries(
            query_manager=query_manager,
            new_query_params=new_query_params,
            prediction=np.round(prediction[0]))
    # For this function, catch both sqlite3.OperationalError and sqlalchemy.exc.OperationalError in one except statement
    # because both exceptions need to be handled in the same way, which is returning the database_error.html page.
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Error occurred updating the historical queries table. %s", database_exception)
        return render_template("database_error.html")

    # Try to update the ActivePrediction table with the prediction value.
    try:
        src.app_module.run_update_active_prediction(
            query_manager=query_manager,
            prediction=np.round(prediction[0]),
            traffic_volume=traffic_volume)
    # For this function, catch both sqlite3.OperationalError and sqlalchemy.exc.OperationalError in one except statement
    # because both exceptions need to be handled in the same way, which is returning the database_error.html page.
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Error occurred updating the active prediction. %s", database_exception)
        return render_template("database_error.html")

    return redirect(url_for("index"))


@app.route("/like", methods=["POST"])
def increment_like_dislike() -> typing.Union[Response, str]:
    """This function increments the like or dislike count whenever the user clicks the "Like" or "Dislike" button on
    the web app page.

    Returns:
        typing.Union[Response, str]: This function returns a redirect for the index.html page or renders the template
        of database_error.html.

    Raises:
        This function does not raise any exceptions.
    """
    # For this function, catch both sqlite3.OperationalError and sqlalchemy.exc.OperationalError in one except statement
    # because both exceptions need to be handled in the same way, which is returning the database_error.html page.

    # First, get the count of the number of rows in the AppMetrics table
    try:
        row_count = query_manager.session.query(AppMetrics).count()
    except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
        logger.error("Not able to query likes/dislikes table in the" "database. %s ",
                     database_exception)
        return render_template("database_error.html")

    # If the table is empty (row count is 0)
    if row_count == 0:
        # Then create a row in the AppMetrics table
        try:
            query_manager.create_like_dislike()
        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
            logger.error(
                "Not able to create row likes/dislikes table in the"
                "database. %s ", database_exception)
            return render_template("database_error.html")
        else:
            logger.info("Created an empty row in likes/dislikes table.")

    # If the user pressed the like button, increment the count of likes in the AppMetrics table.
    if request.form["choice"] == "Like":
        try:
            query_manager.increment_like()
        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
            logger.error("Not able to increment number of likes in the" "database. %s ",
                         database_exception)
            return render_template("database_error.html")
        else:
            logger.info("Incremented the likes.")

    # Else, the user pressed the dislike button, so increment the count of the dislikes in the AppMetrics table.
    else:
        try:
            query_manager.increment_dislike()
        except (sqlite3.OperationalError, sqlalchemy.exc.OperationalError) as database_exception:
            logger.error("Not able to increment number of dislikes in the database. %s ",
                         database_exception)
            return render_template("database_error.html")
        else:
            logger.info("Incremented the dislikes.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])
