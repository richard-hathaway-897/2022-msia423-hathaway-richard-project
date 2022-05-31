import logging.config
import sqlite3
import traceback

import sqlalchemy.exc
from flask import Flask, render_template, request, redirect, url_for
import numpy as np

# For setting up the Flask-SQLAlchemy database session

from src.create_tables_rds import QueryManager, HistoricalQueries, AppMetrics, ActivePrediction
import src.predict
import src.validate
import src.read_write_functions
import src.preprocess_app_input
import src.app_module

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates",
            static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"]) # Give a meaningful name to the logger in executable script
logger.debug(
    'Web app should be viewable at %s:%s if docker run command maps local '
    'port to the same port as configured for the Docker container '
    'in config/flaskconfig.py (e.g. `-p 5000:5000`). Otherwise, go to the '
    'port defined on the left side of the port mapping '
    '(`i.e. -p THISPORT:5000`). If you are running from a Windows machine, '
    'go to 127.0.0.1 instead of 0.0.0.0.', app.config["HOST"]
    , app.config["PORT"])

# Initialize the database session
query_manager = QueryManager(app)


# Different functions that can take place in the app.
# What function should be returned when someone goes to "/"
@app.route('/')
def index():
    """Main view that lists songs in the database.

     app/templates/index.html template.

    Returns:
        Rendered html template

    """

    try:
        user_query = query_manager.session.query(HistoricalQueries).order_by(HistoricalQueries.query_count.desc()).limit(
            app.config["MAX_ROWS_SHOW"]).all()
        logger.debug("Retrieve top 5 historical queries.")

        # Return template if the search to index.html succeeds
    except sqlite3.OperationalError as e:
        logger.error(
            "Error page returned. Not able to query local sqlite database: %s."
            " Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')
    except sqlalchemy.exc.OperationalError as e:
        logger.error(
            "Error page returned. Not able to query MySQL database: %s. "
            "Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')
    except:
        traceback.print_exc()
        logger.error("Not able to display the historical queries, error page returned")
        return render_template('error.html')

    try:
        most_recent_prediction = query_manager.session.query(ActivePrediction).first()
        logger.debug("Retrieve most recent prediction.")
        # Return template if the search to index.html succeeds
    except sqlite3.OperationalError as e:
        logger.error(
            "Error page returned. Not able to retrieve most recent prediction from local sqlite database: %s."
            " Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')
    except sqlalchemy.exc.OperationalError as e:
        logger.error(
            "Error page returned. Not able to retrieve most recent prediction from MySQL database: %s. "
            "Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')
    except:
        traceback.print_exc()
        logger.error("Not able to retrieve most recent prediction, error page returned")
        return render_template('error.html')

    try:
        like_dislike_count = query_manager.session.query(AppMetrics).all()
        logger.info("Retrieve likes and dislikes.")
    except sqlite3.OperationalError as e:
        logger.error(
            "Error page returned. Not able to query local sqlite database: %s."
            " Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')
    except sqlalchemy.exc.OperationalError as e:
        logger.error(
            "Error page returned. Not able to query MySQL database: %s. "
            "Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
        return render_template('error.html')
    except:
        traceback.print_exc()
        logger.error("Not able to display likes and dislikes, error page returned")
        return render_template('error.html')

    logger.debug("Index page accessed")
    return render_template('index.html',
                           user_query=user_query,
                           like_dislike_count=like_dislike_count,
                           prediction=most_recent_prediction)



@app.route('/add', methods=['POST'])
def enter_query_parameters():
    """
    """
    # TODO: Is hardcoding these column names bad?
    new_query_params = {}
    new_query_params["temp"] = request.form["temperature"]
    new_query_params["clouds_all"] = request.form["cloud_percentage"]
    new_query_params["weather_main"] = request.form["weather_description"]
    new_query_params["month"] = request.form["month"]
    new_query_params["hour"] = request.form["hour"]
    new_query_params["day_of_week"] = request.form["day_of_week"]
    new_query_params["holiday"] = request.form["holiday"]
    new_query_params["rain_1h"] = request.form["rainfall_hour"]

    config_dict = src.read_write_functions.read_yaml(app.config["MODEL_CONFIG_PATH"])
    if len(config_dict) == 0:
        logger.error("Could not load app configuration file.")
        return render_template('error.html')

    try:
        prediction, traffic_volume = \
            src.app_module.run_app_prediction(new_query_params,
                                             model_object_path=app.config["PATH_TRAINED_MODEL_OBJECT"],
                                             one_hot_encoder_path=app.config["PATH_TRAINED_ONE_HOT_ENCODER"],
                                             config_dict=config_dict)
    except FileNotFoundError:
        logger.error("Could not load model object.")
        return render_template('error.html')
    except ValueError:
        logger.error("Error: no prediction was made.")
        return render_template('invalid_input.html')

    else:

        update_error = src.app_module.run_update_tables(query_manager=query_manager,
                                                         new_query_params=new_query_params,
                                                         database_uri_string=app.config['SQLALCHEMY_DATABASE_URI'],
                                                         prediction=np.round(prediction[0]),
                                                         traffic_volume=traffic_volume)
        if update_error:
            return render_template('error.html')
        else:
            return redirect(url_for('index'))





@app.route('/like', methods=['POST'])
def increment_like_dislike():
    """
    """
    try:
        row_count = query_manager.session.query(AppMetrics).count()
    except sqlite3.OperationalError as e:
        logger.error(
            "Error page returned. Not able to query likes/dislikes table in local sqlite "
            "database: %s. Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
    except sqlalchemy.exc.OperationalError as e:
        logger.error(
            "Error page returned. Not able to query likes/dislikes table in MySQL database: %s. "
            "Error: %s ",
            app.config['SQLALCHEMY_DATABASE_URI'], e)
    else:

        if row_count == 0:
            try:
                query_manager.create_like_dislike()
            except sqlite3.OperationalError as e:
                logger.error(
                    "Error page returned. Not able to create an empty row in the AppMetrics table in local sqlite "
                    "database: %s. Error: %s ",
                    app.config['SQLALCHEMY_DATABASE_URI'], e)
            except sqlalchemy.exc.OperationalError as e:
                logger.error(
                    "Error page returned. Not able to create an empty row in the AppMetrics table in MySQL database: %s. "
                    "Error: %s ",
                    app.config['SQLALCHEMY_DATABASE_URI'], e)
            else:
                logger.info("Created an empty row in likes/dislikes table.")

        if request.form["choice"] == "Like":
            try:
                query_manager.increment_like()
            except sqlite3.OperationalError as e:
                logger.error(
                    "Error page returned. Not able to increment likes in the AppMetrics table in local sqlite "
                    "database: %s. Error: %s ",
                    app.config['SQLALCHEMY_DATABASE_URI'], e)
            except sqlalchemy.exc.OperationalError as e:
                logger.error(
                    "Error page returned. Not able to increment likes in the AppMetrics table in MySQL database: %s. "
                    "Error: %s ",
                    app.config['SQLALCHEMY_DATABASE_URI'], e)
            else:
                logger.info("Incremented the likes.")
        else:
            try:
                query_manager.increment_dislike()
            except sqlite3.OperationalError as e:
                logger.error(
                    "Error page returned. Not able to increment dislikes in the AppMetrics table in local sqlite "
                    "database: %s. Error: %s ",
                    app.config['SQLALCHEMY_DATABASE_URI'], e)
            except sqlalchemy.exc.OperationalError as e:
                logger.error(
                    "Error page returned. Not able to increment dislikes in the AppMetrics table in MySQL database: %s. "
                    "Error: %s ",
                    app.config['SQLALCHEMY_DATABASE_URI'], e)
            else:
                logger.info("Incremented the dislikes.")
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"],
            host=app.config["HOST"])
