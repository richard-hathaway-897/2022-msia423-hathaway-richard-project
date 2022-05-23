import logging.config
import sqlite3
import traceback

import sqlalchemy.exc
from flask import Flask, render_template, request, redirect, url_for
import yaml

# For setting up the Flask-SQLAlchemy database session
#from src.add_songs import Tracks, TrackManager
from src.create_tables_rds import QueryManager, HistoricalQueries, AppMetrics
import src.predict
import config.config
import src.validate

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

    Create view into index page that uses data queried from Track database and
    inserts it into the app/templates/index.html template.

    Returns:
        Rendered html template

    """

    try:
        user_query = query_manager.session.query(HistoricalQueries).order_by(app.config["ROW_SORT_BY"]).limit(
            app.config["MAX_ROWS_SHOW"]).all()
        logger.debug("Index page accessed")

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
        logger.error("Not able to display tracks, error page returned")
        return render_template('error.html')
    else:
        try:
            like_dislike_count = query_manager.session.query(AppMetrics).all()
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
            logger.error("Not able to display tracks, error page returned")
            return render_template('error.html')
        else:
            logger.debug("Index page accessed")
            return render_template('index.html', user_query=user_query, like_dislike_count=like_dislike_count)



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

    validation_result = src.validate.validate_user_input_dtype(new_query_params)

    if not validation_result["data_status"]:
        logger.error("One or more data entries had an invalid data type.")
        return render_template('error.html')

    try:
        with open(config.config.MODEL_CONFIG_PATH, "r", encoding="utf-8") as preprocess_yaml:
            preprocess_parameters = yaml.load(preprocess_yaml, Loader=yaml.FullLoader)
    except FileNotFoundError:
        logger.error("Could not locate the model configuration file specified in config.config.py: %s.",
                     config.config.MODEL_CONFIG_PATH)
        return render_template('error.html')

    prediction_df = src.predict.predict_preprocess(validation_result["data"], preprocess_parameters["preprocess_data"])
    prediction = src.predict.predict(prediction_df,
                                     model_object_path=app.config["PATH_TRAINED_MODEL_OBJECT"],
                                     ohe_object_path=app.config["PATH_TRAINED_ONE_HOT_ENCODER"],
                                     s3_bool=False)
    logger.info("Prediction: %f", prediction[0])

    query_count = query_manager.search_for_query_count(query_params=new_query_params)

    if query_count == 0:

        try:
            query_manager.add_new_query(query_params=new_query_params,
                                        query_prediction=prediction[0])
            logger.info("Query added")
            return redirect(url_for('index'))
        except sqlite3.OperationalError as e:
            logger.error(
                "Error page returned. Not able to add song to local sqlite "
                "database: %s. Error: %s ",
                app.config['SQLALCHEMY_DATABASE_URI'], e)
            return render_template('error.html')
        except sqlalchemy.exc.OperationalError as e:
            logger.error(
                "Error page returned. Not able to add song to MySQL database: %s. "
                "Error: %s ",
                app.config['SQLALCHEMY_DATABASE_URI'], e)
            return render_template('error.html')
    else:
        query_manager.increment_query_count(query_params=new_query_params)
        return redirect(url_for('index'))


@app.route('/like', methods=['POST'])
def increment_like_dislike():
    """
    """
    logger.info("Starting")
    if query_manager.session.query(AppMetrics).count() == 0:
        query_manager.create_like_dislike()

    if request.form["choice"] == "Like":
        query_manager.increment_like()
        logger.info("Incremented like")
    else:
        query_manager.increment_dislike()
        logger.info("Incremented dislike")
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"],
            host=app.config["HOST"])
