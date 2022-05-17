import logging.config
import sqlite3
import traceback

import sqlalchemy.exc
from flask import Flask, render_template, request, redirect, url_for

# For setting up the Flask-SQLAlchemy database session
#from src.add_songs import Tracks, TrackManager
from src.create_tables_rds import QueryManager, HistoricalQueries

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
        user_query = query_manager.session.query(HistoricalQueries).limit(
            app.config["MAX_ROWS_SHOW"]).all()
        logger.debug("Index page accessed")
        return render_template('index.html', user_query=user_query)
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



@app.route('/add', methods=['POST'])
def enter_query_parameters():
    """View that process a POST with new song input

    Returns:
        redirect to index page
    """
    new_query_params = {}
    new_query_params["temperature"] = request.form["temperature"]
    new_query_params["cloud_percentage"] = request.form["cloud_percentage"]
    new_query_params["weather_description"] = request.form["weather_description"]
    new_query_params["year"] = request.form["year"]
    new_query_params["month"] = request.form["month"]
    new_query_params["day"] = request.form["day"]
    new_query_params["hour"] = request.form["hour"]
    new_query_params["day_of_week"] = request.form["day_of_week"]
    new_query_params["holiday"] = request.form["holiday"]
    new_query_params["rainfall_hour"] = request.form["rainfall_hour"]


    try:
        query_manager.add_new_query(query_params=new_query_params,
                                    query_prediction=1000) #TODO: Need to plug in model prediction
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


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"],
            host=app.config["HOST"])
