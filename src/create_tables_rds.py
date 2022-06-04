import logging

import sqlite3
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String
import flask
from flask_sqlalchemy import SQLAlchemy

import config.database_config

logger = logging.getLogger(__name__)

Base = declarative_base()

class HistoricalQueries(Base):
    """
    This table will store a history of the queries that the web application has received.
    """
    __tablename__ = "historical_queries"

    query_number = Column(Integer, primary_key=True)  # primary key
    query_count = Column(Integer, unique=False, nullable=False)  # Number of times the particular query has been called.
    predicted_traffic_count = Column(Float, unique=False, nullable=False)  # Predicted traffic count
    temperature = Column(Float, unique=False, nullable=False)  # Temperature
    cloud_percentage = Column(Integer, unique=False, nullable=False)  # Percentage of Cloud Cover
    weather_description = Column(String(30), unique=False, nullable=False)  # Descriptor of the weather
    month = Column(Integer, unique = False, nullable=False)  # Month
    hour = Column(Integer, unique = False, nullable=False)  # Hour
    day_of_week = Column(String(15), unique=False, nullable=False)  # Day of week
    holiday = Column(Integer, unique=False, nullable=False)  # Holiday, binary 1 or 0 indicating a holiday
    rainfall_hour = Column(Float, unique=False, nullable=False)  # Total rainfall in millimeters that fell in 1 hour

    def __repr__(self) -> str:
        """This function defines the string representation of the HistoricalQueries table

        Returns:
            repr_string (str): The string representation of the table.

        """
        repr_string = (f"Query_Number: {self.query_number} \n"
                       f"Query_Count: {self.query_count} \n"
                       f"Predicted_Traffic_Count: {self.predicted_traffic_count} \n"
                       f"Temperature: {self.temperature} \n"
                       f"Cloud_Percentage: {self.cloud_percentage} \n"
                       f"Weather_Description: {self.weather_description} \n"
                       f"Month: {self.month} \n"
                       f"Day_Of_Week: {self.day_of_week} \n"
                       f"Holiday: {self.holiday} \n"
                       f"Rainfall_Hour: {self.rainfall_hour}")
        return repr_string

class AppMetrics(Base):
    """
    This table will store the number of likes and dislikes received on the web application.
    """
    __tablename__ = "app_metrics"

    row_id = Column(Integer, primary_key=True)  # primary key
    likes = Column(Integer, unique=False, nullable=False)  # Number of likes the web application has received
    dislikes = Column(Integer, unique=False, nullable=False)  # Number of dislikes the web application has received.

    def __repr__(self) -> str:
        """This function defines the string representation of the AppMetrics table

        Returns:
            repr_string (str): The string representation of the table.

        """
        repr_string = (f"Likes: {self.likes} \n"
                       f"Dislikes: {self.dislikes}")
        return repr_string


class ActivePrediction(Base):
    """
    This table will store the most recent prediction made by the web application.
    """
    __tablename__ = "active_prediction"

    row_id = Column(Integer, primary_key=True)  # primary key
    prediction = Column(Float, unique=False, nullable=False)  # The value of the most recent prediction
    volume = Column(String(10), unique=False, nullable=False)  # The level of traffic for the prediction (e.g. 'light').

    def __repr__(self) -> str:
        """This function defines the string representation of the ActivePredictions table

        Returns:
            repr_string (str): The string representation of the table.

        """
        repr_string = (f"Prediction: {self.prediction} \n"
                       f"Volume: {self.volume}")
        return repr_string


class QueryManager:
    """This class creates a sqlalchemy connection to the database.

    Args:
        app (flask.app.Flask): Flask app object used when connecting to the database through flask. This is optional.
        engine_string (str): This is the engine string specifying the database to write to.
    """
    def __init__(self, app: flask.app.Flask = None,
                 engine_string: str = None):
        if app:
            self.database = SQLAlchemy(app)
            self.session = self.database.session
        elif engine_string:

            engine = sqlalchemy.create_engine(engine_string)

            # Try to create the sqlalchemy session
            try:
                session_maker = sqlalchemy.orm.sessionmaker(bind=engine)
            except sqlalchemy.exc.OperationalError as database_error:
                logger.error("Failed to create the sqlalchemy session.")
                raise database_error
            self.session = session_maker()
        else:
            raise ValueError("Need either an engine string or Flask app to initialize the connection to the database")

    def close(self) -> None:
        """Closes SQLAlchemy session

        Returns: None

        """
        self.session.close()

    def add_new_query(self, query_params: dict, query_prediction: float) -> None:
        """This function adds a new query to the HistoricalQueries table

        Args:
            query_params (dict): The parameters of the query as key-value pairs
            query_prediction (float): The prediction of traffic for the given query as a float.

        Returns:
            This function does not return any object.

        Raises:
            sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
                and fails.
            sqlalchemy.exc.OperationalError: This exception will be raised if the app tries to access the AWS RDS
                instance and fails.

        """
        # Create the HistoricalQueries object
        session = self.session
        user_query = HistoricalQueries(query_count=config.database_config.INITIAL_QUERY_COUNT,
                                       predicted_traffic_count=query_prediction,
                                       temperature=query_params["temp"],
                                       cloud_percentage=query_params["clouds_all"],
                                       weather_description=query_params["weather_main"],
                                       month=query_params["month"],
                                       hour=query_params["hour"],
                                       day_of_week=query_params["day_of_week"],
                                       holiday=query_params["holiday"],
                                       rainfall_hour=query_params["rain_1h"])
        session.add(user_query)
        try:
            session.commit()
        except sqlite3.OperationalError as database_exception:
            logger.error("Failed to add the query to the sqlite database. %s", database_exception)
            raise database_exception
        except sqlalchemy.exc.OperationalError as database_exception:
            logger.error("Failed to add the query to the AWS RDS database. %s", database_exception)
            raise database_exception
        logger.info("Added query to the database.")

    def search_for_query_count(self, query_params: dict) -> int:
        """This function returns the number of rows matching the input data in the HistoricalQueries table.

        Args:
            query_params (dict): The user input from the application as a dictionary.

        Returns:
            query_count (int): A count of the number of queries that match the input query.

        Raises:
            sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
                and fails.
            sqlalchemy.exc.OperationalError: This exception will be raised if the app accesses the AWS RDS instance
                and fails.
        """
        session = self.session
        # Search for the number of rows matching the input query
        try:
            query_count = session.query(HistoricalQueries)\
                .filter(HistoricalQueries.temperature == query_params["temp"],
                        HistoricalQueries.cloud_percentage == query_params["clouds_all"],
                        HistoricalQueries.weather_description == query_params["weather_main"],
                        HistoricalQueries.month == query_params["month"],
                        HistoricalQueries.hour == query_params["hour"],
                        HistoricalQueries.day_of_week == query_params["day_of_week"],
                        HistoricalQueries.holiday == query_params["holiday"],
                        HistoricalQueries.rainfall_hour == query_params["rain_1h"]).count()
        except sqlite3.OperationalError as database_exception:
            logger.error("Failed to query to the sqlite database. %s", database_exception)
            raise database_exception
        except sqlalchemy.exc.OperationalError as database_exception:
            logger.error("Failed to query to the AWS RDS database. %s", database_exception)
            raise database_exception

        logger.info("%d queries matching the input parameters found", query_count)
        return query_count

    def increment_query_count(self, query_params: dict) -> None:
        """This function increments the count for the given query in the HistoricalQueries table.

        Args:
            query_params (dict): The user input from the application as a dictionary.

        Returns:
            This function does not return any object.

        Raises:
            sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
                and fails.
            sqlalchemy.exc.OperationalError: This exception will be raised if the app accesses the AWS RDS instance
                and fails.
        """
        session = self.session
        # Increment the query count for the given query.
        session.query(HistoricalQueries).filter(HistoricalQueries.temperature == query_params["temp"],
                                                HistoricalQueries.cloud_percentage == query_params[
                                                  "clouds_all"],
                                                HistoricalQueries.weather_description == query_params[
                                                  "weather_main"],
                                                HistoricalQueries.month == query_params["month"],
                                                HistoricalQueries.hour == query_params["hour"],
                                                HistoricalQueries.day_of_week == query_params[
                                                  "day_of_week"],
                                                HistoricalQueries.holiday == query_params["holiday"],
                                                HistoricalQueries.rainfall_hour == query_params[
                                                    "rain_1h"]).update(
                                                {"query_count": HistoricalQueries.query_count +
                                                 config.database_config.INCREMENT_VALUE})
        try:
            session.commit()
        except sqlite3.OperationalError as database_exception:
            logger.error("Failed to update the sqlite database. %s", database_exception)
            raise database_exception
        except sqlalchemy.exc.OperationalError as database_exception:
            logger.error("Failed to update the AWS RDS database. %s", database_exception)
            raise database_exception
        logger.info("Record count incremented by %d", config.database_config.INCREMENT_VALUE)

    def increment_like(self) -> None:
        """This function increments the number of likes in the AppMetrics Table

        Returns:
            This function does not return any object.

        Raises:
            sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
                and fails.
            sqlalchemy.exc.OperationalError: This exception will be raised if the app accesses the AWS RDS instance
                and fails.

        """
        session = self.session
        # Increment the number of likes by 1
        session.query(AppMetrics).update({"likes": AppMetrics.likes + config.database_config.INCREMENT_VALUE})
        try:
            session.commit()
        except sqlite3.OperationalError as database_exception:
            logger.error("Failed to increment the likes in the sqlite database. %s", database_exception)
            raise database_exception
        except sqlalchemy.exc.OperationalError as database_exception:
            logger.error("Failed to increment the likes in the the AWS RDS database. %s", database_exception)
            raise database_exception
        logger.info("Incremented likes by %d", config.database_config.INCREMENT_VALUE)

    def increment_dislike(self) -> None:
        """This function increments the number of dislikes in the AppMetrics table

        Returns:
            This function does not return any object.

        Raises:
            sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
                and fails.
            sqlalchemy.exc.OperationalError: This exception will be raised if the app accesses the AWS RDS instance
                and fails.

        """
        session = self.session
        # Increment the number of dislikes.
        session.query(AppMetrics).update({"dislikes": AppMetrics.dislikes + config.database_config.INCREMENT_VALUE})
        try:
            session.commit()
        except sqlite3.OperationalError as database_exception:
            logger.error("Failed to increment the dislikes in the sqlite database. %s", database_exception)
            raise database_exception
        except sqlalchemy.exc.OperationalError as database_exception:
            logger.error("Failed to increment the dislikes in the the AWS RDS database. %s", database_exception)
            raise database_exception
        logger.info("Incremented dislikes by %d", config.database_config.INCREMENT_VALUE)

    def create_like_dislike(self) -> None:
        """This function creates a row in the AppMetrics Table and initializes the values with configuration initial
        like and dislikes values from config.database_config.py

        Returns:
            This function does not return any object.

        Raises:
            sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
                and fails.
            sqlalchemy.exc.OperationalError: This exception will be raised if the app accesses the AWS RDS instance
                and fails.

        """
        session = self.session
        # Initialize the table with 0 likes and 0 dislikes.
        like_dislike_row = AppMetrics(likes=config.database_config.INITIAL_LIKES,
                                      dislikes=config.database_config.INITIAL_DISLIKES)
        session.add(like_dislike_row)
        try:
            session.commit()
        except sqlite3.OperationalError as database_exception:
            logger.error("Failed to create row in the AppMetrics table in the sqlite database. %s", database_exception)
            raise database_exception
        except sqlalchemy.exc.OperationalError as database_exception:
            logger.error("Failed to create row in the AppMetrics table in the sqlite database. %s", database_exception)
            raise database_exception
        logger.info("Added like_dislike_row to the database.")

    def create_most_recent_query(self) -> None:
        """This function creates a row in the ActivePredictions table and initializes the row with values of 0 for
        the prediction and "light" for the volume.

        Returns:
            This function does not return any object.

        Raises:
            sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
                and fails.
            sqlalchemy.exc.OperationalError: This exception will be raised if the app accesses the AWS RDS instance
                and fails.

        """
        session = self.session
        # Initialize the row in ActivePrediction with a prediction 0 and a traffic volume of "light"
        default_prediction = ActivePrediction(prediction=config.database_config.INITIAL_PREDICTION,
                                              volume=config.database_config.INITIAL_VOLUME)
        session.add(default_prediction)
        try:
            session.commit()
        except sqlite3.OperationalError as database_exception:
            logger.error("Failed to create row in the ActivePredictions table in the sqlite database. %s",
                         database_exception)
            raise database_exception
        except sqlalchemy.exc.OperationalError as database_exception:
            logger.error("Failed to create row in the ActivePredictions table in the sqlite database. %s",
                         database_exception)
            raise database_exception
        logger.info("Added active prediction row to the database.")

    def update_active_prediction(self, new_prediction_value: float, new_volume: str) -> None:
        """This function updates the active prediction in the ActivePrediction table.

        Args:
            new_prediction_value (float): The value of the new prediction.
            new_volume (str): The string value for the traffic volume corresponding to the prediction.

        Returns:
            This function does not return any object.

        Raises:
            sqlite3.OperationalError: This exception will be raised if the app tries to access a local sqlite database
                and fails.
            sqlalchemy.exc.OperationalError: This exception will be raised if the app accesses the AWS RDS instance
                and fails.

        """
        session = self.session

        # Update the active prediction.
        session.query(ActivePrediction).update({"prediction": new_prediction_value, "volume": new_volume})
        try:
            session.commit()
        except sqlite3.OperationalError as database_exception:
            logger.error("Failed to update row in the ActivePredictions table in the sqlite database. %s",
                         database_exception)
            raise database_exception
        except sqlalchemy.exc.OperationalError as database_exception:
            logger.error("Failed to update row in the ActivePredictions table in the sqlite database. %s",
                         database_exception)
            raise database_exception
        logger.info("Updated most recent prediction with prediction %f and volume %s.",
                    new_prediction_value, new_volume)

def create_db(engine_string: str) -> None:
    """This function creates all of the tables in the database needed to run the web application.

    Args:
        engine_string (str): The engine string needed to connect to the database

    Returns:
        This function does not return any object.

    """

    # Make sure the environment variable exists
    if engine_string is None:
        logger.error("Environment variable SQLALCHEMY_DATABASE_URI does not exist.")

    # Try to create the sqlalchemy engine.
    try:
        engine = sqlalchemy.create_engine(engine_string)
    except sqlalchemy.exc.OperationalError as o_error:
        # This error will occur if the SQL Alchemy engine can not be created
        logger.error("Could not create the sqlalchemy engine: %s", o_error)
    except sqlalchemy.exc.ArgumentError as arg_error:
        # This error will occur if the SQLALCHEMY_DATABASE_URI does not exist or has an invalid format.
        logger.error("Could not parse the SQLALCHEMY_DATABASE_URI. "
                     "Please make sure this environment variable exists and is of a valid format. %s", arg_error)
    else:
        logger.debug("Successfully created engine.")

        # Try to create the database
        try:
            Base.metadata.create_all(engine)
        except sqlalchemy.exc.OperationalError as o_error:
            # This error will occur if the folder for the SQLALCHEMY_DATABASE_URI is invalid or if the user is not
            # connected to Northwestern VPN.
            logger.error("Could not create the database at the specified location. Ensure the location is correct, "
                         "and ensure connection to the Northwestern VPN. %s", o_error)
        else:
            logger.info("Created database.")
