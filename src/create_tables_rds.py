import logging
import typing
import os

import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# put all of these in try blocks?
# connection_type = "mysql+pymysql"
# mysql_user = os.getenv("MYSQL_USER")
# mysql_password = os.getenv("MYSQL_PASSWORD")
# mysql_host = os.getenv("MYSQL_HOST")
# mysql_port = os.getenv("MYSQL_PORT")
# mysql_db_name = os.getenv("DATABASE_NAME")
# engine_string = f"{connection_type}://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db_name}"
engine_string = os.getenv("SQLALCHEMY_DATABASE_URI")

# Make sure the environment variable exists
if engine_string is None:
    logger.error("Environment variable SQLALCHEMY_DATABASE_URI does not exist.")

# Try to create the sqlalchemy engine.
try:
    engine = sqlalchemy.create_engine(engine_string)
except OperationalError as o_error:
    logger.error("Could not create the sqlalchemy engine: %s", o_error)
    engine = None
except Exception as other_exception:
    logger.error(other_exception)
    engine = None
else:
    logger.debug("Successfully created engine.")

Base = declarative_base()

class HistoricalQueries(Base):
    """
    This table will store a cache of popular queries. If the web application recieves a query that is popular, it can
    retrieve the prediction from the RDS database instead of having to make a new prediction.
    """
    __tablename__ = "historical_queries"

    query_number = Column(Integer, primary_key=True)   # primary key
    query_count = Column(Integer, unique=False, nullable=False) # Number of times the particular query has been called.
    predicted_traffic_count = Column(Float, unique=False, nullable=False)  # Predicted traffic count
    temperature = Column(Float, unique=False, nullable=False) # Temperature
    cloud_percentage = Column(Integer, unique=False, nullable=False) # Percentage of Cloud Cover
    weather_description = Column(String(30), unique=False, nullable=False) # Descriptor of the weather ("Clear", "Thunderstorm", etc.)
    year = Column(Integer, unique = False, nullable = False) # Year
    month = Column(Integer, unique = False, nullable = False) # Month
    day = Column(Integer, unique = False, nullable = False) # Day
    hour = Column(Integer, unique = False, nullable = False) # Hour
    day_of_week = Column(String(15), unique = False, nullable = False) # Day of week
    holiday = Column(Integer, unique = False, nullable = False)  # Holiday, binary 1 or 0 indicating if there is a holiday
    # Can I put a constraint of 0 or 1?
    rainfall_hour = Column(Float, unique = False, nullable = False) # Amount of rainfall in millimeters that fell in 1 hour


    def __repr__(self):
        print(f"Query_Number: {self.query_number} \n"
              f"Query_Count: {self.query_count} \n"
              f"Predicted_Traffic_Count: {self.predicted_traffic_count} \n"
              f"Temperature: {self.temperature} \n"
              f"Cloud_Percentage: {self.cloud_percentage} \n"
              f"Weather_Description: {self.weather_description} \n"
              f"Year: {self.year} \n"
              f"Month: {self.month} \n"
              f"Day: {self.day} \n"
              f"Day_Of_Week: {self.day_of_week} \n"
              f"Holiday: {self.holiday} \n"
              f"Rainfall_Hour: {self.rainfall_hour}")


if __name__ == "__main__":
    try:
        Base.metadata.create_all(engine) # what is this line really doing, and what error could it return??
    # TODO: Add another exception
    except Exception as other_error:
        logger.error("Could not create the table: %s", other_error)
    else:
        logger.info("Created table.")



