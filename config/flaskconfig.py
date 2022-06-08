import os

DEBUG = True
LOGGING_CONFIG = "config/logging/local.conf"
PORT = 5000
APP_NAME = "Traffic Prediction"
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
SQLALCHEMY_ECHO = False
MAX_ROWS_SHOW = 5

PATH_TRAINED_MODEL_OBJECT = './models/trained_model_object.joblib'
PATH_TRAINED_ONE_HOT_ENCODER = './models/ohe_object.joblib'
MODEL_CONFIG_PATH = "./config/model_config.yaml"

SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')
if SQLALCHEMY_DATABASE_URI is None:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data/traffic_prediction.db'


