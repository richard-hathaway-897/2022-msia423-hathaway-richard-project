import logging
import typing

import numpy as np
import pandas as pd
import sklearn.preprocessing
import joblib

import src.s3_actions

logger = logging.getLogger(__name__)

def validate_user_input_dtype(input_dict: dict):
    new_query_params = {}
    valid_input = True
    try:
        new_query_params["temp"] = float(input_dict["temp"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["clouds_all"] = float(input_dict["clouds_all"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["weather_main"] = str(input_dict["weather_main"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["month"] = int(input_dict["month"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["hour"] = int(input_dict["hour"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["day_of_week"] = str(input_dict["day_of_week"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["holiday"] = str(input_dict["holiday"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    try:
        new_query_params["rain_1h"] = float(input_dict["rain_1h"])
    except ValueError:
        logger.error("A float was not entered for temperature.")
        valid_input = False

    return {'data_status': valid_input, 'data': new_query_params}

