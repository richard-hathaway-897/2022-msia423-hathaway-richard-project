import logging.config
import typing
import os

import yaml
import pandas as pd
import joblib
import sklearn
import urllib.error

logger = logging.getLogger(__name__)


def read_yaml(config_path: str) -> dict:
    """This function reads in a configuration yaml file and returns a dictionary

    Args:
        config_path (str): Path to the configuration yaml file.

    Returns:
        config_dict (dict): A dictionary containing the contents of the yaml file

    """
    config_dict = {}
    try:
        with open(config_path) as config_file:
            config_dict = yaml.safe_load(config_file)
    except FileNotFoundError as file_not_found:
        logger.error("Could not locate the specified configuration file. %s", file_not_found)
        logger.warning("Returning empty dictionary.")
    else:
        logger.info("Successfully loaded configuration file.")
    return config_dict


def read_csv_url(data_source: str) -> pd.DataFrame():
    """
    This function fetches a csv file from a URL and reads it into a pandas dataframe.

    Args:
        data_source (str): The URL source of the csv file.

    Returns:
        raw_data (pd.DataFrame): A pandas dataframe with the data read from the source. If reading the URL fails,
        an empty dataframe is returned.

    Raises:
        This function does not raise any exceptions.

    """
    raw_data = pd.DataFrame()
    try:
        raw_data = pd.read_csv(data_source)

    except urllib.error.HTTPError as http_error:
        # This error will occur if file does not exist at the specified location on a website's domain.
        logger.error("Could not read the data because the data source was invalid: %s", http_error)
    except urllib.error.URLError as url_error:
        # This error will occur if the website does not exist.
        logger.error("Could not read the data because the url was invalid: %s", url_error)
    except pd.errors.ParserError:
        # error caused by data_source as https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
        logger.error("Could not read the data. This may occur if you fetch a website instead of a zipped csv file.")
    except Exception as unknown_error:
        # This error will catch any other potential exceptions
        logger.error("Could not read the data because an unknown error occured. %s", unknown_error)
    else:
        # If the data source is successfully read, log the message and try to upload the dataframe to S3.
        logger.info("Successfully read data from: %s", data_source)

    return raw_data


def read_csv(input_source: str) -> pd.DataFrame:
    """This function reads in a csv file into a pandas dataframe.

    Args:
        input_source (str): The path to the input csv file.

    Returns:
        data_input (pd.DataFrame): The dataframe containing the contents of the csv file.
    """
    data_input = pd.DataFrame()
    try:
        data_input = pd.read_csv(input_source)
    except FileNotFoundError as file_not_found:
        logger.error("Could not read the file from the directory. %s", file_not_found)
    except pd.errors.ParserError:
        # This error can occur if the file cannot be read into a pandas dataframe.
        logger.error("File could not be converted to a pandas dataframe.")
    else:
        # Log out the shape of the data.
        logger.debug("Successfully read file with %d records and %d columns from %s",
                     data_input.shape[0], data_input.shape[1], input_source)
    return data_input


def load_model_object(model_input_source: str) -> sklearn.base.BaseEstimator:
    """This function loads a sklearn model object stored in a joblib file.
        It can be used to load in trained model objects.

    Args:
        model_input_source (str): The path to the trained model object

    Returns:
        model (sklearn.base.BaseEstimator): The sklearn trained model object loaded in.
    """
    model = None
    try:
        model = joblib.load(model_input_source)
    except FileNotFoundError as file_not_found:
        logger.error("Could not read the file from the directory. %s", file_not_found)
    # The next 3 errors all catch errors that can occur when a file that is not a .joblib file is loaded in.
    except KeyError:
        logger.error("Unexpected behavior occurred reading the file. Check that the input file is a .joblib file.")
    except ValueError:
        logger.error("Unexpected behavior occurred reading the file. Check that the input file is a .joblib file.")
    except IndexError:
        logger.error("Unexpected behavior occurred reading the file. Check that the input file is a .joblib file.")
    else:
        logger.debug("Successfully read model object from %s", model_input_source)
    return model


def save_csv(data: typing.Union[pd.DataFrame, pd.Series], output_path: str):
    """This function saves a pandas dataframe as a csv file to a specified local output path.

    Args:
        data (pd.DataFrame): The dataframe to save as a csv.
        output_path (str): The path where the file is saved to.

    Returns:
        This function does not return any object.
    """
    try:
        data.to_csv(output_path, index=False)
    except OSError as os_error:
        logger.error("Failed to save data because the folder does not exist. %s", os_error)
    else:
        try:
            logger.info("Successfully saved csv file with %d columns and %d rows to %s.",
                        data.shape[0], data.shape[1], output_path)
        except IndexError:
            # If the data is a pandas series, trying to log data.shape will throw an index error, so log an alternative
            # message if the IndexError arises.
            logger.info("Successfully saved csv file with 1 column and %d rows to %s.",
                        data.size, output_path)

    # Warn the user if a non-csv file format is specified. It could cause problems further along in the pipeline.
    if os.path.splitext(output_path)[1] != ".csv":
        logger.warning("Warning: saving to a non-csv file format.")


def save_model_object(model: sklearn.base.BaseEstimator, output_path: str) -> None:
    """This function saves a sklearn model object as a joblib file to a specified output path.

    Args:
        model (sklearn.base.BaseEstimator): The sklearn model object to save to the file.
        output_path (str): The output path to which to save the joblib file.

    Returns:
        This function does not return any object.
    """
    try:
        joblib.dump(model, output_path)
    except OSError as os_error:
        logger.error("Saving model object failed because the output folder does not exist. %s", os_error)
    else:
        logger.info("Saved the model object to %s", output_path)

def save_dict_as_text(data: dict, output_path: str) -> None:
    try:
        with open(output_path, "w") as data_file:
            for metric, value in data.items():
                data_file.write(metric + ": " + str(value) + "\n")
    except OSError as os_error:
        logger.error("Failed to save text file because the folder does not exist. %s", os_error)
    else:
        logger.info("Successfully saved text file to %s", output_path)

