import argparse
import logging
import urllib.error

import botocore.exceptions
import pandas as pd

logger = logging.getLogger(__name__)

def process_commandline_args(command_line_arguments: argparse.Namespace):

    if command_line_arguments.action == "write":
        s3_write(data_source=command_line_arguments.path_local,
                  s3_destination=command_line_arguments.path_s3,
                  delimiter=command_line_arguments.delimiter)

    if command_line_arguments.action == "read":
        try:
            s3_read(s3_source=command_line_arguments.path_s3,
                    delimiter=command_line_arguments.delimiter)
        except ValueError as value_error:
            logger.error(value_error)


def s3_write(data_source: str, s3_destination: str, delimiter: str = ","):

    # TODO: Potentially add an option to pass in a pandas dataframe instead of a string
    try:
        raw_data = pd.read_csv(data_source)
    except urllib.error.HTTPError as http_error:
        logger.error("Could not read the data because the data source was invalid: %s", http_error)
    except urllib.error.URLError as url_error:
        logger.error("Could not read the data because the url was invalid: %s", url_error)
    except FileNotFoundError as file_error:
        logger.error("Could not read the data because the specified file does not exist! %s", file_error)
    except Exception as unknown_error:
        logger.error("Could not read the data because an unknown error occured. %s", unknown_error)
    else:
        logger.info("Successfully read data from: %s", data_source)

        try:
            raw_data.to_csv(s3_destination, sep=delimiter)
        except botocore.exceptions.NoCredentialsError as no_credentials_error:
            logger.error("Could not write data to AWS S3 bucket. "
                         "Ensure the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are activated."
                         "Error: %s", no_credentials_error)
        except botocore.exceptions.ClientError as client_error:
            logger.error("A client error occurred - could not write data to AWS S3 bucket. %s", client_error)
        except Exception as unknown_error:
            logger.error("An unknown error occurred. Could not write data to AWS S3 bucket. %s", unknown_error)
        else:
            logger.info("Successfully uploaded data to %s", s3_destination)


def s3_read(s3_source: str, delimiter: str = ",") -> pd.DataFrame:

    # set default DataFrame
    data = pd.DataFrame()
    try:
        data = pd.read_csv(s3_source, sep=delimiter)
    except FileNotFoundError as file_error:
        logger.error("The specified file or bucket does not exist: %s", file_error)
    except botocore.exceptions.NoCredentialsError as no_credentials_error:
        logger.error("Could not read data from AWS S3 bucket. "
                     "Ensure the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are activated."
                     "Error: %s", no_credentials_error)
    except botocore.exceptions.ClientError as client_error:
        logger.error("A client error occurred - could not read data from AWS S3 bucket. %s", client_error)
    except Exception as unknown_error:
        logger.error("An unknown error occurred. Could not read data from AWS S3 bucket. %s", unknown_error)
    else:
        logger.info("Successfully read data from AWS S3, %s", s3_source)
    finally:
        if data.empty:
            raise ValueError("Could not read data from AWS S3, returning empty dataframe.")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("action", type = str,
                        choices=["read", "write"],
                        help="Select either `upload` or `download` to indicate the desired action.")
    parser.add_argument("--path_s3", type = str,
                        required = True,
                        help = "Path of the data on s3 to read from or write to.")
    parser.add_argument("--path_local", type = str,
                         # TODO: Is this required? Only for writing or also reading?
                        help = "Local path or URL to read from or write to.")
    parser.add_argument("--delimiter", type = str,
                        default = ",",
                        help = "The delimiter of the file.")

    process_commandline_args(parser.parse_args())



