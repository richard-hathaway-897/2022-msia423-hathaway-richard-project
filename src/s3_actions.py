import argparse
import logging
import urllib.error
import typing

import botocore.exceptions
import pandas as pd

logger = logging.getLogger(__name__)

# TODO: Consider having this function actually return an object.
def process_commandline_args(command_line_arguments: argparse.Namespace) -> None:

    """
    This function parses the command line arguments and calls the appropriate function to achieve the desired task.

    Args:
        command_line_arguments (argparse.Namespace): This is an argparse.Namespace object that contains all of the
            command line arguments passed in by the user

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any exceptions.

    """

    # if the user selects write, call the function to upload to S3
    if command_line_arguments.action == "write":
        s3_write(data_source=command_line_arguments.path_local,
                  s3_destination=command_line_arguments.path_s3,
                  delimiter=command_line_arguments.delimiter)


    # if the user selects read, call the function to read from S3.
    if command_line_arguments.action == "read":
        try:
            s3_read(s3_source=command_line_arguments.path_s3,
                    delimiter=command_line_arguments.delimiter)
        except ValueError as value_error:
            # This exception will occur if s3_read returns an empty dataframe.
            logger.error(value_error)


def s3_write(data_source: typing.Union[str, pd.DataFrame], s3_destination: str, delimiter: str = ",") -> None:
    """
    This function fetches a data source, reads it into a pandas dataframe, and uploads it to S3 as a csv file.

    Args:
        data_source (typing.Union[str, pd.DataFrame]): The source of the data. This can be an existing pandas dataframe
            or a string local file path or a URL.
        s3_destination (str): The file path in S3 to upload the data to.
        delimiter (str): The delimiter character separating entries in the csv file.

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any exceptions.

    """

    raw_data = None
    if isinstance(data_source, pd.DataFrame):
        raw_data = data_source

    else:
        # TODO: Potentially add an option to pass in a pandas dataframe instead of a string
        try:
            raw_data = pd.read_csv(data_source)

        except urllib.error.HTTPError as http_error:
            # This error will occur if file does not exist at the specified location on a website's domain.
            logger.error("Could not read the data because the data source was invalid: %s", http_error)
        except urllib.error.URLError as url_error:
            # This error will occur if the website does not exist.
            logger.error("Could not read the data because the url was invalid: %s", url_error)
        except FileNotFoundError as file_error:
            # This error will occur if the local file path does not exist.
            logger.error("Could not read the data because the specified file does not exist! %s", file_error)
        except pd.errors.ParserError as parser_error:
            # error caused by data_source as https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
            logger.error("Could not read the data. This may occur if you fetch a website instead of a zipped csv file.")
        except Exception as unknown_error:
            # This error will catch any other potential exceptions
            logger.error("Could not read the data because an unknown error occured. %s", unknown_error)
        else:
            # If the data source is successfully read, log the message and try to upload the dataframe to S3.
            logger.info("Successfully read data from: %s", data_source)

    if raw_data is not None:
        try:
            raw_data.to_csv(s3_destination, sep=delimiter, index=False)
        except botocore.exceptions.NoCredentialsError as no_credentials_error:
            # This error will occur if the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are not set.
            logger.error("Could not write data to AWS S3 bucket. "
                         "Ensure the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are activated."
                         "Error: %s", no_credentials_error)
        except botocore.exceptions.ClientError as client_error:
            # This exception will catch any AWS service exceptions. More information at:
            # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
            logger.error("A client error occurred - could not write data to AWS S3 bucket. %s", client_error)
        except PermissionError as permission_error:
            # This error will occur if the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are not set.
            logger.error("Permission Error, could not write data to AWS S3 bucket."
                         "Ensure the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are activated."
                         "Error: %s", permission_error)
        except OSError as os_error:
            # This error will occur if a local directory is specified that does not exist.
            logger.error("The output directory does not exist: %s", os_error)
        except Exception as unknown_error:
            logger.error("An unknown error occurred. Could not write data to AWS S3 bucket. %s", unknown_error)
        else:
            logger.info("Successfully saved data to %s", s3_destination)
            logger.info("Saved data has %d records and %d columns.", raw_data.shape[0], raw_data.shape[1])
    else:
        logger.error("Failed to write data to S3. Data source could not be read.")


def s3_read(s3_source: str, delimiter: str = ",") -> pd.DataFrame:
    """
    This function reads a csv file from S3 into a pandas dataframe.
    Args:
        s3_source (str): The S3 path of the source file.
        delimiter (str): The delimiter character separating entries in the csv file.

    Returns:
        data (pd.DataFrame): This function returns the pandas dataframe that was read from S3.

    Raises:
        ValueError: This function will raise a ValueError if reading from S3 fails.

    """

    # TODO: Potentially add an option to save to a file.
    # set default value for DataFrame
    data = pd.DataFrame()
    try:
        data = pd.read_csv(s3_source, sep=delimiter)
    except FileNotFoundError as file_error:
        # This error will occur if the file specified is not found on S3.
        logger.error("The specified file or bucket does not exist: %s", file_error)
    except botocore.exceptions.NoCredentialsError as no_credentials_error:
        # This error will occur if the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are not set.
        logger.error("Could not read data from AWS S3 bucket. "
                     "Ensure the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are activated."
                     "Error: %s", no_credentials_error)
    except PermissionError as permission_error:
        # This error will occur if the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are not set.
        logger.error("Permission Error, could not read data from AWS S3 bucket."
                     "Ensure the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are activated."
                     "Error: %s", permission_error)
    except botocore.exceptions.ClientError as client_error:
        # This exception will catch any AWS service exceptions. More information at:
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
        logger.error("A client error occurred - could not read data from AWS S3 bucket. %s", client_error)
    #except Exception as unknown_error:
        #logger.error("An unknown error occurred. Could not read data from AWS S3 bucket. %s", unknown_error)
    else:
        logger.info("Successfully read data from AWS S3, %s", s3_source)
    #finally:
        # Check if the dataframe is empty. If it is empty, raise a valueError because it means reading from S3 failed.
        #if data.empty:
            #logger.warning("Dataframe is empty")
            #raise ValueError("Could not read data from AWS S3, returning empty dataframe.")

    return data


if __name__ == "__main__":
    """
    Main function takes in command line arguments and then passes them to process_commandline_args()
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("action", type=str,
                        choices=["read", "write"],
                        help="Select either `read` or `write` to indicate the desired action.")
    parser.add_argument("--path_s3", type=str,
                        required=True,
                        help = "Path of the data on s3 to read from or write to.")
    parser.add_argument("--path_local", type=str,
                         # TODO: Is this required? Only for writing or also reading?
                        help = "Local path or URL to read from or write to.")
    parser.add_argument("--delimiter", type=str,
                        default = ",",
                        help = "The delimiter of the file.")

    process_commandline_args(parser.parse_args())



