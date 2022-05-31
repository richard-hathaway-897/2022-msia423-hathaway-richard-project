import argparse
import logging
import urllib.error
import typing

import botocore.exceptions
import pandas as pd

logger = logging.getLogger(__name__)

def s3_write(data: pd.DataFrame, s3_output_path: str) -> None:
    """
    This function writes a pandas dataframe to S3 as a csv file.

    Args:
        data (pd.DataFrame): A pandas dataframe to be written as a csv file to S3.
        s3_output_path (str): The file path in S3 to upload the data to.

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any exceptions.

    """

    # TODO: Check the exceptions make sense
    try:
        data.to_csv(s3_output_path, index=False)
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
    # except OSError as os_error:
    #     # This error will occur if a local directory is specified that does not exist.
    #     logger.error("The output directory does not exist: %s", os_error)
    except Exception as unknown_error:
        logger.error("An unknown error occurred. Could not write data to AWS S3 bucket. %s", unknown_error)
    else:
        logger.info("Successfully saved data to %s", s3_output_path)
        logger.info("Saved data has %d records and %d columns.", data.shape[0], data.shape[1])


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
    except pd.errors.EmptyDataError as empty_data:
        logger.error("Could not parse the csv file because it is empty.")
    #except Exception as unknown_error:
        #logger.error("An unknown error occurred. Could not read data from AWS S3 bucket. %s", unknown_error)
    else:
        logger.info("Successfully read data from AWS S3, %s", s3_source)
    finally:
        #Check if the dataframe is empty. If it is empty, raise a valueError because it means reading from S3 failed.
        if data.empty:
            raise ValueError("Could not read data from AWS S3, returning empty dataframe.")

    return data



