import logging

import botocore.exceptions
import pandas as pd

logger = logging.getLogger(__name__)


def s3_write(data: pd.DataFrame, s3_output_path: str) -> None:
    """This function writes a pandas dataframe to S3 as a csv file.

    Args:
        data (pd.DataFrame): A pandas dataframe to be written as a csv file to S3.
        s3_output_path (str): The file path in S3 to upload the data to.

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any exceptions.

    """
    try:
        data.to_csv(s3_output_path, index=False)
    except botocore.exceptions.ClientError as client_error:
        # This exception will catch any AWS service exceptions. More information at:
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
        logger.error("A client error occurred - could not write data to AWS S3 bucket. %s", client_error)
    except PermissionError as permission_error:
        # This error will occur if the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are not set.
        logger.error("No permissions found. Could not write data to AWS S3 bucket. "
                     "In order to write to S3, the environment variables AWS_ACCESS_KEY_ID and "
                     "AWS_SECRET_ACCESS_KEY must be available. %s", permission_error)
    except ImportError as import_error:
        # This error will occur if the user does not have s3fs or fsspec installed.
        logger.error("Missing required packages to write to S3. %s", import_error)
    except OSError as os_error:
        # This error will occur if the S3 path is invalid.
        logger.error("Could not save the data to the specified S3 location. %s", os_error)
    else:
        logger.info("Successfully saved data to %s", s3_output_path)
        logger.info("Saved data has %d records and %d columns.", data.shape[0], data.shape[1])


def s3_read(s3_source: str) -> pd.DataFrame:
    """This function reads a csv file from S3 into a pandas dataframe.

    Args:
        s3_source (str): The S3 path of the source file.

    Returns:
        data (pd.DataFrame): This function returns the pandas dataframe that was read from S3.

    Raises:
        ValueError: This function will raise a ValueError if reading from S3 fails. There are many possible scenarios
        that could cause reading to fail, which are caught by various exceptions.

    """
    # set default value for DataFrame
    data = pd.DataFrame()
    try:
        data = pd.read_csv(s3_source)
    except FileNotFoundError as file_error:
        # This error will occur if the file specified is not found on S3.
        logger.error("The specified file or bucket does not exist: %s", file_error)
    except PermissionError as permission_error:
        # This error will occur if the AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are not set.
        logger.error("No permissions found. Could not read data from AWS S3 bucket. "
                     "In order to write to S3, the environment variables AWS_ACCESS_KEY_ID and "
                     "AWS_SECRET_ACCESS_KEY must be available. %s", permission_error)
    except botocore.exceptions.ClientError as client_error:
        # This exception will catch any AWS service exceptions. More information at:
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html
        logger.error("A client error occurred - could not read data from AWS S3 bucket. %s", client_error)
    except pd.errors.EmptyDataError as empty_data:
        # This error will occur if the file read is empty.
        logger.error("Could not parse the csv file because it is empty. %s", empty_data)
    else:
        logger.info("Successfully read data from AWS S3, %s", s3_source)
    finally:
        # Check if the dataframe is empty. If it is empty, raise a valueError because it means reading from S3 failed.
        if data.empty:
            raise ValueError("Could not read data from AWS S3.")

    return data
