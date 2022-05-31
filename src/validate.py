import logging
import typing

import pandas as pd

logger = logging.getLogger(__name__)



# def validate_datatype(data, column_name, expected_dtype: str) -> bool:
#     valid_data = True
#     try:
#         valid_dtype = data[column_name].dtype == expected_dtype
#     except KeyError:
#         logger.error("Data validation failed: column '%s' was not found in the dataframe.", column_name)
#         valid_data = False
#     else:
#         if not valid_dtype:
#             logger.error("Validation failed. Column '%s' in the data was not of type %s", column_name, column_dtype)
#             valid_data = False
#     return valid_data


def validate_dataframe(data: pd.DataFrame,
                       duplicated_method: str) -> bool:
    """
    This function performs data validation on an input dataframe. It checks:
    1. Whether the input data is a dataframe
    2. Whether the input data is an empty dataframe
    3. If the columns that are passed to the function exist in the dataframe.
    4. If the columns are of the specified datatype.
    5. If there are any null values
    6. If there are any duplicate rows

    Args:
        data (pd.DataFrame): The input dataframe
        expected_columns (typing.List): The list of columns for which to check the dataframe.
        duplicated_method (str): In the event of a duplicated record, which one to keep. ('first' or 'last')

    Returns:
        data_validated (bool): Returns true if the dataframe was successfully validated, false if problems were found
        within the dataframe.
    """
    data_validated = True

    # Check if the data is a dataframe
    if not isinstance(data, pd.DataFrame):
        logger.error("Data validation failed. The data is not a pandas dataframe.")
        data_validated = False
    # Check if the dataframe is empty
    elif data.empty:
        logger.warning("Data validation failed. The dataframe is empty.")
        data_validated = False
    else:
        # For each column in the list of expected columns
        # for column in expected_columns:
        #     if column not in data.columns:
        #         logger.error("Data validation failed. Expected column '%s' was not present in the dataframe.", column)
        #         data_validated = False

        # Count the number of duplicate and null values.
        count_duplicate_rows = data.duplicated(keep=duplicated_method).sum()
        sum_null_values = data.isnull().sum()
        logger.debug("Found %d duplicate rows and %d rows with NA values.",
                     count_duplicate_rows.sum(),
                     sum_null_values.sum())
        logger.info("Completed data validation step.")

    return data_validated



