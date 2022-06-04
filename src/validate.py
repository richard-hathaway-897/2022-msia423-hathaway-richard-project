import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataframe(data: pd.DataFrame,
                       duplicated_method: str) -> bool:
    """
    This function performs data validation on an input dataframe. It checks:
    1. Whether the input data is a dataframe
    2. Whether the input data is an empty dataframe
    3. If there are any null values
    4. If there are any duplicate rows

    Args:
        data (pd.DataFrame): The input dataframe
        duplicated_method (str): In the event of a duplicated record, which one to keep. ('first' or 'last')

    Returns:
        data_validated (bool): This function returns a boolean indicating if
            data validation succeeded (True) or failed (False).

    Raises:
        ValueError: This function raises a ValueError if data validation fails.
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

        # Count the number of duplicate values.
        try:
            count_duplicate_rows = data.duplicated(keep=duplicated_method).sum()
        except ValueError as val_error:
            # This error can occur if an invalid value is passed to duplicated() function.
            raise val_error

        # Count the number of null values.
        sum_null_values = data.isnull().sum()
        logger.debug("Found %d duplicate rows and %d rows with NA values.",
                     count_duplicate_rows.sum(),
                     sum_null_values.sum())
        logger.info("Completed data validation step.")

    if not data_validated:
        raise ValueError("Data validation failed. Either the input data is not a pandas dataframe or the dataframe"
                         "is empty.")

    return data_validated
