import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(data: pd.DataFrame, duplicated_method: str = "first") -> pd.DataFrame:
    """This function performs data cleaning operations. It performs two main operations - removal of NA values and
    removal of duplicate values

    Args:
        data (pd.DataFrame): The input dataframe to clean.
        duplicated_method (str): Which of the duplicated rows to keep. Either 'first' or 'last'. Default is 'first'.

    Returns:
        data (pd.DataFrame): This function returns a pandas dataframe with the cleaned data. If the data is not able to
        be cleaned, a TypeError is raised.

    Raises:
        TypeError: A type error is raised if the input dataframe is not pandas dataframe.

    """
    if not isinstance(data, pd.DataFrame):
        logger.error("Could not clean the input data. The input object was not a dataframe.")
        raise TypeError("Input data is not a pandas dataframe.")
    data_before_shape = data.shape
    logger.info("%d records and %d columns in input data.", data_before_shape[0], data_before_shape[1])
    count_duplicate_rows = data.duplicated(keep=duplicated_method).sum()
    data = data.drop_duplicates(keep=duplicated_method)
    count_null_values = data.isnull().sum()
    data = data.dropna()
    data_after_shape = data.shape

    logger.info("Dropped %d duplicate rows and %d rows with NA values.",
                count_duplicate_rows.sum(),
                count_null_values.sum())
    logger.info("Clean data has %d records and %d columns.", data_after_shape[0], data_after_shape[1])
    data = data.reset_index(drop=True)

    return data
