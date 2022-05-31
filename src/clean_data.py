import logging

import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(data: pd.DataFrame, duplicated_method: str = "first") -> pd.DataFrame:

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
