"""This module contains functions that orchestrate the different function calls needed for each step of the pipeline.
Note that all the exception handling occurs in each of the modules called by these orchestrate functions.

"""

import argparse
import logging.config

import src.clean_data
import src.data_preprocessing
import src.train_model
import src.read_write_functions
import src.validate
import src.read_write_s3
import src.predict
import src.evaluate_model

logger = logging.getLogger(__name__)


def fetch_data(command_line_args: argparse.Namespace) -> None:
    """This function calls the functions to pull the raw data from the URL and write it to S3.
    All exception handling for reading/writing files is handled inside the read_csv_url and s3_write functions.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any errors.
    """

    raw_data = src.read_write_functions.read_csv_url(data_source=command_line_args.data_url)
    if not raw_data.empty:
        src.read_write_s3.s3_write(raw_data, command_line_args.path_s3)


def run_clean_data(command_line_args: argparse.Namespace, config_dict: dict) -> None:
    """This function calls the necessary functions to load data, clean data, and save the cleaned data.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.
        config_dict (dict): A dictionary of configuration parameters

    Returns:
        This function does not return any object.
    """
    try:
        raw_data = src.read_write_s3.s3_read(s3_source=command_line_args.input_source)
    except ValueError:
        logger.error("Failed to read raw data from S3 for data cleaning.")
    else:
        valid_data = src.validate.validate_dataframe(
                     raw_data,
                     **config_dict["validate_dataframe"])
        if valid_data:
            cleaned_data = src.clean_data.clean_data(data=raw_data, **config_dict["clean_data"])
            src.read_write_functions.save_csv(cleaned_data, command_line_args.output_path)

def run_generate_features(command_line_args: argparse.Namespace, config_dict: dict) -> None:
    """This function calls the necessary functions to load data, generate features, and save the train/test data.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.
        config_dict (dict): A dictionary of configuration parameters

    Returns:
        This function does not return any object.
    """
    # Read the input cleaned data
    input_data = src.read_write_functions.read_csv(command_line_args.input_source)
    valid_data = src.validate.validate_dataframe(
        input_data,
        **config_dict["validate_dataframe"])

    if valid_data:
        train_data, test_data, one_hot_encoder = \
            src.data_preprocessing.generate_features(data=input_data,
                                                     remove_outlier_params=config_dict["remove_outliers"],
                                                     **config_dict["generate_features"]["pipeline_and_app"],
                                                     **config_dict["generate_features"]["pipeline_only"])
        train_data = src.clean_data.clean_data(data=train_data, **config_dict["clean_data"])
        test_data = src.clean_data.clean_data(data=test_data, **config_dict["clean_data"])

        # Save the train and test data, and print out messages if successful.
        src.read_write_functions.save_csv(train_data, command_line_args.train_output_source)
        src.read_write_functions.save_csv(test_data, command_line_args.test_output_source)

def run_train_model(command_line_args: argparse.Namespace, config_dict: dict) -> None:
    """This function calls the necessary functions to load train data, fit the model object, and save the model_object.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.
        config_dict (dict): A dictionary of configuration parameters

    Returns:
        This function does not return any object.
    """
    # Read in the training data
    training_data = src.read_write_functions.read_csv(command_line_args.train_input_source)
    valid_data = src.validate.validate_dataframe(
        training_data,
        **config_dict["validate_dataframe"])
    if valid_data:
        try:
            trained_model = src.train_model.train_model(train_data=training_data,
                                                        **config_dict["model_training"]["random_forest"])
        except KeyError:
            logger.error("The response column was not found in the training data.")
        else:
            src.read_write_functions.save_model_object(trained_model, command_line_args.model_output_source)


def run_predict(command_line_args: argparse.Namespace, config_dict: dict) -> None:
    """This function calls the necessary functions to load the test data and trained model object,
        make predictions using the model, and save the model predictions.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.
        config_dict (dict): A dictionary of configuration parameters

    Returns:
        This function does not return any object.
    """
    # Read in the test data
    test_data = src.read_write_functions.read_csv(command_line_args.test_input_source)
    model = src.read_write_functions.load_model_object(command_line_args.model_input_source)
    valid_data = src.validate.validate_dataframe(
        test_data,
        **config_dict["validate_dataframe"])
    if valid_data and model is not None:
        predictions = src.predict.make_predictions(new_data=test_data, model=model, is_test_data=True, **config_dict["predict"])
        if not predictions.empty:
            src.read_write_functions.save_csv(predictions,
                                              command_line_args.predictions_output_source)
#
def run_evaluate(command_line_args: argparse.Namespace, config_dict: dict) -> None:
    """This function calls the necessary functions to load the test data and model predictions,
        create model evaluation metrics, and save the model evaluation metrics.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.
        config_dict (dict): A dictionary of configuration parameters

    Returns:
        This function does not return any object.
    """

    # Read in the test data
    test_data = src.read_write_functions.read_csv(command_line_args.test_input_source)
    valid_data = src.validate.validate_dataframe(
        test_data,
        **config_dict["validate_dataframe"])
    predictions = src.read_write_functions.read_csv(command_line_args.predictions_input_source)
    if valid_data and not predictions.empty:
        metrics_dict = src.evaluate_model.evaluate_model(test=test_data,
                                                         predictions=predictions,
                                                         **config_dict["predict"])
        src.read_write_functions.save_dict_as_text(data=metrics_dict,
                                                   output_path=command_line_args.performance_metrics_output_source)