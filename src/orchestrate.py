"""This module contains functions that orchestrate the different function calls needed for each step of the pipeline.
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
    Robust exception handling for reading/writing files is handled inside the read_csv_url and s3_write functions.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any errors.
    """
    try:
        raw_data = src.read_write_functions.read_csv_url(data_source=command_line_args.data_url)
    except ValueError as val_error:
        # This error will be raised if reading the data from the URL fails for any reason.
        # See the logs for the src.read_write_functions module for more specific information on why reading failed.
        logger.error(val_error)
    else:
        src.read_write_s3.s3_write(raw_data, command_line_args.path_s3)


def run_clean_data(command_line_args: argparse.Namespace, config_dict: dict) -> None:
    """This function calls the necessary functions to load data, clean data, and save the cleaned data.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.
        config_dict (dict): A dictionary of configuration parameters

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any errors.
    """
    try:
        raw_data = src.read_write_s3.s3_read(s3_source=command_line_args.input_source)
    except ValueError:
        # This error will be raised if reading the data from S3 fails for any reason.
        # See the logs for the src.read_write_s3 module for more specific information on why reading failed.
        logger.error("Failed to read raw data from S3 for data cleaning. Failed to clean data.")
    else:
        try:
            src.validate.validate_dataframe(raw_data, **config_dict["validate_dataframe"])
        except ValueError as val_error:
            # This error will occur if data validation occurs for any reason
            logger.error("Data cleaning did not occur. Data validation failed. %s", val_error)
        else:
            try:
                cleaned_data = src.clean_data.clean_data(data=raw_data, **config_dict["clean_data"])
            except TypeError:
                # This error will occur if the input is not a pandas dataframe.
                logger.error("Data Cleaning failed. Input dataframe was not a pandas dataframe.")
            else:
                src.read_write_functions.save_csv(cleaned_data, command_line_args.output_path)


def run_generate_features(command_line_args: argparse.Namespace, config_dict: dict) -> None:
    """This function calls the necessary functions to load data, generate features, and save the train/test data.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.
        config_dict (dict): A dictionary of configuration parameters

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any errors.
    """
    # Read the input cleaned data
    try:
        input_data = src.read_write_functions.read_csv(command_line_args.input_source)
    except ValueError:
        logger.error("Failed to read in the cleaned data csv file. Please check the logs for further information.")
    else:
        # Perform data validation
        try:
            src.validate.validate_dataframe(input_data, **config_dict["validate_dataframe"])
        except ValueError as val_error:
            # This error will occur if data validation fails for any reason.
            logger.error("Feature generation did not occur. Data validation failed. %s", val_error)

        else:
            try:
                train_data, test_data, one_hot_encoder = \
                    src.data_preprocessing.generate_features(data=input_data,
                                                             remove_outlier_params=config_dict["remove_outliers"],
                                                             **config_dict["generate_features"]["pipeline_and_app"],
                                                             **config_dict["generate_features"]["pipeline_only"])
            except (KeyError, ValueError, TypeError):
                # More specific logging messages and exception handling occurs in the module. Here, because the program
                # just needs to know whether or not generating features failed, the program can catch all 3 exception
                # types in one block and handle them in the same way.
                logger.error("Generate Features step failed.")
            else:

                # Save the train and test data, and one-hot-encoder and print out messages if successful.
                # Exception handling for saving functions occurs inside the module.
                src.read_write_functions.save_csv(train_data, command_line_args.train_output_source)
                src.read_write_functions.save_csv(test_data, command_line_args.test_output_source)
                src.read_write_functions.save_model_object(one_hot_encoder, command_line_args.one_hot_path)


def run_train_model(command_line_args: argparse.Namespace, config_dict: dict) -> None:
    """This function calls the necessary functions to load train data, fit the model object, and save the model_object.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.
        config_dict (dict): A dictionary of configuration parameters

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any errors.
    """
    # Read in the training data
    try:
        training_data = src.read_write_functions.read_csv(command_line_args.train_input_source)
    except ValueError:
        # This error will occur if reading the data fails for any reason. More specific exception handling
        # occurs inside the module.
        logger.error("Failed to read in the training data csv file. Please check the logs for further information.")
    else:

        # Perform data validation
        try:
            src.validate.validate_dataframe(training_data, **config_dict["validate_dataframe"])
        except ValueError as val_error:
            logger.error("Model training did not occur. Data validation failed. %s", val_error)
        else:

            # Train the model
            try:
                trained_model = src.train_model.train_model(train_data=training_data,
                                                            **config_dict["model_training"]["random_forest"])
            except KeyError:
                logger.error("The response column was not found in the training data.")
            except ValueError:
                logger.error("The model failed to be fit.")
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

    Raises:
        This function does not raise any exceptions.
    """
    # Read in the test data
    data_read_error = False
    try:
        test_data = src.read_write_functions.read_csv(command_line_args.test_input_source)
    except ValueError:
        # This error will occur if reading the data fails for any reason. More specific exception handling
        # occurs inside the module.
        logger.error("Failed to read in the test data csv file. Please check the logs for further information.")
        data_read_error = True

    # Read in the model object
    try:
        model = src.read_write_functions.load_model_object(command_line_args.model_input_source)
    except ValueError:
        # This error will occur if reading the model fails for any reason. More specific exception handling
        # occurs inside the module.
        logger.error("Failed to read in the model object. Please check the logs for further information.")
        data_read_error = True

    # If the data and model are read successfully
    if not data_read_error:
        # Perform data validation
        try:
            src.validate.validate_dataframe(test_data, **config_dict["validate_dataframe"])
        except ValueError as val_error:
            logger.error("Model prediction did not occur. Data validation failed. %s", val_error)
        else:
            # Try to make predictions
            try:
                predictions = src.predict.make_predictions(new_data=test_data,
                                                           model=model,
                                                           is_test_data=True,
                                                           **config_dict["predict"])
            except KeyError:
                logger.error("Failed to make predictions. A column was missing from the dataframe.")
            except ValueError:
                logger.error("Failed to make predictions. The model object is not fitted or has invalid parameters.")
            else:
                # Else, save the predictions
                src.read_write_functions.save_csv(predictions,
                                                  command_line_args.predictions_output_source)


def run_evaluate(command_line_args: argparse.Namespace, config_dict: dict) -> None:
    """This function calls the necessary functions to load the test data and model predictions,
        create model evaluation metrics, and save the model evaluation metrics.

    Args:
        command_line_args (argparse.Namespace): The command line arguments passed to the program.
        config_dict (dict): A dictionary of configuration parameters

    Returns:
        This function does not return any object.

    Raises:
        This function does not raise any exceptions.
    """

    # Read in the test data
    read_error = False
    try:
        test_data = src.read_write_functions.read_csv(command_line_args.test_input_source)
    except ValueError:
        # This error will occur if reading the data fails for any reason. More specific exception handling
        # occurs inside the module.
        logger.error("Failed to read in the test data csv file. Please check the logs for further information.")
        read_error = True

    # Read in the predictions
    try:
        predictions = src.read_write_functions.read_csv(command_line_args.predictions_input_source)
    except ValueError:
        # This error will occur if reading the data fails for any reason. More specific exception handling
        # occurs inside the module.
        logger.error("Failed to read in the predictions csv file. Please check the logs for further information.")
        read_error = True

    # If both test data and predictions are successfully read, validate the data and compute evaluation metrics.
    if not read_error:
        try:
            src.validate.validate_dataframe(test_data, **config_dict["validate_dataframe"])
        except ValueError as val_error:
            logger.error("Model evaluation did not occur. Data validation failed. %s", val_error)
        else:

            try:
                metrics_dict = src.evaluate_model.evaluate_model(test=test_data,
                                                                 predictions=predictions,
                                                                 **config_dict["predict"])
            except KeyError:
                logger.error("Failed to evaluate the model. A required column was missing from the data.")
            except ValueError:
                logger.error("Failed to evaluate the model. An error occurred computing the final model metrics.")
            else:
                # Save the metrics as a text file.
                src.read_write_functions.\
                    save_dict_as_text(data=metrics_dict,
                                      output_path=command_line_args.performance_metrics_output_source)
