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

logging.config.fileConfig("config/logging/local.conf")
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
        # TODO: Add data validation
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

    train_data, test_data, one_hot_encoder = \
        src.data_preprocessing.generate_features(data=input_data,
                                                 remove_outlier_params=config_dict["remove_outliers"],
                                                 **config_dict["generate_features"])

    # Save the train and test data, and print out messages if successful.
    src.read_write_functions.save_csv(train_data, command_line_args.train_output_source)
    src.read_write_functions.save_csv(test_data, command_line_args.test_output_source)

# def run_train_model(command_line_args: argparse.Namespace, config_dict: dict) -> None:
#     """This function calls the necessary functions to load train data, fit the model object, and save the model_object.
#
#     Args:
#         command_line_args (argparse.Namespace): The command line arguments passed to the program.
#         config_dict (dict): A dictionary of configuration parameters
#
#     Returns:
#         This function does not return any object.
#     """
#     # Read in the training data
#     input_data = src.helper_functions.read_csv(command_line_args.train_input_source)
#
#     # Perform data validation
#     all_columns = config_dict["columns"]["predictor_columns"] + [config_dict["columns"]["target_column_name"]]
#     valid_data = src.validate.validate_clouds(data=input_data,
#                                               expected_columns=all_columns, **config_dict["validate_clouds"])
#     # If the data is valid, train the model
#     if valid_data:
#         trained_model = src.train_model.train_model(train_data=input_data,
#                                                     **config_dict["train_model"]["model_columns"],
#                                                     **config_dict["train_model"]["random_forest_classifier"])
#         src.helper_functions.save_model_object(trained_model, command_line_args.model_output_source)
#
#
# def run_score_model(command_line_args: argparse.Namespace, config_dict: dict) -> None:
#     """This function calls the necessary functions to load the test data and trained model object,
#         make predictions using the model, and save the model predictions.
#
#     Args:
#         command_line_args (argparse.Namespace): The command line arguments passed to the program.
#         config_dict (dict): A dictionary of configuration parameters
#
#     Returns:
#         This function does not return any object.
#     """
#     # Read in the test data
#     test_data = src.helper_functions.read_csv(command_line_args.test_input_source)
#
#     # Perform data validation
#     all_columns = config_dict["columns"]["predictor_columns"] + [config_dict["columns"]["target_column_name"]]
#     valid_data = src.validate.validate_clouds(data=test_data,
#                                               expected_columns=all_columns, **config_dict["validate_clouds"])
#
#     # Load the trained model object
#     model = src.helper_functions.load_model_object(command_line_args.model_input_source)
#
#     # If loading data was successful, call the score_model function
#     if valid_data and (model is not None):
#         class_predictions, probability_predictions = src.score_evaluate_model.score_model(model=model,
#                                                                                           test_data=test_data,
#                                                                                           **config_dict["score_model"])
#         # If the score_model function was successful, then save the output files.
#         # Check if the output files from the previous step are empty
#         if (not class_predictions.empty) and (not probability_predictions.empty):
#             class_status = src.helper_functions.save_csv(class_predictions,
#                                                          command_line_args.class_output_source)
#             prob_status = src.helper_functions.save_csv(probability_predictions,
#                                                         command_line_args.prob_output_source)
#             # If saving was successful, log out a message.
#             if class_status and prob_status:
#                 logger.info("Successfully saved class predictions and probability predictions to '%s' and '%s'",
#                             command_line_args.class_output_source,
#                             command_line_args.prob_output_source)
#             else:
#                 logger.error("Failed to save class predictions and probability predictions.")
#
#
# def run_evaluate_model(command_line_args: argparse.Namespace, config_dict: dict) -> None:
#     """This function calls the necessary functions to load the test data and model predictions,
#         create model evaluation metrics, and save the model evaluation metrics.
#
#     Args:
#         command_line_args (argparse.Namespace): The command line arguments passed to the program.
#         config_dict (dict): A dictionary of configuration parameters
#
#     Returns:
#         This function does not return any object.
#     """
#
#     # Read in the test data
#     test_data = src.helper_functions.read_csv(command_line_args.test_input_source)
#     all_columns = config_dict["columns"]["predictor_columns"] + [config_dict["columns"]["target_column_name"]]
#
#     # Perform data validation
#     valid_data = src.validate.validate_clouds(data=test_data,
#                                               expected_columns=all_columns, **config_dict["validate_clouds"])
#     # Read in the predicted values
#     class_predictions = src.helper_functions.read_csv(command_line_args.class_input_source)
#     prob_predictions = src.helper_functions.read_csv(command_line_args.prob_input_source)
#
#     # If the data is valid, call the evaluate_model function
#     if valid_data and (not class_predictions.empty) and (not prob_predictions.empty):
#         metrics, confusion_matrix, class_report = \
#             src.score_evaluate_model.evaluate_model(test=test_data,
#                                                     ypred_proba_test=prob_predictions,
#                                                     ypred_bin_test=class_predictions,
#                                                     response_column=config_dict["columns"]["target_column_name"],
#                                                     **config_dict["evaluate_model"]["confusion_matrix"])
#
#         # Save the output files from evaluate_model
#         metrics_save_status = src.helper_functions.save_csv(metrics, command_line_args.auc_accuracy_output_source)
#         confusion_matrix_save_status = src.helper_functions.save_csv(confusion_matrix,
#                                                                      command_line_args.auc_accuracy_output_source)
#         class_save_status = src.helper_functions.save_csv(class_report, command_line_args.report_output_source)
#
#         # If saving was successful, log out a message.
#         if metrics_save_status and confusion_matrix_save_status and class_save_status:
#             logger.info("Successfully saved model evaluation artifacts to '%s', '%s', '%s'.",
#                         command_line_args.auc_accuracy_output_source,
#                         command_line_args.auc_accuracy_output_source,
#                         command_line_args.report_output_source)
