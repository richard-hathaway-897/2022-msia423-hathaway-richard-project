"""Configures the subparsers for receiving command line arguments for each
 stage in the model pipeline and orchestrates their execution."""
import argparse
import logging.config

import yaml

from config.flaskconfig import SQLALCHEMY_DATABASE_URI
import config.config
from src.create_tables_rds import create_db
import src.data_preprocessing
import src.train_model
#import src.predict
import src.orchestrate
import src.read_write_functions

logging.config.fileConfig('config/logging/local.conf')
logger = logging.getLogger('penny-lane-pipeline')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the traffic prediction database creation, data ingestion,"
                                                 " and model pipeline.")
    # Define the subparsers
    subparsers = parser.add_subparsers(dest="action")
    create_db_subparser = subparsers.add_parser("create_db", description="Create database")
    initialize_tables_subparser = subparsers.add_parser("initialize_tables",
                                                        description="Clear tables in the database and "
                                                                    "recreate the initial states of tables.")
    fetch_data_subparser = subparsers.add_parser("fetch", description="Fetch the raw data.")
    clean_data_subparser = subparsers.add_parser("clean", description="Clean the raw data.")
    generate_features_subparser = subparsers.add_parser("generate_features", description="Generate model features.")
    train_model_subparser = subparsers.add_parser("train_model", description="Train model.")
    score_model_subparser = subparsers.add_parser("score_model",
                                                  description="Score the model (make predictions using the model).")
    evaluate_subparser = subparsers.add_parser("evaluate_performance",
                                               description="Calculate model performance metrics.")

    # Create DB subparser
    create_db_subparser.add_argument("--engine_string",
                                     default=SQLALCHEMY_DATABASE_URI,
                                     help="SQLALCHEMY DATABASE URI connection string.")

    # Initialize RDS Tables SubParser
    initialize_tables_subparser.add_argument("--engine_string",
                                             default=SQLALCHEMY_DATABASE_URI,
                                             help="SQLALCHEMY DATABASE URI connection string.")
    # Fetch Data subparser
    fetch_data_subparser.add_argument("--path_s3", type=str,
                                      required=True,
                                      help="Path of the data on s3 to write to.")
    fetch_data_subparser.add_argument("--data_url", type=str,
                                      help="Local path or URL to read from or write to.")

    # Clean Data subparser
    clean_data_subparser.add_argument("--config_path", type=str,
                                      default="./config/model_config.yaml",
                                      help="Location of the pipeline configuration yaml file.")
    clean_data_subparser.add_argument("--input_source", type=str,
                                      required=True,
                                      help="Location of the input data file.")
    clean_data_subparser.add_argument("--output_path", type=str,
                                      required=True, help="Location of the clean data output file.")

    # Generate Features Subparser
    generate_features_subparser.add_argument("--config_path", type=str,
                                             default="./config/model_config/pipeline_config.yaml",
                                             help="Location of the pipeline configuration yaml file.")
    generate_features_subparser.add_argument("--input_source", type=str,
                                             required=True,
                                             help="Path of the cleaned data to read from.")
    generate_features_subparser.add_argument("--one_hot_path", type=str,
                                             default="./models/ohe_object.joblib",
                                             help="Local path to which to save the one_hot_encoder.")
    generate_features_subparser.add_argument("--train_output_source", type=str,
                                             required=True,
                                             help="Location of the output file for the training data.")
    generate_features_subparser.add_argument("--test_output_source", type=str,
                                             required=True, help="Location of the output file for the test data.")

    # Train Model Subparser
    train_model_subparser.add_argument("--config_path", type=str,
                                       default="./config/model_config/pipeline_config.yaml",
                                       help="Location of the pipeline configuration yaml file.")
    train_model_subparser.add_argument("--train_input_source", type=str,
                                       required=True,
                                       help="Location of the input training data file.")
    train_model_subparser.add_argument("--model_output_source", type=str,
                                       default="./models/trained_model_object.joblib",
                                       required=True,
                                       help="Location of the output file for the trained model object.")

    # Score Model Subparser
    score_model_subparser.add_argument("--config_path", type=str,
                                       default="./config/model_config/pipeline_config.yaml",
                                       help="Location of the pipeline configuration yaml file.")
    score_model_subparser.add_argument("--model_input_source", type=str,
                                       required=True,
                                       help="Location of the input file for trained model object.")
    score_model_subparser.add_argument("--one_hot_input_source", type=str,
                                       required=True,
                                       help="Location of the input file for one-hot-encoder object.")
    score_model_subparser.add_argument("--test_input_source", type=str,
                                       required=True,
                                       help="Location of the input file for the test data.")
    score_model_subparser.add_argument("--predictions_output_source", type=str,
                                       required=True,
                                       help="Location of the output file for the predicted classes.")

    # Evaluate Model Subparser
    evaluate_subparser.add_argument("--config_path", type=str,
                                    default="./config/model_config/pipeline_config.yaml",
                                    help="Location of the pipeline configuration yaml file.")
    evaluate_subparser.add_argument("--test_input_source", type=str,
                                    required=True,
                                    help="Location of the input file for test data (true values).")
    evaluate_subparser.add_argument("--predictions_input_source", type=str,
                                    required=True,
                                    help="Location of the input file for predicted values.")
    evaluate_subparser.add_argument("--performance_metrics_output_source", type=str,
                                    required=True,
                                    help="Location of the input file for the predicted class probabilities.")

    command_line_args = parser.parse_args()
    command_choice = command_line_args.action

    if command_choice == 'create_db':
        create_db(command_line_args.engine_string)
    elif command_choice == 'initialize_tables':
        pass
    elif command_choice == 'fetch':
        src.orchestrate.fetch_data(command_line_args)
    else:
        # Every other option needs the yaml config file, so read it in here first before continuing.
        config_dict = src.read_write_functions.read_yaml(command_line_args.config_path)
        if len(config_dict) > 0:
            if command_choice == 'clean':
                src.orchestrate.run_clean_data(command_line_args, config_dict=config_dict)
            elif command_choice == 'generate_features':
                src.orchestrate.run_generate_features(command_line_args, config_dict=config_dict)
            else:
                pass
            # elif command_choice == 'train_model':
            #     try:
            #         with open(config.config.MODEL_CONFIG_PATH, "r", encoding="utf-8") as preprocess_yaml:
            #             preprocess_parameters = yaml.load(preprocess_yaml, Loader=yaml.FullLoader)
            #     except FileNotFoundError:
            #         logger.error("Could not locate the model configuration file specified in config.config.py: %s.",
            #                      config.config.MODEL_CONFIG_PATH)
            #     else:
            #         src.train_model.train_model(model_data_source=args.data_source,
            #                                     model_training_params=preprocess_parameters["model_training"],
            #                                     model_output_local_path=args.output_path_local,
            #                                     model_output_s3_path=args.output_path_s3,
            #                                     delimiter=args.delimiter)
    # elif command_choice == 'predict':
    #
    #     src.predict.predict(predictors=,
    #                         model_training_params=preprocess_parameters["model_training"],
    #                         model_output_local_path=args.output_path_local,
    #                         model_output_s3_path=args.output_path_s3,
    #                         delimiter=args.delimiter)
        else:
            parser.print_help()
