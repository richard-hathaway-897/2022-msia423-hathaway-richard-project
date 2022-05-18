"""Configures the subparsers for receiving command line arguments for each
 stage in the model pipeline and orchestrates their execution."""
import argparse
import logging.config

import yaml

from config.flaskconfig import SQLALCHEMY_DATABASE_URI
import config.config
#from src.add_songs import create_db, add_song
from src.create_tables_rds import create_db_richard
from src.s3_actions import s3_write
import src.data_preprocessing
import src.train_model

logging.config.fileConfig('config/logging/local.conf')
logger = logging.getLogger('penny-lane-pipeline')

if __name__ == '__main__':

    # Add parsers for both creating a database and adding songs to it
    parser = argparse.ArgumentParser(
        description="Create and/or add data to database")
    subparsers = parser.add_subparsers(dest='subparser_name')

    # Sub-parser for creating a database
    sp_create = subparsers.add_parser("create_db",
                                      description="Create database")
    sp_create.add_argument("--engine_string", default=SQLALCHEMY_DATABASE_URI,
                           help="SQLAlchemy connection URI for database")

    # Sub-parser for ingesting new data
    sp_ingest = subparsers.add_parser("ingest",
                                      description="Add data to database")
    sp_ingest.add_argument("--artist", default="Emancipator",
                           help="Artist of song to be added")
    sp_ingest.add_argument("--title", default="Minor Cause",
                           help="Title of song to be added")
    sp_ingest.add_argument("--album", default="Dusk to Dawn",
                           help="Album of song being added")
    sp_ingest.add_argument("--engine_string",
                           default='sqlite:///data/tracks.db',
                           help="SQLAlchemy connection URI for database")

    sp_fetch_raw_data = subparsers.add_parser("fetch",
                                      description="Fetch raw data and save to S3")
    sp_fetch_raw_data.add_argument("--path_s3", type=str,
                        required=True,
                        help = "Path of the data on s3 to read from or write to.")
    sp_fetch_raw_data.add_argument("--data_url", type=str,
                        help = "Local path or URL to read from or write to.")
    sp_fetch_raw_data.add_argument("--delimiter", type=str,
                        default = ",",
                        help = "The delimiter of the file.")

    sp_clean_data = subparsers.add_parser("clean",
                                      description="Fetch raw data and save to S3")
    sp_clean_data.add_argument("--data_source", type=str,
                        required=True,
                        help = "Path of the data on s3 to read from or write to.")
    sp_clean_data.add_argument("--output_path", type=str,
                        help = "Local path or URL to read from or write to.")
    sp_clean_data.add_argument("--delimiter", type=str,
                        default = ",",
                        help = "The delimiter of the file.")

    sp_generate_features = subparsers.add_parser("create_features",
                                      description="Fetch raw data and save to S3")
    sp_generate_features.add_argument("--data_source", type=str,
                        required=True,
                        help = "Path of the data on s3 to read from or write to.")
    sp_generate_features.add_argument("--output_path", type=str,
                        help = "Local path or URL to read from or write to.")
    sp_generate_features.add_argument("--delimiter", type=str,
                        default = ",",
                        help = "The delimiter of the file.")


    sp_generate_features = subparsers.add_parser("train_model",
                                      description="Fetch raw data and save to S3")
    sp_generate_features.add_argument("--data_source", type=str,
                        required=True,
                        help = "Path of the data on s3 to read from or write to.")
    sp_generate_features.add_argument("--output_path_local", type=str,
                                      default = "./models/trained_model_object.joblib",
                                      help="Local path or URL to read from or write to.")
    sp_generate_features.add_argument("--output_path_s3", type=str,
                                      help="Local path or URL to read from or write to.")
    sp_generate_features.add_argument("--delimiter", type=str,
                        default = ",",
                        help = "The delimiter of the file.")

    args = parser.parse_args()
    command_choice = args.subparser_name
    if command_choice == 'create_db':
        create_db_richard(args.engine_string)
    elif command_choice == 'ingest':
        #add_song(args)
        pass
    elif command_choice == 'fetch':
        s3_write(s3_destination=args.path_s3, data_source=args.data_url, delimiter=args.delimiter)
    elif command_choice == 'clean':
        src.data_preprocessing.clean_data(data_source=args.data_source, clean_data_path=args.output_path,
                                          delimiter=args.delimiter)
    elif command_choice == 'create_features':

        try:
            with open(config.config.MODEL_CONFIG_PATH, "r", encoding="utf-8") as preprocess_yaml:
                preprocess_parameters = yaml.load(preprocess_yaml, Loader=yaml.FullLoader)
        except FileNotFoundError:
            logger.error("Could not locate the model configuration file specified in config.config.py: %s.",
                         config.config.MODEL_CONFIG_PATH)
        else:
            src.data_preprocessing.generate_features(data_source=args.data_source,
                                                     features_path=args.output_path,
                                                     preprocess_params=preprocess_parameters["preprocess_data"],
                                                     delimiter=args.delimiter)
    elif command_choice == 'train_model':
        try:
            with open(config.config.MODEL_CONFIG_PATH, "r", encoding="utf-8") as preprocess_yaml:
                preprocess_parameters = yaml.load(preprocess_yaml, Loader=yaml.FullLoader)
        except FileNotFoundError:
            logger.error("Could not locate the model configuration file specified in config.config.py: %s.",
                         config.config.MODEL_CONFIG_PATH)
        else:
            src.train_model.train_model(model_data_source=args.data_source,
                                        model_training_params=preprocess_parameters["model_training"],
                                        model_output_local_path=args.output_path_local,
                                        model_output_s3_path = args.output_path_s3,
                                        delimiter=args.delimiter)
    else:
        parser.print_help()
