import logging

import pandas as pd
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection
import yaml
import joblib

import src.s3_actions
import src.data_preprocessing

logger = logging.getLogger(__name__)

# predictors = {'temp': 288.3,
#               'clouds_all': 75,
#               'holiday': "None",
#               'rain_1h': 0.01,
#               'weather_main': 'Drizzle',
#               'month': 9,
#               'day_of_week': "Tuesday",
#               'hour': 13
#             }


def predict_preprocess(predictors: dict, preprocess_params: dict) -> pd.DataFrame:
    # Call to validate()
    predictors_dict_for_df = {}
    for column_name, value in predictors.items():
        predictors_dict_for_df[column_name] = [value]
    prediction_df = pd.DataFrame(predictors_dict_for_df)

    prediction_df = src.data_preprocessing.collapse_weather_categories(prediction_df, preprocess_params)
    prediction_df = src.data_preprocessing.binarize_column(prediction_df, preprocess_params)
    prediction_df = src.data_preprocessing.log_transform(prediction_df, preprocess_params)

    prediction_df = prediction_df.drop(list(preprocess_params["log_transform_columns"]) +
                                        list(preprocess_params["binarize_columns"]), axis=1)
    logger.info("Dropped the following columns from the dataset: %s", str(list(preprocess_params["drop_columns"]) +
                                                                          list(preprocess_params[
                                                                                   "log_transform_columns"]) +
                                                                          list(preprocess_params["binarize_columns"])))
    return prediction_df


def predict(new_data: pd.DataFrame, model_object_path: str = "", ohe_object_path: str = "", s3_bool: bool = False):
    one_hot_encode_columns = ["weather_main", "month", "hour", "day_of_week"] # TODO USE YAML
    if s3_bool:
        src.s3_actions.s3_read_from_file(model_object_path, "./trained_model_object_s3.joblib") #TODO Fix hardcoding
        src.s3_actions.s3_read_from_file(ohe_object_path, "./ohe_object.joblib")

    model_object = joblib.load("./models/trained_model_object1.joblib")
    one_hot_encoder = joblib.load("./models/ohe_object.joblib")

    #new_data = pd.DataFrame(predictors)
    one_hot_array = one_hot_encoder.transform(new_data[["weather_main", "month", "hour", "day_of_week"]]) # TODO use yaml config herer
    #print(ohe_new_data)

    one_hot_column_names = one_hot_encoder.get_feature_names_out()
    one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_column_names)
    data_one_hot_encoded = new_data.join(one_hot_df).drop(one_hot_encode_columns, axis=1)
    print(data_one_hot_encoded.head())
    prediction = model_object.predict(data_one_hot_encoded)
    print(prediction)
    return prediction


# with open("/Users/richard/Documents/school_work/SpringQuarter/AVC/2022-msia423-hathaway-richard-project/config/model_config.yaml", "r", encoding="utf-8") as preprocess_yaml:
#     preprocess_parameters = yaml.load(preprocess_yaml, Loader=yaml.FullLoader)
# df = predict_preprocess(predictors, preprocess_parameters["preprocess_data"])
# predict(df)
