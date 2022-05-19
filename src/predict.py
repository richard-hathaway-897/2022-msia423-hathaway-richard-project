import logging

import pandas as pd
import sklearn.linear_model
import sklearn.preprocessing
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection
import joblib

import s3_actions

logger = logging.getLogger(__name__)

predictors = {'temp': [288.3],
              'clouds_all': [75],
              'holiday_binary': [0],
              'log_rain_1h': [0.01],
              'weather_main': ['Drizzle'],
              'month': [9],
              'day_of_week': ["Tuesday"],
              'hour': [13]
            }


def predict(predictors: dict, model_object_path: str, ohe_object_path, s3_bool: bool, output_path: str):
    one_hot_encode_columns = ["weather_main", "month", "hour", "day_of_week"] # TODO USE YAML
    if s3_bool:
        s3_actions.s3_read_from_file(model_object_path, "./models/trained_model_object_s3.joblib") #TODO Fix hardcoding
        s3_actions.s3_read_from_file(model_object_path, "./models/ohe_object.joblib")

    model_object = joblib.load("./models/trained_model_object1.joblib")
    one_hot_encoder = joblib.load("./models/ohe_object.joblib")

    new_data = pd.DataFrame(predictors)
    one_hot_array = one_hot_encoder.transform(new_data[["weather_main", "month", "hour", "day_of_week"]]) # TODO use yaml config herer
    #print(ohe_new_data)

    one_hot_column_names = one_hot_encoder.get_feature_names_out()
    one_hot_df = pd.DataFrame(one_hot_array, columns=one_hot_column_names)
    data_one_hot_encoded = new_data.join(one_hot_df).drop(one_hot_encode_columns, axis=1)
    print(data_one_hot_encoded.head())
    prediction = model_object.predict(data_one_hot_encoded)
    print(prediction)


predict(predictors,
        model_object_path="s3://2022-msia423-hathaway-richard/models/model1.joblib",
        ohe_object_path="s3://2022-msia423-hathaway-richard/models/one_hot_encoder.joblib",
        s3_bool=False,
        output_path="")