python3 run.py fetch --path_s3=s3://2022-msia423-hathaway-richard/raw_data/metro_interstate_traffic_volume.csv --data_url=https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz

python3 run.py clean --config_path=./config/model_config.yaml --input_source=s3://2022-msia423-hathaway-richard/raw_data/metro_interstate_traffic_volume.csv --output_path=./data/clean_data/cleaned_data.csv

python3 run.py generate_features --config_path=./config/model_config.yaml --input_source=./data/clean_data/cleaned_data.csv --one_hot_path=./models/ohe_object.joblib --train_output_source=./data/train_test_data/train_data.csv --test_output_source=./data/train_test_data/test_data.csv

python3 run.py train_model --config_path=./config/model_config.yaml --train_input_source=./data/train_test_data/train_data.csv --model_output_source=./models/trained_model_object.joblib

python3 run.py predict --config_path=./config/model_config.yaml --test_input_source=./data/train_test_data/test_data.csv --model_input_source=./models/trained_model_object.joblib --predictions_output_source=./data/predictions/predictions.csv

python3 run.py evaluate --config_path=./config/model_config.yaml --test_input_source=./data/train_test_data/test_data.csv --predictions_input_source=./data/predictions/predictions.csv --performance_metrics_output_source=./data/model_performance/performance_metrics.txt