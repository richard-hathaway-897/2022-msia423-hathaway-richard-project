# Traffic Prediction in the Twin Cities
**Richard Hathaway, Spring 2022**

This project was completed as part of MSIA 423 Analytics Value Chain course in the M.S. in Analytics program at Northwestern University.

## Project Charter and Background
### Vision
The Minneapolis-St. Paul Metropolitan area is a metropolis of over 3.1 million people as of 2019 [1]. With a large population, traffic congestion is a significant problem for commuters, causing drivers in the Minneapolis-St. Paul region to lose an estimated 52 hours each in 2019, according to a summary of the INRIX 2019 Global Traffic Scorecard reported in the Minneapolis/St. Paul Business Journal [2]. Additionally, adverse weather, particularly in winter months, can cause major traffic issues and cause further disruptions. Understanding traffic patterns at different times of the year and at different hours of the day in the region is vital for city planners to develop solutions for mounting traffic problems and for commuters to be able to plan daily travel.

### Mission
This web application will predict traffic counts along westbound Interstate 94 at Minnesota DoT ATR Station 301, which is located near mile marker 239 in St. Paul [3], to inform users of possible traffic at a given time using a predictive model. Data was obtained from the UCI Machine Learning Repository [https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume) [4], which contains both hourly traffic counts from 2012-2018 from the Minnesota Department of Transportation [5] and weather information, such as temperature, hourly precipitation totals, and weather descriptors originating from OpenWeatherMap [6]. With this web application, a user, such as a daily commuter, will be able to input the forecasted weather conditions for a given time and receive a prediction of hourly traffic volume and a label of “heavy”, “medium”, or “light” traffic. This application will help drivers plan their commute along this stretch of Interstate 94 and can serve as the first step in developing a comprehensive traffic prediction tool for the Minneapolis – St. Paul region.

### Success Criteria
During model development, a 5-fold cross-validation R-squared of at least 0.8 should be achieved before the model is deployed, meaning that a successful predictive model will be able to explain at least 80% of the variation in hourly traffic counts at ATR Station 301. However, this metric can be negotiated with project stakeholders if initial data analysis reveals that there is more variation in the data than is initially anticipated.

Additionally, when the web application is deployed to the public, data on user satisfaction with the web application will be collected by asking users “Was this prediction helpful?”. Responses in the form of clicking a “like” or a “dislike” button will be recorded, and a threshold of 75% likes will need to be achieved to consider this project successful. If successful, the project stakeholders can consider developing additional predictive models and web applications for traffic counts at other locations across the Minneapolis-St. Paul region.

### References
[1] Metropolitan Council. (2020). *2019 Final Population and Household Estimates*. [https://metrocouncil.org/Data-and-Maps/Publications-And-Resources/Files-and-reports/2019-Population-Estimates-(FINAL,-July-2020).aspx](https://metrocouncil.org/Data-and-Maps/Publications-And-Resources/Files-and-reports/2019-Population-Estimates-(FINAL,-July-2020).aspx)

[2] Reilly, M. (2020, March 11). It’s not just you; traffic really is getting worse. *Minneapolis/St. Paul Business Journal*. [https://www.bizjournals.com/twincities/news/2020/03/11/its-not-just-you-traffic-really-is-getting-worse.html](https://www.bizjournals.com/twincities/news/2020/03/11/its-not-just-you-traffic-really-is-getting-worse.html)

[3] Minnesota Department of Transportation. (2018). *Monthly Report, Station No. 301 (ATR) January 2018*. [https://dot.state.mn.us/traffic/data/reports/atr/Monthly_PDFs/Jan18/301.pdf](https://dot.state.mn.us/traffic/data/reports/atr/Monthly_PDFs/Jan18/301.pdf)

[4] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. *Metro Interstate Traffic Volume Data Set*.  [https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

[5] Minnesota Department of Transportation. [https://www.dot.state.mn.us/](https://www.dot.state.mn.us/)

[6] OpenWeatherMap. [https://openweathermap.org/](https://openweathermap.org/)


## Application

This web application enables a user to predict the traffic volume at westbound Interstate 94 at Minnesota DoT ATR Station 301 given several input variables. In order to generate these predictions, the user must input the following items:

**Temperature:** Enter a temperature between -40 and 115 degrees fahrenheit.

**Percentage of Cloud Cover:** Enter a value between 0 and 100.

**Weather Description:** Enter a weather description. Valid values are (case_sensitive): Clouds, Clear, Mist, Rain, Snow, Drizzle, Haze, Thunderstorm, Fog, Smoke, and Squall

**Month:** Enter a numeric value between 1 and 12 for the month.

**Hour:** Enter a numeric value between 0 and 23 for the hour.

**Day Of Week:** Enter a day of the week (case_sensitive): Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, or Saturday.

**Holiday:** Enter "None" if the date is not a holiday. Any other value will be treated as a holiday.

**Hourly Rainfall in mm:** Enter a value between 0 and 300 for the hourly rainfall in mm.


## Project Design Decisions

The repository contains the following scripts, modules, and pipelines which were split out by functionality:

1. Main executable python scripts in the root directory: `run.py` and `app.py`.
2. The bash script `run-pipeline.sh` at the root of the directory for running the entire model pipeline.
3. There are 13 python modules in the `src` folder. These modules organize functions by functionality:
   - `app_module.py` contains orchestration functions that organize functionality of the app such as creating predictions and updating tables.
   - `clean_data.py` contains functions that are used for performing data cleaning operations.
   - `data_preprocessing.py` contains functions for transforming the cleaned data into data suitable for modeling.
   - `evaluate_model.py` contains functions for evaluating the final model.
   - `orchestrate.py` contains orchestration functions for each step of the pipeline.
   - `predict.py` contains functions that generate predictions using a trained model object and test data or user input from the web app.
   - `predict_preprocess_app_input.py` contains functions that validates and processes user input from the web app to so that predictions can be made with it.
   - `read_write_functions.py` contains functions for reading and writing files to/from local folders.
   - `read_write_s3.py` contains functions for reading/writing data to/from S3.
   - `remove_outliers.py` contains functions for removing outliers. It also serves as data validation for user input from the app.
   - `train_model.py` contains functions for training a model object.
   - `validate.py` contains functions for validating data that can be used at each step of the model pipeline.
4. There are 8 modules that contain functions eligible for unit testing. These unit tests are split out by module and are located in the `tests` directory.

## Directory structure 
1. **app** - Folder for storing .html and .css files for the webpage
   1. **static** - Folder for holding static templates like .css files
      1. `basic.css` - Basic css styling for the web app.
   2. **templates** - Folder for holding .html files
      1. `database_error.html` - HTML page for when a database error occurs.
      2. `error.html` - HTML page for when any non-database error occurs.
      3. `index.html` - HTML page for the main web application.
2. **config** - Folder for configuration files
   1. **local** - Folder for storing local configuration files
   2. **logging** - Folder for logging configuration files
      1. `local.conf` - Logging configuration file
   3. `database_config.py` - Configurations used for initializing and updating database tables.
   4. `flaskconfig.py` - Configurations used by the Flask app.
   5. `model_config.yaml` - YAML file to hold configurations for the model pipeline.
3. **data** - Folder to hold data sources
   1. **clean_data** - Folder to store the cleaned data.
   2. **model_performance** - Folder to store the model performance metrics.
   3. **predictions** - Folder to store the model predictions.
   4. **train_test_data** - Folder to store data after feature generation, one-hot-encoding, and train/test split creation. 
4. **deliverables** - Folder for project deliverables
   1. `Application Demo - MSiA 423.pdf` - The slides for the application demonstration.    
6. **dockerfiles** - Folder to hold dockerfiles
   1. `Dockerfile` - The dockerfile for the project to run individual steps of the pipeline
   2. `Dockerfile.pipeline` - The dockerfile used to run the entire model pipeline
   3. `Dockerfile.app` - The Dockerfile used for running the web app
   4. `Dockerfile.test` - The dockerfile for running the unit tests
7. **models** - Folder to store trained model objects
8. **src** - Folder to hold any source python modules
   1. `app_module.py` - module for running app functions such as prediction
   2. `clean_data.py` - module containing the functions for performing data cleaning operations
   3. `create_tables_rds` - module containing database table class and functions for creating, updating, and querying tables
   4. `data_preprocessing.py` - module containing the functions for performing feature transformations and train/test split
   5. `evaluate_model.py` - module for performing final model evaluation and computing final model metrics
   6. `orchestrate.py` - module containing functions that organize and execute the required functions to complete any given step of the pipeline
   7. `predict.py` - module containing functions that make predictions using the model
   8. `preprocess_app_input.py` - module containing functions for validating and processing input into the web app before using it to make a prediction.
   9. `read_write_functions.py` - module containing functions for reading and writing data to/from files
   10. `read_write_s3` - module containing functions for reading and writing to S3 bucket
   11. `remove_outliers` - module containing functions for removing outliers
   12. `train_model.py` - module containing functions for training a model object
   13. `validate.py` - module containing functions for performing data validation in the pipeline.
9. **tests**
   1. `test_clean_data.py` - Unit tests for functions in clean_data.py
   2. `test_data_preprocessing.py` - Unit tests for functions in data_preprocessing.py
   3. `test_evaluate_model.py` - Unit tests for functions in evaluate_model.py
   4. `test_predict.py` - Unit tests for functions in predict.py
   5. `test_preprocess_app_input.py` - Unit tests for functions in preprocess_app_input.py
   6. `test_remove_outliers.py` - Unit tests for functions in remove_outliers.py
   7. `test_train_model.py` - Unit tests for functions in train_model.py
   8. `test_validate.py` - Unit tests for functions in validate.py
10. `app.py` - Executable script that runs the web application
11. `README.md` - Repository README
12. `requirements.txt` - Environment requirements file
13. `run.py` - Executable script for running steps in the model pipeline
14. `run-pipeline.sh` - The bash script for executing all steps in the pipeline at once.

## Running the code

### run.py instructions

The main executable script, `run.py` can take the following options as the first commandline argument: `create_db`, `fetch`, `clean`, `generate_features`, `train_model`, `predict`, and `evaluate`.

The allowable flags for each of these options are discussed in the steps below.


### 1. Initialize the database 
#### Build the image 

To build the image, run this image from the root of the repository: 

```bash
docker build -f dockerfiles/Dockerfile -t final-project .
```

#### Create the database 
To create the database, using the configurations in `flaskconfig.py` and `database_config.py`, run: 

```bash
docker run -e SQLALCHEMY_DATABASE_URI --mount type=bind,source="$(pwd)",target=/app/ final-project create_db  --engine_string=${SQLALCHEMY_DATABASE_URI}
```

This command uses the `create_db` option. Additional allowable flags include:

`--engine_string`: The SQLALCHEMY_DATABASE_URI connection string.

### 2. Acquiring the Raw Data

#### Build the image

Build the same docker image from the previous step and run it from the root of the repository: 

```bash
docker build -f dockerfiles/Dockerfile -t final-project .
```

Run the following command to acquire the raw data and save it to S3:
```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY  --mount type=bind,source="$(pwd)",target=/app/data/ final-project fetch --path_s3=s3://2022-msia423-hathaway-richard/raw_data/metro_interstate_traffic_volume.csv --data_url=https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz
```

This command uses the `fetch` option as the first command-line argument. Additional allowable flags include:

`--path_s3`: S3 Path of the data to save the raw data to.

`--data_url`: The url of the source raw data.

### 3. Run each individual step of the model pipeline

#### Build the image

Build the same docker image from the previous step and run it from the root of the repository: 

```bash
docker build -f dockerfiles/Dockerfile -t final-project .
```

#### Read Data from S3 and Clean Data

To read the raw data from S3 and clean the data, run:
```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY  --mount type=bind,source="$(pwd)",target=/app/ final-project clean --config_path=./config/model_config.yaml --input_source=s3://2022-msia423-hathaway-richard/raw_data/metro_interstate_traffic_volume.csv --output_path=./data/clean_data/cleaned_data.csv 
```
This command uses the `clean` option as the first command-line argument. Additional allowable flags include:

`--config_path`: Location of the pipeline configuration YAML file.

`--input_source`: Location of the input raw data file on S3.

`--output_source`: Location to save the cleaned data output csv file.

#### Generate Features

To generate features for model training, including splitting the data into train and test files, run:
```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ final-project generate_features --config_path=./config/model_config.yaml --input_source=./data/clean_data/cleaned_data.csv --one_hot_path=./models/ohe_object.joblib --train_output_source=./data/train_test_data/train_data.csv --test_output_source=./data/train_test_data/test_data.csv
```

This command uses the `generate_features` option as the first command-line argument. Additional allowable flags include:

`--config_path`: Location of the pipeline configuration YAML file.

`--input_source`: Location of the cleaned data csv file.

`--one_hot_path`: Location to save the fitted one-hot-encoder object.

`--train_output_source`: Location to save the train data csv file.

`--test_output_source`: Location to save the test data csv file.


#### Train Model

To train the model, run:
```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ final-project train_model --config_path=./config/model_config.yaml --train_input_source=./data/train_test_data/train_data.csv --model_output_source=./models/trained_model_object.joblib
```

This command uses the `train_model` option as the first command-line argument. Additional allowable flags include:

`--config_path`: Location of the pipeline configuration YAML file.

`--train_input_source`: Location of the train data csv file.

`--model_output_source`: Location to save the fitted model object.

#### Make Predictions

To make predictions using the trained model, run:
```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ final-project predict --config_path=./config/model_config.yaml --test_input_source=./data/train_test_data/test_data.csv --model_input_source=./models/trained_model_object.joblib --predictions_output_source=./data/predictions/predictions.csv
```

This command uses the `predict` option as the first command-line argument. Additional allowable flags include:

`--config_path`: Location of the pipeline configuration YAML file.

`--model_input_source`: Location of the trained model object.

`--test_input_source`: Location of the the input test data csv file.

`--predictions_output_source`: Location to save the model predictions csv file.

#### Evaluate the Model

To make evaluate the model performance and generate metrics such as R^2 and Mean-Squared-Error, run:
```bash
docker run --mount type=bind,source="$(pwd)",target=/app/ final-project evaluate --config_path=./config/model_config.yaml --test_input_source=./data/train_test_data/test_data.csv --predictions_input_source=./data/predictions/predictions.csv --performance_metrics_output_source=./data/model_performance/performance_metrics.txt
```

This command uses the `evaluate` option as the first command-line argument. Additional allowable flags include:

`--config_path`: Location of the pipeline configuration YAML file.

`--test_input_source`: Location of the the input test data csv file.

`--predictions_input_source`: Location of the input predictions csv file.

`--performance_metrics_output_source`: Location to save the performance metrics test file.

### 4. Run the entire pipeline
To run the entire pipeline, from reading the data from S3 through model evaluation, first build the image in the root of the repository: 

```bash
docker build -f dockerfiles/Dockerfile.pipeline -t final-project-pipeline .
```

Then run the following command:
```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY --mount type=bind,source="$(pwd)",target=/app/ final-project-pipeline run-pipeline.sh
```

### 5. Run the Flask app 

#### Build the image 

Build the image by running this command from the root of the repository: 

```bash
docker build -f dockerfiles/Dockerfile.app -t final-project-app .
```

#### Run the Flask Application

To run the Flask application, run: 

```bash
 docker run -e SQLALCHEMY_DATABASE_URI --mount type=bind,source="$(pwd)",target=/app/ -p 5000:5000 final-project-app
```
You should be able to access the app at http://127.0.0.1:5000/ in your browser.

### 6. Unit Tests

To run the unit tests, first build the image using the following command:

```bash
docker build -f dockerfiles/Dockerfile.test -t final-project-tests .
```

To run the tests, run: 

```bash
docker run final-project-tests
```
