# Traffic Prediction in the Twin Cities
**Richard Hathaway, Spring 2022**

## Project Charter
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














# Template ReadME

# Table of Contents
* [Directory structure ](#Directory-structure)
* [Running the app ](#Running-the-app)
	* [1. Initialize the database ](#1.-Initialize-the-database)
	* [2. Configure Flask app ](#2.-Configure-Flask-app)
	* [3. Run the Flask app ](#3.-Run-the-Flask-app)
* [Testing](#Testing)
* [Mypy](#Mypy)
* [Pylint](#Pylint)



## Directory structure 

```
├── README.md                         <- You are here
├── api
│   ├── static/                       <- CSS, JS files that remain static
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs│    
│
├── config                            <- Directory for configuration files 
│   ├── local/                        <- Directory for keeping environment variables and other local configurations that *do not sync** to Github 
│   ├── logging/                      <- Configuration of python loggers
│   ├── flaskconfig.py                <- Configurations for Flask API 
│
├── data                              <- Folder that contains data used or generated. Only the external/ and sample/ subdirectories are tracked by git. 
│   ├── external/                     <- External data sources, usually reference data,  will be synced with git
│   ├── sample/                       <- Sample data used for code development and testing, will be synced with git
│
├── deliverables/                     <- Any white papers, presentations, final work products that are presented or delivered to a stakeholder 
│
├── docs/                             <- Sphinx documentation based on Python docstrings. Optional for this project.
|
├── dockerfiles/                      <- Directory for all project-related Dockerfiles 
│   ├── Dockerfile.app                <- Dockerfile for building image to run web app
│   ├── Dockerfile.run                <- Dockerfile for building image to execute run.py  
│   ├── Dockerfile.test               <- Dockerfile for building image to run unit tests
│
├── figures/                          <- Generated graphics and figures to be used in reporting, documentation, etc
│
├── models/                           <- Trained model objects (TMOs), model predictions, and/or model summaries
│
├── notebooks/
│   ├── archive/                      <- Develop notebooks no longer being used.
│   ├── deliver/                      <- Notebooks shared with others / in final state
│   ├── develop/                      <- Current notebooks being used in development.
│   ├── template.ipynb                <- Template notebook for analysis with useful imports, helper functions, and SQLAlchemy setup. 
│
├── reference/                        <- Any reference material relevant to the project
│
├── src/                              <- Source data for the project. No executable Python files should live in this folder.  
│
├── test/                             <- Files necessary for running model tests (see documentation below) 
│
├── app.py                            <- Flask wrapper for running the web app 
├── run.py                            <- Simplifies the execution of one or more of the src scripts  
├── requirements.txt                  <- Python package dependencies 
```

## Running the app 

### 1. Initialize the database 
#### Build the image 

To build the image, run from this directory (the root of the repo): 

```bash
docker build -f dockerfiles/Dockerfile -t final-project .
```

#### Write Raw Data to S3
```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY  --mount type=bind,source="$(pwd)"/data,target=/app/data/ final-project fetch -- path_s3=s3://2022-msia423-hathaway-richard/raw_data/metro_interstate_traffic_volume.csv --data_url=https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz
```

#### Read Data from S3 and Clean Data
```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY  --mount type=bind,source="$(pwd)"/data,target=/app/data/ final-project clean --config_path=./config/model_config.yaml --input_source=s3://2022-msia423-hathaway-richard/raw_data/metro_interstate_traffic_volume.csv --output_path=./data/clean_data/cleaned_data.csv 
```

#### Create the database 
To create the database in the location configured in `config.py` run: 

```bash
docker run -e SQLALCHEMY_DATABASE_URI --mount type=bind,source="$(pwd)"/data,target=/app/data/ final-project create_db  --engine_string=${SQLALCHEMY_DATABASE_URI}
```
The `--mount` argument allows the app to access your local `data/` folder and save the SQLite database there so it is available after the Docker container finishes.

#### Run the entire pipeline
To run the entire pipeline, run: 

```bash
docker build -f dockerfiles/Dockerfile.pipeline -t final-project-pipeline .
```

```bash
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY --mount type=bind,source="$(pwd)"/data,target=/app/data/ final-project-pipeline run-pipeline.sh
```


#### Adding songs 
To add songs to the database:

```bash
docker run --mount type=bind,source="$(pwd)"/data,target=/app/data/ pennylanedb ingest --engine_string=sqlite:///data/tracks.db --artist=Emancipator --title="Minor Cause" --album="Dusk to Dawn"
```

#### Defining your engine string 
A SQLAlchemy database connection is defined by a string with the following format:

`dialect+driver://username:password@host:port/database`

The `+dialect` is optional and if not provided, a default is used. For a more detailed description of what `dialect` and `driver` are and how a connection is made, you can see the documentation [here](https://docs.sqlalchemy.org/en/13/core/engines.html). We will cover SQLAlchemy and connection strings in the SQLAlchemy lab session on 
##### Local SQLite database 

A local SQLite database can be created for development and local testing. It does not require a username or password and replaces the host and port with the path to the database file: 

```python
engine_string='sqlite:///data/tracks.db'

```

The three `///` denote that it is a relative path to where the code is being run (which is from the root of this directory).

You can also define the absolute path with four `////`, for example:

```python
engine_string = 'sqlite://///Users/cmawer/Repos/2022-msia423-template-repository/data/tracks.db'
```


### 2. Configure Flask app 

`config/flaskconfig.py` holds the configurations for the Flask app. It includes the following configurations:

```python
DEBUG = True  # Keep True for debugging, change to False when moving to production 
LOGGING_CONFIG = "config/logging/local.conf"  # Path to file that configures Python logger
HOST = "0.0.0.0" # the host that is running the app. 0.0.0.0 when running locally 
PORT = 5000  # What port to expose app on. Must be the same as the port exposed in dockerfiles/Dockerfile.app 
SQLALCHEMY_DATABASE_URI = 'sqlite:///data/tracks.db'  # URI (engine string) for database that contains tracks
APP_NAME = "penny-lane"
SQLALCHEMY_TRACK_MODIFICATIONS = True 
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 100 # Limits the number of rows returned from the database 
```

### 3. Run the Flask app 

#### Build the image 

To build the image, run from this directory (the root of the repo): 

```bash
 docker build -f dockerfiles/Dockerfile.app -t trafficpredictionapp .
```

This command builds the Docker image, with the tag `pennylaneapp`, based on the instructions in `dockerfiles/Dockerfile.app` and the files existing in this directory.

#### Running the app

To run the Flask app, run: 

```bash
 docker run -e SQLALCHEMY_DATABASE_URI --name test-app --mount type=bind,source="$(pwd)"/data,target=/app/data/ -p 5000:5000 trafficpredictionapp
```
You should be able to access the app at http://127.0.0.1:5000/ in your browser (Mac/Linux should also be able to access the app at http://127.0.0.1:5000/ or localhost:5000/) .

The arguments in the above command do the following: 

* The `--name test-app` argument names the container "test". This name can be used to kill the container once finished with it.
* The `--mount` argument allows the app to access your local `data/` folder so it can read from the SQLlite database created in the prior section. 
* The `-p 5000:5000` argument maps your computer's local port 5000 to the Docker container's port 5000 so that you can view the app in your browser. If your port 5000 is already being used for someone, you can use `-p 5001:5000` (or another value in place of 5001) which maps the Docker container's port 5000 to your local port 5001.

Note: If `PORT` in `config/flaskconfig.py` is changed, this port should be changed accordingly (as should the `EXPOSE 5000` line in `dockerfiles/Dockerfile.app`)


#### Kill the container 

Once finished with the app, you will need to kill the container. If you named the container, you can execute the following: 

```bash
docker kill test-app 
```
where `test-app` is the name given in the `docker run` command.

If you did not name the container, you can look up its name by running the following:

```bash 
docker container ls
```

The name will be provided in the right most column. 

## Testing

Run the following:

```bash
 docker build -f dockerfiles/Dockerfile.test -t pennylanetest .
```

To run the tests, run: 

```bash
 docker run pennylanetest
```

The following command will be executed within the container to run the provided unit tests under `test/`:  

```bash
python -m pytest
``` 

## Mypy

Run the following:

```bash
 docker build -f dockerfiles/Dockerfile.mypy -t pennymypy .
```

To run mypy over all files in the repo, run: 

```bash
 docker run pennymypy .
```
To allow for quick iteration, mount your entire repo so changes in Python files are detected:


```bash
 docker run --mount type=bind,source="$(pwd)"/,target=/app/ pennymypy .
```

To run mypy for a single file, run: 

```bash
 docker run pennymypy run.py
```

## Pylint

Run the following:

```bash
 docker build -f dockerfiles/Dockerfile.pylint -t pennylint .
```

To run pylint for a file, run:

```bash
 docker run pennylint run.py 
```

(or any other file name, with its path relative to where you are executing the command from)

To allow for quick iteration, mount your entire repo so changes in Python files are detected:


```bash
 docker run --mount type=bind,source="$(pwd)"/,target=/app/ pennylint run.py
```
