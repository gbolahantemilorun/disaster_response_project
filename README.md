# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

To run the Python scripts and web app, make sure to install the following libraries:

#### **process_data.py**
- pandas
- sqlalchemy

#### **run.py**
- flask
- plotly
- nltk

#### **train_classifier.py**
- pandas
- numpy
- sqlalchemy
- nltk
- scikit-learn

## Project Overview<a name="overview"></a>

This project focuses on analyzing disaster data to build a model for an API that classifies disaster messages. The dataset contains real messages sent during disaster events. The goal is to create a machine learning pipeline to categorize these events, enabling the messages to be sent to the appropriate disaster relief agency.

The project includes a web app where emergency workers can input new messages and receive classification results in several categories. Additionally, the web app displays visualizations of the data, showcasing the software skills of the developer in creating basic data pipelines and writing clean, organized code.

## File Descriptions<a name="files"></a>

### app folder

- **run.py**: This file contains the main script to run the web app.
- **templates folder**: Contains HTML templates for the web app.
    - **go.html**: Template for displaying classification results.
    - **master.html**: Main template for the web app layout.

### data folder

- **disaster_messages.csv**: CSV file containing disaster messages.
- **disaster_categories.csv**: CSV file containing disaster categories.
- **DisasterResponse.db**: SQLite database file storing cleaned data.
- **process_data.py**: Python script for processing and cleaning data.

### models folder

- **classifier.pkl**: Pickle file containing the trained classifier model.
- **train_classifier.py**: Python script for training the classifier model.

## Instructions - Running the Python Scripts and Web App<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity for providing the project idea and dataset and Appen (formerly Figure 8) for providing the disaster data.