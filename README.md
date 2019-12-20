# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/disaster_model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Files:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- disaster_model.pkl  # saved model

- notebooks
|- ETL Pipeline Preparation.ipynb # etl exploration
|- ML Pipeline Preparation.ipynb # ML exploration

- README.md
```

### Required packages:

- flask
- joblib
- jupyter # If you want to view the notebooks
- pandas
- plot.ly
- numpy
- scikit-learn
- sqlalchemy


### Data and cleaning details:

The data in this project comes from a modified version of the figure-eight [disaster response data](https://www.figure-eight.com/dataset/combined-disaster-response-data/). In general it was pretty clean, the primary transform steps of the ETL are to a remove a few values that don't match up between the categories/messages data and remove a few bad remaining values.

For example the yes/no values are 1/2 in the original data but 0/1 in the provided data. However, a few 2s remained and are replaced with 0s.


### Some thoughts on the classifier:

The distribution of many classes is extremely skewed with very few 1 values. The classifier in this project was not optimized very well to work in this case - in a real world application I would increase the weight of recall in the score significantly in order to catch more of these low representation classes.
