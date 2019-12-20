import sys

import joblib
import nltk
import pandas as pd
from sqlalchemy import create_engine
import string

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
STOP_WORDS = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
PUNCTUATION_TABLE = str.maketrans('', '', string.punctuation)

def load_data(database_filepath):
    """Loads X and Y and gets category names
    Args:
        database_filepath (str): string filepath of the sqlite database

    Returns:
        X (pandas dataframe): Feature data, just the messages
        Y (pandas dataframe): Classification labels
        category_names (list): List of the category names for classification
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('labeled_messages', engine)
    engine.dispose()

    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    """Basic tokenizer that removes punctuation and stopwords then lemmatizes
    Args:
        text (string): input message to tokenize

    Returns:
        tokens (list): list of cleaned tokens in the message
    """
    # normalize case and remove punctuation
    text = text.translate(PUNCTUATION_TABLE).lower()

    # tokenize text
    tokens = nltk.word_tokenize(text)

    # lemmatize and remove stop words
    return [lemmatizer.lemmatize(word) for word in tokens
                                                    if word not in STOP_WORDS]


def build_model():
    """Returns the GridSearchCV object to be used as the model
    Args:
        None

    Returns:
        cv (scikit-learn GridSearchCV): Grid search model object
    """

    clf = RandomForestClassifier(n_estimators=100)

    # The pipeline has tfidf, dimensionality reduction, and classifier
    pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('best', TruncatedSVD(n_components=100)),
                    ('clf', MultiOutputClassifier(clf))
                      ])

    # The param grid still takes ~2h to search even at this modest size
    param_grid = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': [0.8, 1.0],
        'tfidf__max_features': [None, 10000],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }

    Initialize a gridsearch object that is parallelized
    cv = GridSearchCV(pipeline, param_grid, cv=3, verbose=10), n_jobs=-1)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Prints multi-output classification results
    Args:
        model (pandas dataframe): the scikit-learn fitted model
        X_text (pandas dataframe): The X test set
        Y_test (pandas dataframe): the Y test classifications
        category_names (list): the category names

    Returns:
        None
    """

    # Generate predictions
    Y_pred = model.predict(X_test)

    # Print out the full classification report
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to

    Returns:
        None
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
