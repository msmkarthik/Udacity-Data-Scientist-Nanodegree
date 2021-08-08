import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
import joblib


def load_data(database_filepath):
    """
    The function loads the table from given database_filepath as a pandas
    dataframe. Splits the message column from the category column.
    Args:
        database_filepath (str): path to the database
    Returns:
        X (pandas_dataframe): Pandas Dataframe containing the messages
        y (pandas_dataframe): Pandas Dataframe the categories
        category_names (list): List containing the column names
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = list(df.columns[4:])
    return X, y, category_names


def tokenize(text):
    """
    Function that tokenizes  the raw text
    Args:
        text (str): raw text
    Returns:
        clean_tokens (list): Text after getting tokenized
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Initializes the pipeline and parameters for hyperparameter tuning.
    GridSearchCV is initialized and the model pipeline is returned

    Returns:
        cv (GridSearchCV instance): Defined GridSearchCV instance with initialised
                                    pipeline and hyperparameter grid
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(RandomForestClassifier()) ) ])
    parameters = {
#         'tfidf__use_idf': (True, False),
        'tfidf__norm':['l2','l1'],
#         'clf__estimator__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        'clf__estimator__min_samples_split': [2, 3, 4],
#         'clf__estimator__min_samples_leaf'   : [1, 2, 4],
        # 'clf__estimator__max_features': ['auto', 'sqrt'],
#         'clf__estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of the model in the test data
    Args:
        model (GridSearchCV instance): Tuned model 
        X_test (pandas_dataframe): Pandas Dataframe containing the messages in  
                                   test data
        y_test (pandas_dataframe): Pandas Dataframe containing the targets(categories)
        category_names (list): List of category names
    """
    y_pred = model.predict(X_test)
    for ix, category in enumerate(Y_test):
        print(category)
        y_pred_col = y_pred[:,ix]
        y_true_col = Y_test[category]
        print(classification_report(y_true_col, y_pred_col))


def save_model(model, model_filepath):
    """
    Saves the model in given path
    Args:
        model (GridSearchCV instance): Tuned model 
        model_filepath (str): path in which the data needs to be dumped 
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