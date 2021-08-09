import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    The function receives the paths for the message data and categories data
    and loads the dataframe. It then merges the both dataframe on the column 
    'id'
    Args: 
        messages_filepath (str): path to messages.csv
        categories_filepath (str): path to categories.csv
    Returns: 
        df (pandas dataframe): Pandas dataframe of the merged messages and 
                               categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how =  'inner', on = 'id')
    return df


def clean_data(df):
    """
    The function cleans the  by parsing the category column.
    The returned dataframe will have compatible labels for training the 
    machine learning algorithm.
    Args: 
        df (pandas dataframe): Pandas dataframe of the merged messages and 
                               categories data
    Returns: 
        df (pandas dataframe): Cleaned dataframe with separate columns for 
                               categories
    """
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0]
    category_colnames = [i.split('-')[0] for i in row]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1]))
    df = df.drop('categories', axis =1 )
    df = pd.concat([df, categories], axis = 1) 
    df = df.drop_duplicates()
    for cat in categories:
        if  len(set(df[cat].unique()) - {0,1}) > 0:
            indexes = df.loc[df[cat] == 2].index
            df.loc[indexes,'related']  = 1 
    return df


def save_data(df, database_filename):
    """
    Saves the dataframe as a table into a SQL database
    Args: 
        df (pandas dataframe): Pandas dataframe to dumped as SQL table
        database_filename (str): Filename for the database in which the df
                                 is to be dumped
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()