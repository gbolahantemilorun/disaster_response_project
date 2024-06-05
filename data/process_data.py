import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
    messages_filepath (str): Filepath of the messages dataset.
    categories_filepath (str): Filepath of the categories dataset.

    Returns:
    pandas.DataFrame: Merged DataFrame of messages and categories.
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    """
    Clean the merged DataFrame.

    Args:
    df (pandas.DataFrame): Merged DataFrame of messages and categories.

    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """
    # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Use the first row to extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    
    # Rename the columns of categories
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
    
    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to an SQLite database.

    Args:
    df (pandas.DataFrame): Cleaned DataFrame.
    database_filename (str): Filename for the SQLite database.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    """
    Main function to load, clean, and save data.

    This function checks if the correct number of command-line arguments are provided,
    loads the messages and categories datasets, cleans the data, and saves the cleaned
    data to a database.

    Command-line arguments:
    - messages_filepath (str): Filepath of the messages dataset.
    - categories_filepath (str): Filepath of the categories dataset.
    - database_filepath (str): Filepath of the database to save the cleaned data.

    Returns:
    None
    """
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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
