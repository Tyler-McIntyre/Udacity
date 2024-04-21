import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Parameters:
        messages_filepath (str): Filepath to the messages dataset.
        categories_filepath (str): Filepath to the categories dataset.

    Returns:
        DataFrame: A merged DataFrame containing messages and categories.

    Notes:
        This function loads two separate CSV files, one containing messages and the other containing categories,
        and merges them based on the 'id' column. It returns a single DataFrame containing both messages and
        categories, ready for further processing.

    Example:
        >>> df = load_data('messages.csv', 'categories.csv')
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df_merged = pd.merge(messages, categories, on='id')
    
    return df_merged

def clean_data(df):
    """
    Clean the input DataFrame by processing the 'categories' column.

    Parameters:
        df (DataFrame): The input DataFrame containing the 'categories' column.

    Returns:
        DataFrame: A cleaned DataFrame with processed category columns.

    Notes:
        This function takes a DataFrame as input, processes the 'categories' column, and performs the following steps:
        - Splits the 'categories' column into individual category columns.
        - Extracts column names for the categories.
        - Renames the category columns.
        - Extracts category values from each cell.
        - Fills NaN values with 0.
        - Converts column types to integers.
        - Drops duplicates from the DataFrame.
        - Returns the cleaned DataFrame.

    Example:
        >>> df_cleaned = clean_data(df)
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    pattern = r'([^-]+)'
    category_colnames = row.apply(lambda x: re.findall(pattern,x)[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # Split each value by '-' and extract the first part
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

    categories.head()
        
    # Fill NaN values
    categories.fillna(0, inplace=True)
    
    # Convert column types to int
    for col in category_colnames:
        categories[col] = categories[col].astype(int)
        
    # drop the original categories column from `df`
    df.drop(labels='categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    columns_to_concat = categories.columns
    df[columns_to_concat] = categories[columns_to_concat].astype(int)
    
    # drop duplicates
    df_no_duplicates = df.drop_duplicates()

    # Drop rows where the 'related' column has a value of 2
    df_cleaned = df_no_duplicates[df_no_duplicates['related'] != 2].copy()
    
    return df_cleaned


def save_data(df, database_filename):
    """
    Save a DataFrame to a SQLite database.

    Parameters:
    df (DataFrame): The DataFrame to be saved.
    database_filename (str): The filename of the SQLite database.

    Returns:
    None

    Example:
    >>> save_data(df, 'disaster_messages.db')
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')


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