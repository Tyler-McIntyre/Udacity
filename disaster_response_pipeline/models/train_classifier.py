import sys
import pandas as pd 
import nltk
import string
import re
import numpy as np
import time
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from column_selector import ColumnSelector
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier

# download data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Load data from a SQLite database file.

    Parameters:
    database_filepath (str): Path to the SQLite database file.

    Returns:
    tuple: A tuple containing X (messages), y (binary data), and category_names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    query = "SELECT * FROM messages"
    
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    X = df[['message', 'genre']]
    y = df.drop(columns=['message', 'id', 'original', 'genre'])
    
    return X, y

def tokenize(text):
    """
    Tokenizes and normalizes input text.

    Parameters:
    text (str): Input text to tokenize.

    Returns:
    list: List of normalized tokens.
    """
    
    # Remove conjunctions
    text = text.replace('\'', '')
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation and set to lowercase
    lowercase_tokens = [token.lower() for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    filtered_tokens = remove_stopwords(lowercase_tokens)

    # Lemmatize
    normalized_tokens = lemmatize_tokens(filtered_tokens)
    
    return normalized_tokens

def remove_stopwords(tokens):
    """Removes stopwords from a list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token.lower() not in stop_words]

def lemmatize_tokens(tokens):
    """
    Lemmatizes a list of tokens.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def build_model():
    """
    Build and return the machine learning model pipeline.
    """
    # Create the MLPClassifier with your desired hyperparameters
    clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001,
                        batch_size=32, learning_rate='constant', max_iter=200)

    # Create the MultiOutputClassifier with MLPClassifier
    pipeline = Pipeline([
        ('text_pipeline', Pipeline([
            ('selector', ColumnSelector(key='message')),
            ('vectorizer', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('classifier', MultiOutputClassifier(RandomForestClassifier(n_jobs=1)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test):
    # Predict on the testing data
    y_pred = model.predict(X_test)
    
    num_predictions_per_sample = np.sum(y_pred, axis=1)

    # Convert DataFrame to NumPy array
    Y_test = np.array(Y_test)
    y_pred = np.array(y_pred)

    # Calculate F1 Score for each label separately
    f1_scores = []
    for i in range(Y_test.shape[1]):
        f1 = f1_score(Y_test[:, i], y_pred[:, i], average='weighted')
        f1_scores.append(f1)

    # Calculate average F1 Score across all labels
    avg_f1 = np.mean(f1_scores)

    print("Average F1 Score on test data:", avg_f1)


def save_model(model, model_filepath):
    filename = model_filepath

    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

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