# Disaster Response Pipeline Project

Project Summary:

<ins>Data Loading and Preprocessing</ins>:

Data is loaded from a SQLite database containing messages and their corresponding categories. Text preprocessing techniques like tokenization, removal of stopwords, and lemmatization are applied to normalize the text data.

<ins>Model Building</ins>:

The machine learning model pipeline is constructed using scikit-learn. The pipeline includes text processing steps such as vectorization and TF-IDF transformation. The model used is a MultiOutputClassifier with a RandomForestClassifier as the base estimator.

<ins>Model Training and Evaluation</ins>:

The model is trained on the preprocessed data, and GridSearchCV is employed to tune hyperparameters for optimal performance. Evaluation metrics, particularly the F1 score, are used to assess the model's performance on test data.

<ins>Model Deployment</ins>:

The trained model is saved to a pickle file for future use. Additionally, there's a Flask frontend UI where users can input new data for classification into relevant categories.

<ins>Files Structure</ins>:

train_classifier.py: Contains the code for loading, preprocessing, training, evaluating, and saving the model.
column_selector.py: Custom module for selecting specific columns from a DataFrame.
run.py: Flask application to host the frontend UI.
templates/: Folder containing HTML templates for the Flask UI.
data/: Directory where the SQLite database is stored.
models/: Directory where the trained model is saved.

### How to run instructions:
1. Create and activate your venv
```
python -m venv venv
```

Activate on windows
```
venv\Scripts\activate
```

MacOS or Linux
```
source venv/bin/activate
```

2. Install the dependencies from requirements.txt
```
pip install -r requirements.txt
```

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

5. Use your browser to navigate to 
```
http://127.0.0.1:3000
```
