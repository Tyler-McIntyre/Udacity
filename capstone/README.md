# Capstone Project: Classification of Astronomical Objects

## Project Overview
In this project, we demonstrate how machine learning models can be employed to classify astronomical objects, specifically Stars, Galaxies, and Quasars, using data from the Sloan Digital Sky Survey-V (SDSS-V). Our goal is to identify and classify these celestial bodies based on their spectral characteristics, facilitating deeper analysis and understanding of the universe.

## Problem Statement
Classifying interstellar bodies within extensive datasets is crucial for advancing our knowledge of the universe. By analyzing spectral data, we can identify and categorize these objects, uncovering patterns and behaviors that enhance our comprehension of cosmic phenomena.

## Libraries Used
pandas: For data manipulation and analysis.
numpy: For numerical operations.
matplotlib: For data visualization.
scipy: For statistical functions, including skewness and Pearson correlation.
imblearn: For handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
sklearn: For machine learning algorithms, including model selection and evaluation.

## Repository Files
- <u>star_classification.csv</u>: Contains the dataset used for classification.
- <u>stellar_classification.ipynb</u>: Jupyter notebook for exploratory data analysis, model training, and evaluation.
- <u>stellar_classification.pdf</u>: Folder with visualizations, analysis and results of model performance.

## Analysis and Results
The project involved training and evaluating K-Nearest Neighbors (KNN) and Support Vector Classification (SVC) models. Despite initial expectations, KNN did not perform as well as SVC. The smaller size and structure of the dataset made KNN's complexity a disadvantage compared to the simpler SVC algorithm. Optimized hyperparameters for SVC, such as linear kernel, balanced class weights, and scaling, led to better performance.

## Conclusion
SVC proved to be more effective than KNN for this classification problem. The dataset's size and characteristics favored the SVC model, highlighting its suitability for smaller datasets compared to KNN. Future work should involve using additional real-world data to validate these findings, as synthetic data may not accurately reflect real-world scenarios.

## Reflection & Improvement
Initially, it was expected that KNN would perform similarly to or better than SVC. However, the results indicated that KNN's performance was hindered by the dataset's limited size and structure. To make a more definitive comparison, incorporating real-world data is essential. Synthetic data augmentation alone may not provide a comprehensive assessment of model performance, especially in the presence of new outliers.

## Acknowledgements
Thank you to the SDSS for providing the data used in this project.