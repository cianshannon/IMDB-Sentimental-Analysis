# IMDB-Sentimental-Analysis

# Sentiment Analysis with Random Forest

This repository contains a Python project for sentimental analysis on IMDB movie reviews. The project uses a Random Forest Classifier to predict sentiment from text data. It demonstrates data loading, cleaning, preprocessing, and modelling using sklearn's machine learning library.

## Dataset

The dataset used is the IMDB Dataset of 50K Movie Reviews available on Kaggle 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data'. Ensure that the dataset file `IMDB_Dataset.csv` is available in the project directory or modify the file path in the script accordingly.

## Project Structure

- `sentiment_analysis.py`: Jupyter Notebook file containing the full pipeline of the project including data cleaning, data preprocessing, model training, and evaluation.

## Features

- Data exploration including checking for missing values and duplicates.
- Text data cleaning and preprocessing.
- Conversion of text data into a suitable format for modeling using TF-IDF vectorization.
- Sentiment prediction using a Random Forest Classifier.
- Evaluation of the model's performance.

## Requirements

This project is built using Python and requires the following libraries:
- pandas
- numpy
- scikit-learn
- re
