# IMDB-Sentimental-Analysis-With-Random-Forest

This repository contains a Python project for sentimental analysis of IMDB movie reviews. The project uses a Random Forest Classifier to predict sentiment from text data. It demonstrates data loading, cleaning, preprocessing, and modelling using Sklearn's machine-learning library.

## Dataset

The dataset used is the IMDB Dataset of 50K Movie Reviews available on Kaggle 'https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data'. I was unable to add the `IMDB_Dataset.csv` file to GitHub as it was too large so it must be downloaded from Kaggle to use this project.

## Project Structure

- `IMDB_Sentimental_Analysis.ipynb`: Jupyter Notebook file containing the full pipeline of the project including data cleaning, data preprocessing, model training, and evaluation.

## Features

- Data exploration including checking for missing values and duplicates.
- Text data cleaning and preprocessing.
- Conversion of text data into a suitable format for modelling using TF-IDF vectorization.
- Sentiment prediction using a Random Forest Classifier.
- Evaluation of the model's performance.

## Requirements

This project is built using Python and requires the following libraries:
- pandas
- NumPy
- scikit-learn
- re
