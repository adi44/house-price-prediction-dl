"""
Load dataset from CSV file and return as a pandas DataFrame.
"""
import numpy as np
import pandas as pd


from feature_engineering import split_and_engineer, remove_outliers



def load_dataset(file_path):

    """
    Load dataset from CSV file and return as a pandas DataFrame.
    """
    data = pd.read_csv(file_path)
    return data



def load_and_preprocess_dataset(file_path, test_size=0.2, remove_outliers_flag=True):
    """
    Load dataset from CSV file, preprocess it, and split into training and testing sets.
    """
    data = load_dataset(file_path)

    if remove_outliers_flag:
        data = remove_outliers(data, 'price')

    X_train, X_test, y_train, y_test, scaler = split_and_engineer(data, test_size=test_size)

    return X_train, X_test, y_train, y_test, scaler



