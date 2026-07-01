'''
lets figure out feature engineering for the synthetic dataset. We can create new features based on existing ones, such as price per square foot, age of the property in decades, and a binary feature indicating if the property is considered "luxury" based on certain thresholds.
check if normalization, one hot encoding or some feature removal and outlier calculations
'''

NUMERIC_FEATURES = ['area_sqft', 'bedrooms', 'bathrooms', 'floors', 'age_years', 'garage_spaces', 'distance_city_km', 'school_rating', 'crime_index', 'price_per_sqft', 'age_decades']

def feature_engineering(data):
    data = data.copy()
    data['price_per_sqft'] = data['price'] / data['area_sqft']
    data['age_decades'] = data['age_years'] / 10
    data['is_luxury'] = ((data['area_sqft'] > 3000) & (data['bedrooms'] >= 4) & (data['school_rating'] >= 8)).astype(int)
    data['nearby_metro'] = (data['nearby_metro'] == 'Yes').astype(int)
    return data


def feature_scaling(train_data, test_data):
    import numpy as np

    train_data = train_data.copy()
    test_data = test_data.copy()

    mean = train_data[NUMERIC_FEATURES].mean()
    std = train_data[NUMERIC_FEATURES].std().replace(0, 1)

    train_data[NUMERIC_FEATURES] = (train_data[NUMERIC_FEATURES] - mean) / std
    test_data[NUMERIC_FEATURES] = (test_data[NUMERIC_FEATURES] - mean) / std

    return train_data, test_data, (mean, std)


# we will split the dataset into training and testing sets, and then apply feature engineering to both sets. We will also check for outliers in the numeric features and remove them if necessary.

def split_and_engineer(data, test_size=0.2):
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(data) * (1 - test_size))
    train_data, test_data = data.iloc[:split_idx], data.iloc[split_idx:]

    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)
    train_data, test_data, scaler = feature_scaling(train_data, test_data)

    y_train = train_data['price']
    y_test = test_data['price']
    X_train = remove_target_variable(train_data)
    X_test = remove_target_variable(test_data)

    return X_train, X_test, y_train, y_test, scaler


# since we have price as the target variable, we will also check for outliers in the price column and remove them if necessary. We can use the IQR method to identify outliers.
def remove_outliers(data, column, multiplier=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return filtered_data


# remove price column from the dataset after feature engineering, as it is the target variable and should not be used as a feature for training the model.
def remove_target_variable(data, target_column='price'):
    return data.drop(columns=[target_column])




