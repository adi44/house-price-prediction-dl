'''
This script will be used to generate a syntehtic dataset for training and testing the model.
Here are some of the features that will be included in the dataset:
area_sqft: The area of the property in square feet. Continuous 500-5000
bedrooms: The number of bedrooms in the property. Discrete 1-5
bathrooms: The number of bathrooms in the property. Discrete 1-3
floors: The number of floors in the property. Discrete 1-3
age_years: The age of the property in years. Continuous 0-100
garage_spaces: The number of garage spaces in the property. Discrete 0-3
distance_city_km: The distance of the property from the city center in kilometers. Continuous 0-50
school_rating: The rating of the nearest school. Discrete 1-10
crime_index: The crime index of the neighborhood. Continuous 0-100
nearby_metro: Whether the property is near a metro station. Categorical Yes/No
'''

import random


def generate_synthetic_data(num_samples):
    data = []
    for _ in range(num_samples):
        area_sqft = random.uniform(500, 5000)
        bedrooms = random.randint(1, 5)
        bathrooms = random.randint(1, 3)
        floors = random.randint(1, 3)
        age_years = random.uniform(0, 100)
        garage_spaces = random.randint(0, 3)
        distance_city_km = random.uniform(0, 50)
        school_rating = random.randint(1, 10)
        crime_index = random.uniform(0, 100)
        nearby_metro = random.choice(['Yes', 'No'])

        data.append({
            'area_sqft': area_sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'floors': floors,
            'age_years': age_years,
            'garage_spaces': garage_spaces,
            'distance_city_km': distance_city_km,
            'school_rating': school_rating,
            'crime_index': crime_index,
            'nearby_metro': nearby_metro
        })
    return data


def calculate_price(sample):
    # Base price calculation based on area and number of bedrooms
    base_price = sample['area_sqft'] * 200 + sample['bedrooms'] * 10000
    
    # Adjust price based on other features
    base_price += sample['bathrooms'] * 5000
    base_price += sample['floors'] * 3000
    base_price -= sample['age_years'] * 1000
    base_price += sample['garage_spaces'] * 2000
    base_price -= sample['distance_city_km'] * 500
    base_price += sample['school_rating'] * 1000
    base_price -= sample['crime_index'] * 200
    
    # Adjust price based on proximity to metro station
    if sample['nearby_metro'] == 'Yes':
        base_price += 5000
    
    return max(base_price, 0)  # Ensure price is not negative


def add_noise(price, noise_level=0.1):
    noise = random.uniform(-noise_level, noise_level) * price
    return price + noise


# Add non linear relationships and interactions between features
def calculate_price_with_interactions(sample):
    price = calculate_price(sample)
    
    # Interaction between area and number of bedrooms
    price += (sample['area_sqft'] / 1000) * sample['bedrooms'] * 500
    
    # Interaction between school rating and crime index
    price += (sample['school_rating'] - sample['crime_index'] / 10) * 1000
    
    return max(price, 0)  # Ensure price is not negative

# lets generate a synthetic dataset with 10000 samples
if __name__ == "__main__":
    num_samples = 10000
    synthetic_data = generate_synthetic_data(num_samples)
    
    # Calculate price for each sample and add noise
    for sample in synthetic_data:
        price = calculate_price_with_interactions(sample)
        noisy_price = add_noise(price)

        sample['price'] = noisy_price
    
    # Save the dataset to a CSV file
    import pandas as pd
    df = pd.DataFrame(synthetic_data)
    df.to_csv('synthetic_real_estate_data.csv', index=False)
    print("Synthetic dataset generated and saved to 'synthetic_real_estate_data.csv'")

