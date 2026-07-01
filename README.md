# House Price Prediction with Deep Learning

A feedforward neural network built with PyTorch to predict house prices from a synthetic real estate dataset.

## Project Structure

```
├── data_generation_script.py   # Generates the synthetic dataset
├── eda.py                      # Exploratory data analysis and visualizations
├── feature_engineering.py      # Feature creation, encoding, scaling, and train/test split
├── load_dataset.py             # Dataset loading and preprocessing pipeline
├── house_prediction.py         # PyTorch Dataset and DataLoader wrappers
├── neural_network.py           # Model architecture and training loop
├── synthetic_real_estate_data.csv
└── requirements.txt
```

## Features

**Input features (13 total):**
- `area_sqft`, `bedrooms`, `bathrooms`, `floors`, `age_years`, `garage_spaces`
- `distance_city_km`, `school_rating`, `crime_index`
- `nearby_metro` — binary (0/1)
- `price_per_sqft`, `age_decades` — engineered
- `is_luxury` — binary flag (area > 3000 sqft, bedrooms ≥ 4, school rating ≥ 8)

**Target:** `price`

## Model Architecture

```
Input (13) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)
```

- Optimizer: Adam (lr=0.001)
- Loss: MSELoss
- Batch size: 32

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Generate dataset:**
```bash
python data_generation_script.py
```

**Run EDA (saves plots to `plots/`):**
```bash
python eda.py
```

**Train the model:**
```bash
python neural_network.py
```

## Results

| Metric | Value |
|--------|-------|
| Epochs | 100 |
| Test RMSE | ~$6,406 |
