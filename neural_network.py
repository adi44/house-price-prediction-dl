"""
This module is used for designing a neural network
model for predicting house prices based on various features. 
It includes the definition of a custom dataset class, 
a data loader class, and the neural network architecture itself. 
The model is designed to be trained using PyTorch, with support 
for early stopping and 
saving the best model based on validation loss.
"""

import torch
import torch.nn as nn
from house_prediction import HousePriceDataLoader, HousePriceDataset
from load_dataset import load_and_preprocess_dataset

class HousePricePredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()


        # Hidden layer 1
        self.hidden1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()

        # Hidden layer 2
        self.hidden2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()  


        # Hidden layer 3
        self.hidden3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()

        # Output layer
        self.output = nn.Linear(32, 1)


    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)

        x = self.hidden2(x)
        x = self.relu2(x)

        x = self.hidden3(x)
        x = self.relu3(x)

        x = self.output(x)
        return x
    


X_train, X_test, y_train, y_test, scaler = load_and_preprocess_dataset('synthetic_real_estate_data.csv')

data_loader = HousePriceDataLoader(X_train, y_train, X_test, y_test, batch_size=32)

model = HousePricePredictor(input_size=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(500):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in data_loader.get_train_loader():
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch [{epoch+1}/100], Loss: {epoch_loss / len(data_loader.get_train_loader()):.4f}')

print('Training complete. Evaluating on test set...')

model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in data_loader.get_test_loader():
        test_loss += criterion(model(inputs), targets).item()

rmse = (test_loss / len(data_loader.get_test_loader())) ** 0.5
print(f'Test RMSE: ${rmse:.2f}')
print('Evaluation complete.')

