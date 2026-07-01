"""
This will contain the script to apply neural networks to predict house prices based on the features in the dataset. We will use a feedforward neural network with multiple hidden layers and ReLU activation functions. The model will be trained using the Adam optimizer and mean squared error loss function. We will also implement early stopping to prevent overfitting and save the best model based on validation loss.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset

class HousePriceDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)

        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


from torch.utils.data import DataLoader

class HousePriceDataLoader:
    def __init__(self, X_train, y_train, X_test, y_test, batch_size=32):
        self.train_dataset = HousePriceDataset(X_train, y_train)
        self.test_dataset = HousePriceDataset(X_test, y_test)
        self.batch_size = batch_size

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
