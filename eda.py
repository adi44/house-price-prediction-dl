"""
we will load the data and perform some basic exploratory 
data analysis (EDA) on it. This will 
include checking for missing values, 
understanding the distribution of the data, 
and visualizing relationships between different features.
"""

import pandas as pd

dataset = pd.read_csv('synthetic_real_estate_data.csv')

class EDA:
    def __init__(self, data):
        self.data = data

    def check_missing_values(self):
        """
        Check for missing values in the dataset.
        """
        missing_values = self.data.isnull().sum()
        return missing_values[missing_values > 0]

    def describe_data(self):
        """
        Get a statistical summary of the dataset.
        """
        return self.data.describe()

    def visualize_distributions(self, output_dir='plots'):
        """
        Visualize the distributions of numerical features.
        """
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(output_dir, exist_ok=True)
        numeric_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        for feature in numeric_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[feature], kde=True)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.savefig(f'{output_dir}/dist_{feature}.png', bbox_inches='tight')
            plt.close()

    def visualize_relationships(self, output_dir='plots'):
        """
        Visualize relationships between different features.
        """
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(output_dir, exist_ok=True)
        pair_plot = sns.pairplot(self.data)
        pair_plot.savefig(f'{output_dir}/pairplot.png', bbox_inches='tight')
        plt.close()


eda = EDA(dataset)
eda.visualize_distributions()
eda.visualize_relationships()
