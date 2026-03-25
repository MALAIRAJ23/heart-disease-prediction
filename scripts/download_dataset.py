import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import os

def download_heart_disease_dataset():
    """
    Download the Heart Disease dataset from UCI repository
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # URL for the Heart Disease dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names based on the dataset documentation
    column_names = [
        'age',
        'sex',
        'cp',  # chest pain type
        'trestbps',  # resting blood pressure
        'chol',  # serum cholesterol
        'fbs',  # fasting blood sugar
        'restecg',  # resting electrocardiographic results
        'thalach',  # maximum heart rate achieved
        'exang',  # exercise induced angina
        'oldpeak',  # ST depression induced by exercise
        'slope',  # slope of peak exercise ST segment
        'ca',  # number of major vessels
        'thal',  # thalassemia
        'target'  # diagnosis of heart disease
    ]
    
    try:
        # Download the dataset
        print("Downloading Heart Disease dataset...")
        urllib.request.urlretrieve(url, "data/heart_disease.csv")
        
        # Read the dataset and add column names
        df = pd.read_csv("data/heart_disease.csv", names=column_names, na_values='?')
        
        # Save the dataset with proper column names
        df.to_csv("data/heart_disease.csv", index=False)
        
        print(f"Dataset downloaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Dataset columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        # Create a sample dataset for demonstration purposes
        print("Creating a sample dataset for demonstration...")
        sample_data = create_sample_dataset()
        return sample_data

def create_sample_dataset():
    """
    Create a sample dataset for demonstration purposes
    """
    if not os.path.exists('data'):
        os.makedirs('data')
        
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.round(np.random.uniform(0, 6, n_samples), 1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv("data/heart_disease.csv", index=False)
    print("Sample dataset created successfully!")
    return df

if __name__ == "__main__":
    df = download_heart_disease_dataset()