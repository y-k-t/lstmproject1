# models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
import joblib

# Load and preprocess data (assuming 'transactions.csv' contains Date, Amount, Category)
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data.sort_values('Date', inplace=True)
    return data

# Anomaly Detection Model
def train_anomaly_detection_model(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['Amount']])
    model = IsolationForest(contamination=0.05)
    model.fit(X)
    joblib.dump(model, 'anomaly_detection_model.pkl')
    joblib.dump(scaler, 'anomaly_scaler.pkl')

# Clustering Model
def train_clustering_model(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['Amount']])
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    joblib.dump(kmeans, 'clustering_model.pkl')
    joblib.dump(scaler, 'clustering_scaler.pkl')

# Time Series Prediction Model (LSTM)
def train_time_series_model(data):
    # Prepare data for LSTM
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data[['Amount']])
    sequence_data = data.set_index('Date')['Amount'].resample('D').sum().fillna(0)
    
    # Generate time series data for LSTM
    n_input = 30
    n_features = 1
    generator = TimeseriesGenerator(sequence_data.values, sequence_data.values, length=n_input, batch_size=1)
    
    # Define LSTM model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(generator, epochs=10)
    model.save('time_series_model.h5')
    joblib.dump(scaler, 'time_series_scaler.pkl')

# Load and train models
data = load_data('transactions.csv')
train_anomaly_detection_model(data)
train_clustering_model(data)
train_time_series_model(data)
