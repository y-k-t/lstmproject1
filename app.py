# app.py
from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from keras.models import load_model
import plotly.express as px
import plotly.graph_objs as go
from io import BytesIOAC
import base64

# Load the trained models
anomaly_model = joblib.load('anomaly_detection_model.pkl')
anomaly_scaler = joblib.load('anomaly_scaler.pkl')
clustering_model = joblib.load('clustering_model.pkl')
clustering_scaler = joblib.load('clustering_scaler.pkl')
time_series_model = load_model('time_series_model.h5')
time_series_scaler = joblib.load('time_series_scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Handle file upload and process
        file = request.files['file']
        data = pd.read_csv(file, parse_dates=['Date'])
        
        # Anomaly detection
        X_anomaly = anomaly_scaler.transform(data[['Amount']])
        data['Anomaly'] = anomaly_model.predict(X_anomaly)

        # Clustering
        X_cluster = clustering_scaler.transform(data[['Amount']])
        data['Cluster'] = clustering_model.predict(X_cluster)

        # Save processed data
        data.to_csv('processed_transactions.csv', index=False)
        return redirect(url_for('dashboard'))

    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    # Load processed data
    data = pd.read_csv('processed_transactions.csv', parse_dates=['Date'])
    
    # Plot transaction history
    fig = px.line(data, x='Date', y='Amount', title='Transaction History')
    graph = fig.to_html(full_html=False)

    # Plot clustering results
    fig_cluster = px.scatter(data, x='Date', y='Amount', color='Cluster', title='Transaction Clusters')
    graph_cluster = fig_cluster.to_html(full_html=False)
    # Pass the anomaly detection data to the template
    anomaly_data = data.to_dict(orient='records')

    return render_template('dashboard.html', graph=graph, graph_cluster=graph_cluster)

if __name__ == '__main__':
    app.run(debug=True)
