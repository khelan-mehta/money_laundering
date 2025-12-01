from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from collections import defaultdict, deque
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all endpoints and origins

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

class MoneyLaunderingDetector:
    def __init__(self):
        self.autoencoder = pickle.load(open(os.path.join(MODEL_DIR, 'autoencoder.pkl'), 'rb'))
        self.xgb_model = pickle.load(open(os.path.join(MODEL_DIR, 'xgb_model.pkl'), 'rb'))
        self.lstm_model = pickle.load(open(os.path.join(MODEL_DIR, 'lstm_model.pkl'), 'rb'))
        self.scaler = pickle.load(open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb'))
        self.vectorizer = pickle.load(open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb'))
        self.minmax_scaler = pickle.load(open(os.path.join(MODEL_DIR, 'minmax_scaler.pkl'), 'rb'))
        self.encoder = pickle.load(open(os.path.join(MODEL_DIR, 'encoder.pkl'), 'rb'))
        self.best_threshold = pickle.load(open(os.path.join(MODEL_DIR, 'best_threshold.pkl'), 'rb'))
        self.le_dict = pickle.load(open(os.path.join(MODEL_DIR, 'le_dict.pkl'), 'rb'))
        self.feature_order = pickle.load(open(os.path.join(MODEL_DIR, 'feature_order.pkl'), 'rb'))
        self.lstm_feature_order = pickle.load(open(os.path.join(MODEL_DIR, 'lstm_feature_order.pkl'), 'rb'))
        self.sequence_length = 10
        self.cat_cols = ['Payment_currency', 'Received_currency', 'Sender_bank_location',
                         'Receiver_bank_location', 'Payment_type', 'Laundering_type']
        self.text_columns = ["Sender_bank_location", "Receiver_bank_location", "Payment_type", "Laundering_type"]
        self.numerical_columns = ["Amount", "anomaly"]
        self.categorical_columns = ["Payment_currency", "Received_currency"]
        self.user_sequences = defaultdict(lambda: deque(maxlen=self.sequence_length))

    def preprocess_single_transaction(self, transaction_data):
        df = pd.DataFrame([transaction_data])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['day'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
        df.drop(columns=['Date'], inplace=True)
        df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
        df['hour'] = df['Time'].dt.hour
        df['minute'] = df['Time'].dt.minute
        df['second'] = df['Time'].dt.second
        df.drop(columns=['Time'], inplace=True)
        for col in ['Sender_account', 'Receiver_account']:
            if col in df.columns:
                df[col + '_hash'] = df[col].astype(str).apply(lambda x: hash(x) % 1000)
                df.drop(columns=col, inplace=True)
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).apply(lambda x: self.le_dict[col].transform([x])[0] if x in self.le_dict[col].classes_ else 0)
        df = df.fillna(0)
        if 'Is_laundering' in df.columns:
            df = df.drop(columns=['Is_laundering'])
        df = df.reindex(columns=self.feature_order, fill_value=0)
        return df

    def get_anomaly_score(self, X_processed):
        X_scaled = self.scaler.transform(X_processed)
        X_pred = self.autoencoder.predict(X_scaled, verbose=0)
        recon_error = np.mean(np.square(X_scaled - X_pred), axis=1)
        anomaly_flag = (recon_error > self.best_threshold).astype(int)
        return anomaly_flag[0], recon_error[0]

    def get_xgb_prediction(self, X_processed, anomaly_flag):
        X_with_anomaly = X_processed.copy()
        X_with_anomaly['anomaly'] = anomaly_flag
        X_with_anomaly[self.text_columns] = X_with_anomaly[self.text_columns].fillna("")
        X_with_anomaly[self.categorical_columns] = X_with_anomaly[self.categorical_columns].fillna("Unknown")
        text_data = X_with_anomaly[self.text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        text_features = self.vectorizer.transform(text_data).toarray()
        numerical_features = self.minmax_scaler.transform(X_with_anomaly[self.numerical_columns])
        categorical_features = self.encoder.transform(X_with_anomaly[self.categorical_columns])
        X_combined = np.hstack((text_features, numerical_features, categorical_features))
        prediction_proba = self.xgb_model.predict_proba(X_combined)[0, 1]
        prediction = int(prediction_proba > 0.5)
        return prediction, prediction_proba

    def get_lstm_prediction(self, X_processed, anomaly_flag, xgb_proba, user_id):
        X_lstm = X_processed.copy()
        X_lstm['anomaly'] = anomaly_flag
        X_lstm['xgb_pred'] = xgb_proba
        X_lstm = X_lstm.reindex(columns=self.lstm_feature_order, fill_value=0)
        self.user_sequences[user_id].append(X_lstm.values[0])
        if len(self.user_sequences[user_id]) < self.sequence_length:
            return 0, 0.5
        sequence = np.array(list(self.user_sequences[user_id])).reshape(1, self.sequence_length, len(self.lstm_feature_order))
        lstm_proba = self.lstm_model.predict(sequence, verbose=0)[0, 0]
        lstm_prediction = int(lstm_proba > 0.5)
        return lstm_prediction, float(lstm_proba)

    def predict_single_transaction(self, transaction_data):
        X_processed = self.preprocess_single_transaction(transaction_data)
        user_id = X_processed['Sender_account_hash'].iloc[0] if 'Sender_account_hash' in X_processed else 0
        anomaly_flag, anomaly_score = self.get_anomaly_score(X_processed)
        xgb_prediction, xgb_proba = self.get_xgb_prediction(X_processed, anomaly_flag)
        lstm_prediction, lstm_proba = self.get_lstm_prediction(X_processed, anomaly_flag, xgb_proba, user_id)
        anomaly_proba = 1 / (1 + np.exp(-anomaly_score + self.best_threshold))
        ensemble_proba = (0.3 * anomaly_proba + 0.4 * xgb_proba + 0.3 * lstm_proba)
        ensemble_prediction = int(ensemble_proba > 0.5)
        risk_level = 'CRITICAL' if ensemble_proba >= 0.8 else 'HIGH' if ensemble_proba >= 0.6 else 'MEDIUM' if ensemble_proba >= 0.4 else 'LOW'
        return {
            'timestamp': datetime.now().isoformat(),
            'user_id': int(user_id),
            'predictions': {
                'anomaly_detection': {'prediction': int(anomaly_flag), 'probability': float(anomaly_proba)},
                'xgboost': {'prediction': int(xgb_prediction), 'probability': float(xgb_proba)},
                'lstm': {'prediction': int(lstm_prediction), 'probability': float(lstm_proba)},
                'ensemble': {'prediction': int(ensemble_prediction), 'probability': float(ensemble_proba)}
            },
            'risk_level': risk_level
        }

detector = MoneyLaunderingDetector()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        transaction_data = request.get_json()
        if not transaction_data:
            return jsonify({'error': 'No transaction data provided'}), 400
        required_fields = ['Date', 'Time', 'Amount', 'Payment_currency', 'Received_currency',
                          'Sender_account', 'Receiver_account', 'Sender_bank_location',
                          'Receiver_bank_location', 'Payment_type']
        missing_fields = [field for field in required_fields if field not in transaction_data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        result = detector.predict_single_transaction(transaction_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)