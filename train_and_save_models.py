import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras import layers, models
from xgboost import XGBClassifier
import lime
from lime.lime_tabular import LimeTabularExplainer
from pymongo import MongoClient
from collections import defaultdict
from sklearn.metrics import accuracy_score
import time
import threading
import queue
import logging
import os
import pickle
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
MODEL_DIR = 'models'
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MoneyLaunderingTrainer:
    def __init__(self):
        self.autoencoder = None
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=150)
        self.minmax_scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        self.le_dict = {col: LabelEncoder() for col in ['Payment_currency', 'Received_currency', 'Sender_bank_location',
                                                        'Receiver_bank_location', 'Payment_type', 'Laundering_type']}
        self.best_threshold = None
        self.sequence_length = 10
        self.cat_cols = ['Payment_currency', 'Received_currency', 'Sender_bank_location',
                         'Receiver_bank_location', 'Payment_type', 'Laundering_type']
        self.text_columns = ["Sender_bank_location", "Receiver_bank_location", "Payment_type", "Laundering_type"]
        self.numerical_columns = ["Amount"] # anomaly is added later
        self.categorical_columns = ["Payment_currency", "Received_currency"]
        self.feature_order = None
        self.lstm_feature_order = None

    def preprocess_data(self, df, for_autoencoder=True):
        df = df.copy()
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
            df[col + '_hash'] = df[col].astype(str).apply(lambda x: hash(x) % 1000)
            df.drop(columns=col, inplace=True)

        for col in self.cat_cols:
            df[col] = self.le_dict[col].fit_transform(df[col].astype(str))

        df = df.fillna(0)

        if for_autoencoder:
            X = df.drop(columns=['Is_laundering'])
            # IMPORTANT: Save the feature order WITHOUT Is_laundering and without anomaly initially
            self.feature_order = X.columns.tolist()
        else:
            X = df

        return X

    def train_autoencoder(self, X_scaled, y):
        X_train_all = X_scaled[y == 0]
        X_train, X_val = train_test_split(X_train_all, test_size=0.2, random_state=42)

        input_dim = X_train.shape[1]
        self.autoencoder = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(24, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(12, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(24, activation='relu'),
            layers.Dense(input_dim, activation='linear')
        ])
        self.autoencoder.compile(optimizer='adam', loss='mse')

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.autoencoder.fit(X_train, X_train, epochs=50, batch_size=256,
                            validation_data=(X_val, X_val), callbacks=[early_stop], verbose=1)

        X_pred = self.autoencoder.predict(X_scaled)
        recon_error = np.mean(np.square(X_scaled - X_pred), axis=1)
        fpr, tpr, thresholds = roc_curve(y, recon_error)
        self.best_threshold = thresholds[np.argmax(tpr - fpr)]

    def train_xgboost(self, X_original, y):
        X_original[self.text_columns] = X_original[self.text_columns].fillna("")
        X_original[self.categorical_columns] = X_original[self.categorical_columns].fillna("Unknown")

        text_data = X_original[self.text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        text_features = self.vectorizer.fit_transform(text_data).toarray()
        # Ensure numerical_columns includes 'anomaly' for training
        numerical_features = self.minmax_scaler.fit_transform(X_original[self.numerical_columns])
        noise = np.random.normal(0, 0.02, numerical_features.shape)
        numerical_features += noise
        categorical_features = self.encoder.fit_transform(X_original[self.categorical_columns])

        X = np.hstack((text_features, numerical_features, categorical_features))
        class_counts = y.value_counts()
        target_majority = int(class_counts.min() * 1.5)
        sampling_strategy_under = {class_counts.idxmax(): target_majority, class_counts.idxmin(): class_counts.min()}
        undersample = RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42)
        X_resampled, y_resampled = undersample.fit_resample(X, y)
        sampling_strategy_smote = {y_resampled.value_counts().idxmin(): target_majority}
        smote = SMOTE(sampling_strategy=sampling_strategy_smote, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
        self.xgb_model = XGBClassifier(n_estimators=30, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42, eval_metric='logloss')
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    def train_lstm(self, df_lstm, y_lstm):
        X_lstm_processed = self.preprocess_data(df_lstm.copy(), for_autoencoder=False)
        # Drop Is_laundering for LSTM feature preparation
        X_lstm_for_scaling = X_lstm_processed.drop(columns=['Is_laundering'], errors='ignore')
        X_lstm_scaled = self.scaler.transform(X_lstm_for_scaling)
        X_lstm_for_scaling['anomaly'] = (np.mean(np.square(X_lstm_scaled - self.autoencoder.predict(X_lstm_scaled, verbose=0)), axis=1) > self.best_threshold).astype(int)

        X_lstm_for_scaling[self.text_columns] = X_lstm_for_scaling[self.text_columns].fillna("")
        X_lstm_for_scaling[self.categorical_columns] = X_lstm_for_scaling[self.categorical_columns].fillna("Unknown")

        text_data = X_lstm_for_scaling[self.text_columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        text_features = self.vectorizer.transform(text_data).toarray()

        # Select the numerical columns (including 'anomaly') before scaling
        numerical_features = self.minmax_scaler.transform(X_lstm_for_scaling[self.numerical_columns])
        noise = np.random.normal(0, 0.02, numerical_features.shape)
        numerical_features += noise
        categorical_features = self.encoder.transform(X_lstm_for_scaling[self.categorical_columns])
        X_for_xgb = np.hstack((text_features, numerical_features, categorical_features))
        X_lstm_for_scaling['xgb_pred'] = self.xgb_model.predict_proba(X_for_xgb)[:, 1]

        self.lstm_feature_order = X_lstm_for_scaling.columns.tolist()

        sequences, labels = [], []
        user_transactions = defaultdict(list)

        # Create sequences using only the feature columns (without Is_laundering)
        for idx, row in X_lstm_processed.iterrows():
            row_features = X_lstm_for_scaling.loc[idx]  # This includes anomaly and xgb_pred
            user_transactions[row_features['Sender_account_hash']].append([*row_features.values, y_lstm.iloc[idx]])

        for user, transactions in user_transactions.items():
            # Sort by time
            transactions = sorted(transactions, key=lambda x: x[self.lstm_feature_order.index('year')] * 100000000 +
                                 x[self.lstm_feature_order.index('month')] * 1000000 + x[self.lstm_feature_order.index('day')] * 10000 +
                                 x[self.lstm_feature_order.index('hour')] * 100 + x[self.lstm_feature_order.index('minute')] * 10 +
                                 x[self.lstm_feature_order.index('second')])
            for i in range(0, len(transactions) - self.sequence_length + 1):
                seq = transactions[i:i + self.sequence_length]
                sequences.append([s[:-1] for s in seq])  # Exclude the label
                labels.append(seq[-1][-1])  # Take the label from the last transaction

        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.sequence_length, dtype='float32', padding='pre', truncating='pre')
        labels = np.array(labels)
        X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

        self.lstm_model = models.Sequential([
            layers.LSTM(64, input_shape=(self.sequence_length, len(self.lstm_feature_order)), return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        self.lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.lstm_model.fit(X_seq_train, y_seq_train, epochs=20, batch_size=32, validation_data=(X_seq_test, y_seq_test), verbose=1)

    def save_models(self):
        with open(os.path.join(MODEL_DIR, 'autoencoder.pkl'), 'wb') as f:
            pickle.dump(self.autoencoder, f)
        with open(os.path.join(MODEL_DIR, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(self.xgb_model, f)
        with open(os.path.join(MODEL_DIR, 'lstm_model.pkl'), 'wb') as f:
            pickle.dump(self.lstm_model, f)
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(MODEL_DIR, 'minmax_scaler.pkl'), 'wb') as f:
            pickle.dump(self.minmax_scaler, f)
        with open(os.path.join(MODEL_DIR, 'encoder.pkl'), 'wb') as f:
            pickle.dump(self.encoder, f)
        with open(os.path.join(MODEL_DIR, 'best_threshold.pkl'), 'wb') as f:
            pickle.dump(self.best_threshold, f)
        with open(os.path.join(MODEL_DIR, 'le_dict.pkl'), 'wb') as f:
            pickle.dump(self.le_dict, f)
        with open(os.path.join(MODEL_DIR, 'feature_order.pkl'), 'wb') as f:
            pickle.dump(self.feature_order, f)
        with open(os.path.join(MODEL_DIR, 'lstm_feature_order.pkl'), 'wb') as f:
            pickle.dump(self.lstm_feature_order, f)

        logger.info("Models and preprocessors saved successfully")
        logger.info(f"Feature order saved: {self.feature_order}")
        logger.info(f"LSTM Feature order saved: {self.lstm_feature_order}")


def main():
    # MongoDB connection setup
    mongo_uri = "mongodb+srv://Khelan05:KrxRwjRwkhgYUdwh@cluster0.c6y9phd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(mongo_uri)
    db = client['fd2']  # Replace with your database name
    collection = db['transactions']  # Replace with your collection name

    # Fetch data from MongoDB and convert to DataFrame
    data = list(collection.find())
    df = pd.DataFrame(data)

    # Drop MongoDB's default '_id' column if not needed
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)

    trainer = MoneyLaunderingTrainer()

    # Initialize le_dict before preprocessing
    for col in trainer.cat_cols:
        trainer.le_dict[col].fit(df[col].astype(str))

    # Process data for autoencoder
    df_processed = trainer.preprocess_data(df.copy(), for_autoencoder=True)
    X_original = df_processed
    y = df['Is_laundering']
    X_scaled = trainer.scaler.fit_transform(X_original)

    # Train autoencoder
    trainer.train_autoencoder(X_scaled, y)

    # Add anomaly column for XGBoost training
    X_original['anomaly'] = (np.mean(np.square(X_scaled - trainer.autoencoder.predict(X_scaled, verbose=0)), axis=1) > trainer.best_threshold).astype(int)
    trainer.numerical_columns.append('anomaly')

    # Train XGBoost
    trainer.train_xgboost(X_original.copy(), y)

    # Train LSTM
    trainer.train_lstm(df.copy(), y)

    # Save all models
    trainer.save_models()

    # Close MongoDB connection
    client.close()

if __name__ == "__main__":
    main()