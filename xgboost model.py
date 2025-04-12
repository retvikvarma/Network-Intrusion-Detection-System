import pandas as pd
import numpy as np
import logging
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib


# ----------------- Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ----------------- Start Pipeline -----------------
logging.info("Starting NIDS model training pipeline...")
print("🚀 Starting NIDS model training pipeline... [0%]")

# ----------------- Load and Combine Datasets -----------------
folder_path = "C:/Users/91905/Downloads/MachineLearningCSV/MachineLearningCVE"
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

dataframes = []
for file in csv_files:
    full_path = os.path.join(folder_path, file)
    try:
        df = pd.read_csv(full_path)
        df['source_file'] = file  # Optional: track origin
        dataframes.append(df)
        print(f"✅ Loaded: {file}")
    except Exception as e:
        print(f"❌ Failed to load {file}: {e}")

df = pd.concat(dataframes, ignore_index=True)
logging.info(f"Combined {len(csv_files)} datasets.")
print("📂 All datasets loaded and combined. [10%]")

# ----------------- Clean Column Names -----------------
df.columns = df.columns.str.strip()
logging.info("Stripped column names.")
print("🧼 Cleaned column names. [20%]")

# ----------------- Convert Labels -----------------
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
logging.info("Converted labels to binary (0: BENIGN, 1: Attack).")
print("✅ Labels converted to binary. [30%]")

# ----------------- Handle NaN / Infinite -----------------
before_drop = len(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
after_drop = len(df)
logging.info(f"Dropped {before_drop - after_drop} rows with NaN or inf values.")
print(f"🧹 Dropped {before_drop - after_drop} invalid rows. [40%]")

# ----------------- Feature and Label Split -----------------
X = df.drop(['Label', 'source_file'], axis=1, errors='ignore')  # Ignore if not present
y = df['Label']
logging.info(f"Features and label separated. Feature shape: {X.shape}, Labels: {y.shape}")
print("📊 Features and labels separated. [50%]")

# ----------------- Standardize Features -----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logging.info("Features standardized using StandardScaler.")
print("⚖️ Features standardized. [60%]")

# ----------------- Train-Test Split -----------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
logging.info("Split data into train and test sets.")
print("📤 Data split into train/test sets. [70%]")

# ----------------- Train XGBoost Model -----------------
model = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    scale_pos_weight=5,  # helps if attack class is smaller
    random_state=42
)
model.fit(X_train, y_train)
logging.info("XGBoost model trained.")
print("🤖 XGBoost Model training complete. [85%]")

# ----------------- Evaluation -----------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
logging.info(f"Model Accuracy: {acc}")
logging.info("Classification Report:\n" + report)
print(f"📈 Accuracy: {acc:.4f}")
print("📋 Classification Report Generated. [100%] ✅")

# Save the model, scaler, and feature columns
joblib.dump(model, "nids_xgboost_model.pkl")
joblib.dump(scaler, "nids_scaler.pkl")
joblib.dump(X.columns.tolist(), "nids_feature_columns.pkl")
print("💾 Model, scaler, and feature columns saved.")