from scapy.all import sniff, IP
import pandas as pd
import joblib

# Load model components
model = joblib.load("nids_xgboost_model.pkl")
scaler = joblib.load("nids_scaler.pkl")
feature_columns = joblib.load("nids_feature_columns.pkl")

def extract_features(packet):
    if IP in packet:
        features = {
            'Flow Duration': len(packet),                     # Simple approximation
            'Total Fwd Packets': 1,
            'Total Backward Packets': 1,
            'Total Length of Fwd Packets': len(packet),
            'Total Length of Bwd Packets': len(packet),
            'Fwd Packet Length Max': len(packet),
            'Bwd Packet Length Max': len(packet),
            # Add more static/default values as needed to match training feature count
        }
        return features
    return None

# ----------------- Real-Time Detection -----------------
def detect_intrusion(packet):
    features = extract_features(packet)
    if features is None:
        return

    try:
        # Fill missing features with zero (or expected default)
        for col in feature_columns:
            if col not in features:
                features[col] = 0


        features_df = pd.DataFrame([features])
        features_scaled = scaler.transform(features_df[feature_columns])
        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            print("🚨 Intrusion Detected! 🚨")
        else:
            print("✅ Normal Traffic")

    except Exception as e:
        print(f"⚠️ Error during prediction: {e}")

# ----------------- Start Live Sniffing -----------------
print("🌐 Starting live network traffic monitoring...")
sniff(filter="ip", prn=detect_intrusion, store=0)