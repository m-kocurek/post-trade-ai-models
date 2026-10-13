"""
Settlement Anomaly Detection on real SEC Fails-to-Deliver data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import urllib.request
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')


FILE = "cnsfails202501b"
DATA_DIR = "data"
FILE_PATH = os.path.join(DATA_DIR, FILE)

def read_ftd():
    df = pd.read_csv(FILE_PATH, sep='|', dtype=str)
    return df

def preprocess(df):
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        'SETTLEMENT DATE': 'settlement_date',
        'CUSIP': 'cusip',
        'SYMBOL': 'symbol',
        'QUANTITY (FAILS)': 'fails_quantity',
        'DESCRIPTION': 'description',
        'PRICE': 'price'
    })
    
    df['fails_quantity'] = pd.to_numeric(df['fails_quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['fails_quantity', 'price'])
    df = df[df['fails_quantity'] > 0].copy()
    
    # Notional value of fails
    df['fails_value'] = df['fails_quantity'] * df['price']
    
    # Log features (heavy right tail)
    df['log_fails_qty'] = np.log1p(df['fails_quantity'])
    df['log_fails_value'] = np.log1p(df['fails_value'])
    df['log_price'] = np.log1p(df['price'])
    
    return df.reset_index(drop=True)

def main():
    raw = read_ftd()
    df = preprocess(raw)
    
    print(f"Records after cleaning: {len(df):,}")
    print(f"Unique symbols: {df['symbol'].nunique():,}")
    print(f"Date range: {df['settlement_date'].min()} – {df['settlement_date'].max()}\n")
    
    features = ['log_fails_qty', 'log_fails_value', 'log_price']
    X = df[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Expected anomaly rate ~1% (extreme fails)
    contamination = 0.01
    
    models = {
        "Isolation Forest": IsolationForest(
            n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1
        ),
        "One-Class SVM": OneClassSVM(
            nu=contamination, kernel='rbf', gamma='scale'
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=35, contamination=contamination, novelty=False
        ),
        "Elliptic Envelope": EllipticEnvelope(
            contamination=contamination, random_state=42
        ),
    }
    
    print("=" * 90)
    print(f"{'Model':<25} {'# Anomalies':>12} {'Median fails_qty':>18} {'Median fails_value':>20}")
    print("=" * 90)
    
    results = {}
    
    for name, model in models.items():
        if name == "Local Outlier Factor":
            y_pred_raw = model.fit_predict(X_scaled)
            scores = -model.negative_outlier_factor_
        else:
            model.fit(X_scaled)
            y_pred_raw = model.predict(X_scaled)
            scores = -model.decision_function(X_scaled) if hasattr(model, 'decision_function') else np.zeros(len(df))
        
        is_anomaly = (y_pred_raw == -1)
        n_anom = is_anomaly.sum()
        
        med_qty = df.loc[is_anomaly, 'fails_quantity'].median() if n_anom > 0 else np.nan
        med_val = df.loc[is_anomaly, 'fails_value'].median() if n_anom > 0 else np.nan
        
        results[name] = {
            'mask': is_anomaly,
            'scores': scores,
            'n': n_anom
        }
        
        print(f"{name:<25} {n_anom:12,} {med_qty:18,.0f} {med_val:20,.0f}")
    
    print("=" * 90)
    
    # Show top anomalies from Isolation Forest (usually the most stable)
    print("\n=== Top 15 Anomalies (Isolation Forest) ===")
    iso_mask = results["Isolation Forest"]['mask']
    iso_scores = results["Isolation Forest"]['scores']
    
    top_idx = np.argsort(iso_scores)[-15:][::-1]
    top = df.iloc[top_idx][[
        'settlement_date', 'symbol', 'description',
        'fails_quantity', 'price', 'fails_value'
    ]].copy()
    top['anomaly_score'] = iso_scores[top_idx]
    print(top.round(2).to_string(index=False))
    
    print("\nNote: No ground-truth labels available in public FTD data.")
    print("Anomalies are securities with unusually high fails quantity / value.")
    print("Change FTD_URL to analyze a different period.")

if __name__ == "__main__":
    main()
