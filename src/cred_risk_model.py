"""
Credit Risk Scoring Model - Multi-model comparison
Uses the public UCI German Credit dataset.
Compares several classifiers by accuracy, ROC-AUC and other metrics.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import urllib.request
import os
import warnings
warnings.filterwarnings('ignore')

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATA_PATH = "data/german.data"

columns = [
    'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
    'savings', 'employment', 'installment_rate', 'personal_status', 'other_debtors',
    'residence_since', 'property', 'age', 'other_installment_plans', 'housing',
    'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker', 'credit_risk'
]

def load_data():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print("Downloading German Credit dataset...")
        urllib.request.urlretrieve(url, DATA_PATH)
    
    df = pd.read_csv(DATA_PATH, sep=' ', header=None, names=columns)
    df['credit_risk'] = (df['credit_risk'] == 2).astype(int)  # 1 = high risk
    return df

def preprocess(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def main():
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"High-risk rate: {df['credit_risk'].mean():.2%}\n")
    
    df = preprocess(df)
    
    X = df.drop('credit_risk', axis=1)
    y = df['credit_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "SVM (RBF)": SVC(
            probability=True, class_weight='balanced', random_state=42
        ),
        "MLP Neural Net": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        ),
    }
    
    results = []
    
    print("=" * 70)
    print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10} {'F1 (High)':>12}")
    print("=" * 70)
    
    for name, model in models.items():
        # Use scaled data for linear / neural / SVM
        if name in ["Logistic Regression", "SVM (RBF)", "MLP Neural Net"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        
        results.append((name, acc, auc, f1))
        print(f"{name:<25} {acc:10.3f} {auc:10.3f} {f1:12.3f}")
    
    print("=" * 70)
    
    # Best model details
    best = max(results, key=lambda x: x[2])  # by AUC
    print(f"\nBest model by ROC-AUC: {best[0]} (AUC = {best[2]:.3f})")
    
    # Detailed report for the best one
    print(f"\n=== Detailed report for {best[0]} ===")
    best_model = models[best[0]]
    if best[0] in ["Logistic Regression", "SVM (RBF)", "MLP Neural Net"]:
        y_pred_best = best_model.predict(X_test_scaled)
    else:
        y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=['Low Risk', 'High Risk']))

if __name__ == "__main__":
    main()
