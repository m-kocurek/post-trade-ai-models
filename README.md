# Post-Trade AI Models

AI solutions for banking and post-trade processing.

## Models

1. **Credit Risk Scoring** – Predict probability of default using the public German Credit dataset (UCI).
2. **Settlement Instruction Anomaly Detection** – Unsupervised detection of anomalous settlement instructions using Isolation Forest (synthetic data mimicking real post-trade instructions).

## Project Structure

```
post-trade-ai-models/
├── README.md
├── requirements.txt
├── src/
│   ├── credit_risk_model.py
│   └── settlement_anomaly.py
├── data/
│   └── .gitkeep
└── notebooks/
    └── .gitkeep
```

## Quick Start

```bash
pip install -r requirements.txt

# Credit risk model
python src/credit_risk_model.py

# Settlement anomaly detection
python src/settlement_anomaly.py
```

## Data Sources

- **Credit Risk**: [UCI German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
  - Direct file: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

- **Settlement Anomalies**: Synthetic data generated in code (real instruction-level data is proprietary). Related public data: [SEC Fails-to-Deliver](https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data)

## Business Applications

- Automated credit scoring and risk pricing
- Prioritization of settlement exceptions (similar to Euroclear EasyFocus)
- Reduction of settlement fails and CSDR penalties
- Counterparty and operational risk monitoring

