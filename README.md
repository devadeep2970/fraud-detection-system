# Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.14-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-100%25_Accuracy-green)
![Flask](https://img.shields.io/badge/Flask-REST_API-red)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow)

## Live Demo
Access the live fraud detection system here:
**https://fraud-detection-system-q850.onrender.com**

## Project Overview
An end-to-end fraud detection system built using machine learning, data analysis, and web deployment. The system analyzes credit card transactions in real-time and predicts whether a transaction is fraudulent or legitimate.

## Dataset
- Source: Kaggle Credit Card Fraud Detection Dataset
- 284,807 transactions over 2 days
- 492 fraud cases (0.17% of all transactions)
- Features: 28 PCA-transformed features + Amount + Time

## Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| PostgreSQL | Database storage |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | ML preprocessing & evaluation |
| XGBoost | Fraud detection model |
| SMOTE | Handling class imbalance |
| Flask | REST API & web interface |
| Power BI | Interactive dashboard |
| Jupyter Notebook | Development environment |
| Render.com | Cloud deployment |
| GitHub | Version control |

## Project Structure
```
fraud-detection-system/
├── app.py                  # Flask API + Frontend
├── requirements.txt        # Python dependencies
├── Procfile               # Render deployment config
├── xgb_fraud_model.pkl    # Trained XGBoost model
├── notebooks/
│   ├── 01_load_data.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_data_prep.ipynb
│   ├── 04_ml_model.ipynb
│   └── 05_deploy.ipynb
```

## Model Performance
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~97% | ~97% |
| Random Forest | ~100% | ~100% |
| XGBoost | **100%** | **100%** |

## Key Findings from EDA
- Only 0.17% of transactions are fraudulent
- Fraud transactions tend to be smaller amounts
- Fraudsters strike at any time (no day/night pattern)
- Features V14, V12, V10 are most correlated with fraud

## How to Run Locally
```bash
# Clone the repository
git clone https://github.com/devadeep2970/fraud-detection-system.git
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Open browser
http://127.0.0.1:5000
```

## API Usage
```python
import requests

transaction = {
    "v1": -1.35, "v2": -0.07, "v3": 2.53, "v4": 1.37,
    "v5": -0.33, "v6": 0.46, "v7": 0.23, "v8": 0.09,
    "v9": 0.36, "v10": 0.09, "v11": -0.18, "v12": 0.17,
    "v13": 0.12, "v14": -0.28, "v15": 0.16, "v16": 0.06,
    "v17": -0.08, "v18": 0.08, "v19": 0.08, "v20": 0.07,
    "v21": -0.01, "v22": 0.27, "v23": -0.11, "v24": 0.06,
    "v25": 0.12, "v26": -0.18, "v27": 0.13, "v28": -0.02,
    "amount": 149.62, "time": 30490.0
}

response = requests.post("https://fraud-detection-system-q850.onrender.com/predict", 
                         json=transaction)
print(response.json())
```

## Author
**Devadeep**
- GitHub: [@devadeep2970](https://github.com/devadeep2970)
- Project built as part of data analytics portfolio

## License
MIT License
