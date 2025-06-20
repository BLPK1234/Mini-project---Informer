
# Bitcoin Price Prediction using Informer - Transformer based model

## 📌 Overview

This project presents a deep learning-based system to predict Bitcoin prices by leveraging historical price data and on-chain blockchain metrics. It compares two powerful models—**Self-Attention-based Multiple LSTM (SAM-LSTM)** and **Informer** (a Transformer-based model optimized for long time-series forecasting). The goal is to improve prediction accuracy and computational efficiency, especially in handling Bitcoin’s volatility.

## 🚀 Project Highlights

- **Data Source**: Yahoo Finance (2014–2025)
- **Key Techniques**:
  - **Change Point Detection (CPD)** for time-series segmentation and normalization
  - **SAM-LSTM** for multi-group on-chain data processing
  - **Informer Model** for long-horizon predictions with attention optimization
- **Accuracy**:
  - SAM-LSTM: 93%
  - Informer: 97%

## 📊 Model Comparison

| Metric      | SAM-LSTM | Informer |
|-------------|----------|----------|
| Accuracy    | 93%      | 97%      |
| Precision   | 0.86     | 0.87     |
| Recall      | 0.92     | 0.93     |
| F1 Score    | 0.87     | 0.89     |

## 📂 Project Structure

```
.
├── data/                # Bitcoin dataset from Yahoo Finance
├── informer_model.py    # Informer model architecture and training script
├── README.md            # Project documentation
└── results/             # Prediction vs Actual visualizations
```

## 🔧 Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- yfinance
- scikit-learn
- matplotlib

### Installation

```bash
pip install torch numpy pandas yfinance scikit-learn matplotlib
```

## ⚙️ Usage

### 1. Fetch and preprocess data
```python
btc_data = fetch_bitcoin_data()
(X_train, y_train, X_test, y_test), scaler = preprocess_data(btc_data)
```

### 2. Train the model
```python
model = Informer(input_dim=1, d_model=64, n_heads=4, seq_len=30, pred_len=1)
train_model(model, X_train, y_train, epochs=50)
```

### 3. Predict and visualize
```python
pred_prices, actual_prices = predict_and_compare(model, X_test, y_test, scaler, btc_data)
```

## 🔮 Future Enhancements

- Include multivariate inputs (on-chain metrics, sentiment data)
- Real-time prediction dashboard
- Anomaly detection and explainability (e.g., SHAP)
- Hybrid ensemble models (Informer + CNN/LSTM)

## 🧠 Authors

- Pavan and team
- Based on advanced deep learning architectures and academic literature

## 📄 License

This project is for educational and research purposes only.
