#Install libraries torch, pandas, numpy, sklearn, matplotlib using the command
#pip install  torch,pandas,numpy,sklearn,matplotlib
import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
#Fetch the data from yfinance library
def fetch_bitcoin_data():
    btc = yf.download('BTC-USD', start='2014-01-01', end='2025-03-25', interval='1d')
    btc = btc[['Close']].dropna()
    return btc
#Preprocessing the data and converting the data into torch data structure
def preprocess_data(data, seq_len=30, pred_len=1):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - seq_len - pred_len):
        X.append(scaled_data[i:i + seq_len])
        y.append(scaled_data[i + seq_len:i + seq_len + pred_len])

    X = np.array(X)
    y = np.array(y)

    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return (torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)), scaler
#informer model
class Informer(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_heads=4, seq_len=30, pred_len=365):
        super(Informer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.enc_embedding = nn.Linear(input_dim, d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model * seq_len, pred_len)

    def forward(self, x):
        x = self.enc_embedding(x)
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        x = x.transpose(0, 1).reshape(x.shape[1], -1)
        output = self.decoder(x)
        return output
#Training the model
def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.6f}")

#  Prediction and Comparison
def predict_and_compare(model, X_test, y_test, scaler, btc_data):
    model.eval()
    with torch.no_grad():
        pred = model(X_test)
    pred_prices = scaler.inverse_transform(pred.numpy())
    actual_prices = scaler.inverse_transform(y_test.squeeze(-1).numpy())
    test_dates = btc_data.index[-len(y_test):]
    print("\nPredicted vs Actual Prices (Sample of 5):")
    for i in range(min(5, len(pred_prices))):
        print(f"Date: {test_dates[i].date()}, Predicted: {pred_prices[i][0]:.2f}, Actual: {actual_prices[i][0]:.2f}")
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, actual_prices, label='Actual Price', color='blue')
    plt.plot(test_dates, pred_prices, label='Predicted Price', color='orange', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title('Bitcoin Price Prediction: Actual vs Predicted')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return pred_prices, actual_prices

if __name__ == "__main__":
    btc_data = fetch_bitcoin_data()
    (X_train, y_train, X_test, y_test), scaler = preprocess_data(btc_data)
    model = Informer(input_dim=1, d_model=64, n_heads=4, seq_len=30, pred_len=1)
    train_model(model, X_train, y_train, epochs=50)
    pred_prices, actual_prices = predict_and_compare(model, X_test, y_test, scaler, btc_data)
