import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_utils import load_data
from model.stock_lstm import StockLSTM
import pandas as pd
import os.path
from os import path

TRAIN_PATH = r'C:\Users\nguyetnvm\Documents\VCB-Stock-Prediction\vcb_2009_2018.csv'
TEST_PATH = r'C:\Users\nguyetnvm\Documents\VCB-Stock-Prediction\vcb_2019.csv'

X_train, y_train, sc, dataset_test, real_stock_price, inputs, dataset_train, dataset_total = load_data(TRAIN_PATH, TEST_PATH)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = StockLSTM()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model_path = "C:\\Users\\nguyetnvm\\Documents\\VCB-Stock-Prediction\\model\\mymodel.pt"
if path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    model.train()
    for epoch in range(100):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.6f}")
    torch.save(model.state_dict(), model_path)

# Prediction
X_test = []
no_of_sample = len(inputs)
for i in range(60, no_of_sample):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    predicted_stock_price = model(X_test).cpu().numpy()
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Predict next days until 30/10
dataset_test_seq = dataset_test['CLOSE'][len(dataset_test)-60:len(dataset_test)].to_numpy()
dataset_test_seq = np.array(dataset_test_seq)
inputs = dataset_test_seq.reshape(-1, 1)
inputs = sc.transform(inputs)

for i in range(28):
    X_pred = inputs[-60:, 0]
    X_pred = torch.tensor(X_pred, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred_price = model(X_pred).cpu().numpy()
    pred_price = sc.inverse_transform(pred_price)
    dataset_test_seq = np.append(dataset_test_seq, pred_price[0, 0])
    inputs = dataset_test_seq.reshape(-1, 1)
    inputs = sc.transform(inputs)
    print(f'Stock price {i+3}/10/2019 of VCB : {pred_price[0, 0]:.2f}')
