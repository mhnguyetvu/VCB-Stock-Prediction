import torch
import numpy as np
from src.data_utils import load_data
from model.stock_lstm import StockLSTM

TRAIN_PATH = r'C:\Users\nguyetnvm\Documents\VCB-Stock-Prediction\vcb_2009_2018.csv'
TEST_PATH = r'C:\Users\nguyetnvm\Documents\VCB-Stock-Prediction\vcb_2019.csv'

X_train, y_train, sc, dataset_test, real_stock_price, inputs, dataset_train, dataset_total = load_data(TRAIN_PATH, TEST_PATH)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

model = StockLSTM()
model.load_state_dict(torch.load("mymodel.pt", map_location=torch.device('cpu')))
model.eval()

# Test prediction shape
X_test = []
no_of_sample = len(inputs)
for i in range(60, no_of_sample):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
with torch.no_grad():
    predicted_stock_price = model(X_test).cpu().numpy()
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

assert predicted_stock_price.shape[0] == real_stock_price.shape[0], "Prediction and real price length mismatch"
print("Test passed: Prediction shape matches real stock price shape.")
