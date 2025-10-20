
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, render_template, send_file
import matplotlib.pyplot as plt
import io
import torch
import numpy as np
import pandas as pd
from src.data_utils import load_data
from model.stock_lstm import StockLSTM

app = Flask(__name__)
UPLOAD_FOLDER = 'web/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            test_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(test_path)
            train_path = r'C:\Users\nguyetnvm\Documents\VCB-Stock-Prediction\vcb_2009_2018.csv'
            X_train, y_train, sc, dataset_test, real_stock_price, inputs, dataset_train, dataset_total = load_data(train_path, test_path)
            X_test = []
            no_of_sample = len(inputs)
            for i in range(60, no_of_sample):
                X_test.append(inputs[i-60:i, 0])
            X_test = np.array(X_test)
            X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
            model = StockLSTM()
            model.load_state_dict(torch.load("C:\\Users\\nguyetnvm\\Documents\\VCB-Stock-Prediction\\model\\mymodel.pt", map_location=torch.device('cpu')))
            model.eval()
            with torch.no_grad():
                predicted_stock_price = model(X_test).cpu().numpy()
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            # Plot chart to PNG
            fig, ax = plt.subplots()
            ax.plot(real_stock_price, color='red', label='Real VCB Stock Price')
            ax.plot(predicted_stock_price, color='blue', label='Predicted VCB Stock Price')
            ax.set_title('VCB Stock Price Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('VCB Stock Price')
            ax.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close(fig)
            return send_file(buf, mimetype='image/png')
        else:
            return 'Please upload a CSV file.'
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
