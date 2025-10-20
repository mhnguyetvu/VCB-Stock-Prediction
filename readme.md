# VCB Stock Price Prediction

This project predicts VCB stock prices using an LSTM model built with PyTorch. It includes a web interface for uploading test CSV files and visualizing predictions.

## Project Structure

- `src/` — Main logic and data utilities
  - `main.py` — Main script for training and prediction
  - `data_utils.py` — Data loading and preprocessing
- `model/` — Model definition
  - `stock_lstm.py` — PyTorch LSTM model
- `test/` — Test scripts
  - `test_stock_predict.py` — Basic integration test
- `web/` — Flask web app
  - `app.py` — Web server for file upload and chart display
  - `templates/upload.html` — Upload form
- `requirements.txt` — Python dependencies

## Setup

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train and Predict (Script)
Run the main script to train the model and predict stock prices:
```bash
python src/main.py
```

### Web App
Start the Flask web server:
```bash
python web/app.py
```
Then open your browser and go to `http://127.0.0.1:5000/` to upload a test CSV file and view the prediction chart.

## Notes
- The model expects training and test CSV files in the project directory (see `main.py` and `web/app.py` for paths).
- The trained model is saved as `mymodel.pt`.
- The web app returns a PNG chart comparing real and predicted prices.

## Requirements
See `requirements.txt` for all dependencies.

## License
MIT License
