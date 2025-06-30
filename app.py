from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

def prepare_data(stock_symbol):
    df = yf.download(stock_symbol, start="2020-01-01", end="2023-12-31")
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['Daily Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily Return'].rolling(window=10).std()
    df.dropna(inplace=True)
    df['Target'] = df['Close'].shift(-1)
    features = ['Close', 'MA50', 'MA200', 'Daily Return', 'Volatility']
    X = df[features][:-1]
    return X.tail(1), df

@app.route('/', methods=['GET', 'POST'])
def index():
    price = None
    stock_symbol = 'AAPL'
    if request.method == 'POST':
        stock_symbol = request.form['symbol']
        X_latest, df = prepare_data(stock_symbol)
        price = model.predict(X_latest)[0]
    return render_template('index.html', price=price, symbol=stock_symbol)

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

