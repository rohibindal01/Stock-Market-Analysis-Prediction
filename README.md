# 📈 Stock Market Direction Prediction

A complete intermediate-level Data Science project that predicts whether a stock's price will go **UP or DOWN** the next day using technical indicators and machine learning.

---

## 🗂️ Dataset

**S&P 500 Stock Data** from Kaggle  
🔗 [https://www.kaggle.com/datasets/camnugent/sandp500](https://www.kaggle.com/datasets/camnugent/sandp500)

Download `all_stocks_5yr.csv` and place it in the project root.

---

## 🔧 Project Structure

```
stock-market-analysis/
│
├── stock_analysis.py       # Main pipeline script
├── requirements.txt        # Dependencies
├── README.md
│
└── outputs/                # Auto-generated after running
    ├── eda_plots.png
    ├── model_evaluation.png
    └── model_comparison.png
```

---

## 🚀 Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/rohibindal01/stock-market-analysis.git
cd stock-market-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place all_stocks_5yr.csv here

# 4. Run the pipeline
python stock_analysis.py
```

To change the stock ticker, edit line in `stock_analysis.py`:
```python
TICKER = "AAPL"   # Change to GOOGL, MSFT, AMZN, etc.
```

---

## 📊 What the Project Does

### 1. Data Loading & Cleaning
- Loads 5 years of daily OHLCV data for any S&P 500 ticker
- Parses dates, handles missing values, filters by ticker

### 2. Feature Engineering (20+ features)
| Category | Features |
|---|---|
| Moving Averages | SMA (5, 10, 20, 50), EMA (12, 26) |
| Momentum | MACD, MACD Signal, MACD Histogram |
| Oscillators | RSI (14-period) |
| Volatility | Bollinger Bands width/position, Rolling std (10, 20) |
| Volume | Volume ratio vs SMA, Price×Volume |
| Lag Features | 1, 2, 3, 5-day return lags |
| Calendar | Day of week, Month, Quarter |

### 3. EDA & Visualizations
- Price chart with SMAs and Bollinger Bands
- Volume bar chart
- Daily return distribution with KDE
- RSI and MACD plots
- Feature correlation heatmap

### 4. ML Models
- Logistic Regression (baseline)
- Random Forest Classifier
- Gradient Boosting Classifier

All models use `sklearn.pipeline.Pipeline` with `StandardScaler`.

### 5. Evaluation
- **Time-series aware** train/test split (80/20, no shuffle)
- **TimeSeriesSplit** cross-validation (5 folds)
- Metrics: ROC-AUC, Precision, Recall, F1
- Confusion matrix, ROC curve, Feature importance plots
- Side-by-side model comparison chart

---

## 📉 Sample Results (AAPL)

| Model | Test AUC | CV AUC |
|---|---|---|
| Logistic Regression | ~0.52 | ~0.51 |
| Random Forest | ~0.57 | ~0.55 |
| Gradient Boosting | ~0.58 | ~0.56 |

> Stock direction prediction is inherently noisy — AUC above 0.55 on daily data is considered meaningful.

---

## 🧠 Key Concepts Demonstrated

- Time-series feature engineering with pandas
- Avoiding data leakage (target is `shift(-1)`, split is chronological)
- Sklearn Pipelines for clean ML workflows
- TimeSeriesSplit for proper temporal cross-validation
- Multi-model comparison with visual evaluation

---

## 📌 Future Improvements

- [ ] Add LSTM / deep learning model
- [ ] Incorporate sentiment analysis (news headlines)
- [ ] Multi-ticker portfolio analysis
- [ ] Backtesting with a simple trading strategy
- [ ] Hyperparameter tuning with Optuna

---

## 📄 License

MIT
