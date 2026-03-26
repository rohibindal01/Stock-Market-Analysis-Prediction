# ============================================================
# Stock Market Analysis & Price Direction Prediction
# Dataset: S&P 500 Stock Data (Kaggle)
# https://www.kaggle.com/datasets/camnugent/sandp500
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────
TICKER    = "AAPL"          # Change to any ticker present in the CSV
DATA_PATH = "all_stocks_5yr.csv"   # Kaggle file name
RANDOM_STATE = 42
plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


# ──────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ──────────────────────────────────────────────
def load_data(path: str, ticker: str) -> pd.DataFrame:
    """Load CSV, filter to one ticker, parse dates."""
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = df.columns.str.lower().str.strip()

    # Kaggle column may be 'Name' or 'symbol'
    name_col = next((c for c in df.columns if c in ("name", "symbol")), None)
    if name_col is None:
        raise ValueError("Could not find ticker column in CSV.")

    df = df[df[name_col].str.upper() == ticker.upper()].copy()
    if df.empty:
        raise ValueError(f"Ticker '{ticker}' not found. Check the CSV.")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.dropna(inplace=True)
    print(f"✅  Loaded {len(df):,} rows for {ticker}")
    return df


# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators and lag features."""
    d = df.copy()

    # --- Returns & Target ---
    d["daily_return"]   = d["close"].pct_change()
    d["target"]         = (d["daily_return"].shift(-1) > 0).astype(int)  # 1=UP, 0=DOWN

    # --- Moving Averages ---
    for w in [5, 10, 20, 50]:
        d[f"sma_{w}"] = d["close"].rolling(w).mean()

    d["ema_12"] = d["close"].ewm(span=12, adjust=False).mean()
    d["ema_26"] = d["close"].ewm(span=26, adjust=False).mean()

    # --- MACD ---
    d["macd"]        = d["ema_12"] - d["ema_26"]
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    # --- RSI (14-period) ---
    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    mid              = d["close"].rolling(20).mean()
    std20            = d["close"].rolling(20).std()
    d["bb_upper"]    = mid + 2 * std20
    d["bb_lower"]    = mid - 2 * std20
    d["bb_width"]    = (d["bb_upper"] - d["bb_lower"]) / (mid + 1e-9)
    d["bb_position"] = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-9)

    # --- Volatility ---
    d["volatility_10"] = d["daily_return"].rolling(10).std()
    d["volatility_20"] = d["daily_return"].rolling(20).std()

    # --- Volume features ---
    d["volume_sma_10"]   = d["volume"].rolling(10).mean()
    d["volume_ratio"]    = d["volume"] / (d["volume_sma_10"] + 1e-9)
    d["price_volume"]    = d["close"] * d["volume"]

    # --- Lagged returns ---
    for lag in [1, 2, 3, 5]:
        d[f"return_lag_{lag}"] = d["daily_return"].shift(lag)

    # --- Price position vs highs/lows ---
    d["high_low_ratio"]  = (d["close"] - d["low"]) / (d["high"] - d["low"] + 1e-9)
    d["open_close_diff"] = (d["close"] - d["open"]) / (d["open"] + 1e-9)

    # --- Calendar features ---
    d["day_of_week"]  = d["date"].dt.dayofweek
    d["month"]        = d["date"].dt.month
    d["quarter"]      = d["date"].dt.quarter

    d.dropna(inplace=True)
    d.reset_index(drop=True, inplace=True)
    print(f"✅  Engineered {d.shape[1]} columns, {len(d):,} rows after dropping NaNs")
    return d


# ──────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────
def plot_eda(df: pd.DataFrame, ticker: str):
    print("\n📊  Generating EDA plots…")
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f"{ticker} — Exploratory Data Analysis", fontsize=18, fontweight="bold", y=1.01)

    # 3a. Closing price + SMAs
    ax = axes[0, 0]
    ax.plot(df["date"], df["close"], color=PALETTE[0], lw=1.2, label="Close")
    ax.plot(df["date"], df["sma_20"], color=PALETTE[1], lw=1, ls="--", label="SMA 20")
    ax.plot(df["date"], df["sma_50"], color=PALETTE[2], lw=1, ls="--", label="SMA 50")
    ax.fill_between(df["date"], df["bb_lower"], df["bb_upper"], alpha=0.1, color=PALETTE[0], label="BB")
    ax.set_title("Closing Price with SMAs & Bollinger Bands")
    ax.legend(fontsize=8); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 3b. Volume
    ax = axes[0, 1]
    ax.bar(df["date"], df["volume"], color=PALETTE[0], alpha=0.6, width=1)
    ax.plot(df["date"], df["volume_sma_10"], color=PALETTE[1], lw=1.2, label="Vol SMA 10")
    ax.set_title("Trading Volume"); ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 3c. Daily returns distribution
    ax = axes[1, 0]
    sns.histplot(df["daily_return"], bins=60, kde=True, ax=ax, color=PALETTE[0])
    ax.axvline(df["daily_return"].mean(), color=PALETTE[1], ls="--", label="Mean")
    ax.set_title("Daily Return Distribution"); ax.legend(fontsize=8)

    # 3d. RSI
    ax = axes[1, 1]
    ax.plot(df["date"], df["rsi_14"], color=PALETTE[3], lw=1)
    ax.axhline(70, color=PALETTE[1], ls="--", lw=0.8, label="Overbought (70)")
    ax.axhline(30, color=PALETTE[2], ls="--", lw=0.8, label="Oversold (30)")
    ax.set_title("RSI (14-period)"); ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 3e. MACD
    ax = axes[2, 0]
    ax.plot(df["date"], df["macd"], color=PALETTE[0], lw=1, label="MACD")
    ax.plot(df["date"], df["macd_signal"], color=PALETTE[1], lw=1, label="Signal")
    ax.bar(df["date"], df["macd_hist"], color=np.where(df["macd_hist"] >= 0, PALETTE[2], PALETTE[1]),
           alpha=0.4, width=1, label="Histogram")
    ax.set_title("MACD"); ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 3f. Correlation heatmap (top features)
    ax = axes[2, 1]
    feature_cols = ["close", "daily_return", "rsi_14", "macd", "bb_width",
                    "volatility_20", "volume_ratio", "sma_20"]
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   Saved → eda_plots.png")


# ──────────────────────────────────────────────
# 4. MODEL TRAINING
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "rsi_14", "macd", "macd_hist", "bb_width", "bb_position",
    "volatility_10", "volatility_20", "volume_ratio",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "sma_5", "sma_10", "sma_20",
    "high_low_ratio", "open_close_diff",
    "day_of_week", "month", "quarter"
]

def prepare_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df["target"]
    return X, y


def train_models(X_train, y_train):
    """Train three models and return fitted pipelines."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_leaf=10,
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4,
                random_state=RANDOM_STATE
            ))
        ]),
    }
    print("\n🤖  Training models…")
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        print(f"   ✔  {name}")
    return models


# ──────────────────────────────────────────────
# 5. EVALUATION
# ──────────────────────────────────────────────
def evaluate_models(models: dict, X_train, X_test, y_train, y_test):
    print("\n📈  Model Evaluation\n" + "=" * 60)
    results = {}

    fig, axes = plt.subplots(len(models), 3, figsize=(18, 5 * len(models)))
    fig.suptitle("Model Evaluation Results", fontsize=16, fontweight="bold")

    tscv = TimeSeriesSplit(n_splits=5)

    for i, (name, pipe) in enumerate(models.items()):
        y_pred      = pipe.predict(X_test)
        y_proba     = pipe.predict_proba(X_test)[:, 1]
        auc         = roc_auc_score(y_test, y_proba)
        cv_scores   = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring="roc_auc")

        results[name] = {"AUC": auc, "CV_AUC_mean": cv_scores.mean(), "CV_AUC_std": cv_scores.std()}

        print(f"\n🔹 {name}")
        print(f"   Test AUC : {auc:.4f}")
        print(f"   CV  AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))

        # --- Confusion matrix ---
        ax = axes[i, 0]
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["DOWN", "UP"],
            colorbar=False, ax=ax, cmap="Blues"
        )
        ax.set_title(f"{name}\nConfusion Matrix")

        # --- ROC Curve ---
        ax = axes[i, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name}\nROC Curve"); ax.legend()

        # --- Feature importance (only tree models) ---
        ax = axes[i, 2]
        clf = pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            fi = pd.Series(clf.feature_importances_, index=FEATURE_COLS).nlargest(12)
            fi.sort_values().plot(kind="barh", ax=ax, color=PALETTE[i % len(PALETTE)])
            ax.set_title(f"{name}\nTop Feature Importances")
        else:
            coef = pd.Series(np.abs(clf.coef_[0]), index=FEATURE_COLS).nlargest(12)
            coef.sort_values().plot(kind="barh", ax=ax, color=PALETTE[i % len(PALETTE)])
            ax.set_title(f"{name}\n|Coefficients|")

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n   Saved → model_evaluation.png")
    return results


def plot_model_comparison(results: dict):
    names  = list(results.keys())
    aucs   = [results[n]["AUC"] for n in names]
    cv_avg = [results[n]["CV_AUC_mean"] for n in names]
    cv_std = [results[n]["CV_AUC_std"]  for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - 0.2, aucs,   0.35, label="Test AUC",   color=PALETTE[0])
    bars2 = ax.bar(x + 0.2, cv_avg, 0.35, label="CV AUC",     color=PALETTE[2],
                   yerr=cv_std, capsize=5)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("AUC Score"); ax.set_title("Model Comparison: Test vs Cross-Validated AUC")
    ax.legend()
    for b in [*bars1, *bars2]:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   Saved → model_comparison.png")


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Stock Market Direction Prediction Pipeline")
    print("=" * 60)

    # Load & feature engineer
    df = load_data(DATA_PATH, TICKER)
    df = engineer_features(df)

    # EDA
    plot_eda(df, TICKER)

    # Train / test split (time-aware — no shuffle!)
    X, y = prepare_xy(df)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"\n📦  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"   Class balance (train) — UP: {y_train.mean():.1%}  DOWN: {1-y_train.mean():.1%}")

    # Train
    models = train_models(X_train, y_train)

    # Evaluate
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # Compare
    plot_model_comparison(results)

    # Best model
    best = max(results, key=lambda k: results[k]["AUC"])
    print(f"\n🏆  Best model: {best}  (Test AUC = {results[best]['AUC']:.4f})")
    print("\n✅  Pipeline complete. Check the saved PNG files for visualizations.")


if __name__ == "__main__":
    main()
