import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


os.makedirs("outputs", exist_ok=True)

# ===============================
# 1. Load datasets
# ===============================

sentiment = pd.read_csv("data/fear_greed_index.csv")
trades = pd.read_csv("data/historical_data.csv")

print("Sentiment data shape:", sentiment.shape)
print("Trader data shape:", trades.shape)

print("\nSentiment Columns:")
print(sentiment.columns)

print("\nTrader Columns:")
print(trades.columns)


# ===============================
# 2. Data cleaning
# ===============================

sentiment["date"] = pd.to_datetime(sentiment["date"])

trades["Timestamp IST"] = pd.to_datetime(
    trades["Timestamp IST"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

trades["date"] = trades["Timestamp IST"].dt.date
trades["date"] = pd.to_datetime(trades["date"])

sentiment.drop_duplicates(inplace=True)
trades.drop_duplicates(inplace=True)

trades = trades.dropna(subset=["date", "Closed PnL", "Size USD", "Fee"])


# ===============================
# 3. Merge datasets
# ===============================

merged = pd.merge(
    trades,
    sentiment,
    on="date",
    how="inner"
)

print("\nMerged data shape:", merged.shape)

merged.to_csv("outputs/merged_trader_sentiment_data.csv", index=False)


# ===============================
# 4. Basic summary
# ===============================

summary = merged.groupby("classification").agg(
    total_trades=("Trade ID", "count"),
    avg_closed_pnl=("Closed PnL", "mean"),
    total_closed_pnl=("Closed PnL", "sum"),
    avg_trade_size_usd=("Size USD", "mean"),
    total_trade_size_usd=("Size USD", "sum"),
    avg_fee=("Fee", "mean")
).reset_index()

summary.to_csv("outputs/sentiment_summary.csv", index=False)

print("\nSentiment Summary:")
print(summary)


# ===============================
# 5. Win rate
# ===============================

merged["Profitable"] = (merged["Closed PnL"] > 0).astype(int)

win_rate = merged.groupby("classification")["Profitable"].mean().reset_index()
win_rate["Win Rate %"] = win_rate["Profitable"] * 100

win_rate.to_csv("outputs/win_rate_by_sentiment.csv", index=False)

print("\nWin Rate:")
print(win_rate)


# ===============================
# 6. Charts
# ===============================

sns.set_theme(style="whitegrid")

# Chart 1: Average Profit by Sentiment
plt.figure(figsize=(8, 5))
sns.barplot(data=merged, x="classification", y="Closed PnL")
plt.title("Average Profit by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Closed PnL")
plt.tight_layout()
plt.savefig("outputs/avg_profit_by_sentiment.png")
plt.close()


# Chart 2: Total Profit by Sentiment
total_profit = merged.groupby("classification")["Closed PnL"].sum().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=total_profit, x="classification", y="Closed PnL")
plt.title("Total Profit by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Total Closed PnL")
plt.tight_layout()
plt.savefig("outputs/total_profit_by_sentiment.png")
plt.close()


# Chart 3: Average Trade Size by Sentiment
plt.figure(figsize=(8, 5))
sns.barplot(data=merged, x="classification", y="Size USD")
plt.title("Average Trade Size by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Size USD")
plt.tight_layout()
plt.savefig("outputs/avg_trade_size_by_sentiment.png")
plt.close()


# Chart 4: Buy vs Sell Performance
plt.figure(figsize=(8, 5))
sns.barplot(data=merged, x="Side", y="Closed PnL")
plt.title("BUY vs SELL Average Performance")
plt.xlabel("Trade Side")
plt.ylabel("Average Closed PnL")
plt.tight_layout()
plt.savefig("outputs/buy_sell_performance.png")
plt.close()


# Chart 5: Top 10 Profitable Coins
top_coins = (
    merged.groupby("Coin")["Closed PnL"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
top_coins.plot(kind="bar")
plt.title("Top 10 Profitable Coins")
plt.xlabel("Coin")
plt.ylabel("Total Closed PnL")
plt.tight_layout()
plt.savefig("outputs/top_10_profitable_coins.png")
plt.close()


# Chart 6: Trade Count by Sentiment
trade_count = merged["classification"].value_counts().reset_index()
trade_count.columns = ["classification", "count"]

plt.figure(figsize=(8, 5))
sns.barplot(data=trade_count, x="classification", y="count")
plt.title("Number of Trades by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Trade Count")
plt.tight_layout()
plt.savefig("outputs/trade_count_by_sentiment.png")
plt.close()


# Chart 7: Fee Analysis
plt.figure(figsize=(8, 5))
sns.barplot(data=merged, x="classification", y="Fee")
plt.title("Average Fee by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Fee")
plt.tight_layout()
plt.savefig("outputs/fee_by_sentiment.png")
plt.close()


# Chart 8: Win Rate by Sentiment
plt.figure(figsize=(8, 5))
sns.barplot(data=win_rate, x="classification", y="Win Rate %")
plt.title("Win Rate by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Win Rate (%)")
plt.tight_layout()
plt.savefig("outputs/win_rate_by_sentiment.png")
plt.close()


# Chart 9: Risk Analysis using Start Position
plt.figure(figsize=(8, 5))
sns.barplot(data=merged, x="classification", y="Start Position")
plt.title("Average Start Position by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Start Position")
plt.tight_layout()
plt.savefig("outputs/start_position_by_sentiment.png")
plt.close()


# ===============================
# 7. Hidden pattern analysis
# ===============================

side_sentiment = merged.groupby(
    ["classification", "Side"]
)["Closed PnL"].mean().reset_index()

side_sentiment.to_csv("outputs/side_sentiment_analysis.csv", index=False)

coin_sentiment = merged.groupby(
    ["classification", "Coin"]
)["Closed PnL"].sum().reset_index()

coin_sentiment.to_csv("outputs/coin_sentiment_analysis.csv", index=False)


# ===============================
# 8. Bonus ML Model
# ===============================

ml_data = merged[
    [
        "Execution Price",
        "Size Tokens",
        "Size USD",
        "Side",
        "Coin",
        "Fee",
        "classification",
        "Profitable"
    ]
].copy()

ml_data.dropna(inplace=True)

label_cols = ["Side", "Coin", "classification"]

for col in label_cols:
    le = LabelEncoder()
    ml_data[col] = le.fit_transform(ml_data[col].astype(str))

X = ml_data.drop("Profitable", axis=1)
y = ml_data["Profitable"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

with open("outputs/ml_model_results.txt", "w") as f:
    f.write("Bonus ML Model: Random Forest Classifier\n")
    f.write("Target: Predict whether a trade is profitable or not\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("\nML Model Accuracy:", accuracy)
print(report)


# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance.to_csv("outputs/feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x="Importance", y="Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()


# ===============================
# 9. Final insights file
# ===============================

with open("outputs/final_insights.txt", "w") as f:
    f.write("Final Insights\n")
    f.write("====================\n\n")
    f.write("1. The project analyzes the relationship between Bitcoin market sentiment and trader performance.\n")
    f.write("2. Trader data was merged with Fear & Greed Index data using date.\n")
    f.write("3. Closed PnL was used as the main performance metric.\n")
    f.write("4. Size USD, Fee, Side, Coin, and Start Position were used to study trading behavior.\n")
    f.write("5. Win rate was calculated to identify profitable trading conditions.\n")
    f.write("6. Charts were generated to compare profit, volume, fee, and trading activity across sentiments.\n")
    f.write("7. A Random Forest model was added as a bonus to predict whether a trade is profitable.\n")
    f.write("8. Feature importance shows which variables influence profitability the most.\n")

print("\nAnalysis completed successfully.")
print("Check the outputs folder for charts, CSV files, and ML results.")