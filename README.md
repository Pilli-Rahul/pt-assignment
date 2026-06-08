# Bitcoin Market Sentiment vs Trader Performance Analysis

## Objective
Analyze the relationship between Bitcoin market sentiment (Fear/Greed) and trader performance using Hyperliquid trading data.

## Datasets
- Bitcoin Fear & Greed Index
- Hyperliquid Historical Trader Data

## Methodology
1. Data Cleaning
2. Date Alignment and Merging
3. Exploratory Data Analysis (EDA)
4. Visualization
5. Sentiment-Based Performance Analysis
6. Bonus Machine Learning Model

## Analysis Performed
- Average Profit by Sentiment
- Total Profit by Sentiment
- Average Trade Size by Sentiment
- Buy vs Sell Performance
- Top 10 Profitable Coins
- Fee Analysis
- Win Rate Analysis
- Trade Count Analysis

## Bonus ML Model
Random Forest Classifier used to predict whether a trade is profitable.

Features:
- Execution Price
- Size Tokens
- Size USD
- Side
- Coin
- Fee
- Market Sentiment

Target:
- Profitable (Closed PnL > 0)

## Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure
data/
outputs/
analysis.py
requirements.txt
README.md

## Conclusion
The project explores how market sentiment influences trader profitability, trading volume, and trading behavior. The results provide insights that can be used to design sentiment-aware trading strategies.