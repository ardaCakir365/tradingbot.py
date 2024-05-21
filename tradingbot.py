import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

if os.path.exists("sp500.csv"):
    DATA = pd.read_csv("sp500.csv", index_col=0)
else:
    DATA = yf.Ticker("^GSPC").history(period="max")
    DATA.to_csv("sp500.csv")

DATA.index = pd.to_datetime(DATA.index)

plt.plot(DATA.index, DATA["Close"])
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("S&P 500 Index")
plt.show()

DATA.drop(columns=["Dividends", "Stock Splits"], inplace=True)

DATA["Tomorrow"] = DATA["Close"].shift(-1)
DATA["Target"] = (DATA["Tomorrow"] > DATA["Close"]).astype(int)
DATA = DATA.loc["1980-06-01":].copy()

train = DATA.iloc[:-100]
test = DATA.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
model.fit(train[predictors], train["Target"])

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
print("Precision Score:", precision_score(test["Target"], preds))

# Combine actual and predicted values
combined = pd.concat([test["Target"], preds], axis=1)
combined.columns = ['Actual', 'Predicted']

combined.plot(kind='line', color=['green', 'blue'])
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Actual vs Predicted")
plt.show()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=3000, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(DATA, model, predictors)
print(predictions["Predictions"].value_counts())
print("Overall Precision Score:", precision_score(predictions["Target"], predictions["Predictions"]))
print(predictions["Target"].value_counts() / predictions.shape[0])

list_of_ma = [3, 5, 30, 90, 250]
new_predictors = []

for ma in list_of_ma:
    rolling_averages = DATA.rolling(ma).mean()
    ratio_column = f"Close_Ratio_{ma}"
    DATA[ratio_column] = DATA["Close"] / rolling_averages["Close"]
    trend_column = f"Trend_{ma}"
    DATA[trend_column] = DATA.shift(1).rolling(ma).sum()["Target"]
    new_predictors += [ratio_column, trend_column]

DATA = DATA.dropna(subset=DATA.columns[DATA.columns != "Tomorrow"])

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

predictions = backtest(DATA, model, new_predictors)
print(predictions["Predictions"].value_counts())
print("Final Precision Score:", precision_score(predictions["Target"], predictions["Predictions"]))
print(predictions["Target"].value_counts() / predictions.shape[0])

print(predictions)
