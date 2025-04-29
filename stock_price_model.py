import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.title("S&P 500 Closing Price Model")


@st.cache_data
def load_data():
    csv = "sp500_stocks.csv"
    df = pd.read_csv(csv)
    df.dropna(inplace=True)
    return df

df = load_data()

symbols = sorted(df["Symbol"].unique())
currentSymbol = st.selectbox("Choose a stock symbol", symbols)

history_days = st.slider("Select number of historical days to use for prediction:", min_value=1, max_value=250, value=5)

dfSymbol = df[df['Symbol'] == currentSymbol].copy()
dfSymbol.sort_values(by=['Date'], inplace=True)

dfSymbol['Next_Close'] = dfSymbol['Close'].shift(-1)
dfSymbol.dropna(inplace=True)

dfSymbol['Prev_Close'] = dfSymbol['Close'].shift(1)
dfSymbol['Change'] = dfSymbol['Close'] - dfSymbol['Prev_Close']
dfSymbol['averageClose'] = dfSymbol['Close'].rolling(window=history_days).mean()

dfSymbol.dropna(inplace=True)

st.subheader("Historical Closing Prices")
st.line_chart(dfSymbol.set_index("Date")["Close"])

dfRecent = dfSymbol.tail(100)
st.subheader("Close vs Next Closing Price")
fig, ax= plt.subplots()
ax.scatter(dfRecent['Close'], dfRecent['Next_Close'])
ax.set_xlabel("Close")
ax.set_ylabel("Next_Close (Predicted Target)")
ax.set_title("Close vs Next Day Close")
st.pyplot(fig)

st.subheader("Change Distribution")
fig, axis = plt.subplots()
sns.histplot(dfSymbol['Change'])
axis.set_title("Distribution of Daily Price Changes")
axis.set_xlabel("Change ($)")
axis.set_ylabel("Frequency")
st.pyplot(fig)

features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Change', 'averageClose']
x = dfSymbol[features]
y = dfSymbol['Next_Close']

stockModel = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
stockModel.fit(X_train, y_train)
prediction = stockModel.predict(X_test)
score = r2_score(y_test, prediction)
mse = mean_squared_error(y_test, prediction)

st.write(f"R² Score: {score}")
st.write(f"Mean Square Error: {mse}")

stockModel.fit(x, y)

latest_data = dfSymbol[features].tail(1)
predicted_next_close = stockModel.predict(latest_data)[0]

st.subheader(f"Prediction for {currentSymbol}")

st.write("Latest available stock data:")
st.dataframe(latest_data)

st.write(f"Predicted next closing price: ${predicted_next_close:.2f}")