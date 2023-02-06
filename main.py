import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.model_selection import train_test_split
import datetime as dt

finance = ['BTC','ETH','ADA', 'BNB','USDT','DOT','XRP','UNI','LTC', 'LINK',
           'BCH','USDC', 'XLM','THETA','XTZ','DOGE','LUNA', 'FIL','VET','ATOM']
names = ['Bitcoin','Ethereum','Cardano','Binance Coin', 'Tether','Polkadot','Ripple', 'Uniswap','Litecoin','Chainlink',
         'Bitcoin Cash', 'USD Coin', 'Stellar', 'Theta','Tezos','Dogecoin','Terra','Filecoin','VeChain','Cosmos']

# ordinal encoding-----mapping desired order with a dictionary
coins_dict = {'BTC': 1, 'ETH': 2, 'ADA': 3, 'BNB': 4, 'USDT': 5, 'DOT': 6, 'XRP': 7, 'UNI': 8, 'LTC': 9, 'LINK': 10,
           'BCH': 11, 'USDC': 12, 'XLM': 13, 'THETA': 14, 'XTZ': 15, 'DOGE': 16, 'LUNA': 17, 'FIL': 18, 'VET': 19, 'ATOM': 20}


#Prediction for individual coins
#define function to collect data and returns data for b†c only
def retrieve(abb):
    global finance
    for coin in finance:
        mold_2= pd.DataFrame()
        start = dt.datetime(2020, 3, 15)
        end = dt.datetime.now()
        index = finance.index(coin)
        df = yf.Ticker(f'{coin}-USD').history(start=start, end=end, interval='1d')
        df = pd.DataFrame(df)
        df['names'] = names[index]
        df['Symbol'] = coin
        mold = mold_2.append(df)
        mold=mold.reset_index()
        df_3= mold.drop(['Dividends','Stock Splits','Volume',"names"], axis=1)
        df_4= df_3[df_3['Symbol'] == abb].copy()  # collecting data for bitcoin only

        return df_4


#function to encode symbol column and create new target variables
def create(df):
    df = df.drop(["Open","High","Low","Date"], axis=1) # picking only close
    df['predict_1'] = df['Close'].shift(-1)  #Creating new columns for prediction
    df['predict_7d'] = df['Close'].shift(-7)
    df['predict_30'] = df['Close'].shift(-30)
    df['predict_90'] = df['Close'].shift(-90)
    dfw = df.dropna().copy()  #drop not available data after shifting
    return dfw



#Convert df to a numpy array and drop prediction
#Separating independent from dependent variables
#makes prediction with trained model
def create_predict(df_f):
    prediction_days = 90
    x = np.array(df_f['Close']).reshape(-1, 1)  #.values.reshape(-1,1))
    X = x[:len(x)-prediction_days]-1   #Remove last n rows from x data
    Y= df_f.drop(['Close'],  axis=1)  # create dependent data
    y=Y[:len(Y)-prediction_days]-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)  #60-40
    loaded_model=joblib.load('k.sav')
    load = loaded_model.fit(X_train, y_train)
    y_get= load.predict(X_test)
    return y_get

#function to display predicted value specified by user
def display(array_orig, period):
    array_data= list(array_orig)
    # for arr in array_data:
    if period == '1day':
        value = array_data[0][0]
    elif period == '7days':
        value = array_data[0][1]
    elif period == '30days':
        value= array_data[0][2]
    elif period == 'Quarter':
        value = array_data[0][3]
    else:
        value = "Check your input"
    return value


# coin = 'BTC'
# # duration = 7
# crypto_df = retrieve(coin)
# # print(crypto_df)
#
# encoded_df = encode(crypto_df)
# # print(encoded_df)
# predict_df = create_predict(encoded_df)
# # print(predict_df)
# display=display(predict_df, '1day')
# print(display)
