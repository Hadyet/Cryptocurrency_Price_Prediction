import streamlit as st
import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import yfinance as yf
from sklearn.model_selection import train_test_split
import datetime as dt
from plotly import graph_objects as go

#Creating title
st.title('*********Solligence********')
st.title('Cryptocurrency Prediction App')
image = Image.open('pr.jpg')
st.image(image, width=700)


coins = ('BTC','ETH','ADA', 'BNB','USDT','DOT','XRP','UNI','LTC', 'LINK',
           'BCH','USDC', 'XLM','THETA','XTZ','DOGE','LUNA', 'FIL','VET','ATOM')

selected_coins = st.selectbox('Select coin for prediction', coins)


period = ('1day','7days', '30days', 'Quarter')
selected_periods = st.selectbox('Select period for prediction', period)


#catch the data
#Prediction for individual coins
#define function to collect data and returns data for selected coin
@st.cache
def retrieve(abb):
    start = dt.datetime(2021, 3, 15)
    end = dt.datetime.now()
    df = yf.download(f'{abb}-USD', start=start, end=end, interval='1d')
    df.reset_index(inplace=True)
    return df

data_load_update = st.text('loading data...')
data_c = retrieve(selected_coins)
data_load_update.text("Data loading ....completed!")

st.subheader('Fetched data (Prices in USD)')
st.write(data_c.head())

#plotting original data from yfinance
def plot_collected():
    global selected_coins
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_c['Date'], y=data_c['High'], name="Coin_High"))
    fig.add_trace(go.Scatter(x=data_c['Date'], y=data_c['Close'], name="Coin_Close"))
    fig.layout.update(title_text=f'Historical Data of {selected_coins}', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_collected()


#function to collect close column and create new target variables
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)  #60-40
    loaded_model=joblib.load('k.sav')
    load = loaded_model.fit(X_train, y_train)
    y_get= load.predict(X_test)
    return y_get

#function to display predicted value specified by user
def display(array_orig, period):
    array_data = list(array_orig)
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


predict_update = st.text('Predicting...')
create = create(data_c)
predict = create_predict(create)
period_c = selected_periods
display1 = display(predict, period_c)
predict_update.text("Predicting ....done!")


st.subheader(f'Predicted price for {period_c}')
st.write(display1)



