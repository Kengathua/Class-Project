import logging
from os import close
from django.shortcuts import render


import pytz
import datetime

# Create your views here.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from django.shortcuts import render, redirect, get_object_or_404

from django.http import HttpResponse, response
from django.core.exceptions import ObjectDoesNotExist
import base64
from io import BytesIO

from rest_framework import viewsets
from .models import Results, Ticker, Stock
from .serializers import TickerSerializer,StockSerializer, ResultsSerializer

LOGGER = logging.getLogger(__name__)

class TickerViewSet(viewsets.ModelViewSet):
    queryset = Ticker.objects.all().order_by('name')
    serializer_class = TickerSerializer


class StockViewSet(viewsets.ModelViewSet):
    queryset = Stock.objects.all().order_by('ticker')
    serializer_class = StockSerializer


class ResultsViewSet(viewsets.ModelViewSet):
    queryset = Results.objects.all().order_by('date')
    serializer_class = ResultsSerializer

def load_df(ticker):
    result=[]
    try:
        df = pd.read_csv('nse_dfs/{}.csv'.format(ticker),
                         index_col='Date', parse_dates=True)
        result = df

    except FileNotFoundError as e:
        msg = "The Ticker {} does not exist".format(ticker)
        msg += "and returned a {} error".format(e)
        LOGGER.error(msg)
    
    return result

def analyze_stock(ticker):
    output=[]
    df = load_df(ticker)
    if type(df) == list:
        msg="No dataframe is present for processing"
        print(msg)
        LOGGER.info(msg)
        return

    else:
        msg="Processing {}".format(ticker)
        print(msg)
        LOGGER.info(msg)
    
    try:
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.01)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
    except KeyError as err:
        msg="Exiting with error{}".format(err)
        return

    scaler = MinMaxScaler(feature_range=(0, 1))

    data_training_array = scaler.fit_transform(data_training)

    # Load model
    model = load_model(
        'trained_models/nse_stock_price_prediction_model_1.h5')

    past_30_days = data_training.tail(30)

    final_df = past_30_days.append(data_testing, ignore_index=True)

    input_data = scaler.fit_transform(final_df)

    X_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        X_test.append(input_data[i-30: i])
        y_test.append(input_data[i, 0])

    # print('y test data',y_test)

    X_test, y_test = np.array(X_test), np.array(y_test)
    y_predicted = model.predict(X_test)

    print(len(y_predicted))
    print(len(y_test))

    scaler = scaler.scale_
    scale_factor = 1/scaler[0]

    y_predicted = y_predicted*scale_factor
    y_test = y_test*scale_factor

    n = 0
    y_predicted = y_predicted[:, n, n]

    values = y_predicted[-20:]
    predicted_data = values.tolist()
    output.append(predicted_data)
    predicted_average = np.mean(values.flatten())
    msg = "The average is", predicted_average
    LOGGER.info(msg)

    current = y_predicted[-21]
    msg = "The current price is", current
    output.append(current)
    LOGGER.info(msg)

    if current > predicted_average:
        if predicted_average <= 0.9*current:
            msg = "STRONG SELL"
            output.append(msg)
            print(msg)
        else:
            msg="WEAK SELL"
            output.append(msg)
            print(msg)

    elif current<predicted_average:
        if predicted_average >= 1.1*current:
            msg="STRONG BUY"
            output.append(msg)
            print(msg)
        else:
            msg="WEAK BUY"
            output.append(msg)
            print(msg)

    else:
        msg="HOLD"
        output.append(msg)
        print(msg)

    return output


def save_stocks_data_to_db():
    tickers = ['ABSA','COOP','EQTY','HFCK','IMH','KCB','NBK','NCBA','SBIC','SCBK']

    for ticker in tickers:
        try:
            stock_object = Ticker.objects.get(name=ticker)
            stock_name = str(stock_object)
            msg = "found {}".format((ticker))
            print(msg)
        
        except Ticker.DoesNotExist:
            msg = "{} ticker not found".format((ticker))
            print(msg)
            model = Ticker(name=ticker)
            model.save()

    for ticker in tickers:
        df = load_df(ticker)
        df.reset_index(inplace=True)
        for index, row in df.iterrows():
            date = row[0]
            start = row[1]
            low = row[2]
            high = row[3]
            close = row[4]
            adj_close = row[5]
            volume = row[6]

            stock_object = Ticker.objects.get(name=ticker)
            stock_name = str(stock_object)
            print(type(stock_name), type(ticker))

            if stock_name == ticker:
                model = Stock(
                    ticker=stock_object, date=date, open=start, high=high,
                    low=low, close=close, adj_close=adj_close, volume=volume
                )
                model.save()
                msg = "Saving {}'s data for {}".format(ticker, date)
                LOGGER.info(msg)
                print(msg)

            else:
                msg = 'The ticker {}'.format(
                    stock_name), 'is not the same as {}'.format(ticker)
                LOGGER.info(msg)
                print(msg)
                

def load_stock(request):
    if request.method == 'POST':
        date = str(datetime.datetime.now(pytz.timezone('Africa/Nairobi')))
        ticker = request.POST.get('ticker')
        msg="Will be analyzing {}".format(ticker)
        results=analyze_stock(ticker)
        current_price=results[1]
        signal=results[2]
        output_data = results[0]

        model = Results(
            date=date, ticker=ticker, output_data=output_data, signal=signal
        )
        model.save()

        print("Current price",current_price)
        print("signal",signal)
        
    return render(request, 'index.html')
