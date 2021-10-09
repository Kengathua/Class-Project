"""
Getting the NYSE fortune 500 companies data
Aims at using this readily available stock data to train the models
"""
from functools import wraps
import logging
import time
import os
import pickle
import requests
import datetime as dt
from collections import Counter


from django.shortcuts import render

import numpy as np
import pandas as pd
import bs4 as bs
import mplfinance as mpf

import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError

from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from rest_framework.response import Response
from rest_framework import status

from .serializers import *
from .models import *
from .forms import StockForm

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def alpha_args_logger(orig_func):
    logging.basicConfig(filename='{}.log'.format(
        orig_func.__name__), level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs)
        )
        return orig_func(*args, **kwargs)
    return wrapper


def alpha_timer(orig_func):
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in: {} sec'. format(orig_func.__name__, t2))

    return wrapper
    
def status(ticker):
    try:
        def save_sp500_tickers():
            resp = requests.get(
                'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            soup = bs.BeautifulSoup(resp.text, "lxml")
            table = soup.find('table', {'class': 'wikitable sortable'})
            tickers = []
            retained_tickers = []

            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text
                tickers.append(ticker)

            if not os.path.exists("sp500tickers.pickle"):
                with open("sp500tickers.pickle", "wb") as f:
                    pickle.dump(tickers, f)
            else:
                pass

            if not os.path.exists('stock_dfs'):
                os.makedirs('stock_dfs')

            for ticker in tickers[:100]:
                retained_tickers.append(ticker.replace("\n", ""))

            return retained_tickers

        save_sp500_tickers()  # Remember to uncomment

        def add_data_to_csv():
            # Get all 100 retained tickers
            retained_tickers = save_sp500_tickers()

            # Will contain a list of the tickers with data
            populated_tickers = []

            start = dt.datetime(2000, 1, 1)
            end = dt.datetime(2021, 9, 30)

            for ticker in retained_tickers:
                """ add ticker data to csv files """
                if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                    try:
                        df = web.DataReader('{}'.format(
                            ticker), 'yahoo', start, end)
                        df.to_csv('stock_dfs/{}.csv'.format(ticker))
                        populated_tickers.append(ticker)
                    except RemoteDataError as error:
                        print("Error, Cannot add data to {}.csv".format(ticker))
                        pass
                    except KeyError as e:
                        print("Skipping {}".format(ticker))
                        pass

                else:
                    print('The file {} exists'.format(ticker))

            if not os.path.exists("sp500populatedtickers.pickle"):
                with open("sp500populatedtickers.pickle", "wb") as f:
                    pickle.dump(populated_tickers, f)

            with open("sp500populatedtickers.pickle", "rb") as f:
                populated_tickers = pickle.load(f)

            # print(populated_tickers)

            return populated_tickers

        add_data_to_csv()

        def populated_tickers():
            with open("sp500populatedtickers.pickle", "rb") as f:
                tickers = pickle.load(f)

            return tickers

        # populated_tickers()

        def compile_data():
            with open("sp500populatedtickers.pickle", "rb") as f:
                tickers = pickle.load(f)

            main_df = pd.DataFrame()
            print("When starting the main df is", main_df)
            all_data = pd.DataFrame()

            for count, ticker in enumerate(tickers):
                try:
                    df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
                    df.set_index('Date', inplace=True)
                except FileNotFoundError:
                    print("FileNotFoundError")

                df.rename(columns={'Adj Close': ticker}, inplace=True)

                df.drop(columns=["Volume", "Close", "Open",
                                 "High", "Low"], axis=1, inplace=True)

                all_data.join(df, how='outer')

                if main_df.empty:
                    main_df = df
                    # print("Main df is", main_df)

                elif not main_df.empty:
                    main_df = main_df.join(df, how='outer')
                    # print("Main df is", main_df)

                else:
                    pass

                if count % 10 == 0:
                    # Monitor progress by printing how far the progress is
                    print(count)

            print("Finally main df head is", main_df.head())
            main_df.to_csv('sp500_joined_closes.csv')

        compile_data()

        def process_data_for_label(ticker):
            hm_days = 7
            df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
            df.fillna(0, inplace=True)

            for i in range(1, hm_days+1):
                df['{}_{}d'.format(ticker, i)] = (
                    df[ticker].shift(-i) - df[ticker]) / df[ticker]

            df.fillna(0, inplace=True)
            print("Processed joined closes data as", df)
            return df

        process_data_for_label(ticker)

        def buy_sell_hold(*args):
            cols = [c for c in args]
            requirement = 0.025

            for col in cols:
                if col > requirement:
                    return 1   # buy
                elif col < -requirement:
                    return-1   # sell
                else:
                    return 0   # hold

        def extract_featuredsets(ticker):
            df = process_data_for_label(ticker)
            tickers = populated_tickers()

            df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                                      df['{}_1d'.format(
                                                          ticker)],
                                                      df['{}_2d'.format(
                                                          ticker)],
                                                      df['{}_3d'.format(
                                                          ticker)],
                                                      df['{}_4d'.format(
                                                          ticker)],
                                                      df['{}_5d'.format(
                                                          ticker)],
                                                      df['{}_6d'.format(
                                                          ticker)],
                                                      df['{}_7d'.format(ticker)]))

            vals = df['{}_target'.format(ticker)].values.tolist()

            str_vals = [str(i) for i in vals]

            print('Data spread', Counter(str_vals))

            df.fillna(0, inplace=True)

            df = df.replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace=True)

            df_vals = df[[ticker for ticker in tickers]].pct_change()
            df_vals = df_vals.replace([np.inf, -np.inf], 0)
            df_vals.fillna(0, inplace=True)

            X = df_vals.values
            y = df['{}_target'.format(ticker)].values

            return X, y, df

        # extract_featuredsets(ticker)

        def check_correlation():
            df = pd.read_csv('sp500_joined_closes.csv')
            # df.plot()
            print(df.index)
            print(type(df))
            print(df.shape)
            df_corr = df.corr()
            # If working with the cell above would produce a correlation table of all the data
            print(pd)
            print("Correlation:", df_corr)

        # check_correlation()

        # Will need to join data of all datasets to have all fields

        def show_graph(ticker):
            # Will by querying different tickers for they high,low,open,close and adjusted close
            print("Working on", ticker)

            intraday = pd.read_csv(
                'sp500_joined_closes.csv', index_col=0, parse_dates=True)
            # Volume is zero anyway for this intraday data set
            intraday.index.name = 'Date'
            intraday.shape
            iday = intraday.loc['2020-01-01':'2020-06-01', :]
            print(type(iday))
            mpf.plot(iday, type='candle', mav=(7, 12, 20), volume=True)

        # show_graph(ticker)

        def do_ml(ticker):
            X, y, df = extract_featuredsets(ticker)

            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, test_size=0.25)

            # clf = neighbors.KNeighborsClassifier()
            clf = VotingClassifier(
                [('lsvc', svm.LinearSVC()), ('knn', neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier())])
            clf.fit(X_train, y_train)

            confidence = clf.score(X_test, y_test)

            predictions = clf.predict(X_test)

            print('Predicted spread', Counter(predictions))

            print("Confidence is:", confidence)

            y_pred = (confidence > 0.80)
            result = "Yes" if y_pred else "No"
            return result

        # do_ml(ticker)

    except AttributeError as e:
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)


def FormView(request):
    if request.method == 'POST':
        form = StockForm(request.POST or None)

        if form.is_valid():
            Ticker = form.cleaned_data['ticker']

            ticker = pd.DataFrame({'ticker': [Ticker]})
            # df["gender"] = 1 if "male" else 2
            result = status(ticker)
            return render(request, 'status.html', {"data": result})

    form = StockForm()
    return render(request, 'form.html', {'form': form})
