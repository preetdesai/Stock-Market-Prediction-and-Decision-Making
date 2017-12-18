from __future__ import division
import warnings
warnings.filterwarnings("ignore")
import math
from matplotlib import pyplot as plt
from ggplot import *
from collections import defaultdict
from math import *
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.finance import candlestick_ohlc
from matplotlib import legend_handler
from pandas import ExcelWriter
from scipy import stats
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import googlefinance.client as gfc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import datetime as dt
import statsmodels.api as sm
import string
import csv
import operator
import itertools
import matplotlib as mpl

n1 = pd.read_csv('NASDAQ.csv', 'rb')
NASDAQ = n1['Symbol'].tolist()

stocklist = str(raw_input('Whcih stock you want to sell? (Enter tickers): '))
stocklist = [word.strip(string.punctuation) for word in stocklist.split()]
print 'What was the purchase amount? (Enter Dollar Amount) : '
purchasePrice = int(input())
print 'How long you would like to wait for (Enter Days): '
Term = int(input())
print 'What is your profit margin? (Enter Percentage) : '
profitMargin = int(input())
U = {}
slopes = {}
diffdict = {}   
payoff = {}

def predictStockPrice(purchasePrice, Term, profitMargin,stockname):
#     f, axarr = plt.subplots(2, 2)  
    print '******************************%s*****************************' %stockname
    predictedPrices = {}
    counter = 0   
    for history in range(3,13,3):
        
        if history == 3:
            df_ohlc = stock
            df_ohlc = df_ohlc.reset_index()
            del df_ohlc['index']
            df_ohlc = df_ohlc.sort_index(ascending = False)
            df_ohlc = df_ohlc.head(62)
            df_ohlc.columns = ["Date","Open","High",'Low',"Close"]
            df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
            new = np.arange(len(df_ohlc)+1)
            new = new[len(df_ohlc)+1:0:-1]
            #print new, len(new)
            df_ohlc['y'] = new
            #print df_ohlc
            #axarr[1,1].plt.subplot2grid((6,1), (0,0), rowspan=6, colspan=1)   
#             axarr[1,1].xaxis_date()
#             axarr[1,1].set_xlabel("Date")   
#             candlestick_ohlc(axarr[1,1],df_ohlc.values,width=1, colorup='g', colordown='r',alpha=0.75)
#             axarr[1,1].set_ylabel("Price")
#             axarr[1,1].set_title('Performance of last %d months of %s' %(history, stockname))
#             axarr[1,1].legend()  
            X = df_ohlc['Date']
            x = df_ohlc['y']
            y = df_ohlc['Close']
            model = sm.OLS(y, x).fit()
            #print model.summary()
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            m, b = np.polyfit(df_ohlc['Date'], df_ohlc['Close'], 1)
#             print 'Slope: ', slope
#             print 'Intercept', intercept
#             axarr[1,1].plot(X, m*X + b, '-')
        if history == 6:
            df_ohlc = stock
            df_ohlc = df_ohlc.reset_index()
            del df_ohlc['index']
            df_ohlc = df_ohlc.sort_index(ascending = False)
            df_ohlc = df_ohlc.head(124)
            df_ohlc.columns = ["Date","Open","High",'Low',"Close"]
            df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
            new = np.arange(len(df_ohlc)+1)
            new = new[len(df_ohlc)+1:0:-1]
            #print new, len(new)
            df_ohlc['y'] = new
#             axarr[1,0].xaxis_date()
#             axarr[1,0].set_xlabel("Date")   
#             candlestick_ohlc(axarr[1,0],df_ohlc.values,width=1, colorup='g', colordown='r',alpha=0.75)
#             axarr[1,0].set_ylabel("Price")
#             axarr[1,0].set_title('Performance of last %d months of %s' %(history, stockname))
#             axarr[1,0].legend()
            X = df_ohlc['Date']  
            x = df_ohlc['y']
            y = df_ohlc['Close']
            model = sm.OLS(y, x).fit()
            #print model.summary()
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            m, b = np.polyfit(df_ohlc['Date'], df_ohlc['Close'], 1)
#             print 'Slope: ', slope
#             print 'Intercept', intercept
#             axarr[1,0].plot(X, m*X + b, '-')
        if history == 9:
            df_ohlc = stock
            df_ohlc = df_ohlc.reset_index()
            del df_ohlc['index']
            df_ohlc = df_ohlc.sort_index(ascending = False)
            df_ohlc = df_ohlc.head(186)
            df_ohlc.columns = ["Date","Open","High",'Low',"Close"]
            df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
            new = np.arange(len(df_ohlc)+1)
            new = new[len(df_ohlc)+1:0:-1]
            #print new, len(new)
            df_ohlc['y'] = new
#             axarr[0,1].xaxis_date()  
#             candlestick_ohlc(axarr[0,1],df_ohlc.values,width=1, colorup='g', colordown='r',alpha=0.75)
#             axarr[0,1].set_ylabel("Price")
#             axarr[0,1].set_title('Performance of last %d months of %s' %(history, stockname))
#             axarr[0,1].legend()  
            X = df_ohlc['Date']
            x = df_ohlc['y']
            y = df_ohlc['Close']
            model = sm.OLS(y, x).fit()
            #print model.summary()
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            m, b = np.polyfit(df_ohlc['Date'], df_ohlc['Close'], 1)
#             print 'Slope: ', slope
#             print 'Intercept', intercept
#             axarr[0,1].plot(X, m*X + b, '-')
        if history == 12:
            df_ohlc = stock
            df_ohlc = df_ohlc.reset_index()
            del df_ohlc['index']
            df_ohlc = df_ohlc.sort_index(ascending = False)
            df_ohlc = df_ohlc.head(248)
            df_ohlc.columns = ["Date","Open","High",'Low',"Close"]
            df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
            new = np.arange(len(df_ohlc)+1)
            new = new[len(df_ohlc)+1:0:-1]
            #print new, len(new)
            df_ohlc['y'] = new
#             axarr[0,0].xaxis_date()  
#             candlestick_ohlc(axarr[0,0],df_ohlc.values,width=1, colorup='g', colordown='r',alpha=0.75)
#             axarr[0,0].set_ylabel("Price")
#             axarr[0,0].set_title('Performance of last %d months of %s' %(history, stockname))
#             axarr[0,0].legend()  
            X = df_ohlc['Date']
            x = df_ohlc['y']
            y = df_ohlc['Close']
            model = sm.OLS(y, x).fit()
            #print model.summary()
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            m, b = np.polyfit(df_ohlc['Date'], df_ohlc['Close'], 1)
#             print 'Slope: ', slope
            #print 'Intercept', intercept
#             axarr[0,0].plot(X, m*X + b, '-')
#             print 'Intercept', intercept
#         print '\n-------------------------The prediction Model based on %d months history-----------------------' %(history)
#         print '\nBudget: %d months' %(Budget)
#         print '\n-------------------------The prediction Model based on %d months history-----------------------' %(history)
        #print '\nBudget: %d months' %(Budget)
        todayPrice = df_ohlc['Close'].iloc[0]
        todayDate = df_ohlc['y'].iloc[0]
        nextDate = todayDate + Term
        p1 = slope*todayDate + intercept
        p2 = slope*nextDate + intercept
        dif = todayPrice - p1
        #print p1, p2, dif
        predictedPrice = p2 + dif
        
        diffdict[history] = abs(predictedPrice - todayPrice)
        expectations = predictedPrice - todayPrice
        payoff[history] = expectations
#         print diffdict
#         print 'The predicted price after %d months based on last %d months performance is %f' %(Term, history, predictedPrice)
        slopes[history] = slope        
        predictedPrices[history] = predictedPrice
    return (predictedPrices, slopes, payoff)
      
# def avgPredictedPrice(predictedPrices, payoff):
#     avgprice = 0
#     EMV = 0
#     pctcntr = 0.10
#     for key, value in sorted(diffdict.iteritems(), key=lambda (k,v): (v,k)):
# #             print pctcntr, predictedPrices[key]
#         avgprice += pctcntr*predictedPrices[key]
#         pctcntr += 0.10 
#     return avgprice, EMV 
    
def avgPredictedPrice(predictedPrices, payoff):
    avgprice = 0
    EMV = 0
    pctcntr = 0.10
    for key, value in sorted(diffdict.iteritems(), key=lambda (k,v): (v,k)):
#         print pctcntr, predictedPrices[key]
        avgprice += pctcntr*predictedPrices[key]
        EMV += pctcntr*payoff[key]
        pctcntr += 0.10 
    return avgprice, EMV
    
for stockname in stocklist: 
#     print '******************************%s*********************************' %stockname
    if stockname in NASDAQ:       
        param = {
            'q': "%s" %stockname, # Stock symbol (ex: "AAPL")
            'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
            'x': "NASDAQ", # Stock exchange symbol on which stock is traded (ex: "NASD")
            'p': "1Y" # Period (Ex: "1Y" = 1 year)
        }
        # get price data (return pandas dataframe)
        #print param
    else:
        param = {
            'q': "%s" %stockname, # Stock symbol (ex: "AAPL")
            'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
            'x': "NYSE", # Stock exchange symbol on which stock is traded (ex: "NASD")
            'p': "1Y" # Period (Ex: "1Y" = 1 year)
        }    
    # get price data (return pandas dataframe)
    df = gfc.get_price_data(param)
    stockData = pd.DataFrame(df)
    stockData['Date'] = pd.DatetimeIndex(stockData.index).date
    stockData['Year'] = pd.DatetimeIndex(stockData.index).year
    stockData['Month'] = pd.DatetimeIndex(stockData.index).month
    stockData['Day'] = pd.DatetimeIndex(stockData.index).dayofweek
    #Draw a candlestick plot
    stock = stockData[['Date', 'Open', 'High', 'Low', 'Close']]
    df_ohlc = stock.reset_index()
    del df_ohlc['index']
    df_ohlc.columns = ["Date","Open","High",'Low',"Close"]
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
    new = np.arange(1,len(df_ohlc)+1)
    df_ohlc['y'] = new
    #print df_ohlc
    todayPrice = df_ohlc['Close'][len(df_ohlc)-1]
    desiredPrice = purchasePrice + purchasePrice*(profitMargin/100)
    print '***************************************DECISION**************************************'
    print 'purchase price : ', purchasePrice
    print 'desired price : ', desiredPrice
    print 'today\'s price : ', todayPrice
    if desiredPrice <= todayPrice:
        profit = todayPrice - purchasePrice
        actualProfitMargin = (profit/purchasePrice)*100
        print 'You can sell stocks, returns will be %f per stock, %f percent profit per stock' %(profit, actualProfitMargin)
        exit()
    elif (desiredPrice > todayPrice): 
        for stockname in stocklist:
            investmentList = []
            TermList = []
            wholeEMVList = []
            wholeEMV = {}
            wholeEMV = defaultdict(dict)
            U = {}
            slopes = {}
            diffdict = {}   
            payoff = {}
#             print '******************************%s*****************************' %stockname
            #print history  
            if stockname in NASDAQ:       
                param = {
                    'q': "%s" %stockname, # Stock symbol (ex: "AAPL")
                    'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
                    'x': "NASDAQ", # Stock exchange symbol on which stock is traded (ex: "NASD")
                    'p': "1Y" # Period (Ex: "1Y" = 1 year)
                }
                # get price data (return pandas dataframe)
                #print param
            else:
                param = {
                    'q': "%s" %stockname, # Stock symbol (ex: "AAPL")
                    'i': "86400", # Interval size in seconds ("86400" = 1 day intervals)
                    'x': "NYSE", # Stock exchange symbol on which stock is traded (ex: "NASD")
                    'p': "1Y" # Period (Ex: "1Y" = 1 year)
                }    
            df = gfc.get_price_data(param)
            stockData = pd.DataFrame(df)
            stockData['Date'] = pd.DatetimeIndex(stockData.index).date
            stockData['Year'] = pd.DatetimeIndex(stockData.index).year
            stockData['Month'] = pd.DatetimeIndex(stockData.index).month
            stockData['Day'] = pd.DatetimeIndex(stockData.index).dayofweek
            #Draw a candlestick plot
            stock = stockData[['Date', 'Open', 'High', 'Low', 'Close']]
                   
        predictedPrices, slopes, payoff = predictStockPrice(purchasePrice, Term, profitMargin,stockname)
        avgprice, EMV = avgPredictedPrice(predictedPrices, payoff)
        print 'The predicted price after %d days is %f'%(Term, avgprice)
        if (avgprice >= desiredPrice):
            print 'You should wait for %d days, the predicted price after %d days is %f' %(Term, Term, avgprice)
        else:
            if (todayPrice >= avgprice):
                print 'WARNING : You should better sell your stocks today, because today\'s price is $%f higher then the predicted price after %d days' %(todayPrice-avgprice, Term)
            else:
                profit = avgprice - purchasePrice
                predictedProfitMargin = (profit/purchasePrice)*100 
                print 'WAIT: you might get better price after %d days, with expected profit of %f per stock, with %f percent profit margin' %(Term, (avgprice-todayPrice), predictedProfitMargin) 
        
        plt.show()
   