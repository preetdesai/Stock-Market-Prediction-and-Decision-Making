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

# print 'how much you would like to invest? (Enter Dollar Amount) : '
# Budget = int(input())
# print 'How long you would like to keep the stocks (Enter Months): '
# Term = int(input())
stocklist = str(raw_input('Whcih shares you would like to buy? (Enter tickers): '))
stocklist = [word.strip(string.punctuation) for word in stocklist.split()]

# stocklist = ['MSFT','AAPL','TSLA']
BudgetDict = {1:1000, 2:2000, 3:3000, 4:4000, 5:5000, 6:6000, 7:7000, 8:8000, 9:9000, 10:10000}
TermDict = {1:1,2:2,3:3,4:8,5:12,6:18,7:24,8:36,9:48,10:52}

n1 = pd.read_csv('NASDAQ.csv', 'rb')
NASDAQ = n1['Symbol'].tolist()

def predictStockPrice(Budget, Term, stockname):
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
        todayPrice = df_ohlc['Close'].iloc[0]
        stocks = math.floor(Budget/todayPrice)
        Investment = stocks*todayPrice
#         print 'Today\'s Price', todayPrice
#         print 'You can buy %d stocks with %f dollar investment on today\'s price' %(stocks, Investment)
#         print 'Term: %d months' %(Term)
        todayDate = df_ohlc['y'].iloc[0]
        nextDate = todayDate + (7*Term)
        p1 = slope*todayDate + intercept
        p2 = slope*nextDate + intercept
        dif = todayPrice - p1
#         print p1, p2, dif
        predictedPrice = p2 + dif
        if todayPrice > Budget:
            U[history] = 0
            payoff[history] = 0
            expectations = 0
        else:
            diffdict[history] = abs(predictedPrice - todayPrice)
#             print 'The predicted price after %d months based on last %d months performance is %f' %(Term, history, predictedPrice)
            expectations = predictedPrice*stocks - Investment
            if expectations > 0:
                U[history] = math.log(expectations)
            else:
                U[history] = -math.log(abs(expectations))
            payoff[history] = expectations
#         if (expectations>0):
#             print 'The expected profit is $', expectations
#         else:
#             print 'The expected loss is $', expectations 
        predictedPrices[history] = predictedPrice 
        slopes[history] = slope        
    
    return (predictedPrices, slopes, U, payoff, Investment)


def avgPredictedPrice(predictedPrices, U, payoff):
    avgprice = 0
    EU = 0
    EMV = 0
    pctcntr = 0.10
    for key, value in sorted(diffdict.iteritems(), key=lambda (k,v): (v,k)):
#         print pctcntr, predictedPrices[key]
        avgprice += pctcntr*predictedPrices[key]
        EU += pctcntr*U[key]
        EMV += pctcntr*payoff[key]
        pctcntr += 0.10 
    return avgprice, EU, EMV 

fig = plt.figure()
ax = fig.gca(projection='3d')
CEDict = {}
EMVDict = {}
colors = iter(cm.rainbow(np.linspace(0, 1, len(stocklist))))

counter = 0
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
    print '******************************%s*****************************' %stockname
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
    
    for i in range(1,len(BudgetDict)+1):
        Budget= BudgetDict[i]
        print Budget
        for i in range(1,len(BudgetDict)+1):
            Term =  TermDict[i]
            print Term
#             print '******************************%s*********************************' %stockname         
            predictedPrices, slopes, U, payoff, Investment = predictStockPrice(Budget, Term, stockname)
#             print predictedPrices
            avgprice, EU, EMV = avgPredictedPrice(predictedPrices, U, payoff)
#             print EU
            CE = e**EU
#             print "CE", CE
#             print 'The predicted price for %s after %d month is %f' %(stockname, Term, avgprice)
#             print 'slopes'
#             print slopes
#             print 'U', U
#             print 'EMV', EMV
#             print 'payoff',payoff
            CEDict[stockname] = CE
            EMVDict[stockname] = EMV
            wholeEMV[Budget][Term] = EMV
            investmentList.append(Investment)
            TermList.append(Term)
            wholeEMVList.append(EMV)
            #plt.show()
#             print BudgetList
#             print TermList
#             print wholeEMVList
    next_color = colors.next()
    ax.plot_trisurf(investmentList, TermList, wholeEMVList, color=next_color, linewidth=0.2, antialiased=True, alpha = 0.9)
    ax.scatter(investmentList, TermList, wholeEMVList, color=next_color, linewidth=0.2, antialiased=True, alpha = 0.9, label = stockname)
    fake2Dline = mpl.lines.Line2D([counter],[counter], linestyle="none", c=next_color, marker = 'o')        
    counter += 1
# print payoff
# print EU
# print diffdict
# print EMVDict
print CEDict
ax.legend()
best_CE = max(CEDict.iteritems(), key=operator.itemgetter(1))[0]
# print 'Best stock you should buy based on CE is %s with %f CE' %(best_CE, CEDict[best_CE])
print EMVDict
best_EMV = max(EMVDict.iteritems(), key=operator.itemgetter(1))[0]
# print 'Best stock you should buy based on EMV is %s with %f EMV' %(best_CE, EMVDict[best_EMV])    
ax.set_xlabel('Investment')
ax.set_ylabel('Term(Weeks)')
ax.set_zlabel('Retruns')
plt.show()            
           
