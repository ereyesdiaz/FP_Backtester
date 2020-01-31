# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:15:41 2017
@author: Enrique Reyes
Descrption.- This code loops a Stratgy function over several Futures time series to execute
Backtesting maximizing by SR to find optimar parameters. Output variables are produce with some
detail to get information about every contract backtest. 
IMPORTANT NOTE: Carefull running takes quite a while depending on the set of inut params and the 
group of contract tickers.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import talib as ta
import os
import itertools
import timeit

##################################################### Starts Strategy Function Code ############################################################

def Strategy(data, input_parameters, start_date, end_date, capital, ann_vol_tgt, point_value): #start_date and end_date should be provided in 'YYYY-MM-DD' format
    
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    
    SemaBars = input_parameters[0]       #Inout parameter for Short Exponential Moving Average Bars Back
    LemaBars = input_parameters[1]       #Inout parameter for Long Exponential Moving Average Bars Back
    StopmaBars = input_parameters[2]     #Inout parameter for Exponential Moving Average Bars Back used as Default Trailing Stop
    BBbarsback = input_parameters[3]     #Inout parameter for Bollinger Bands Bars Back
    BBvol = input_parameters[4]          #Inout parameter for Bollinger Bands Volatility
    ADXwindow = input_parameters[5]      #Inout parameter for Bars Back used to calculate ADXscore which is a MA of ADXbin variable
    ATRmultiple = input_parameters[6]    #Inout parameter for Short Exponential Moving Average Bars Back
    AddPosMult = input_parameters[7]
    acc = input_parameters[8]            #Input parameter for PSAR indicator means acceleration

    maxacc = 0.20
    Capital = float(capital)                    #Capital
    Point_Value = float(point_value)            #NQ Multiplier per point of Index
    Risk_Tolerance = float(ann_vol_tgt)         #Annual Volatility Target
    
    data['FCVol'] = (data['Close'].rolling(60).std())*Point_Value  #Futures Contract Standard Deviation using a rolling window of 3 months 60 trading days
    data['SEMA'] = ta.EMA(np.asarray(data['Close']), timeperiod = SemaBars )  #Short term Exponential MA using talib
    data['LEMA'] = ta.EMA(np.asarray(data['Close']), timeperiod = LemaBars)   #Long term Exopnential MA using talibtecn
    data['MA_Stp1'] = ta.EMA(np.asarray(data['Close']), timeperiod = StopmaBars) #Exponential MA for Trailing Stops using talib
    data['PSAR'] = ta.SAR(np.asarray(data['High']), np.asarray(data['Low']), acc, maxacc)   #Parabolic SAR using talib
    data['SAR_Xt'] = np.zeros(len(data))
    data['PSAR_Side'] = np.ones(len(data))                                    #PSAR_Side will be used to know the current PSAR signal  
    data.loc[data['PSAR']<data['Close'], 'PSAR_Side'] = 1    #PSAR_Side = 1 means PSAR is signaling a Long Trade
    data.loc[data['PSAR']>data['Close'], 'PSAR_Side'] = -1   #PSAR_Side = 1 means PSAR is signaling a Short Trade
    data['SAR_PE'] = np.zeros(len(data))
    data['SAR_FA'] = np.zeros(len(data))
    N = len(data)
    curracc = acc
    SARsignalbar = 1
    data.index = range(0, len(data))
    for j in range(N):
        if (data['PSAR'][j] < data['Close'][j]):
            if (data['PSAR'][j-1] > data['Close'][j-1]):
                SARsignalbar = j
                data.set_value(j,'SAR_PE', data['High'][j])
                curracc = acc
                data.set_value(j,'SAR_FA', curracc)
            elif (data['PSAR'][j-1] < data['Close'][j-1]):
                data.set_value(j,'SAR_PE', data['High'][SARsignalbar:(j+1)].max())
                if (data['SAR_PE'][j] > data['SAR_PE'][j-1]):
                    curracc = curracc + acc
                data.set_value(j,'SAR_FA', min(curracc,maxacc))
        elif (data['PSAR'][j] > data['Close'][j]):
            if (data['PSAR'][j-1] < data['Close'][j-1]):
                SARsignalbar = j
                data.set_value(j,'SAR_PE', data['Low'][j])
                curracc = acc
                data.set_value(j,'SAR_FA', curracc)
            elif (data['PSAR'][j-1] > data['Close'][j-1]):
                data.set_value(j,'SAR_PE', data['Low'][SARsignalbar:(j+1)].min())
                if (data['SAR_PE'][j] < data['SAR_PE'][j-1]):
                    curracc = curracc + acc
                data.set_value(j,'SAR_FA', min(curracc,maxacc))

    data.loc[data['PSAR'] < data['Close'], 'SAR_Xt'] = data['PSAR'] + (data['SAR_FA']*(data['SAR_PE']-data['PSAR']))
    data.loc[data['PSAR'] > data['Close'], 'SAR_Xt'] = data['PSAR'] - (data['SAR_FA']*(data['PSAR']-data['SAR_PE'])) 
    data['ATR'] = ta.ATR(np.asarray(data['High']), np.asarray(data['Low']), np.asarray(data['Close']), 20) #Average True Range using 20 Bars Back fixed input
    data['ADX'] = ta.ADX(np.asarray(data['High']), np.asarray(data['Low']), np.asarray(data['Close']), timeperiod=14) #ADX using 14 Bars Back fixed input
    data['ADXbin'] = np.zeros(len(data))    #ADXbin will be = 1 when ADX(14) prints a value over 20 points
    data.set_value(data['ADX']>20, 'ADXbin', 1) #ADX threshold = 20 points used as fixed input
    data['ADXscore'] = ta.MA(np.asarray(data['ADXbin']), timeperiod = ADXwindow) #ADXscore = % of days of rolling window where ADX(14) > 20
    data = data.dropna()
    data['Trend'] = np.zeros(len(data)) #Trend Filter variable will be used to filter when our Entry Rule could buy (Trend=1) or sell (Trend+-1)
    data.loc[(data['SEMA'].shift(1)>data['LEMA'].shift(1))&(data['ADXscore'].shift(1)>0.50)&(data['PSAR_Side']==1), 'Trend'] = 1    
    data.loc[(data['SEMA'].shift(1)<data['LEMA'].shift(1))&(data['ADXscore'].shift(1)>0.50)&(data['PSAR_Side']==-1), 'Trend'] = -1  
    data['BBupper'], data['BBmiddle'], data['BBlower'] = ta.BBANDS(np.asarray(data['Close']), BBbarsback, BBvol, BBvol) #Bollinger Bands using talib
    data['UpperBd'] = data['BBupper'] #Upper Band - Breakout Entry level for Long Entries assuming we are executing using a Stop market order
    data['LowerBd'] = data['BBlower'] #Lower Band - Breakout Entry Level for Short Entries assuming we are executing using a Stop market order
    data['XtLE1'] = data['MA_Stp1']   #Assigning the default Trailing Stop Mechanism in this case we are using a single MA to trigger exit signals
    data['XtSE1'] = data['MA_Stp1']   #for this Strategy version XtLE1 = XtSE1 but it could be a a diferent value in next implementations
    N = len(data)
    data['LongXt'] = np.zeros(N)      #Long Exit is a variable that can shift between Trailing Exit #1 System or #2 System
    data['ShortXt'] = np.zeros(N)     #Long Exit is a variable that can shift between Trailing Exit #1 System or #2 System
    data['Signal'] = np.zeros(len(data))   #Signal = 1 means Bought at UpperBd with Stop Order
    data['MktPos'] = np.zeros(len(data))   #Signal = -1 means Sold at LowerBd with Stop order
    data['EntryPrice'] = np.zeros(len(data))
    data['ExitPrice'] = np.zeros(len(data))
    data['MtoM'] = np.zeros(len(data))   # Mark to Market
    data['MtoM_Chg'] = np.zeros(len(data))
    data['Contracts'] = np.zeros(len(data))
    data['PosSiz'] = np.zeros(len(data))
    data['2ndEntry'] = np.zeros(len(data))
    data['2ndMtoM'] = np.zeros(len(data))
    data['2ndMtoM_Chg'] = np.zeros(len(data))
    data['PnL2ndTrade'] = np.zeros(len(data))
    data['Dly_PnL'] = np.zeros(len(data))
    data['Longs'] = np.zeros(len(data))
    data['Shorts'] = np.zeros(len(data))
    data['MaxMtM'] = np.zeros(len(data)) # Max of MtoM per trade will be used to compute when m*ATR profit is achieved
    data['MaxClose'] = data['Close']
    data['StateTE'] = np.zeros(len(data))
    data['ExitFlag'] = np.zeros(len(data))
    data['LN_Rtn'] = np.zeros(len(data)) #MtoM Returns
    data['Cum_TRtn'] = np.zeros(len(data))
    data['Return'] = np.zeros(len(data))   #Simple Returns calculated when Exit a position 
    data['Equity'] = np.ones(len(data))*Capital
    N = len(data)
    data.index = range(0, len(data))
    for i in range(1,N):
        if (data['MktPos'][i-1]==0):
            if ((data['Trend'][i]==1) and (data['High'][i]>data['UpperBd'][i])):
                data.set_value(i,'Signal', 1)
                data.set_value(i,'MktPos', 1)
                if (data['Open'][i] > data['UpperBd'][i]):
                    entryprice = data['Open'][i]
                else:
                    entryprice = data['UpperBd'][i]
                InvestedEq = data['Equity'][i-1]
                contracts = np.trunc((InvestedEq*Risk_Tolerance/16)/float(data['FCVol'][i-1]))
                data.set_value(i,'Contracts', contracts)
                data.set_value(i,'EntryPrice', entryprice)
                data.set_value(i,'MtoM', data['Close'][i] - entryprice)
                data.set_value(i,'MtoM_Chg', data['MtoM'][i])
                data.set_value(i,'Dly_PnL', (data['MtoM_Chg'][i]*Point_Value*contracts)+data['PnL2ndTrade'][i])
                data.set_value(i,'Equity', InvestedEq + data['Dly_PnL'][i])
                data.set_value(i,'LN_Rtn', np.log(data['Equity'][i]/data['Equity'][i-1]))
                data.set_value(i,'Cum_TRtn', data['LN_Rtn'][i])
                entrybar = i
            elif ((data['Trend'][i]==-1) and (data['Low'][i]<data['LowerBd'][i])):
                data.set_value(i,'Signal', -1)
                data.set_value(i,'MktPos', -1)
                if (data['Open'][i] < data['UpperBd'][i]):
                    entryprice = data['Open'][i]
                else:
                    entryprice = data['UpperBd'][i]
                InvestedEq = data['Equity'][i-1]
                contracts = np.trunc((InvestedEq*Risk_Tolerance/16)/float(data['FCVol'][i-1]))
                data.set_value(i,'Contracts', contracts)
                data.set_value(i,'EntryPrice', entryprice)
                data.set_value(i,'MtoM', entryprice - data['Close'][i])
                data.set_value(i,'MtoM_Chg', data['MtoM'][i])
                data.set_value(i,'Dly_PnL', (data['MtoM_Chg'][i]*Point_Value*contracts)+data['PnL2ndTrade'][i])
                data.set_value(i,'Equity', InvestedEq + data['Dly_PnL'][i])
                data.set_value(i,'LN_Rtn', np.log(data['Equity'][i]/data['Equity'][i-1]))
                data.set_value(i,'Cum_TRtn', data['LN_Rtn'][i])
                entrybar = i
            else:
                data.set_value(i,'Equity', data['Equity'][i-1])
        elif (data['MktPos'][i-1]==1):
            if (data['MaxMtM'][i-1]>(AddPosMult*data['ATR'][i-1])):
                data.set_value(i,'PosSiz', 1)
                if(data['PosSiz'][i-1] == 0):
                    PosSizEntry = data['Open'][i]
                    data.set_value(i,'2ndEntry', PosSizEntry)
                    data.set_value(i,'2ndMtoM', data['Close'][i] - PosSizEntry)
                    data.set_value(i,'2ndMtoM_Chg', data['2ndMtoM'][i])
                    data.set_value(i,'PnL2ndTrade', data['2ndMtoM_Chg'][i]*Point_Value*contracts)
            if (data['MaxMtM'][i-1]>(ATRmultiple*data['ATR'][i-1])):
                data.set_value(i,'LongXt', data['SAR_Xt'][i-1])
            else:
                data.set_value(i,'LongXt', data['XtLE1'][i-1])
            if (data['Low'][i]<data['LongXt'][i]):
                data.set_value(i,'Signal', -1)
                data.set_value(i,'Longs', 1)
                data.set_value(i,'MktPos', 0)
                exitprice = data['LongXt'][i]
                data.set_value(i,'ExitPrice', exitprice)
                data.set_value(i,'MtoM', 0)
                data.set_value(i,'MtoM_Chg', exitprice - data['Close'][i-1])
                if(data['PosSiz'][i-1] == 1):
                    data.set_value(i,'2ndMtoM', 0)
                    data.set_value(i,'2ndMtoM_Chg', exitprice - data['Close'][i-1])
                    data.set_value(i,'PnL2ndTrade', data['2ndMtoM_Chg'][i]*Point_Value*contracts)
                data.set_value(i,'Dly_PnL', (data['MtoM_Chg'][i]*Point_Value*contracts)+data['PnL2ndTrade'][i])
                data.set_value(i,'Equity', data['Equity'][i-1] + data['Dly_PnL'][i])
                data.set_value(i,'LN_Rtn', np.log(data['Equity'][i]/data['Equity'][i-1]))
                data.set_value(i,'Cum_TRtn', data['LN_Rtn'][entrybar:(i+1)].sum())
                ret = (((exitprice - entryprice)*Point_Value)/InvestedEq)*100
                data.set_value(i,'Return',ret)
            else:
                if(data['PosSiz'][i-1] == 1):
                    data.set_value(i,'2ndMtoM', data['Close'][i] - PosSizEntry)
                    data.set_value(i,'2ndMtoM_Chg', data['2ndMtoM'][i]-data['2ndMtoM'][i-1])
                    data.set_value(i,'PnL2ndTrade', data['2ndMtoM_Chg'][i]*Point_Value*contracts)
                data.set_value(i,'MktPos', 1)
                data.set_value(i,'MtoM', data['Close'][i] - entryprice)
                data.set_value(i,'MtoM_Chg', data['MtoM'][i]-data['MtoM'][i-1])
                data.set_value(i,'Dly_PnL', (data['MtoM_Chg'][i]*Point_Value*contracts)+data['PnL2ndTrade'][i])
                data.set_value(i,'MaxMtM', data['MtoM'][entrybar:(i+1)].max())
                data.set_value(i,'MaxClose', data['Close'][entrybar:(i+1)].max())
                data.set_value(i,'Equity', data['Equity'][i-1] + data['Dly_PnL'][i])
                data.set_value(i,'LN_Rtn', np.log(data['Equity'][i]/data['Equity'][i-1]))
                data.set_value(i,'Cum_TRtn', data['LN_Rtn'][entrybar:(i+1)].sum())
        elif (data['MktPos'][i-1]==-1):
            if (data['MaxMtM'][i-1]>(AddPosMult*data['ATR'][i-1])):
                data.set_value(i,'PosSiz', -1)
                if(data['PosSiz'][i-1] == 0):
                    PosSizEntry = data['Open'][i]
                    data.set_value(i,'2ndEntry', PosSizEntry)
                    data.set_value(i,'2ndMtoM', PosSizEntry - data['Close'][i])
                    data.set_value(i,'2ndMtoM_Chg', data['2ndMtoM'][i])
                    data.set_value(i,'PnL2ndTrade', data['2ndMtoM_Chg'][i]*Point_Value*contracts)
            if (data['MaxMtM'][i-1]>(ATRmultiple*data['ATR'][i-1])):
                data.set_value(i,'ShortXt', data['SAR_Xt'][i-1])
            else:
                data.set_value(i,'ShortXt', data['XtSE1'][i-1])
            if (data['High'][i]>data['ShortXt'][i]):
                data.set_value(i,'Signal', 1)
                data.set_value(i,'Shorts', 1)
                data.set_value(i,'MktPos', 0)
                exitprice = data['ShortXt'][i]
                data.set_value(i,'ExitPrice', exitprice)
                data.set_value(i,'MtoM', 0)
                data.set_value(i,'MtoM_Chg', data['Close'][i-1] - exitprice)
                if(data['PosSiz'][i-1] == -1):
                    data.set_value(i,'2ndMtoM', 0)
                    data.set_value(i,'2ndMtoM_Chg', data['Close'][i-1] - exitprice)
                    data.set_value(i,'PnL2ndTrade', data['2ndMtoM_Chg'][i]*Point_Value*contracts)
                data.set_value(i,'Dly_PnL', (data['MtoM_Chg'][i]*Point_Value*contracts)+data['PnL2ndTrade'][i])
                data.set_value(i,'Equity', data['Equity'][i-1] + data['Dly_PnL'][i])
                data.set_value(i,'LN_Rtn', np.log(data['Equity'][i]/data['Equity'][i-1]))
                data.set_value(i,'Cum_TRtn', data['LN_Rtn'][entrybar:(i+1)].sum())
                ret = (((entryprice-exitprice)*Point_Value)/InvestedEq)*100
                data.set_value(i,'Return',ret)
            else:
                if(data['PosSiz'][i-1] == -1):
                    data.set_value(i,'2ndMtoM', PosSizEntry - data['Close'][i])
                    data.set_value(i,'2ndMtoM_Chg', data['2ndMtoM'][i]-data['2ndMtoM'][i-1])
                    data.set_value(i,'PnL2ndTrade', data['2ndMtoM_Chg'][i]*Point_Value*contracts)
                data.set_value(i,'MktPos', -1)
                data.set_value(i,'MtoM', entryprice-data['Close'][i])
                data.set_value(i,'MtoM_Chg', data['MtoM'][i]-data['MtoM'][i-1])
                data.set_value(i,'Dly_PnL', (data['MtoM_Chg'][i]*Point_Value*contracts)+data['PnL2ndTrade'][i])
                data.set_value(i,'MaxMtM', data['MtoM'][entrybar:(i+1)].max())
                data.set_value(i,'MaxClose', data['Close'][entrybar:(i+1)].min())
                data.set_value(i,'Equity', data['Equity'][i-1] + data['Dly_PnL'][i])
                data.set_value(i,'LN_Rtn', np.log(data['Equity'][i]/data['Equity'][i-1]))
                data.set_value(i,'Cum_TRtn', data['LN_Rtn'][entrybar:(i+1)].sum())

    Longs_Rtn = data['Return']*data['Longs']
    Shorts_Rtn = data['Return']*data['Shorts']
    Long_Winners = float(sum(float(num)>0 for num in Longs_Rtn))
    Short_Winners = float(sum(float(num)>0 for num in Shorts_Rtn))
    Total_Longs = data['Longs'].sum()
    Total_Shorts = data['Shorts'].sum()
    Long_Losers = float(sum(float(num)<0 for num in Longs_Rtn))
    Short_Losers = float(sum(float(num)<0 for num in Shorts_Rtn))
    Hit_R_Longs = round((Long_Winners/Total_Longs)*100,4)
    Hit_R_Shorts = round((Short_Winners/Total_Shorts)*100,4)
    Winners_Rtn_L = round(Longs_Rtn[Longs_Rtn>0].sum(),4)
    Losers_Rtn_L = round(Longs_Rtn[Longs_Rtn<0].sum(),4)
    Winners_Rtn_S = round(Shorts_Rtn[Shorts_Rtn>0].sum(),4)
    Losers_Rtn_S = round(Shorts_Rtn[Shorts_Rtn<0].sum(),4)
    if (Long_Winners == 0):
        Avg_Profit_L = 0
    else:
        Avg_Profit_L = round(Winners_Rtn_L/Long_Winners,8)
    if (Long_Losers == 0):
        Avg_Loss_L = 0
        Avg_PL_R_L = 'NA'
    else:
        Avg_Loss_L = round(Losers_Rtn_L/Long_Losers,8)
        Avg_PL_R_L = round(Avg_Profit_L/(-Avg_Loss_L),8)
    
    if (Short_Winners == 0):
        Avg_Profit_S = 0
    else:
        Avg_Profit_S = round(Winners_Rtn_S/Short_Winners,8)
    if (Short_Losers == 0):
        Avg_Loss_S = 0
        Avg_PL_R_S = 'NA'
    else:
        Avg_Loss_S = round(Losers_Rtn_S/Short_Losers,8)
        Avg_PL_R_S = round(Avg_Profit_S/(-Avg_Loss_S),8)
    
    data['MaxEquity'] = np.ones(N)*Capital                                           #The nexts lines are used to calculate Performance Variables 
    for j in range(1,N):
        data.set_value(j,'MaxEquity',data['Equity'][0:(j+1)].max())
    data['DD'] = round((data['Equity']/data['MaxEquity']-1)*100,2)
    MaxDD = data['DD'].min()
    tempcolnum = data.columns.get_loc('Equity')
    Total_Return = round(((data.iloc[N-1, tempcolnum]/Capital)-1)*100,2)
    Total_Trades = float(sum(float(num)!=0 for num in data['Return']))
    PctLongs = Total_Longs/Total_Trades
    Winners = float(sum(float(num)>0 for num in data['Return']))
    Losers = float(sum(float(num)<0 for num in data['Return']))
    Years = N/252
    Trades_xYr = round(Total_Trades / Years,1)
    if (Total_Trades == 0):
        Hit_Ratio = 0
    else:
        Hit_Ratio = round((Winners/Total_Trades)*100,2)
    Winners_Rtn = round(data[data.Return>0].sum()['Return'],4)
    Losers_Rtn = round(data[data.Return<0].sum()['Return'],4)
    if (Winners == 0):
        Avg_Profit = 0
    else:
        Avg_Profit = round(Winners_Rtn/Winners,4)
    if (Losers == 0):
        Avg_Loss = 0
        Avg_PL_Ratio = 'NA'
    else:
        Avg_Loss = round(Losers_Rtn/Losers,4)
        Avg_PL_Ratio = round(Avg_Profit/(-Avg_Loss),4)
        
    Daily_AvgLnRtn = np.mean(data['LN_Rtn'])
    Daily_Std = np.std(data['LN_Rtn'])
    if (Daily_Std == 0):
        SharpeRatio = 0
    else:
        SharpeRatio = round((Daily_AvgLnRtn/Daily_Std)*np.sqrt(252),2)      #Annualized Sharpe Ratio its calculated using Daily LN Returns
    Cagr = round((((1+(Total_Return/100))**(float(1)/Years))-1)*100,2) 
    data = data.set_index(['Date'])
    Returns = data['LN_Rtn']
    Equity = data['Equity']
    DD_curve = data['DD']   
    
    return Returns,Equity,DD_curve,Total_Return,Cagr,SharpeRatio,Trades_xYr,Hit_Ratio,Avg_PL_Ratio,PctLongs,Hit_R_Longs,Avg_PL_R_L,Hit_R_Shorts,Avg_PL_R_S,MaxDD,data

############################################################## Ends Strat Function ###################################################
######################################################### Starts Backtesting Function ################################################

def Backtesting(data, parameters, start_date, end_date, capital, ann_vol_tgt, point_value):

    BT_Runs = list(itertools.product(*parameters))
    BT_Results = pd.DataFrame(index=range(len(BT_Runs)))
    BT_Results['PARAMETERS'] = BT_Runs
    N = len(BT_Results)
    BT_Results['TOTAL_RTN'] = np.zeros(N)
    BT_Results['CAGR'] = np.zeros(N)
    BT_Results['SHARPE_RATIO'] = np.zeros(N)
    BT_Results['#TRADESxYR'] = np.zeros(N)
    BT_Results['HIT_RATIO'] = np.zeros(N)
    BT_Results['AVG_P/L_RATIO'] = np.zeros(N)
    BT_Results['PCT_LONGS'] = np.zeros(N)
    BT_Results['HIT_R_LONGS'] = np.zeros(N)
    BT_Results['AVG_P/L_R_L'] = np.zeros(N)
    BT_Results['HIT_R_SHORTS'] = np.zeros(N)
    BT_Results['AVG_P/L_R_S'] = np.zeros(N)
    BT_Results['MAX_DD'] = np.zeros(N)
    for i in range(0,N):
        Strat_Run = Strategy(data, BT_Runs[i], start_date, end_date, capital, ann_vol_tgt, point_value)
        for j in range(3,15):
            BT_Results.iloc[i,j-2] = Strat_Run[j]
        print('Backtesting', i+1,'from', N, 'its done')
        
    return BT_Results

######################################################### Ends Backtesting Function #######################################################
################################################## Create a Dictionary for Contract Units #################################################    

dict_FCUnits = {}
dict_FCUnits['AD'] = 100000
dict_FCUnits['BP'] = 62500
dict_FCUnits['C'] = 50
dict_FCUnits['CL'] = 1000
dict_FCUnits['EC'] = 125000
dict_FCUnits['ES'] = 50
dict_FCUnits['GC'] = 100
dict_FCUnits['HG'] = 25000
dict_FCUnits['JY'] = 12.50
dict_FCUnits['NG'] = 10000
dict_FCUnits['NQ'] = 20
dict_FCUnits['S'] = 50
dict_FCUnits['TY'] = 1000
dict_FCUnits['W'] = 50
dict_FCUnits['YM'] = 5

###########################################################################################################################################
######################################## Starts Loop for In Sample Backtesting a set of Instruments #######################################

start = timeit.default_timer()

start_date = '1999-11-24'
end_date = '2009-11-24'
capital = 1000000
ann_vol_tgt = 0.50

assets_list = os.listdir('FP_Data')
assets_list = [x.split('.')[0] for x in assets_list]

dict_assets_data = {}
for asset in assets_list:
    df = pd.read_csv('FP_Data'+os.sep+asset+'.csv')
    dict_assets_data[asset] = df
    
dict_best_data ={}

dict_sort_results = {}
df_best_results = pd.DataFrame(index=range(len(assets_list)), columns=['ASSET','PARAMETERS','TOTAL_RTN','CAGR','SHARPE_RATIO', 
                             '#TRADESxYR','HIT_RATIO','AVG_P/L_RATIO','PCT_LONGS','HIT_R_LONGS','AVG_P/L_R_L','HIT_R_SHORTS',
                             'AVG_P/L_R_S','MAX_DD'])
asset_count = 0
for asset in assets_list:
    data = dict_assets_data[asset]
    ########################################################### Input Parameters #######################################################
    SemaBars = [20,30]          #Inout parameter for Short Exponential Moving Average Bars Back
    LemaBars = [100]            #Inout parameter for Long Exponential Moving Average Bars Back
    StopmaBars = [40,50]        #Inout parameter for Exponential Moving Average Bars Back used as Default Trailing Stop
    BBbarsback = [20,30]           #Inout parameter for Bollinger Bands Bars Ba
    BBvol = [1,2]                 #Inout parameter for Bollinger Bands Volatility
    ADXwindow = [20,40]            #Inout parameter for Bars Back used to calculate ADXscore which is a MA of ADXbin variable
    ATRmultiple = [2,3,4]           #Multiple of ATRs to trigger second TE mechanism
    AddPosMult = [1,2]        #Multiple of ATRs to trigger Pos Sizing
    acc = [0.02]                #Input parameter for PSAR indicator means acceleration
    
    parameters_values = [SemaBars, LemaBars, StopmaBars, BBbarsback, BBvol, ADXwindow, ATRmultiple, AddPosMult, acc] 
    Test = Backtesting(data, parameters_values, start_date, end_date, capital, ann_vol_tgt, dict_FCUnits[asset])
    Test = Test.sort_values('SHARPE_RATIO', ascending = False)
    best_params = Test.loc[Test['SHARPE_RATIO']==np.max(Test['SHARPE_RATIO'])]
    dict_sort_results[asset] = Test
    df_best_results.iloc[asset_count,0] = asset
    for i in range(0,13):
        df_best_results.iloc[asset_count,i+1] = best_params.iloc[0,i]
    asset_count = asset_count + 1
    best_run = Strategy(data, list(best_params.iloc[0,0]), start_date, end_date, capital, ann_vol_tgt, dict_FCUnits[asset])
    dict_best_data[asset] = best_run[15]
    print('Backtesting for', asset,'its done')

stop = timeit.default_timer()
print('Running time =', round((stop-start)/60,1),'min')

#################################################### Ends Loop for Testing a set of Instruments ########################################
############################################ Get Equity and DD curve plots for best results on each asset ##############################

for i in range(len(df_best_results)):
    asset = df_best_results.iloc[i,0]
    parameters = list(df_best_results.iloc[i,1])
    data = dict_assets_data[asset]
    Strat_results = Strategy(data, parameters, start_date, end_date, 1000000, 0.25, dict_FCUnits[asset])
    EqCurve = Strat_results[1]/capital
    EqCurve.plot(figsize=(9,5))
    plt.title('Equity Curve '+ asset)
    plt.show()
    Strat_results[2].plot(figsize=(9,3), color='r',kind='area')
    plt.title('Draw Down Curve '+ asset)
    plt.show()
    print(parameters)

#########################################################################################################################################











    
