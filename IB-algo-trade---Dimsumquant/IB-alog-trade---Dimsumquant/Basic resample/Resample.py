# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:41:51 2018

@author: Aqueous Carlos
"""

import pandas as pd
from pandas_datareader import data
import numpy as np

data = pd.read_csv('.csv', 
                   parse_dates={'DateTime': ['Date', 'Time']}, 
                   usecols=['Index', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Count', 'WAP'], 
                   na_values=['nan']).set_index('DateTime')  

data1 = pd.read_csv('.csv',
                    parse_dates={'DateTime': ['Date', 'Time']}, 
                    usecols=['Index', 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Count', 'WAP'], 
                    na_values=['nan']).set_index('DateTime') 

data_0 = pd.concat([data, data1], axis=0)

data_0_dict = {'Open':'first', 'High':'max', 'Low':'min', 
               'Close':'last', 'Volume':'sum', 'Count':'sum', 'WAP':'mean'}

# For period in 1T, 2T, 5T, 10T, 30S, 60S, 1H, 4H, 1D, 1W, 1M, 1Y
'''Alias   Description
B       business day frequency
C       custom business day frequency (experimental)
D       calendar day frequency
W       weekly frequency
M       month end frequency
BM      business month end frequency
CBM     custom business month end frequency
MS      month start frequency
BMS     business month start frequency
CBMS    custom business month start frequency
Q       quarter end frequency
BQ      business quarter endfrequency
QS      quarter start frequency
BQS     business quarter start frequency
A       year end frequency
BA      business year end frequency
AS      year start frequency
BAS     business year start frequency
BH      business hour frequency
H       hourly frequency
T, min  minutely frequency
S       secondly frequency
L, ms   milliseonds
U, us   microseconds
N       nanoseconds
'''
data_0_resample = data_0.resample('1 D', how=data_0_dict).dropna(how='any')

# change within the bar
Change = data_0_resample['Change'] = data_0_resample['High'] - data_0_resample['Low']

# Rows diff and add column
CloseDiff = data_0_resample['CloseDiff'] = data_0_resample['Close'].diff()

# Close pct change
Close_pct_change = data_0_resample['Close_pct_change'] = data_0_resample['Close'].pct_change()*100

# Time diff
# TimeDiff = data_0_resample['TimeDiff'] = data_0_resample['DateTime'].diff()

#pivot table
Ptable = pd.pivot_table(data_0_resample,index=['DateTime'])

PtableHMax = data_0_resample.groupby(pd.TimeGrouper('Y')).High.idxmax()
PtableLMin = data_0_resample.groupby(pd.TimeGrouper('Y')).Low.idxmin()

# describe data()
# de = Change.describe()

print(data_0_resample)

# print(de)
# print(Ptable)

