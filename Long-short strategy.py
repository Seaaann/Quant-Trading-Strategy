import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import _pickle as cPickle
import gzip
import functools
import dask
from dask import compute, delayed
from itertools import chain
from collections import OrderedDict

warnings.filterwarnings('ignore')

HEAD_PATH = '/Users/sean/Desktop/Plan B/Quant'
DATA_PATH = HEAD_PATH + '/energy pkl tick/'

import os
os.chdir(DATA_PATH)

import multiprocessing
print(multiprocessing.cpu_count())
print(os.cpu_count())

CORE_NUM = int(multiprocessing.cpu_count())

print(os.getcwd())

# Product list
product_list = ['bu', 'ru', 'v', 'pp', 'l', 'jd']
product = product_list[0]

dire = DATA_PATH + product
print(dire)

# sort dates
all_dates = list(map(lambda x: x, os.listdir(dire)))
dates = []
for i in range(len(all_dates)):
    dates.append(int(all_dates[i][0:8]))
dates = np.sort(dates)
all_dates = []
for i in dates:
    all_dates.append(str(i) + '.pkl')
    
print(len(all_dates))
print(all_dates[0:6])

def get_daily_pnl_fast(date, product, period=4096, tranct_ratio=False, threshold=0.001, tranct=0.21, noise=0):
    
    with gzip.open(dire + '/' + date, 'rb', compresslevel=1) as file_object:
        raw_data = file_object.read()
    ori_data = cPickle.loads(raw_data) # 解压数据
    
    data = ori_data[ori_data['good']] # 中间部分的数据用来回测
    
    n_bar = len(data)
    unit = np.std(data['ret'])
    np.random.seed(10)
    
    ## 计算每日收益率
    ret_period = (data['ret'].rolling(period).sum()).dropna().reset_index(drop=True) # 未来收益率，用来当作交易信号
    ret_period = ret_period.append(pd.Series([0] * (len(data) - len(ret_period)))).reset_index(drop=True)
    
    signal = pd.Series([0] * n_bar)
    signal[(ret_period > threshold) & (np.array(data['next.ask']) > 0)] = 1
    signal[(ret_period < -threshold) & (np.array(data['next.bid']) > 0)] = -1
    
    position = signal
    position[0] = 0
    position[n_bar-1] = 0
    position[n_bar-2] = 0
    change_of_position = position - position.shift(1)
    change_of_position[0] = 0
    change_base = np.zeros(n_bar)
    change_buy = np.array(change_of_position > 0)
    change_sell = np.array(change_of_position < 0)
    
    # 交易佣金
    if tranct_ratio:
        change_base[change_buy] = data['next.ask'][change_buy] * (1 + tranct) # 比率交易费用
        change_base[change_sell] = data['next.bid'][change_sell] * (1 - tranct)
    else:
        change_base[change_buy] = data['next.ask'][change_buy] + tranct # 固定交易费用
        change_base[change_sell] = data['next.bid'][change_sell] - tranct
    
    final_pnl = -sum(change_base * change_of_position) # pnl, 买 - ， 卖 +
    turnover = sum(change_base * abs(change_of_position)) 
    num = sum((position != 0) & (change_of_position != 0)) # 交易次数
    hld_period = sum(position != 0)
    
    result = OrderedDict([('date', date), ('final.pnl', final_pnl), ('turnover', turnover), ('num', num), ('hld.period', hld_period)])
    return result

# 一日盈亏
get_daily_pnl_fast(all_dates[0], product='bu', period=4096, tranct_ratio=True, threshold=0.001, tranct=1.1e-4)


with dask.config.set(scheduler='processes', num_workers=CORE_NUM):
    f_par = functools.partial(get_daily_pnl_fast, product='bu', period=4096, tranct_ratio=True, threshold=0.001, tranct=1.1e-4, noise=0)
    result = compute([delayed(f_par)(date) for date in all_dates])[0]