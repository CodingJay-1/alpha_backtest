import copy
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import sys
sys.path.append("/home/yjsun/alpha_dataserver")
import DataServer as ds

sys.path.append("/home/yjsun/alpha_backtest")
import AlphaBackTest as abt
from AlphaStrategy_v4_3 import AlphaStrategy

# 准备Funding数据
start_date = "20230901"
end_date = "20231130"
spot_data_dict, future_data_dict = abt.get_prepared_kline_from_basic_data(start_date,end_date, basic_data_path = "/update/alpha_data/basic_data_AlphaFM")
funding_data = future_data_dict['fundingRate']
del future_data_dict['fundingRate']

twap_data_dict = ds.get_min_feature(feature_list =
                            [
                            'm1_future_forward_twap_10min',
                            ]
                            , where = "/mnt1/alpha_data/features_AlphaFMSyj" ,data_tag = "AlphaFM", start_date = start_date, end_date = end_date)
future_data_dict.update(twap_data_dict)


YrMthList = ["202309","202310","202311"]
# YrMthList = ["202311"]
res0 = []
res1 = []
res2 = []
for Mth in YrMthList:
    res0.append(pd.read_pickle(f"/mnt1/alpha_data/bn_v6_factor/xgb_predict_future_FMV11BMD_f120m10normed1440_roll1_{Mth}.pkl"))
    res1.append(pd.read_pickle(f"/mnt1/alpha_data/bn_v6_factor/xgb_predict_future_FMV11BMD_f240m10normed2880_roll1_{Mth}.pkl"))
    res2.append(pd.read_pickle(f"/mnt1/alpha_data/bn_v6_factor/xgb_predict_future_FMV11BMD_f480m10normed2880_roll1_{Mth}.pkl"))
rolling_f0 = pd.concat(res0,axis=0)
rolling_f1 = pd.concat(res1,axis=0)
rolling_f2 = pd.concat(res2,axis=0)

res0 = []
res1 = []
res2 = []
for Mth in YrMthList:
    res0.append(pd.read_pickle(f"/mnt1/alpha_data/bn_v6_factor/xgb_predict_future_FMV11BMD_f120m10normed1440_roll2_{Mth}.pkl"))
    res1.append(pd.read_pickle(f"/mnt1/alpha_data/bn_v6_factor/xgb_predict_future_FMV11BMD_f240m10normed2880_roll2_{Mth}.pkl"))
    res2.append(pd.read_pickle(f"/mnt1/alpha_data/bn_v6_factor/xgb_predict_future_FMV11BMD_f480m10normed2880_roll2_{Mth}.pkl"))
rolling_f3 = pd.concat(res0,axis=0)
rolling_f4 = pd.concat(res1,axis=0)
rolling_f5 = pd.concat(res2,axis=0)

YrMthList = ["202309A","202309B","202309C","202309D","202310A","202310B","202310C","202310D","202311A","202311B","202311C","202311D"]
res0 = []
res1 = []
res2 = []
for Mth in YrMthList:
    res0.append(pd.read_pickle(f"/mnt1/alpha_data/bn_v6_factor/xgb_predict_future_FMV11BMD_f120m10normed1440_rollw_{Mth}.pkl"))
    res1.append(pd.read_pickle(f"/mnt1/alpha_data/bn_v6_factor/xgb_predict_future_FMV11BMD_f240m10normed2880_rollw_{Mth}.pkl"))
    res2.append(pd.read_pickle(f"/mnt1/alpha_data/bn_v6_factor/xgb_predict_future_FMV11BMD_f480m10normed2880_rollw_{Mth}.pkl"))
rolling_f6 = pd.concat(res0,axis=0)
rolling_f7 = pd.concat(res1,axis=0)
rolling_f8 = pd.concat(res2,axis=0)


# 将因子值转为spot_data_dict格式
start_ts = spot_data_dict["open_time"].index[0]
end_ts = spot_data_dict["open_time"].index[-1]
rolling_f0 = rolling_f0.loc[start_ts:end_ts]
rolling_f1 = rolling_f1.loc[start_ts:end_ts]
rolling_f2 = rolling_f2.loc[start_ts:end_ts]
rolling_f3 = rolling_f3.loc[start_ts:end_ts]
rolling_f4 = rolling_f4.loc[start_ts:end_ts]
rolling_f5 = rolling_f5.loc[start_ts:end_ts]
rolling_f6 = rolling_f6.loc[start_ts:end_ts]
rolling_f7 = rolling_f7.loc[start_ts:end_ts]
rolling_f8 = rolling_f8.loc[start_ts:end_ts]
# rolling_f9 = rolling_f9.loc[start_ts:end_ts]
# rolling_f10 = rolling_f10.loc[start_ts:end_ts]
# rolling_f11 = rolling_f11.loc[start_ts:end_ts]


trading_signal_dict = {
    "alpha1": rolling_f0.rank(axis=1,pct=True),
    "alpha2": rolling_f1.rank(axis=1,pct=True),
    "alpha3": rolling_f2.rank(axis=1,pct=True),
    "alpha4": rolling_f3.rank(axis=1,pct=True),
    "alpha5": rolling_f4.rank(axis=1,pct=True),
    "alpha6": rolling_f5.rank(axis=1,pct=True),
    "alpha7": rolling_f6.rank(axis=1,pct=True),
    "alpha8": rolling_f7.rank(axis=1,pct=True),
    "alpha9": rolling_f8.rank(axis=1,pct=True),
}

Strategy = AlphaStrategy("/home/yjsun/alpha_backtest/MarkStrategy/StrategyConfig_AlphaTradingRobot_v4.ini")

Simulator = abt.TradingSimulator(Strategy, spot_data_dict, future_data_dict, funding_data, trading_signal_dict = trading_signal_dict, TradingParams={"FeeRate":0.0002,"FutureBal": {"USDT":1000000},"freq":5,"start_ts":start_ts},)
Simulator.run()
# ds.save_pickle(Simulator.RealizedPnl,"testPnl","/home/yjsun/")
