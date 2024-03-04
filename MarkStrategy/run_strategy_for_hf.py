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
from AlphaStrategy_v4_4 import AlphaStrategy

# 准备Funding数据
start_date = "20240101"
end_date = "20240225"
spot_data_dict, future_data_dict = abt.get_prepared_kline_from_basic_data(start_date,end_date, basic_data_path = "/mnt2/alpha_data/basic_data_AlphaFMV3")
funding_data = future_data_dict['fundingRate']
del future_data_dict['fundingRate']

twap_data_dict = ds.get_min_feature(feature_list =
                            [
                            'm1_future_forward_twap_5min',
                            # 'm1_future_tradingval_maskv2_b7200g1440p80t20',
                            'm1_future_tradingval_maskv2_b7200g1440p40t20',
                            ]
                            , where = "/mnt2/alpha_data/features_AlphaFMV3" ,data_tag = "AlphaFM", start_date = start_date, end_date = end_date)
future_data_dict.update(twap_data_dict)

YrMthList = ["202401","202402"]
# YrMthList = ["202310","202311","202312","202401"]
res0 = []
res1 = []
res2 = []
res3 = []
for Mth in YrMthList:
    res0.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f10m5normed2880_m12n300l0.05_roll1_{Mth}.pkl"))
    res1.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f20m5normed2880_m12n300l0.05_roll1_{Mth}.pkl"))
    res2.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f30m5normed2880_m12n300l0.05_roll1_{Mth}.pkl"))
    res3.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f60m5normed2880_m12n300l0.05_roll1_{Mth}.pkl"))
rolling_f0 = pd.concat(res0,axis=0)
rolling_f1 = pd.concat(res1,axis=0)
rolling_f2 = pd.concat(res2,axis=0)
rolling_f3 = pd.concat(res3,axis=0)

res0 = []
res1 = []
res2 = []
res3 = []
for Mth in YrMthList:
    res0.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f10m5normed2880_m12n300l0.05_roll2_{Mth}.pkl"))
    res1.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f20m5normed2880_m12n300l0.05_roll2_{Mth}.pkl"))
    res2.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f30m5normed2880_m12n300l0.05_roll2_{Mth}.pkl"))
    res3.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f60m5normed2880_m12n300l0.05_roll2_{Mth}.pkl"))
rolling_f4 = pd.concat(res0,axis=0)
rolling_f5 = pd.concat(res1,axis=0)
rolling_f6 = pd.concat(res2,axis=0)
rolling_f7 = pd.concat(res3,axis=0)



YrMthList = ["202401A","202401B","202401C","202401D","202402A","202402B","202402C","202402D"]
# YrMthList = ["202310A","202310B","202310C","202310D","202311A","202311B","202311C","202311D","202312A","202312B","202312C","202312D","202401A","202401B","202401C"]
res0 = []
res1 = []
res2 = []
res3 = []
for Mth in YrMthList:
    res0.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f10m5normed2880_m10n300l0.01_rollw_{Mth}.pkl"))
    res1.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f20m5normed2880_m10n300l0.01_rollw_{Mth}.pkl"))
    res2.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f30m5normed2880_m10n300l0.01_rollw_{Mth}.pkl"))
    res3.append(pd.read_pickle(f"/mnt2/alpha_data/bn_v6_factor/xgb_predict_future_FMV15HF_f60m5normed2880_m10n300l0.01_rollw_{Mth}.pkl"))

rolling_f8 = pd.concat(res0,axis=0)
rolling_f9 = pd.concat(res1,axis=0)
rolling_f10 = pd.concat(res2,axis=0)
rolling_f11 = pd.concat(res3,axis=0)



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
rolling_f9 = rolling_f9.loc[start_ts:end_ts]
rolling_f10 = rolling_f10.loc[start_ts:end_ts]
rolling_f11 = rolling_f11.loc[start_ts:end_ts]

trading_signal_dict = {
    "alpha1": rolling_f0,
    "alpha2": rolling_f1,
    "alpha3": rolling_f2,
    "alpha4": rolling_f3,
    "alpha5": rolling_f4,
    "alpha6": rolling_f5,
    "alpha7": rolling_f6,
    "alpha8": rolling_f7,
    "alpha9": rolling_f8,
    "alpha10": rolling_f9,
    "alpha11": rolling_f10,
    "alpha12": rolling_f11,
}


Strategy = AlphaStrategy("/home/yjsun/alpha_backtest/MarkStrategy/StrategyConfig_AlphaTradingRobot_hf.ini")

Simulator = abt.TradingSimulator(Strategy, spot_data_dict, future_data_dict, funding_data, trading_signal_dict = trading_signal_dict, TradingParams={"FeeRate":0.000125,"FutureBal": {"USDT":1000000},"freq":1},)
Simulator.run()
# ds.save_pickle(Simulator.RealizedPnl,"testPnl","/home/yjsun/")
