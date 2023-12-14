import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams,gridspec,ticker
from datetime import datetime, timedelta, timezone
import logging
from logging.handlers import TimedRotatingFileHandler
from . import __config__ as cfg
import pandas as pd
import sys
sys.path.append(cfg.DS_PACKAGE_PATH)
import DataServer as ds



def MyLogger(write = True, write_folder_path = "logs"):
    """
    记录的时候都用 mylogger.critical(msg) warning(msg)的形式
    :param write:
    :param write_folder_path:
    :return:
    """
    logger = logging.getLogger()
    LOG_FORMAT = logging.Formatter("%(asctime)s - %(process)d - %(message)s")

    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    ch = logging.StreamHandler()
    ch.setFormatter(LOG_FORMAT)
    ch.setLevel(logging.CRITICAL)
    logger.addHandler(ch)

    if write:
        # 可以分时间段记录
        if not os.path.exists(write_folder_path):
            os.makedirs(write_folder_path)
        fh = TimedRotatingFileHandler(f'{write_folder_path}/log', when='D', interval=1 )
        fh.setFormatter(LOG_FORMAT)
        logger.addHandler(fh)

    return logger


def save_pickle(obj, fileName, folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    filePath = os.path.join(folderPath, fileName)
    with open(filePath, 'wb') as file:
        pickle.dump(obj, file)

    print(f"{filePath} saved")
    pass


def convert_timestamp_to_utctime(ts):
    """
    将交易所时间戳转为utc+0:00时间，注意北京时间是utc+8:00
    :param ts:
    :return:
    """
    if np.isnan(ts):
        local_time = ts
    else:
        local_time = datetime.utcfromtimestamp(int(ts)/1000).strftime('%Y-%m-%d %H:%M:%S')
    return local_time

def convert_timestamp_to_utctime_all(ts,type="full"):
    """
    将交易所时间戳转为utc+0:00时间，注意北京时间是utc+8:00
    :param ts:
    :return:
    """
    if np.isnan(ts):
        local_time = ts
    else:
        local_time = datetime.utcfromtimestamp(int(ts)/1000).strftime('%Y%m%d%H%M%S')
        if type == "full":
            local_time = local_time
        elif type == "month":
            local_time = local_time[0:6]
        elif type == "day":
            local_time = local_time[0:8]
        elif type == "min":
            local_time = str(int(local_time[10:12]) + int(local_time[8:10])*60)
        elif type == "hour":
            local_time = local_time[8:10]
    return local_time


def convert_utctime_to_timestamp(Year = 2021, Month = 1, Day = 1, Hour =0, Min =0, Sec =0):
    """
    将utc+0:00时间转换为交易所时间戳
    :param ts:
    :return:
    """
    dt = datetime(Year, Month, Day, Hour, Min, Sec)
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp() * 1000
    return timestamp


def convert_date_to_timestamp(date):
    return convert_utctime_to_timestamp(Year=int(date[:4]),Month=int(date[4:6]),Day=int(date[6:8]))


def get_prev_date(date,count = 1):
    return (datetime.strptime(date, '%Y%m%d') - timedelta(days=count)).strftime( '%Y%m%d')


def get_next_date(date,count = 1):
    return (datetime.strptime(date, '%Y%m%d') + timedelta(days=count)).strftime( '%Y%m%d')


def get_date_range(start_date:str, end_date:str):
    """
    获取起始日期到结束日期中间所有日期
    :param start_date:
    :param end_date:
    :return:
    """
    start_date = datetime(int(start_date[:4]),int(start_date[4:6]),int(start_date[6:]))
    end_date = datetime(int(end_date[:4]),int(end_date[4:6]),int(end_date[6:]))
    delta = end_date - start_date  # as timedelta
    days = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
    days = [x.strftime( '%Y%m%d') for x in days]
    return days


def slice_dict_of_df(dict_of_df:dict,iloc_start,iloc_end):
    """
    对dict of df 进行时序slice
    :param dict_of_df:
    :param iloc_start:
    :param iloc_end:
    :return:
    """
    res = {}
    for key in dict_of_df.keys():
        res[key] = dict_of_df[key].iloc[iloc_start:iloc_end]
    return res


def selCol_dict_of_dict(dict_of_df:dict, columns):
    """
    对dict of df 进行columns筛选
    :param dict_of_df:
    :param columns:
    :return:
    """
    res = {}
    for key in dict_of_df.keys():
        res[key] = dict_of_df[key][columns]
    return res


def CalUnRealizedPnl(Strategy, LatestQuotes, quote_coin):
    """
    计算策略的未平仓收益
    对于处于Quotes价格为na的票，pnl作为0处理
    :param LatestQuotes:
    :return:
    """
    symbols = [x for x in Strategy.future_position.keys() if Strategy.ContractInfo[x]["quote"] == quote_coin]
    long_pnl = np.nansum([Strategy.future_position[x]["LONG"]["pa"] * (LatestQuotes[x] - Strategy.future_position[x]["LONG"]["ep"]) for x in symbols])
    short_pnl = np.nansum([np.abs(Strategy.future_position[x]["SHORT"]["pa"]) * (Strategy.future_position[x]["SHORT"]["ep"] - LatestQuotes[x]) for x in symbols])

    return long_pnl + short_pnl


def CalUsedMargin(Strategy, LatestQuotes, quote_coin):
    """
    计算策略已经使用的保证金
    :param Strategy:
    :param LatestQuotes:
    :return:
    """
    symbols = [x for x in Strategy.future_position.keys() if Strategy.ContractInfo[x]["quote"] == quote_coin]
    long_margin = np.nansum([Strategy.future_position[x]["LONG"]["pa"] * LatestQuotes[x] for x in symbols]) / Strategy.leverage
    short_margin = np.nansum([np.abs(Strategy.future_position[x]["SHORT"]["pa"]) * LatestQuotes[x] for x in symbols]) / Strategy.leverage

    return long_margin + short_margin


def CalAvailableBal(Strategy, LatestQuotes, quote_coin):
    """
    根据仓位，账户余额，以及最新报价计算可用资金
    可用余额 = 账户余额 - 仓位已经使用保证金 + 未实现利润
    """
    Balance = Strategy.future_balance[quote_coin]
    UnPnl = CalUnRealizedPnl(Strategy, LatestQuotes, quote_coin)
    UsedMargin = CalUsedMargin(Strategy, LatestQuotes, quote_coin)
    AvailableBal = Balance + UnPnl - UsedMargin
    return AvailableBal


def CalSpotAccValue(Strategy, LatestQuotes, quote_coin):
    """
    根据持有现货，计算账户
    :param Strategy:
    :param LatestQuotes:
    :param quote_coin:
    :return:
    """
    Position = Strategy.balance
    res = []
    for symbol in Strategy.spot_symbol_list:
        # 判断是否base coin是否一致
        base_coin = Strategy.SpotInfo[symbol]["base"]
        if Strategy.SpotInfo[symbol]["quote"] == quote_coin:
            res.append((Position[base_coin][0] + Position[base_coin][1]) * LatestQuotes[symbol])

    return np.nansum(res) + np.nansum(Strategy.balance[quote_coin])



def get_prepared_kline_from_basic_data(start_date, end_date, spot_symbol_list = None, future_symbol_list = None, basic_data_path = None, exchange = "bn"):
    """
    根据起止时间选出相关k线数据
    :param start_date:
    :param end_date:
    :param basic_data_path:
    :return:
    """
    if basic_data_path is None:
        basic_data_path = cfg.BASIC_DATA_PATH

    if exchange == "bn":
        spot_data_dict = pd.read_pickle(os.path.join(basic_data_path, f"spot_data_dict.pkl"))
        future_data_dict = pd.read_pickle(os.path.join(basic_data_path, f"future_data_dict.pkl"))
    else:
        if exchange in ["bn"]:
            spot_data_dict = pd.read_pickle(os.path.join(basic_data_path, f"spot_data_dict_{exchange}.pkl"))
        else:
            spot_data_dict = None
        future_data_dict = pd.read_pickle(os.path.join(basic_data_path, f"future_data_dict_{exchange}.pkl"))

    start_time = ds.convert_date_to_timestamp(start_date)
    data_start_time = ds.convert_date_to_timestamp(start_date) - 24 * 60 * 60000
    end_time = ds.convert_date_to_timestamp(end_date) + 24 * 60 * 60000 - 1

    if exchange in ["bn"]:
        if spot_symbol_list is None:
            spot_symbol_list = spot_data_dict["open_time"].columns.tolist()
        for key in spot_data_dict.keys():
            spot_data_dict[key] = spot_data_dict[key].loc[data_start_time:end_time, spot_symbol_list]

        if future_symbol_list is None:
            future_symbol_list = future_data_dict["open_time"].columns.tolist()
        for key in future_data_dict.keys():
            future_data_dict[key] = future_data_dict[key].loc[data_start_time:end_time, future_symbol_list]

    future_data_dict['trade_value_sum30min'] = future_data_dict["trade_value"].rolling(30, min_periods=1).sum()
    future_data_dict['trade_value_sum5min'] = future_data_dict["trade_value"].rolling(5, min_periods=1).sum()

    future_data_dict['Rtn1'] = (future_data_dict["close"] / future_data_dict["close"].shift(1) - 1)
    future_data_dict['Rtn1'] = future_data_dict['Rtn1'].where(future_data_dict['Rtn1'] < 1, 0)
    future_data_dict['Rtn5'] = (future_data_dict["close"] / future_data_dict["close"].shift(5) - 1)
    future_data_dict['Rtn1440'] = (future_data_dict["close"] / future_data_dict["close"].shift(1440) - 1)
    index = (future_data_dict['Rtn1'].mean(axis=1).fillna(0)+1).cumprod()
    tmp = pd.DataFrame((index.rolling(1440).max() - index.rolling(1440).min()) / ((index.rolling(1440).max() + index.rolling(1440).min())/2))
    future_data_dict['Market24HVol'] = pd.concat([tmp for x in range(len(future_symbol_list))],axis=1)
    future_data_dict['Market24HVol'].columns = future_symbol_list

    tmp = index / index.shift(1440) - 1
    future_data_dict['Market24HRtn'] = pd.concat([tmp for x in range(len(future_symbol_list))],axis=1)
    future_data_dict['Market24HRtn'].columns = future_symbol_list

    future_data_dict['Vol24H'] = (future_data_dict["high"].rolling(1440).max() - future_data_dict["low"].rolling(1440).min()) / ((future_data_dict["high"].rolling(1440).max() + future_data_dict["low"].rolling(1440).min())/2)

    if exchange in ["bn"]:
        for key in spot_data_dict.keys():
            spot_data_dict[key] = spot_data_dict[key].loc[start_time:end_time, spot_symbol_list]

        for key in future_data_dict.keys():
            future_data_dict[key] = future_data_dict[key].loc[start_time:end_time, future_symbol_list]

    return spot_data_dict, future_data_dict


def read_trading_log(LogPath, my_pid=None):
    curpid = 0
    #     my_pid = None
    res1 = []
    res2 = []
    res3 = []
    res4 = []
    res5 = []
    res6 = []
    with open(LogPath) as file:
        for line in file.readlines():
            split_res = line.split("\n")[0].split(" - ")
            if (len(split_res) == 3):
                pid = int(split_res[1])
                if my_pid is None:
                    if pid != curpid:
                        res1 = []
                        res2 = []
                        res3 = []
                        res4 = []
                        res5 = []
                        curpid = pid
                else:
                    if my_pid != pid:
                        continue
                msg = split_res[2]
                if msg[:9] == "SimAccMsg":
                    tmp_dict = {}
                    for x in msg[10:].split("#"):
                        tmp_res = x.split(":")
                        tmp_dict[tmp_res[0]] = tmp_res[1]
                        tmp_dict["pid"] = pid
                    res1.append(tmp_dict)

                if msg[:9] == "SimEvtMsg":
                    tmp_dict = {}
                    for x in msg[10:].split("#"):
                        tmp_res = x.split(":")
                        tmp_dict[tmp_res[0]] = tmp_res[1]
                        tmp_dict["pid"] = pid
                    if tmp_dict["EvtType"] in ["OpenLong", "CloseLong", "OpenShort", "CloseShort"]:
                        res2.append(tmp_dict)

                if msg[:8] == "StyleMsg":
                    tmp_dict = {}
                    for x in msg[9:].split("#"):
                        tmp_res = x.split(":")
                        tmp_dict[tmp_res[0]] = tmp_res[1]
                        tmp_dict["pid"] = pid
                    res3.append(tmp_dict)

                if msg[:7] == "RiskMsg":
                    tmp_dict = {}
                    for x in msg[8:].split("#"):
                        tmp_res = x.split(":")
                        tmp_dict[tmp_res[0]] = tmp_res[1]
                        tmp_dict["pid"] = pid
                    res4.append(tmp_dict)

                if (msg[:6] == "AlpMsg") and (("StopProfit" in msg) or ("StopLoss" in msg)):
                    tmp_dict = {}
                    for x in msg[7:].split("#"):
                        tmp_res = x.split(":")
                        tmp_dict[tmp_res[0]] = tmp_res[1]
                        tmp_dict["pid"] = pid
                    res5.append(tmp_dict)

                if (msg[:6] == "AlpMsg") and ("ChgPos" in msg):
                    tmp_dict = {}
                    for x in msg[7:].split("#"):
                        tmp_res = x.split(":")
                        tmp_dict[tmp_res[0]] = tmp_res[1]
                        tmp_dict["pid"] = pid
                    res6.append(tmp_dict)

    Accdf = pd.DataFrame(res1)
    Accdf["AccNetVal"] = Accdf["AccNetVal"].astype(float)
    Capital_t0 = Accdf.iloc[0]["AccNetVal"]

    Accdf["LongVal"] = Accdf["LongVal"].astype(float)
    Accdf["LongMax"] = Accdf["LongMax"].astype(float)
    Accdf["LongMaxRatio"] = Accdf["LongMax"] / (Capital_t0 /2)
    Accdf["ShortVal"] = Accdf["ShortVal"].astype(float)
    Accdf["ShortMax"] = Accdf["ShortMax"].astype(float)
    Accdf["ShortMaxRatio"] = Accdf["ShortMax"] / (Capital_t0 /2)
    Accdf["ts"] = Accdf["ts"].astype(int)
    Accdf["Date"] = Accdf["ts"].apply(lambda x: ds.convert_timestamp_to_utctime(x))


    Tradedf = pd.DataFrame(res2)
    Tradedf['TS'] = Tradedf['TS'].astype(int)
    Tradedf['Price'] = Tradedf['Price'].astype(float)
    Tradedf['Quantity'] = Tradedf['Quantity'].astype(float)
    Tradedf['Val'] = Tradedf['Price'] * Tradedf['Quantity']
    Tradedf['Date'] = Tradedf['TS'].apply(lambda x: ds.convert_timestamp_to_utctime(x)[:10])

    Chgposdf = pd.DataFrame(res6)
    Chgposdf['TS'] = Chgposdf['TS'].astype(int)
    Chgposdf['Vol'] = Chgposdf['Vol'].astype(float)
    Chgposdf['Quotes'] = Chgposdf['Quotes'].astype(float)
    Chgposdf['Val'] = Chgposdf['Vol']* Chgposdf['Quotes']
    Chgposdf['Predict'] = Chgposdf['Predict'].astype(float)
    Chgposdf['Gap'] = Chgposdf['Gap'].astype(float)

    try:
        Styledf = pd.DataFrame(res3)
        Styledf["TS"] = Styledf["TS"].astype(int)
        Styledf["24Hdd"] = Styledf["24Hdd"].astype(np.float64)
        Styledf["1Hdd"] = Styledf["1Hdd"].astype(np.float64)
        Styledf["MarketRtn"] = Styledf["MarketRtn"].astype(np.float64)
        Styledf["MarketMedianVol"] = Styledf["MarketMedianVol"].astype(np.float64)
        Styledf["LongVol24H"] = Styledf["LongVol24H"].astype(np.float64)
        Styledf["ShortVol24H"] = Styledf["ShortVol24H"].astype(np.float64)
        Styledf["LongMOM24H"] = Styledf["LongMOM24H"].astype(np.float64)
        Styledf["ShortMOM24H"] = Styledf["ShortMOM24H"].astype(np.float64)

    except:
        Styledf = pd.DataFrame()
    if res4:
        Riskdf = pd.DataFrame(res4)
        Riskdf["TS"] = Riskdf["TS"].astype(int)

    if res4:
        Other = {
            "Risk":Riskdf,
            "Style":Styledf,
            "Stop":pd.DataFrame(res5)
        }
    else:
        Other = {
            "Style":Styledf,
            "Stop": pd.DataFrame(res5),
            "Chg":Chgposdf,
        }
    return Capital_t0, Accdf, Tradedf, Other


def plot_strategy_backtest(Capital_t0, Accdf, Tradedf, Other):
    endTS = Accdf['ts'].iloc[-1] - 60000 * 60
    Accdf = Accdf.loc[Accdf['ts']<endTS].copy()
    Tradedf = Tradedf.loc[Tradedf['TS']<endTS].copy()

    TotalTrade = Tradedf.groupby("Date").agg({"Val": "sum"})
    freq = Accdf["ts"].diff().median() / 60000
    MaxDD = (((Accdf["AccNetVal"].cummax() - Accdf["AccNetVal"])).cummax() / Capital_t0).iloc[-1]
    rtn = Accdf["AccNetVal"].diff() / Capital_t0
    AnnRtn = rtn.mean() * (1440 / freq * 360)
    DailyStd = rtn.std() * (1440 / freq) ** 0.5
    Sharp = rtn.mean() / rtn.std() * (1440 / freq * 360) ** 0.5
    TvrRate = TotalTrade["Val"].mean() / 2 / Capital_t0
    Accdf["day"] = Accdf["Date"].map(lambda x:x[:10])
    DaySampleNum = int(1440 / freq)
    Accdf["24Hdd"] = (Accdf["AccNetVal"].rolling(DaySampleNum).max() - Accdf["AccNetVal"]) / Capital_t0
    Accdf["1Hdd"] = (Accdf["AccNetVal"].rolling(int(DaySampleNum/24)).max() - Accdf["AccNetVal"]) / Capital_t0
    Accdf['mth'] = Accdf['Date'].map(lambda x: x[:7])

    tmp_dd = Accdf.groupby("day").agg({"24Hdd": "max","1Hdd":"max"})
    dd24h_001 = (tmp_dd["24Hdd"] > 0.01).mean()
    dd24h_0015 = (tmp_dd["24Hdd"] > 0.015).mean()
    dd24h_002 = (tmp_dd["24Hdd"] > 0.02).mean()
    dd24h_qt50 = tmp_dd["24Hdd"].quantile(0.5)
    dd24h_qt95 = tmp_dd["24Hdd"].quantile(0.95)
    dd24h_qt99 = tmp_dd["24Hdd"].quantile(0.99)

    dailynv = Accdf.groupby("day").agg({"AccNetVal": "last"})
    WinRatio = (dailynv["AccNetVal"] > dailynv["AccNetVal"].shift()).mean()
    dd_list = ((dailynv > dailynv.shift(-1)) * 1).iloc[:, 0].tolist()
    res = []
    cur_dd = 0
    for idx, tmp in enumerate(dd_list):
        if idx == 0:
            continue
        if tmp == 0 and dd_list[idx - 1] == 1:
            cur_dd = 0
        elif tmp == 1:
            cur_dd += 1
        res.append(cur_dd)
    consecutive_dd = np.nanmax(res)
    idx = np.nanargmax(res)
    dd_time = dailynv.index.tolist()[idx-consecutive_dd]

    # 计算风格暴露
    df = Other['Style']
    voldiff = df["LongVol24H"] - df["ShortVol24H"]
    momdiff = df["LongMOM24H"] - df["ShortMOM24H"]
    LSVolStyle = voldiff.mean() / voldiff.abs().std()
    LSMomStyle = momdiff.mean() / momdiff.abs().std()

    # 计算按月收益率
    mthDf = Accdf.groupby("mth").agg({"AccNetVal": ["first", "last", "count"]})
    mthDf.columns = ["first", "last", "count"]
    mthDf['DateNum'] = mthDf["count"] // 1440
    mthDf = mthDf[mthDf['count'] > 5]
    mthDf['Rtn'] = (mthDf['last'] - mthDf['first']) / Capital_t0

    res = {
        "AnnRtn": AnnRtn,
        "DailyStd": DailyStd,
        "Sharp": Sharp,
        "TvrRate": TvrRate,
        "WinRatio": WinRatio,
        "MaxDD": MaxDD,
        "Max24HDD":tmp_dd["24Hdd"].max(),
        "Max24HDDTime":tmp_dd["24Hdd"][tmp_dd["24Hdd"] == tmp_dd["24Hdd"].max()].index.tolist()[0],
        "24hdd_001":dd24h_001,
        "24hdd_0015":dd24h_0015,
        "24hdd_002":dd24h_002,
        "dd24h_qt50":dd24h_qt50,
        "24hdd_qt95":dd24h_qt95,
        "24hdd_qt99":dd24h_qt99,
        "Max1HDD": tmp_dd["1Hdd"].max(),
        "Max1HDDTime": tmp_dd["1Hdd"][tmp_dd["1Hdd"] == tmp_dd["1Hdd"].max()].index.tolist()[0],
        "LSVolStyle":LSVolStyle,
        "LSMomStyle":LSMomStyle,
        "consecutive_dd":consecutive_dd,
        "consecutive_dd_day":dd_time,
        "MthReport":mthDf[['DateNum','Rtn']].to_dict()
    }
    print(res)
    print("Drawdown 0.025 dates",sorted(list(set(tmp_dd["24Hdd"][tmp_dd["24Hdd"] > 0.025].index.tolist()))))

    fig = plt.figure(figsize=(25, 50), dpi=128)
    fig.set_facecolor('#FFFFFF')
    plt.subplots_adjust(wspace=1, hspace=0.25)
    spec = gridspec.GridSpec(ncols=12, nrows=8)
    # Pic 1
    ax = fig.add_subplot(spec[0, 0:12])
    plt.plot(Accdf["Date"], (Accdf["AccNetVal"] / Capital_t0))
    ax_num = np.maximum(int(Accdf.shape[0] / 10),1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ax_num))
    ax.set_title('策略净值曲线', size=15)

    # Pic2 a
    ax = fig.add_subplot(spec[1, 0:6])
    plt.plot(Accdf["Date"], Accdf["LongVal"])
    plt.plot(Accdf["Date"], -Accdf["ShortVal"])
    ax_num = np.maximum(int(Accdf.shape[0] / 5),1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ax_num))
    ax.set_title('多空持仓名义金额', size=15)

    # Pic2 b
    ax = fig.add_subplot(spec[1, 6:12])
    plt.plot(Accdf["Date"], Accdf["LongMaxRatio"])
    plt.plot(Accdf["Date"], Accdf["ShortMaxRatio"])
    ax_num = np.maximum(int(Accdf.shape[0] / 5),1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ax_num))
    ax.set_title('最大持仓比例', size=15)

    ax = fig.add_subplot(spec[2, 0:12])
    plt.plot(TotalTrade.index, TotalTrade / 2 / Capital_t0)
    ax.set_title('交易换手率', size=15)
    ax_num = np.maximum(int(TotalTrade.shape[0] / 10),1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(ax_num))

    ax = fig.add_subplot(spec[3, 0:12])
    mth_index = mthDf.index.tolist()
    y = mthDf["Rtn"].tolist()
    x = np.arange(len(mth_index))  # the label locations
    width = 0.25
    rects = ax.bar(x,y,width)
    ax.bar_label(rects, padding=3)
    ax.set_xticks(x, mth_index)
    # plt.plot(mthDf["Rtn"])
    ax.set_title('按月收益率', size=15)

    plt.show()

    return res


def decompose_strategy_pnl(data_path, close, Capital_t0):
    recordTS = pd.read_pickle(f"{data_path}/RecordingTS.pkl")[1:]
    holdingVol = pd.read_pickle(f"{data_path}HoldingVol.pkl")

    long_vol = pd.DataFrame(data=holdingVol["LONG"], index=recordTS).iloc[:-1]
    short_vol = pd.DataFrame(data=holdingVol["SHORT"], index=recordTS).iloc[:-1]

    rtn = close.pct_change().iloc[1:]
    diff = close.diff().iloc[1:]

    index = (rtn.mean(axis=1).fillna(0) + 1).cumprod() - 1

    ((long_vol * diff).sum(axis=1) / Capital_t0).cumsum().plot(figsize=(15, 5))
    index.plot(figsize=(15, 5), color="gray")
    plt.show()

    (((long_vol * diff).sum(axis=1) / Capital_t0).cumsum() - index).plot(figsize=(15, 5))
    plt.show()

    ((short_vol * diff).sum(axis=1) / Capital_t0).cumsum().plot(figsize=(15, 5))
    (-index).plot(figsize=(15, 5), color="gray")
    plt.show()

    (index + ((short_vol * diff).sum(axis=1) / Capital_t0).cumsum()).plot(figsize=(15, 5))
    plt.show()


# cal market ratio
def cal_each_signal(Tradedf, symbol, TradeTime, future_data_dict, direction):
    # 对交易进行合并的规则，如果这次交易后的TradeTime中还有交易，则后续交易并入这次交易中
    if direction == "buy":
        tmpTradePlan = Tradedf[(Tradedf["Symbol"] == symbol) & (Tradedf['EvtType'].map(lambda x: x in ["OpenLong","CloseShort"]))]
    elif direction == "sell":
        tmpTradePlan = Tradedf[(Tradedf["Symbol"] == symbol) & (Tradedf['EvtType'].map(lambda x: x in ["OpenShort","CloseLong"]))]

    tmp_group = 0
    group_data = []
    for idx in range(tmpTradePlan.shape[0]):
        if idx == 0:
            group_data.append(tmp_group)
            continue
        now_col = tmpTradePlan.iloc[idx]
        last_col = tmpTradePlan.iloc[idx - 1]

        now_order_ts = now_col["TS"]
        last_order_ts = last_col["TS"]
        if (now_order_ts - last_order_ts) > TradeTime * 60000:
            tmp_group += 1
        group_data.append(tmp_group)

    tmpTradePlan["group"] = group_data
    start_ts = tmpTradePlan.groupby("group").agg({"TS": "first"})
    end_ts = tmpTradePlan.groupby("group").agg({"TS": "last"}) + TradeTime * 60000
    planed_val = tmpTradePlan.groupby("group").agg({"Val": "sum"})
    resList = []
    for idx in range(start_ts.shape[0]):
        tmp_start_ts = start_ts.iloc[idx, 0]
        tmp_end_ts = end_ts.iloc[idx, 0]
        tmp_planed_val = planed_val.iloc[idx, 0]
        market_val = future_data_dict["trade_value"][symbol].loc[tmp_start_ts:tmp_end_ts].sum()
        prev_market_val = future_data_dict["trade_value"][symbol].loc[(tmp_end_ts - 2* TradeTime * 60000):tmp_end_ts].sum()
        resList.append({"symbol": symbol, "planed_val": tmp_planed_val,"prev_market_val":prev_market_val, "market_val": market_val, "start_ts": tmp_start_ts,"end_ts":tmp_end_ts})

    tmp_df = pd.DataFrame(resList)
    if tmp_df.empty:
        tmp_df = pd.DataFrame(columns=["symbol", "planed_val","prev_market_val", "market_val", "start_ts","end_ts", "ratio"])
    tmp_df["ratio"] = tmp_df["planed_val"] / tmp_df["market_val"]
    tmp_df['prev_ratio'] = tmp_df['planed_val'] / tmp_df['prev_market_val']
    tmp_df["direction"] = direction
    tmp_df = tmp_df[tmp_df["market_val"] > 1]
    return tmp_df


def cal_trade_market_ratio(trading_df,future_tickers, future_data_dict, TradeTime=30):
    resList = []
    for ticker in future_tickers:
        tmp_df = cal_each_signal(trading_df, ticker, TradeTime, future_data_dict,"buy")
        resList.append(tmp_df)
        tmp_df = cal_each_signal(trading_df, ticker, TradeTime, future_data_dict,"sell")
        resList.append(tmp_df)

    df = pd.concat(resList)
    market_ratio_mean = (df["ratio"] * df["planed_val"]).sum() / df["planed_val"].sum()
    market_ratio_qt95 = df["ratio"].quantile(0.95)
    return market_ratio_mean, market_ratio_qt95, df


def plot_factor_distribution(f, not_track_symbols, close):
    track_symbols = [x for x in f.columns if x not in not_track_symbols]
    factor_df = f[track_symbols]

    fig = plt.figure(figsize=(25, 50), dpi=128)
    fig.set_facecolor('#FFFFFF')
    plt.subplots_adjust(wspace=1, hspace=0.25)
    spec = gridspec.GridSpec(ncols=12, nrows=8)

    # 按票排序
    ax = fig.add_subplot(spec[0, 0:12])
    symbol_mean = factor_df.mean().sort_values()
    symbol_mean.plot()
    plt.title("Factor Mean GroupBy Symbol")

    ax = fig.add_subplot(spec[1, 0:12])
    symbol_std = factor_df.std().sort_values()
    symbol_std.plot()
    plt.title("Factor Std GroupBy Symbol")

    # 按月统计分位数
    mth_index = [ds.convert_timestamp_to_utctime(x)[:7] for x in factor_df.index]
    mth_list = list(set(mth_index))
    resList1 = []
    resList2 = []
    index_list = []
    for mth in mth_list:
        select = [True if x == mth else False for x in mth_index]
        tmp_df = factor_df.loc[select]
        tmp_array = tmp_df.values.reshape(-1)
        qt_list = [0,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1]
        stats = np.nanquantile(tmp_array,qt_list)
        res = {}
        for idx in range(len(qt_list)):
            res[f"qt_{qt_list[idx]}"] = stats[idx]
        res["mean"] = np.nanmean(tmp_array)
        res["std"] = np.nanstd(tmp_array)
        res["mth"] = mth
        resList1.append(res)

        tmp_ts = tmp_df.mean(axis=1)
        tmp_ts = tmp_ts.groupby((tmp_ts.index // 60000) % 1440).mean()
        resList2.append(tmp_ts)
        index_list.append(mth)

    qt_df = pd.DataFrame(resList1).set_index("mth").sort_index()
    ax = fig.add_subplot(spec[2, 0:12])
    sub_col = ['qt_0.01','qt_0.05','qt_0.1','qt_0.25','qt_0.5','qt_0.75','qt_0.9','qt_0.95','qt_0.99']
    ax.plot(qt_df[sub_col].T,label = qt_df.index, marker = ".")
    plt.legend()
    plt.title("Factor Quantile GroupBy Month")

    ax = fig.add_subplot(spec[3, 0:12])
    sub_col = ['mean','std']
    ax.plot(qt_df[sub_col],label = sub_col)
    plt.legend()
    plt.title("Factor Mean Std GroupBy Month")

    total_std = np.nanstd(factor_df.values.reshape(-1))
    inday_mean = pd.concat(resList2,axis=1)
    inday_mean.columns = index_list
    ax = fig.add_subplot(spec[4, 0:12])
    ax.plot(inday_mean,label = inday_mean.columns)
    plt.hlines(total_std,xmin=inday_mean.index.min(),xmax=inday_mean.index.max(),label="std",color="gray",linestyle="--")
    plt.hlines(-total_std,xmin=inday_mean.index.min(),xmax=inday_mean.index.max(),label="-std",color="gray",linestyle="--")
    plt.legend()
    plt.title("Factor Inday Mean")

    # 按照截面波动统计
    if close is not None:
        rolling_max = close.rolling(60).max()
        rolling_min = close.rolling(60).min()
        vol = (rolling_max - rolling_min) / ((rolling_max + rolling_min)/2)
        vol = vol.loc[factor_df.index.min():factor_df.index.max()][factor_df.columns]
        vol = vol.subtract(vol.mean(axis=1),axis=0).divide(vol.std(axis=1),axis=0)
        tmp_df = pd.DataFrame(data = {"value":factor_df.values.reshape(-1), "vol":vol.values.reshape(-1)})
        tmp_df = tmp_df[~tmp_df['vol'].isnull()]
        tmp_df['vol_rank'] = (tmp_df['vol'].rank(pct=True,axis=0).clip(upper=0.999)*100).astype(int)
        tmp_df['value_abs'] = tmp_df['value'].abs()

        ax = fig.add_subplot(spec[5, 0:12])
        res = tmp_df.groupby("vol_rank").agg({"value_abs":"mean","value":"mean"})
        ax.plot(res["value_abs"],label="value_abs")
        ax.plot(res["value"],label="value")
        plt.title("Factor Groupby CrossVol")

    plt.show()
    pass