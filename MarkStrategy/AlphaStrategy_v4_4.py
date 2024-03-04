import copy
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from collections import defaultdict
from AlphaBackTest import FutureBaseStrategy, MyLogger
import traceback
import configparser
import re


def next_nonezero_idx(nowIdx, MatchDict, MatchOrder):
    for local_idx in range(nowIdx, len(MatchOrder)):
        if local_idx == nowIdx:
            continue
        if MatchDict[MatchOrder[local_idx]] != 0:
            return local_idx
    return len(MatchOrder)


class AlphaStrategy(FutureBaseStrategy):
    def __init__(self, StrategyConfigPath):
        super().__init__()

        self.ConfigPath = StrategyConfigPath
        self.Parser = configparser.ConfigParser()
        self.Parser.read(self.ConfigPath)
        # 确认日志路径
        self.strategy_tag = str(self.Parser.get(section ="strategy" ,option="strategy_tag"))
        self.strategy_base_path = str(self.Parser.get(section ="strategy" ,option="strategy_base_path"))
        self.strategy_info_path = os.path.join(self.strategy_base_path,self.strategy_tag)
        # 建立日志
        self.STLogger = MyLogger(write=True, write_folder_path=self.strategy_info_path)
        self.exchange = str(self.Parser.get(section ="strategy" ,option="exchange"))

        # 确定票池
        if self.exchange == "bn":
            pattern = re.compile(r'\w{1,10}USDT')
        elif self.exchange == "okex":
            pattern = re.compile(r'\w{1,10}-USDT-SWAP')
        self.future_symbol_list = pattern.findall(self.Parser.get(section ="strategy" ,option="future_symbol_list"))
        self.STLogger.critical(f"Trading Future Target: {self.future_symbol_list}")
        self.spot_symbol_list = pattern.findall(self.Parser.get(section ="strategy" ,option="spot_symbol_list"))
        self.black_symbol_list = pattern.findall(self.Parser.get(section="strategy", option="black_symbol_list"))

        self.LatestMinQuotes = {x: np.nan for x in self.future_symbol_list}
        self.TradingStatusDict = None

        # 设定交易参数
        self.quote_coin = self.Parser.get(section ="strategy" ,option="quote_coin")
        self.TargetHoldingVal = {
            "LONG": float(self.Parser.get(section ="strategy" ,option="TargetHoldingValLONG")),
            "SHORT": float(self.Parser.get(section ="strategy" ,option="TargetHoldingValSHORT"))
        }
        self.ExpUpperBound = float(self.Parser.get(section ="strategy" ,option="ExpBoundUpper"))
        self.ExpLowerBound = float(self.Parser.get(section ="strategy" ,option="ExpBoundLower"))
        self.MaxCapRatio = float(self.Parser.get(section ="strategy" ,option="MaxCapRatio"))
        self.MaxCapRatioDeviation = float(self.Parser.get(section="strategy", option="MaxCapRatioDeviation"))
        self.leverage = int(self.Parser.get(section ="strategy" ,option="leverage"))

        self.TradingRatioUpThres100w = float(self.Parser.get(section="strategy", option="TradingRatioUpThres100w"))
        self.TradingRatioUpThres = (self.TargetHoldingVal["LONG"] + np.abs(self.TargetHoldingVal["SHORT"])) / 1000000 * self.TradingRatioUpThres100w

        self.factor_rank = str(self.Parser.get(section ="strategy" ,option="factor_rank"))
        if self.factor_rank == "True":
            self.factor_rank = True
        else:
            self.factor_rank = False

        # 设定资金使用情况
        self.EachChanceVal = float(self.Parser.get(section ="strategy" ,option="EachChanceValue"))
        self.SmallValToClean = float(self.Parser.get(section ="strategy" ,option="SmallValToClean"))

        # 风控参数
        self.Market24HVolThresIn = float(self.Parser.get(section ="strategy" ,option="Market24HVolThresIn"))
        self.Market24HVolThresOut = float(self.Parser.get(section ="strategy" ,option="Market24HVolThresOut"))
        self.Market24HVolBlack = False

        self.MaxDDBlackMin = float(self.Parser.get(section ="strategy" ,option="MaxDDBlackMin"))
        self.Max24HDDBlackThres = float(self.Parser.get(section ="strategy" ,option="Max24HDDBlackThres"))
        self.Max1HDDBlackThres = float(self.Parser.get(section ="strategy" ,option="Max1HDDBlackThres"))
        self.MaxDDTS = 0
        self.MaxDDBlack = False

        self.Symbol24HVolThresIn = float(self.Parser.get(section ="strategy" ,option="Symbol24HVolThresIn"))
        self.Symbol24HVolThresOut = float(self.Parser.get(section ="strategy" ,option="Symbol24HVolThresOut"))
        self.SymbolBlackThres = int(self.Parser.get(section ="strategy" ,option="SymbolBlackThres"))
        self.Black24HVolSymbols = []
        self.SymbolBlack = False

        # 策略执行参数
        self.TradeTime = float(self.Parser.get(section="strategy", option="TradeTime"))
        self.ThresLongPosChange = float(self.Parser.get(section ="strategy" ,option="ThresLongPosChange"))
        self.ThresShortPosChange = float(self.Parser.get(section ="strategy" ,option="ThresShortPosChange"))
        self.ThresStopLoss = float(self.Parser.get(section ="strategy" ,option="ThresStopLoss"))
        self.ThresStopProfit = float(self.Parser.get(section ="strategy" ,option="ThresStopProfit"))
        self.StopLossTimeInterval = float(self.Parser.get(section ="strategy" ,option="StopLossTimeInterval"))
        self.PredictPath = self.Parser.get(section ="strategy" ,option="PredictPath")
        if (not os.path.exists(self.PredictPath)) and (not self.local_test):
            self.STLogger.critical(f"Predict Path Not Exists {self.PredictPath}")
        self.LatestUpPredictTimeStamp = 0

        self.pool_mask_name = str(self.Parser.get(section ="strategy" ,option="pool_mask_name"))
        self.alpha_name_list = str(self.Parser.get(section ="strategy" ,option="alpha_name_list")).split(',')

        self.alpha_update_freq = int(self.Parser.get(section ="strategy" ,option="alpha_update_freq"))
        self.future_twap_name = str(self.Parser.get(section ="strategy" ,option="future_twap_name"))
        self.detail_log = str(self.Parser.get(section ="strategy" ,option="detail_log"))
        if self.detail_log == "True":
            self.detail_log = True
        else:
            self.detail_log = False

        # 策略记录
        # 交易计划， 全部数值必定为正
        self.TradingPlan = {
            "Open": {"LONG": defaultdict(lambda: defaultdict(np.float64)),"SHORT": defaultdict(lambda: defaultdict(np.float64))},
            "Close": {"LONG": defaultdict(lambda: defaultdict(np.float64)),"SHORT": defaultdict(lambda: defaultdict(np.float64))},
        }
        self.tradingplan_info_path = str(self.Parser.get(section="strategy", option="tradingplan_info_path"))
        if self.tradingplan_info_path != "None":
            sub_trading_df = pd.read_pickle(self.tradingplan_info_path)
            plan_ts_list = sub_trading_df['Plan'].unique().tolist()
            for tmp_ts in plan_ts_list:
                for direction in ["Open", "Close"]:
                    for pside in ["LONG", "SHORT"]:
                        tmp_df = sub_trading_df[
                            (sub_trading_df['Plan'] == tmp_ts) & (sub_trading_df['direction'] == direction) & (
                                        sub_trading_df['pside'] == pside)]
                        for row in tmp_df.iterrows():
                            tmp_symbol = row[1]['symbol']
                            tmp_vol = row[1]['PlanedVol']
                            self.TradingPlan[direction][pside][tmp_ts][tmp_symbol] = tmp_vol

        # 挂单中的交易
        self.ExecutingPlan = {
            "Open":{"LONG":defaultdict(np.float64),"SHORT":defaultdict(np.float64)},
            "Close": {"LONG": defaultdict(np.float64), "SHORT": defaultdict(np.float64)},
        }
        # 完成的交易计划
        self.TradedPlan = {
            "Open":{"LONG":defaultdict(lambda: defaultdict(np.float64)),"SHORT":defaultdict(lambda: defaultdict(np.float64))},
            "Close": {"LONG": defaultdict(lambda: defaultdict(np.float64)), "SHORT": defaultdict(lambda: defaultdict(np.float64))},
        }
        # 没有完成并且并不会执行的交易计划
        self.MissedPlan = {
            "Open":{"LONG":defaultdict(lambda: defaultdict(np.float64)),"SHORT":defaultdict(lambda: defaultdict(np.float64))},
            "Close": {"LONG": defaultdict(lambda: defaultdict(np.float64)), "SHORT": defaultdict(lambda: defaultdict(np.float64))},
        }
        # 止损时间
        self.LatestStopLossTimeStamp = {x: {"LONG":0,"SHORT":0} for x in self.future_symbol_list}

        # 记录最近的建仓时间
        self.position_info_path = str(self.Parser.get(section="strategy", option="position_info_path"))
        if self.position_info_path == "None":
            self.InitPosInfo = False
            self.LatestBuildPosTimeStamp = {x: {"LONG":0,"SHORT":0} for x in self.future_symbol_list}
            self.LatestBuildPrice = {x: {"LONG":np.nan,"SHORT":np.nan} for x in self.future_symbol_list}
            self.LatestBuildPosAlpha = {x: {"LONG":np.nan,"SHORT":np.nan} for x in self.future_symbol_list}
        else:
            self.InitPosInfo = True
            position_info = pd.read_pickle(self.position_info_path)
            self.LatestBuildPosTimeStamp = position_info['LatestBuildPosTimeStamp']
            self.LatestBuildPrice = position_info['LatestBuildPrice']
            self.LatestBuildPosAlpha = position_info['LatestBuildPosAlpha']

        if self.local_test:
            self.AccountNetValue = []

    def init_my_strategy(self):
        try:
            event_list = []
            print("init_my_strategy")
            # 读取基本合约信息
            if self.local_test:
                self.ContractInfo = pd.read_pickle(self.Parser.get(section ="strategy" ,option="ContractInfoPath"))
            else:
                # self.STLogger.critical("Collecting Contract Info")
                # self.ContractInfo = CollectContractInfo()
                pass

            for symbol in self.future_symbol_list:
                if symbol not in self.ContractInfo.keys():
                    raise Exception(f"No Contract Info, Del {symbol} or collect relative info")

            return event_list
        except Exception as e:
            self.STLogger.critical(traceback.format_exc())

    def checkLatestCleanTS(self, nowTS, long_vol_res, short_vol_res):
        # 如果预期仓位是0，更新最近清仓时间
        for symbol in self.future_symbol_list:
            long_vol = np.abs(long_vol_res[symbol])
            if long_vol <1e-6:
                self.LatestBuildPosTimeStamp[symbol]["LONG"] = nowTS
                self.LatestBuildPrice[symbol]["LONG"] = np.nan
                self.LatestBuildPosAlpha[symbol]["LONG"] = np.nan

            short_vol = np.abs(short_vol_res[symbol])
            if short_vol < 1e-6:
                self.LatestBuildPosTimeStamp[symbol]["SHORT"] = nowTS
                self.LatestBuildPrice[symbol]["SHORT"] = np.nan
                self.LatestBuildPosAlpha[symbol]["SHORT"] = np.nan

    def updatePosCostAndTs(self, symbol, side, nowTS):
        # 如果进行了买卖，调用此函数，查看是否需要更新建仓成本
        if np.isnan(self.LatestBuildPrice[symbol][side]):
            self.LatestBuildPosTimeStamp[symbol][side] = nowTS
            self.LatestBuildPrice[symbol][side] = self.LatestMinQuotes[symbol]
            self.LatestBuildPosAlpha[symbol][side] = 0

    def updatePosAlpha(self, AlphaDict):
        for symbol in self.future_symbol_list:
            # 更新多头alpha
            if not np.isnan(self.LatestBuildPrice[symbol]["LONG"]):
                self.LatestBuildPosAlpha[symbol]["LONG"] += AlphaDict[symbol]

            # 更新空头alpha
            if not np.isnan(self.LatestBuildPrice[symbol]["SHORT"]):
                self.LatestBuildPosAlpha[symbol]["SHORT"] -= AlphaDict[symbol]

    # 各算法通用
    def round_price(self,symbol,price):
        price = int((np.round(price,self.ContractInfo[symbol]['price_lot'] + 2) + 1e-8) // self.ContractInfo[symbol]['price_tickSize']) * self.ContractInfo[symbol]['price_tickSize']
        price = np.round(price,self.ContractInfo[symbol]['price_lot'])
        if self.ContractInfo[symbol]['price_lot'] == 0:
            price = int(price)
        return price

    def round_quantity(self,symbol,quantity):
        quantity1 = int((np.round(quantity,self.ContractInfo[symbol]['quantity_lot'] + 2)+ 1e-8) //self.ContractInfo[symbol]['quantity_stepSize']) * self.ContractInfo[symbol]['quantity_stepSize']
        quantity = np.round(quantity1,self.ContractInfo[symbol]['quantity_lot'])
        if self.ContractInfo[symbol]['quantity_lot'] == 0:
            quantity = int(quantity)
        return quantity

    def aggTradingPlan(self, symbol, nowTS, backPeriod, side):
        trading = []

        # 确认要查找的方向
        if side == "buy":
            TradingDict1 = self.TradingPlan["Open"]["LONG"]
            TradingDict2 = self.TradingPlan["Close"]["SHORT"]

        elif side == "sell":
            TradingDict1 = self.TradingPlan["Close"]["LONG"]
            TradingDict2 = self.TradingPlan["Open"]["SHORT"]

        # 确认有多少股票正在被买入或卖出
        for histTS in list(TradingDict1.keys()):
            # 找到最近的交易计划，并且记录
            if (nowTS - (histTS + 60000)) < backPeriod:
                if TradingDict1[histTS][symbol] > 1e-8:
                    trading.append(TradingDict1[histTS][symbol])

        for histTS in list(TradingDict2.keys()):
            # 找到最近的交易计划，并且记录
            if (nowTS - (histTS + 60000)) < backPeriod:
                if TradingDict2[histTS][symbol] > 1e-8:
                    trading.append(TradingDict2[histTS][symbol])

        totalPlanVol = np.nansum(trading)
        return totalPlanVol

    def calTradingMarketLimit(self, MarketValDict, nowTS):
        # 计算所有标的的交易占比
        TradingUplimit = {x:{'buy':False,'sell':False} for x in self.future_symbol_list}
        for symbol in self.future_symbol_list:
            tmpBuyVal = self.aggTradingPlan(symbol, nowTS, self.TradeTime * 60000, "buy") * self.LatestMinQuotes[symbol]
            buyLeftQuota = MarketValDict[symbol] * self.TradingRatioUpThres - tmpBuyVal

            tmpSellVal = self.aggTradingPlan(symbol, nowTS, self.TradeTime * 60000, "sell") * self.LatestMinQuotes[symbol]
            sellLeftQuota = MarketValDict[symbol] * self.TradingRatioUpThres - tmpSellVal

            TradingUplimit[symbol]['buy'] = buyLeftQuota < self.EachChanceVal
            TradingUplimit[symbol]['sell'] = sellLeftQuota < self.EachChanceVal
        return TradingUplimit

    def OnBacktestMinBar(self, spot_data_dict, future_data_dict, trading_signal_dict):
        nowTS = future_data_dict["open_time"].index.tolist()[-1]
        # 更新最新报价
        NewQuotes = future_data_dict["close"].iloc[-1].to_dict()
        for symbol in NewQuotes.keys():
            if not np.isnan(NewQuotes[symbol]):
                self.LatestMinQuotes[symbol] = NewQuotes[symbol]

        # 更新当前交易信息
        TradingStatus = ((~future_data_dict["close"].iloc[-1].isna()) & (future_data_dict[self.pool_mask_name].iloc[-1]))

        # 计算超额收益
        Rtns = future_data_dict[f"Rtn{self.alpha_update_freq}"].iloc[-1,:]
        Alphas = (Rtns - Rtns.where(TradingStatus).mean()).to_dict()

        # 计算当前因子值
        if not self.factor_rank:
            TradingSignal = trading_signal_dict[self.alpha_name_list[0]].iloc[-1, :]
            for idx, alpha_name in enumerate(self.alpha_name_list):
                if idx ==0:
                    continue
                TradingSignal += trading_signal_dict[self.alpha_name_list[idx]].iloc[-1, :]
            TradingSignal = (TradingSignal / len(self.alpha_name_list)).where(TradingStatus)
        else:
            TradingSignal = trading_signal_dict[self.alpha_name_list[0]].iloc[-1, :].where(TradingStatus).rank(pct=True)
            for idx, alpha_name in enumerate(self.alpha_name_list):
                if idx ==0:
                    continue
                TradingSignal += trading_signal_dict[self.alpha_name_list[idx]].iloc[-1, :].where(TradingStatus).rank(pct=True)
            TradingSignal = (TradingSignal / len(self.alpha_name_list)).where(TradingStatus)

        SignalNaList = TradingStatus.loc[TradingStatus & (TradingSignal.isna())].index.tolist()
        TradingStatus = TradingStatus & (~TradingSignal.isna())
        if SignalNaList:
            self.STLogger.critical(f"Signal Na Symbols:{SignalNaList}")
        self.TradingStatusDict = TradingStatus.to_dict()

        # 计算24h市场涨跌
        Market24HVol = future_data_dict['Market24HVol'].iloc[-1, 0]
        Market24HRtn = future_data_dict['Market24HRtn'].iloc[-1, 0]
        Rtn1440 = future_data_dict['Rtn1440'].iloc[-1, :]
        # 根据24h波动删去个别标的
        Vol24H = future_data_dict['Vol24H'].iloc[-1, :]

        InfoDict = {
            "Market24HVol":Market24HVol,
            "Market24HRtn":Market24HRtn,
            "Vol24H":Vol24H,
            "Rtn1440":Rtn1440,
            "Alphas":Alphas,
            "trade_value_sum30min": future_data_dict['trade_value_sum30min'].iloc[-1, :],
            "trade_value_sum5min": future_data_dict['trade_value_sum5min'].iloc[-1, :],
        }

        # 获取未来30分钟成交价
        if self.local_test:
            future_twap = future_data_dict[self.future_twap_name].iloc[-1].to_dict()
        else:
            future_twap = None

        # 回调调仓函数
        self.OnPreparedTradingSignal(nowTS,TradingSignal, InfoDict, future_twap=future_twap)

    # 换仓算法
    def OnPreparedTradingSignal(self, nowTS, TradingSignal, InfoDict, future_twap = None):
        t0 = time.time()

        TargetHoldingVal_Long = self.TargetHoldingVal["LONG"]
        TargetHoldingVal_Short = self.TargetHoldingVal["SHORT"]

        # 第0步：获取现阶段仓位信息，计算风险程度
        if self.local_test:
            long_vol_res = {x: self.future_position[x]["LONG"]["pa"] for x in self.future_symbol_list}
            long_val_res = {x: long_vol_res[x] * self.LatestMinQuotes[x] for x in self.future_symbol_list}
            long_pct_res = {x: long_val_res[x] / TargetHoldingVal_Long for x in self.future_symbol_list}

            short_vol_res = {x: self.future_position[x]["SHORT"]["pa"] for x in self.future_symbol_list}
            short_val_res = {x: short_vol_res[x] * self.LatestMinQuotes[x] for x in self.future_symbol_list}
            short_pct_res = {x: short_val_res[x] / TargetHoldingVal_Short for x in self.future_symbol_list}

        else:
            long_vol_res, long_val_res, long_pct_res = self.getExpectedFuturePosition(nowTS, "LONG")
            short_vol_res, short_val_res, short_pct_res = self.getExpectedFuturePosition(nowTS, "SHORT")
            pass

        # 确认最近被清仓时间
        self.checkLatestCleanTS(nowTS,long_vol_res,short_vol_res)
        AlphaDict = InfoDict['Alphas']
        self.updatePosAlpha(AlphaDict)

        # 计算风格暴露
        Vol24H = InfoDict['Vol24H']
        MarketMedianVol24H = InfoDict['Vol24H'].median()
        Vol24HDict = InfoDict['Vol24H'].to_dict()

        exp_long_VolStyle = np.nansum([long_val_res[x] * Vol24HDict[x] for x in self.future_symbol_list]) / TargetHoldingVal_Long
        exp_short_VolStyle = np.nansum([short_val_res[x] * Vol24HDict[x] for x in self.future_symbol_list]) / TargetHoldingVal_Short

        # 计算因子值
        scoreDict = TradingSignal.to_dict()
        for symbol in self.future_symbol_list:
            if symbol not in scoreDict.keys():
                scoreDict[symbol] = np.nan
        exp_long_score = np.nansum([long_val_res[x] * scoreDict[x] for x in self.future_symbol_list]) / TargetHoldingVal_Long
        exp_short_score = np.nansum([short_val_res[x] * scoreDict[x] for x in self.future_symbol_list]) / TargetHoldingVal_Short

        Rtn24HDict = InfoDict["Rtn1440"].to_dict()
        exp_long_RtnStyle = np.nansum([long_val_res[x] * Rtn24HDict[x] for x in self.future_symbol_list]) / TargetHoldingVal_Long
        exp_short_RtnStyle = np.nansum([short_val_res[x] * Rtn24HDict[x] for x in self.future_symbol_list]) / TargetHoldingVal_Short
        MarketMeanRtn24H = InfoDict["Rtn1440"].mean()


        # 计算风险程度
        if self.local_test:
            Balance = self.future_balance[self.quote_coin]
        else:
            Balance = float(self.future_balance[self.quote_coin]['wb'])
        LongNotional = np.nansum([long_val_res[x] for x in self.future_symbol_list])
        ShortNotional = -np.nansum([short_val_res[x] for x in self.future_symbol_list])
        NotionalValOverBalance =(LongNotional + ShortNotional)/ Balance
        TargetValOverBalance = (TargetHoldingVal_Long - TargetHoldingVal_Short)/Balance
        LongOverShort = (LongNotional + 1) / (ShortNotional + 1)
        self.STLogger.critical(f"{nowTS}, Risk Indicators:AccBal = {Balance}, LongVal = {LongNotional}, ShortVal = {ShortNotional},  Current Leverage = {np.round(NotionalValOverBalance,2)}, Target Leverage = {np.round(TargetValOverBalance,2)}, L/S Balance = {LongOverShort}, Market24HVol = {InfoDict['Market24HVol']}")

        # 判断回撤
        drawdown = 0
        drawdown1h = 0
        ExpLongNum = 0
        ExpShortNum = 0
        for x in self.future_symbol_list:
            if long_val_res[x] > 1e-8:
                ExpLongNum += 1
            if short_val_res[x] < -1e-8:
                ExpShortNum += 1

        if (self.local_test) and (len(self.AccountNetValue)>2):
            drawdown = -(self.AccountNetValue[-1] - np.nanmax(self.AccountNetValue[-int(1440/self.alpha_update_freq):])) / (TargetHoldingVal_Long + np.abs(TargetHoldingVal_Short))
            drawdown1h = -(self.AccountNetValue[-1] - np.nanmax(self.AccountNetValue[-int(60/self.alpha_update_freq):])) / (TargetHoldingVal_Long + np.abs(TargetHoldingVal_Short))
        if not self.local_test:
            # 实盘的时候从其他地方获取
            drawdown = self.drawdown
        self.STLogger.critical(f"StyleMsg#TS:{nowTS}#24Hdd:{drawdown}#1Hdd:{drawdown1h}#ExpLongNum:{ExpLongNum}#ExpShortNum:{ExpShortNum}#MarketRtn:{MarketMeanRtn24H}#MarketMedianVol:{MarketMedianVol24H}#LongVol24H:{exp_long_VolStyle}#ShortVol24H:{exp_short_VolStyle}#LongMOM24H:{exp_long_RtnStyle}#ShortMOM24H:{exp_short_RtnStyle}#LongScore:{exp_long_score}#ShortScore:{exp_short_score}")

        if (drawdown > self.Max24HDDBlackThres) or (drawdown1h > self.Max1HDDBlackThres):
            self.MaxDDTS = nowTS
        if (nowTS - self.MaxDDTS) < self.MaxDDBlackMin * 60000:
            self.ForceClear = True
            self.MaxDDBlack = True
            self.STLogger.critical(f"RiskMsg#TS:{nowTS}#Type:MaxDD#Max24HDD:{drawdown}#Max1HDD:{drawdown1h}")
            self.STLogger.critical(f"Max24HDD = {drawdown} > {self.Max24HDDBlackThres}, Max1HDD = {drawdown1h} > {self.Max1HDDBlackThres}, Force Clear, Finish Time = {int(self.MaxDDTS + self.MaxDDBlackMin * 60000)}")
            for symbol in long_val_res.keys():
                if (long_vol_res[symbol] != 0):
                    self.STLogger.critical(f"Force Clear {symbol}, long_vol = {long_vol_res[symbol]}")
                    temp_quantity = self.round_quantity(symbol, long_vol_res[symbol])
                    self.TradingPlan["Close"]["LONG"][nowTS][symbol] += temp_quantity
                    if self.local_test:
                        self.creat_backtest_virtual_order("Close", "LONG", symbol, temp_quantity, future_twap[symbol])

                if (short_vol_res[symbol] != 0):
                    self.STLogger.critical(f"Force Clear {symbol}, short_vol = {short_vol_res[symbol]}")
                    temp_quantity = self.round_quantity(symbol, -short_vol_res[symbol])
                    self.TradingPlan["Close"]["SHORT"][nowTS][symbol] += temp_quantity
                    if self.local_test:
                        self.creat_backtest_virtual_order("Close", "SHORT", symbol, temp_quantity, future_twap[symbol])
            return
        else:
            self.MaxDDBlack = False

        # 判断市场行情
        if not self.Market24HVolBlack:
            if InfoDict['Market24HVol'] > self.Market24HVolThresIn:
                self.STLogger.critical(f"RiskMsg#TS:{nowTS}#Type:MarketLargeVol#Market24HVol:{InfoDict['Market24HVol']}")
                self.Market24HVolBlack = True
                self.ForceClear = True

        if self.Market24HVolBlack:
            self.STLogger.critical(f"Market24HVol = {InfoDict['Market24HVol']} > {self.Market24HVolThresIn}, Force Clear")
            for symbol in long_val_res.keys():
                if (long_vol_res[symbol] != 0):
                    self.STLogger.critical(f"Force Clear {symbol}, long_vol = {long_vol_res[symbol]}")
                    temp_quantity = self.round_quantity(symbol, long_vol_res[symbol])
                    self.TradingPlan["Close"]["LONG"][nowTS][symbol] += temp_quantity
                    if self.local_test:
                        self.creat_backtest_virtual_order("Close", "LONG", symbol, temp_quantity, future_twap[symbol])

                if (short_vol_res[symbol] != 0):
                    self.STLogger.critical(f"Force Clear {symbol}, short_vol = {short_vol_res[symbol]}")
                    temp_quantity = self.round_quantity(symbol, -short_vol_res[symbol])
                    self.TradingPlan["Close"]["SHORT"][nowTS][symbol] += temp_quantity
                    if self.local_test:
                        self.creat_backtest_virtual_order("Close", "SHORT", symbol, temp_quantity, future_twap[symbol])

            if InfoDict['Market24HVol'] < self.Market24HVolThresOut:
                self.Market24HVolBlack = False

            return

        # 判断需要剔除票池的票
        TradeableSymbols = [x for x in self.TradingStatusDict.keys() if self.TradingStatusDict[x] == True]
        LargeVolSymbols = Vol24H[Vol24H>self.Symbol24HVolThresIn].index.tolist()
        self.Black24HVolSymbols = list(set(self.Black24HVolSymbols + LargeVolSymbols))
        tmp = Vol24H.loc[self.Black24HVolSymbols]
        self.Black24HVolSymbols = tmp[tmp>self.Symbol24HVolThresOut].index.tolist()
        auto_black_symbols = [x for x in TradeableSymbols if x in self.Black24HVolSymbols]

        if not self.local_test:
            auto_black_symbols = list(set(auto_black_symbols + self.trading_algo_banned_symbols))

        if auto_black_symbols:
            self.STLogger.critical(f"auto black symbols (large 24HVol >{self.Symbol24HVolThresIn}), {auto_black_symbols}")

        if len(auto_black_symbols) >= self.SymbolBlackThres:
            self.ForceClear = True
            self.SymbolBlack = True
            self.STLogger.critical(f"RiskMsg#TS:{nowTS}#Type:TooManyLargeVol")
            self.STLogger.critical(f"24HVol Black Symbols Num = {len(auto_black_symbols)} > {self.SymbolBlackThres}, Force Clear")
            for symbol in long_val_res.keys():
                if (long_vol_res[symbol] != 0):
                    self.STLogger.critical(f"Force Clear {symbol}, long_vol = {long_vol_res[symbol]}")
                    temp_quantity = self.round_quantity(symbol, long_vol_res[symbol])
                    self.TradingPlan["Close"]["LONG"][nowTS][symbol] += temp_quantity
                    if self.local_test:
                        self.creat_backtest_virtual_order("Close", "LONG", symbol, temp_quantity, future_twap[symbol])

                if (short_vol_res[symbol] != 0):
                    self.STLogger.critical(f"Force Clear {symbol}, short_vol = {short_vol_res[symbol]}")
                    temp_quantity = self.round_quantity(symbol, -short_vol_res[symbol])
                    self.TradingPlan["Close"]["SHORT"][nowTS][symbol] += temp_quantity
                    if self.local_test:
                        self.creat_backtest_virtual_order("Close", "SHORT", symbol, temp_quantity, future_twap[symbol])
            return
        else:
            self.SymbolBlack = False

        status_black_list = [x for x in self.future_symbol_list if self.TradingStatusDict[x] == False]
        total_banned_symbols = list(set(status_black_list + auto_black_symbols + self.black_symbol_list))

        # 第0.5步 判断过去一段时间的交易量占比
        if self.local_test:
            # localtest 中删除过久的trading plan，清理内存，实盘中会在OnTimer中删除
            for direction in ["Open","Close"]:
                for positionSide in ["LONG","SHORT"]:
                    tmp_tsList = list(self.TradingPlan[direction][positionSide])
                    to_del_ts = [x for x in tmp_tsList if (nowTS - x > (self.TradeTime + 30) * 60000)]
                    for tmp_ts in to_del_ts:
                        del self.TradingPlan[direction][positionSide][tmp_ts]
            pass

        MarketValDict = InfoDict[f'trade_value_sum{int(self.TradeTime)}min'].to_dict()
        TradingRatioLimit = self.calTradingMarketLimit(MarketValDict, nowTS)
        banned_buy_list = [x for x in self.future_symbol_list if TradingRatioLimit[x]['buy']]
        banned_sell_list = [x for x in self.future_symbol_list if TradingRatioLimit[x]['sell']]
        if self.detail_log:
            self.STLogger.critical(f"RatioLimitMsg#TS:{nowTS}#BannedBuy:{str(banned_buy_list)}#BannedSell:{str(banned_sell_list)}")

        # 第1步：准备可开可平量
        # 初始化可买入次数
        GlobalPred = TradingSignal.sort_values(ascending=False).dropna()
        GlobalPredDict = GlobalPred.to_dict()

        # 全局多头预测
        LongGlobalPredOrder = GlobalPred.index.to_list()
        LongGlobalMatchTimes = {}
        for symbol in LongGlobalPredOrder:
            cond1 = self.TradingStatusDict[symbol]
            cond2 = (long_pct_res[symbol] < self.MaxCapRatio)
            cond3 = ((nowTS - self.LatestStopLossTimeStamp[symbol]["LONG"]) > self.StopLossTimeInterval * 60000)
            cond4 = (symbol not in total_banned_symbols)
            cond5 = True
            cond6 = True
            cond7 = not TradingRatioLimit[symbol]['buy']

            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7: # 在交易，并且还没有达到最大值，并且已经脱离止损时间
                LongGlobalMatchTimes[symbol] = 1
            else:
                LongGlobalMatchTimes[symbol] = 0
                if cond1 and cond7 and self.detail_log:
                    tmpMsg = f"AlpStatMsg#TS:{nowTS}#symbol:{symbol}#Side:Long#Trading:{cond1}#MaxCap:{cond2}#StopLoss:{cond3}#Banned:{cond4}#LVolCtrl:{cond5}#SVCtrl:{cond6}#TradingRatio:{cond7}"
                    self.STLogger.critical(tmpMsg)

        LongGlobalMatchIdx = 0
        if len(LongGlobalPredOrder)>0 and (LongGlobalMatchTimes[LongGlobalPredOrder[LongGlobalMatchIdx]] == 0):
            LongGlobalMatchIdx = next_nonezero_idx(LongGlobalMatchIdx,LongGlobalMatchTimes,LongGlobalPredOrder)

        # 全局空头预测
        ShortGlobalPredOrder = GlobalPred.index.to_list()[::-1]
        ShortGlobalMatchTimes = {}
        for symbol in ShortGlobalPredOrder:
            cond1 = self.TradingStatusDict[symbol]
            cond2 = (short_pct_res[symbol] < self.MaxCapRatio)
            cond3 = ((nowTS - self.LatestStopLossTimeStamp[symbol]["SHORT"]) > self.StopLossTimeInterval * 60000)
            cond4 = (symbol not in total_banned_symbols)
            # 增加风格过滤条件
            cond5 = True
            cond6 = True
            cond7 = not TradingRatioLimit[symbol]['sell']

            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7:
                ShortGlobalMatchTimes[symbol] = 1
            else:
                ShortGlobalMatchTimes[symbol] = 0
                if cond1 and cond7 and self.detail_log:
                    tmpMsg = f"AlpStatMsg#TS:{nowTS}#symbol:{symbol}#Side:Short#Trading:{cond1}#MaxCap:{cond2}#StopLoss:{cond3}#Banned:{cond4}#LVolCtrl:{cond5}#SVCtrl:{cond6}#TradingRatio:{cond7}"
                    self.STLogger.critical(tmpMsg)

        ShortGlobalMatchIdx = 0
        if len(ShortGlobalPredOrder)>0 and (ShortGlobalMatchTimes[ShortGlobalPredOrder[ShortGlobalMatchIdx]] == 0):
            ShortGlobalMatchIdx = next_nonezero_idx(ShortGlobalMatchIdx,ShortGlobalMatchTimes,ShortGlobalPredOrder)

        # 持仓多头预测，并且清理碎股
        LongHoldingList = []
        for symbol in long_val_res.keys():
            if (long_val_res[symbol] < self.SmallValToClean) and (long_vol_res[symbol] !=0) and (not np.isnan(self.LatestMinQuotes[symbol])):
                self.STLogger.critical(f"Clean Odd Long {symbol}, long_val = {long_val_res[symbol]}")
                # 存在碎股进行清理
                temp_quantity = np.maximum(self.round_quantity(symbol, long_vol_res[symbol]),self.ContractInfo[symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["LONG"][nowTS][symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:CleanOdds#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:LONG#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "LONG", symbol, temp_quantity, future_twap[symbol])

                continue

            if long_val_res[symbol] >= self.SmallValToClean:
                LongHoldingList.append(symbol)

        LongHoldingPreOrder = TradingSignal.loc[LongHoldingList].sort_values(ascending=True).dropna().index.to_list()
        LongHoldingMatchTimes = {}
        for symbol in LongHoldingPreOrder:
            if (self.TradingStatusDict[symbol]) and (not TradingRatioLimit[symbol]['sell']):
                LongHoldingMatchTimes[symbol] = 1
            else:
                LongHoldingMatchTimes[symbol] = 0
                # self.STLogger.critical(f"Holding Warning: long {symbol} not in trading pool")
        LongHoldingMatchIdx = 0
        if len(LongHoldingPreOrder)>0 and (LongHoldingMatchTimes[LongHoldingPreOrder[LongHoldingMatchIdx]] == 0):
            LongHoldingMatchIdx = next_nonezero_idx(LongHoldingMatchIdx, LongHoldingMatchTimes, LongHoldingPreOrder)


        # 持仓空头预测
        ShortHoldingList = []
        for symbol in short_val_res.keys():
            if (short_val_res[symbol] > -self.SmallValToClean) and (short_vol_res[symbol] !=0) and (not np.isnan(self.LatestMinQuotes[symbol])):
                self.STLogger.critical(f"Clean Odd Short {symbol}, short_val = {short_val_res[symbol]}")
                # 存在碎股进行清理
                temp_quantity = np.maximum(self.round_quantity(symbol,-short_vol_res[symbol]),self.ContractInfo[symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["SHORT"][nowTS][symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:CleanOdds#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:SHORT#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "SHORT", symbol, temp_quantity, future_twap[symbol])

                continue

            if short_val_res[symbol] <= -self.SmallValToClean:
                ShortHoldingList.append(symbol)

        ShortHoldingPreOrder = TradingSignal.loc[ShortHoldingList].sort_values(ascending=False).dropna().index.to_list()
        ShortHoldingMatchTimes = {}
        for symbol in ShortHoldingPreOrder:
            if self.TradingStatusDict[symbol] and (not TradingRatioLimit[symbol]['buy']):
                ShortHoldingMatchTimes[symbol] = 1
            else:
                ShortHoldingMatchTimes[symbol] = 0
                # self.STLogger.critical(f"Holding Warning: short {symbol} not in trading pool")
        ShortHoldingMatchIdx = 0
        if len(ShortHoldingPreOrder)>0 and (ShortHoldingMatchTimes[ShortHoldingPreOrder[ShortHoldingMatchIdx]] == 0):
            ShortHoldingMatchIdx = next_nonezero_idx(ShortHoldingMatchIdx, ShortHoldingMatchTimes, ShortHoldingPreOrder)


        # 第2步建仓
        # 多头
        Total_Long_Val = np.nansum([long_val_res[x] for x in self.future_symbol_list])
        LongLowerBound = TargetHoldingVal_Long * self.ExpLowerBound
        LongUpperBound = TargetHoldingVal_Long * self.ExpUpperBound
        # 判断是否要建仓
        if LongLowerBound > Total_Long_Val:
            LongPosToBuildTimes = (TargetHoldingVal_Long - Total_Long_Val) // self.EachChanceVal
        else:
            LongPosToBuildTimes = 0
        LongPosBuilded = 0
        # 退出循环的条件：
        # 1. 找到足够的建仓对象
        # 2. global 的预测值小于要求阈值
        # 3. global idx超出长度
        while (LongPosBuilded < LongPosToBuildTimes) and (LongGlobalMatchIdx < len(LongGlobalPredOrder)):
            # 当前匹配位置
            global_symbol = LongGlobalPredOrder[LongGlobalMatchIdx]

            # 是否还剩余匹配次数 (其实是必然有剩余次数)
            if (LongGlobalMatchTimes[global_symbol] > 0):
                # 下单建仓
                temp_quantity = self.round_quantity(global_symbol, self.EachChanceVal / self.LatestMinQuotes[global_symbol])
                temp_quantity = np.maximum(temp_quantity,self.ContractInfo[global_symbol]['limit_min_qunatity'])
                self.TradingPlan["Open"]["LONG"][nowTS][global_symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:BuildPos#TS:{nowTS}#symbol:{global_symbol}#direction:Open#pside:LONG#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[global_symbol]}#Predict:{GlobalPredDict[global_symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Open", "LONG", global_symbol, temp_quantity, future_twap[global_symbol])

                # 对策略产生影响
                LongPosBuilded += 1
                LongGlobalMatchTimes[global_symbol] -= 1
                if LongGlobalMatchTimes[global_symbol] == 0:
                    LongGlobalMatchIdx = next_nonezero_idx(LongGlobalMatchIdx,LongGlobalMatchTimes,LongGlobalPredOrder)
            del global_symbol

        # 空头
        Total_Short_Val = np.nansum([short_val_res[x] for x in self.future_symbol_list])
        ShortLowerBound = TargetHoldingVal_Short * self.ExpLowerBound
        ShortUpperBound = TargetHoldingVal_Short * self.ExpUpperBound
        # 判断是否要建仓
        if Total_Short_Val > ShortLowerBound:
            ShortPosToBuildTimes = (-TargetHoldingVal_Short + Total_Short_Val) // self.EachChanceVal
        else:
            ShortPosToBuildTimes = 0
        ShortPosBuilded = 0
        # 退出循环的条件：
        # 1. 找到足够的建仓对象
        # 2. global 的预测值小于要求阈值
        # 3. global idx超出长度
        while (ShortPosBuilded < ShortPosToBuildTimes) and (ShortGlobalMatchIdx < len(ShortGlobalPredOrder)) :
            # 当前匹配位置
            global_symbol = ShortGlobalPredOrder[ShortGlobalMatchIdx]

            # 是否还剩余匹配次数 (其实是必然有剩余次数)
            if (ShortGlobalMatchTimes[global_symbol] >0):
                # 下单建仓
                temp_quantity = self.round_quantity(global_symbol, self.EachChanceVal / self.LatestMinQuotes[global_symbol])
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[global_symbol]['limit_min_qunatity'])
                self.TradingPlan["Open"]["SHORT"][nowTS][global_symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:BuildPos#TS:{nowTS}#symbol:{global_symbol}#direction:Open#pside:SHORT#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[global_symbol]}#Predict:{GlobalPredDict[global_symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Open", "SHORT", global_symbol,temp_quantity, future_twap[global_symbol])

                # 对策略产生影响
                ShortPosBuilded += 1
                ShortGlobalMatchTimes[global_symbol] -= 1
                if ShortGlobalMatchTimes[global_symbol] == 0:
                    ShortGlobalMatchIdx = next_nonezero_idx(ShortGlobalMatchIdx,ShortGlobalMatchTimes,ShortGlobalPredOrder)

            del global_symbol

        # 第3步清仓
        # 多头
        if Total_Long_Val > LongUpperBound:
            LongPosToCleanTimes = (Total_Long_Val - TargetHoldingVal_Long) // self.EachChanceVal
        else:
            LongPosToCleanTimes = 0
        LongPosCleaned = 0

        while (LongPosCleaned < LongPosToCleanTimes) and (LongHoldingMatchIdx < len(LongHoldingPreOrder)):
            # 当前匹配位置
            holding_symbol = LongHoldingPreOrder[LongHoldingMatchIdx]

            # 是否还剩余匹配次数 (其实是必然有剩余次数)
            if (LongHoldingMatchTimes[holding_symbol] > 0):
                # 下单清仓
                # 量
                temp_quantity = self.round_quantity(holding_symbol, np.minimum(long_vol_res[holding_symbol], self.EachChanceVal / self.LatestMinQuotes[holding_symbol]))
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[holding_symbol]['limit_min_qunatity'])
                temp_price = self.LatestMinQuotes[holding_symbol]

                self.TradingPlan["Close"]["LONG"][nowTS][holding_symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:CleanPos#TS:{nowTS}#symbol:{holding_symbol}#direction:Close#pside:LONG#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[holding_symbol]}#Predict:{GlobalPredDict[holding_symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "LONG", holding_symbol, temp_quantity, future_twap[holding_symbol])

                # 对策略产生影响
                LongPosCleaned += 1
                LongHoldingMatchTimes[holding_symbol] -= 1
                if LongHoldingMatchTimes[holding_symbol] ==0:
                    LongHoldingMatchIdx = next_nonezero_idx(LongHoldingMatchIdx, LongHoldingMatchTimes, LongHoldingPreOrder)

            del holding_symbol

        # if (LongPosCleaned == 0) and (LongPosToCleanTimes >0):
        #     self.STLogger.critical(f"Pos Warning: LongToClean {LongPosToCleanTimes}, Not Cleaned, Holding Match Predict = {GlobalPredDict[LongHoldingPreOrder[LongHoldingMatchIdx]]}, Min Holding Predict = {GlobalPredDict[LongHoldingPreOrder[0]]}")


        # 空头
        if Total_Short_Val < ShortUpperBound:
            ShortPosToCleanTimes = (-Total_Short_Val + TargetHoldingVal_Short) // self.EachChanceVal
        else:
            ShortPosToCleanTimes = 0
        ShortPosCleaned = 0
        while (ShortPosCleaned < ShortPosToCleanTimes) and (ShortHoldingMatchIdx < len(ShortHoldingPreOrder)):
            # 当前匹配位置
            holding_symbol = ShortHoldingPreOrder[ShortHoldingMatchIdx]

            # 是否还剩余匹配次数 (其实是必然有剩余次数)
            if (ShortHoldingMatchTimes[holding_symbol] > 0):
                # 下单清仓
                # 量
                temp_quantity = self.round_quantity(holding_symbol, np.minimum(-short_vol_res[holding_symbol], self.EachChanceVal / self.LatestMinQuotes[holding_symbol]))
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[holding_symbol]['limit_min_qunatity'])
                temp_price = self.LatestMinQuotes[holding_symbol]

                self.TradingPlan["Close"]["SHORT"][nowTS][holding_symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:CleanPos#TS:{nowTS}#symbol:{holding_symbol}#direction:Close#pside:SHORT#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[holding_symbol]}#Predict:{GlobalPredDict[holding_symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "SHORT", holding_symbol, temp_quantity, future_twap[holding_symbol])

                # 对策略产生影响
                ShortPosCleaned += 1
                ShortHoldingMatchTimes[holding_symbol] -= 1
                if ShortHoldingMatchTimes[holding_symbol] ==0:
                    ShortHoldingMatchIdx = next_nonezero_idx(ShortHoldingMatchIdx, ShortHoldingMatchTimes, ShortHoldingPreOrder)

            del holding_symbol

        # 第3.5步 根据市值占比清仓，实际上是一种止盈
        for symbol in self.future_symbol_list:
            if long_pct_res[symbol] > (self.MaxCapRatio + self.MaxCapRatioDeviation):
                temp_quantity = self.round_quantity(symbol,self.EachChanceVal / self.LatestMinQuotes[symbol])
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["LONG"][nowTS][symbol] += temp_quantity
                self.STLogger.critical(
                    f"AlpMsg#Type:CleanProfit#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:LONG#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[symbol]}#PosPct:{long_pct_res[symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "LONG", symbol, temp_quantity, future_twap[symbol])

            if short_pct_res[symbol] > (self.MaxCapRatio + self.MaxCapRatioDeviation):
                temp_quantity = self.round_quantity(symbol, self.EachChanceVal / self.LatestMinQuotes[symbol])
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["SHORT"][nowTS][symbol] += temp_quantity
                self.STLogger.critical(
                    f"AlpMsg#Type:CleanProfit#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:SHORT#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[symbol]}#PosPct:{short_pct_res[symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "SHORT", symbol, temp_quantity, future_twap[symbol])

        # 第3.6步 止损，需要用真实仓位进行计算
        # 多头
        stop_loss = 0
        stop_profit = 0
        for symbol in self.future_symbol_list:  # symbol必定是存在较高仓位的symbol
            tmp_alpha = self.LatestBuildPosAlpha[symbol]["LONG"]
            tmp_build_price = self.LatestBuildPrice[symbol]["LONG"]
            if not np.isnan(tmp_build_price) and (tmp_alpha < self.ThresStopLoss) and (stop_loss < 2):
                # 止损
                temp_quantity = self.round_quantity(symbol, long_vol_res[symbol])
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["LONG"][nowTS][symbol] = temp_quantity
                self.STLogger.critical(
                    f"AlpMsg#Type:StopLoss#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:LONG#Vol:{temp_quantity}#Alpha:{tmp_alpha}")
                self.LatestStopLossTimeStamp[symbol]["LONG"] = nowTS
                stop_loss += 1

            if not np.isnan(tmp_build_price) and (tmp_alpha > self.ThresStopProfit) and (stop_profit < 2) and (symbol not in list(GlobalPredDict.keys()) or GlobalPredDict[symbol] < 0):
                # 止盈
                temp_quantity = self.round_quantity(symbol, long_vol_res[symbol])
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["LONG"][nowTS][symbol] = temp_quantity
                self.STLogger.critical(
                    f"AlpMsg#Type:StopProfit#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:LONG#Vol:{temp_quantity}#Alpha:{tmp_alpha}")
                stop_profit += 1

        # 空头
        stop_loss = 0
        stop_profit = 0
        for symbol in self.future_symbol_list:  # symbol必定是存在较高仓位的symbol
            tmp_alpha = self.LatestBuildPosAlpha[symbol]["SHORT"]
            tmp_build_price = self.LatestBuildPrice[symbol]["SHORT"]
            if not np.isnan(tmp_build_price) and (tmp_alpha < self.ThresStopLoss) and (stop_loss < 2):
                # 止损
                temp_quantity = self.round_quantity(symbol, -short_vol_res[symbol])
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["SHORT"][nowTS][symbol] = temp_quantity
                self.STLogger.critical(
                    f"AlpMsg#Type:StopLoss#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:SHORT#Vol:{temp_quantity}#Alpha:{tmp_alpha}")
                self.LatestStopLossTimeStamp[symbol]["SHORT"] = nowTS

                stop_loss += 1

            if not np.isnan(tmp_build_price) and (tmp_alpha > self.ThresStopProfit) and (stop_profit < 2) and (symbol not in list(GlobalPredDict.keys()) or GlobalPredDict[symbol] > 0):
                # 止盈
                temp_quantity = self.round_quantity(symbol, -short_vol_res[symbol])
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["SHORT"][nowTS][symbol] = temp_quantity
                self.STLogger.critical(
                    f"AlpMsg#Type:StopProfit#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:SHORT#Vol:{temp_quantity}#Alpha:{tmp_alpha}")

                stop_profit += 1


        # 第4步换仓
        # 多头
        Gap = 10000000000
        while (Gap > self.ThresLongPosChange) and (LongHoldingMatchIdx < len(LongHoldingPreOrder)) and (LongGlobalMatchIdx < len(LongGlobalPredOrder)):
            global_symbol = LongGlobalPredOrder[LongGlobalMatchIdx]
            global_predict = GlobalPredDict[global_symbol]
            global_match_time = LongGlobalMatchTimes[global_symbol]

            holding_symbol = LongHoldingPreOrder[LongHoldingMatchIdx]
            holding_predict = GlobalPredDict[holding_symbol]
            holding_match_time = LongHoldingMatchTimes[holding_symbol]

            if global_predict - holding_predict > Gap: # 确认gap是在不断缩小的
                raise Exception('Gap Not Decrease')
            Gap = global_predict - holding_predict

            if Gap > self.ThresLongPosChange:
                # 成功匹配
                temp_match_times = np.minimum(global_match_time,holding_match_time)

                # 卖出
                temp_price = self.LatestMinQuotes[holding_symbol]
                temp_quantity = self.round_quantity(holding_symbol, np.minimum(long_vol_res[holding_symbol], self.EachChanceVal / temp_price))
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[holding_symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["LONG"][nowTS][holding_symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:ChgPos#TS:{nowTS}#symbol:{holding_symbol}#direction:Close#pside:LONG#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[holding_symbol]}#PosPct:{long_pct_res[holding_symbol]}#Gap:{Gap}#Predict:{GlobalPredDict[holding_symbol]}")

                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "LONG", holding_symbol, temp_quantity, future_twap[holding_symbol])

                # 买入
                temp_price = self.LatestMinQuotes[global_symbol]
                temp_quantity = self.round_quantity(global_symbol,self.EachChanceVal / temp_price)
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[global_symbol]['limit_min_qunatity'])
                self.TradingPlan["Open"]["LONG"][nowTS][global_symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:ChgPos#TS:{nowTS}#symbol:{global_symbol}#direction:Open#pside:LONG#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[global_symbol]}#PosPct:{long_pct_res[global_symbol]}#Gap:{Gap}#Predict:{GlobalPredDict[global_symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Open", "LONG", global_symbol, temp_quantity, future_twap[global_symbol])

                # 匹配后削减匹配次数
                LongGlobalMatchTimes[global_symbol] -= temp_match_times
                if LongGlobalMatchTimes[global_symbol] == 0:
                    LongGlobalMatchIdx = next_nonezero_idx(LongGlobalMatchIdx,LongGlobalMatchTimes,LongGlobalPredOrder)

                LongHoldingMatchTimes[holding_symbol] -= temp_match_times
                if LongHoldingMatchTimes[holding_symbol] ==0:
                    LongHoldingMatchIdx = next_nonezero_idx(LongHoldingMatchIdx, LongHoldingMatchTimes, LongHoldingPreOrder)

        # 空头
        Gap = 10000000000
        while (Gap > self.ThresShortPosChange) and (ShortHoldingMatchIdx < len(ShortHoldingPreOrder)) and (ShortGlobalMatchIdx < len(ShortGlobalPredOrder)):
            global_symbol = ShortGlobalPredOrder[ShortGlobalMatchIdx]
            global_predict = GlobalPredDict[global_symbol]
            global_match_time = ShortGlobalMatchTimes[global_symbol]

            holding_symbol = ShortHoldingPreOrder[ShortHoldingMatchIdx]
            holding_predict = GlobalPredDict[holding_symbol]
            holding_match_time = ShortHoldingMatchTimes[holding_symbol]

            if holding_predict - global_predict > Gap: # 确认gap是在不断缩小的
                raise Exception('Gap Not Decrease')
            Gap = holding_predict - global_predict
            if Gap > self.ThresShortPosChange:
                # 成功匹配
                temp_match_times = np.minimum(global_match_time,holding_match_time)

                # 卖出
                temp_price = self.LatestMinQuotes[holding_symbol]
                temp_quantity = self.round_quantity(holding_symbol,np.minimum(-short_vol_res[holding_symbol], self.EachChanceVal / temp_price))
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[holding_symbol]['limit_min_qunatity'])
                self.TradingPlan["Close"]["SHORT"][nowTS][holding_symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:ChgPos#TS:{nowTS}#symbol:{holding_symbol}#direction:Close#pside:SHORT#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[holding_symbol]}#PosPct:{short_pct_res[holding_symbol]}#Gap:{Gap}#Predict:{GlobalPredDict[holding_symbol]}")

                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "SHORT", holding_symbol, temp_quantity, future_twap[holding_symbol])

                # 买入
                temp_price = self.LatestMinQuotes[global_symbol]
                temp_quantity = self.round_quantity(global_symbol,self.EachChanceVal / temp_price)
                temp_quantity = np.maximum(temp_quantity, self.ContractInfo[global_symbol]['limit_min_qunatity'])
                self.TradingPlan["Open"]["SHORT"][nowTS][global_symbol] += temp_quantity
                self.STLogger.critical(f"AlpMsg#Type:ChgPos#TS:{nowTS}#symbol:{global_symbol}#direction:Open#pside:SHORT#Vol:{temp_quantity}#Quotes:{self.LatestMinQuotes[global_symbol]}#PosPct:{short_pct_res[global_symbol]}#Gap:{Gap}#Predict:{GlobalPredDict[global_symbol]}")

                # if self.local_test:
                #     self.creat_backtest_virtual_order("Open", "SHORT", global_symbol, temp_quantity, future_twap[global_symbol])

                # 匹配后削减匹配次数
                ShortGlobalMatchTimes[global_symbol] -= temp_match_times
                if ShortGlobalMatchTimes[global_symbol] == 0:
                    ShortGlobalMatchIdx = next_nonezero_idx(ShortGlobalMatchIdx,ShortGlobalMatchTimes,ShortGlobalPredOrder)

                ShortHoldingMatchTimes[holding_symbol] -= temp_match_times
                if ShortHoldingMatchTimes[holding_symbol] ==0:
                    ShortHoldingMatchIdx = next_nonezero_idx(ShortHoldingMatchIdx, ShortHoldingMatchTimes, ShortHoldingPreOrder)

        # 第5步清仓被加入黑名单的仓位
        black_list = [x for x in total_banned_symbols if x in self.future_symbol_list]
        for symbol in black_list:
            temp_long_quantity = self.round_quantity(symbol,long_vol_res[symbol])
            temp_short_quantity = self.round_quantity(symbol,-short_vol_res[symbol])
            temp_price = self.LatestMinQuotes[symbol]

            if temp_long_quantity >0:
                self.TradingPlan["Close"]["LONG"][nowTS][symbol] += temp_long_quantity
                self.STLogger.critical(f"AlpMsg#Type:Black#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:LONG#Vol:{temp_long_quantity}#Quotes:{self.LatestMinQuotes[symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "LONG", symbol, temp_long_quantity, future_twap[symbol])

            if temp_short_quantity >0:
                self.TradingPlan["Close"]["SHORT"][nowTS][symbol] += temp_short_quantity
                self.STLogger.critical(f"AlpMsg#Type:Black#TS:{nowTS}#symbol:{symbol}#direction:Close#pside:SHORT#Vol:{temp_short_quantity}#Quotes:{self.LatestMinQuotes[symbol]}")
                # if self.local_test:
                #     self.creat_backtest_virtual_order("Close", "SHORT", symbol, temp_short_quantity, future_twap[symbol])


        # 第6步 核查Close部分的交易不能超过预期持有的vol
        for symbol in self.TradingPlan["Close"]["LONG"][nowTS].keys():
            plan_vol = self.TradingPlan["Close"]["LONG"][nowTS][symbol]
            self.TradingPlan["Close"]["LONG"][nowTS][symbol] = np.minimum(plan_vol, long_vol_res[symbol])

        for symbol in self.TradingPlan["Close"]["SHORT"][nowTS].keys():
            plan_vol = self.TradingPlan["Close"]["SHORT"][nowTS][symbol]
            self.TradingPlan["Close"]["SHORT"][nowTS][symbol] = np.minimum(plan_vol, -short_vol_res[symbol])

        # 第7步 更新建仓成本
        for symbol in self.TradingPlan["Open"]["LONG"][nowTS].keys():
            plan_vol = self.TradingPlan["Open"]["LONG"][nowTS][symbol]
            if plan_vol > 0:
                self.updatePosCostAndTs(symbol, "LONG", nowTS)

        for symbol in self.TradingPlan["Open"]["SHORT"][nowTS].keys():
            plan_vol = self.TradingPlan["Open"]["SHORT"][nowTS][symbol]
            if plan_vol > 0:
                self.updatePosCostAndTs(symbol, "SHORT", nowTS)

        if self.local_test:
            for symbol in self.TradingPlan["Close"]["LONG"][nowTS].keys():
                self.creat_backtest_virtual_order("Close", "LONG", symbol, self.TradingPlan["Close"]["LONG"][nowTS][symbol], future_twap[symbol])
            for symbol in self.TradingPlan["Close"]["SHORT"][nowTS].keys():
                self.creat_backtest_virtual_order("Close", "SHORT", symbol, self.TradingPlan["Close"]["SHORT"][nowTS][symbol], future_twap[symbol])
            for symbol in self.TradingPlan["Open"]["LONG"][nowTS].keys():
                self.creat_backtest_virtual_order("Open", "LONG", symbol, self.TradingPlan["Open"]["LONG"][nowTS][symbol], future_twap[symbol])
            for symbol in self.TradingPlan["Open"]["SHORT"][nowTS].keys():
                self.creat_backtest_virtual_order("Open", "SHORT", symbol, self.TradingPlan["Open"]["SHORT"][nowTS][symbol], future_twap[symbol])



        t2 = time.time()
        # print(f"Cal Position Change Using = {t2 - t0}")

        pass
