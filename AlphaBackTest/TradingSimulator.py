import os
import numpy as np
import pandas as pd
import time
import pickle
import copy
from .utils import *
from . import __config__ as cfg
import sys
sys.path.append(cfg.DS_PACKAGE_PATH)
import DataServer as ds


def TradingSimulatorConfigure(TradingParams = None, save=False, strategy_info_package_path = None):
    """
    输入策略config
    :param paras:
    :param save:
    :return:
    """
    default_paras = {
        # 回测基本参数
        # "FutureBal": {"USDT":100000000},
        "FutureBal": {"USDT":10000},
        "FeeRate": 0.0006,
        # 播放参数设置：
        "hist_bar_num": 5,
        "start_ts": None,
        "freq": 1,
    }

    if TradingParams is not None:
        for k in TradingParams.keys():
            default_paras[k] = TradingParams[k]

    # 存储Configure 用于predict或者其他用途
    if save:
        save_pickle(default_paras, f"trading_cfg.pkl", strategy_info_package_path)

    return default_paras


class TradingSimulator():
    def __init__(self, Strategy, spot_data_dict, future_data_dict, funding_data, trading_signal_dict = None, TradingParams = None, StartFuturePos = None):
        # 初始化Strategy
        self.Strategy = Strategy

        # 设定TradingConfig
        self.strategy_info_package_path = os.path.join(Strategy.strategy_base_path, Strategy.strategy_tag)
        simulator_cfg = TradingSimulatorConfigure(TradingParams=TradingParams,save=True,strategy_info_package_path=self.strategy_info_package_path)
        self.Strategy.STLogger.critical(f"SimMsg:FeeRate = {simulator_cfg['FeeRate']}")

        # 准备数据
        if spot_data_dict is not None:
            self.spot_data_dict = selCol_dict_of_dict(spot_data_dict, Strategy.spot_symbol_list)
        else:
            self.spot_data_dict = None
        self.future_data_dict = selCol_dict_of_dict(future_data_dict, Strategy.future_symbol_list)
        if trading_signal_dict is not None:
            self.trading_signal_dict = selCol_dict_of_dict(trading_signal_dict, Strategy.future_symbol_list)
        else:
            self.trading_signal_dict = None

        if spot_data_dict is not None:
            shape1 = self.spot_data_dict['open_time'].shape[0]
            shape2 = self.future_data_dict['open_time'].shape[0]
            if shape1 != shape2:
                raise Exception(f"spot future data dict not align, spot shape = {shape1}, future shape = {shape2}")


        # 准备播放的时间列表
        self.hist_bar_num = simulator_cfg["hist_bar_num"]
        self.start_ts = simulator_cfg["start_ts"]
        self.freq = simulator_cfg["freq"]
        self.ts_list = future_data_dict["open_time"].index.tolist()
        if self.start_ts is None:
            self.start_ts = self.ts_list[0]

        if funding_data is not None:
            self.funding_ts_list = [x//1000 * 1000 for x in funding_data.index.tolist()]
        self.funding_data = funding_data

        # 测试ts是否正确
        ts_array = np.array(self.ts_list)
        ts_diff = ts_array[1:] - ts_array[:-1]
        na_ts = np.sum(np.isnan(ts_diff))
        max_diff = np.nanmax(ts_diff)
        min_diff = np.nanmin(ts_diff)
        if (na_ts > 0) or (max_diff != 60000 * 1) or (min_diff != 60000 * 1):
            raise Exception("Error in Timestamp list")

        # 初始化设定账户信息
        Strategy.future_balance = simulator_cfg["FutureBal"]

        # 初始化期货仓位
        position_template = {
            "LONG": {"pa": 0, "ep": 0},
            "SHORT": {"pa": 0, "ep": 0},
            "BOTH": {"pa": 0, "ep": 0},
        }
        if StartFuturePos is None:
            # 赋值
            Strategy.future_position = {x: copy.deepcopy(position_template) for x in Strategy.future_symbol_list}

        else:
            # 检查FuturePos，填补不存在的仓位
            for x in Strategy.future_symbol_list:
                if x not in StartFuturePos.keys():
                    StartFuturePos[x] = position_template
            # 赋值
            Strategy.future_position = StartFuturePos

        # 读取必要交易信息
        self.FeeRate = simulator_cfg["FeeRate"]

        # 回测相关记录
        self.EventHistory = []
        self.ComissionFee = {x:0 for x in Strategy.future_balance.keys()}
        self.FundingFee = {x:0 for x in Strategy.future_balance.keys()}
        self.RealizedPnl = {x:0 for x in Strategy.future_symbol_list}
        self.AccountNetValue = []
        self.AccountNetValue_t0 = [] # 记录还没有进行操作前的净值
        self.RecordingTS = []
        self.HoldingVol = {"LONG":[],"SHORT":[]}
        self.HoldingVal = {"LONG":[],"SHORT":[]}

    def run(self):
        self.Strategy.init_my_strategy()

        LatestQuotes = self.future_data_dict["close"].iloc[0].to_dict()  # LatestQuotes原则上对于已经上市票应该不能出现空值
        TradingStatus = (~self.future_data_dict["close"].iloc[0].isna()).to_dict()

        # 计算账户USDT净值
        Balance = self.Strategy.future_balance["USDT"]
        UnPnl = CalUnRealizedPnl(self.Strategy, LatestQuotes, "USDT")

        self.AccountNetValue.append(Balance + UnPnl)
        self.AccountNetValue_t0.append(Balance + UnPnl)
        self.RecordingTS.append(self.future_data_dict["close"].index[0])
        self.Strategy.STLogger.critical(
            f"SimAccMsg#ts:{self.future_data_dict['close'].index[0]}#AccNetVal:{Balance + UnPnl}#LongVal:{np.nan}#LongMax:{np.nan}#ShortVal:{np.nan}#ShortMax:{np.nan}")

        t0 = time.time()

        funding_idx = 0
        for idx, ts in enumerate(self.ts_list):
            if self.funding_data is not None:
                # step1 根据funding fee进行扣费
                if funding_idx < len(self.funding_ts_list):
                    # break
                    funding_ts = self.funding_ts_list[funding_idx]
                    if ts>= funding_ts:
                        FundingFeeRateDict = self.funding_data.iloc[funding_idx].fillna(0).to_dict()
                        # 扣费
                        for symbol in self.Strategy.future_position.keys():
                            base_coin = self.Strategy.ContractInfo[symbol]["quote"]
                            temp_position = self.Strategy.future_position[symbol]
                            for side in temp_position.keys():
                                position_num = temp_position[side]["pa"]
                                if position_num !=0:
                                    self.Deal_FundingFee(FundingFeeRateDict[symbol], LatestQuotes[symbol] * position_num, base_coin, funding_ts, symbol)

                        # funding idx +1
                        funding_idx += 1
                        # funding_idx = np.minimum(len(self.funding_ts_list)-1, funding_idx) # 不能超过funding列表

            # step2 截取播放数据并且传给策略
            start_idx = np.maximum(idx - (self.hist_bar_num - 1), 0)
            end_idx = idx + 1
            if self.spot_data_dict is not None:
                cutted_spot = slice_dict_of_df(self.spot_data_dict, start_idx, end_idx)
            else:
                cutted_spot = None
            cutted_future = slice_dict_of_df(self.future_data_dict, start_idx, end_idx)
            if self.trading_signal_dict is not None:
                cutted_trading_signal = slice_dict_of_df(self.trading_signal_dict, start_idx, end_idx)

            nowTS = cutted_future["close"].index.tolist()[-1] + 60000

            # 如果全市场全部都是na，则跳过这个时间戳的播放
            if cutted_future["open_time"].iloc[-1].isna().sum() == cutted_future["open_time"].shape[1]:
                self.Strategy.STLogger.critical(f"SimMsg: ALL NA Close Price at {ts}")
                continue

            # step3 更新实时行情，用来计算仓位、市值、未平仓利润
            NewQuotes = cutted_future["close"].iloc[-1].to_dict()
            for symbol in NewQuotes.keys():
                if not np.isnan(NewQuotes[symbol]):
                    LatestQuotes[symbol] = NewQuotes[symbol]
                else:
                    if not np.isnan(LatestQuotes[symbol]):
                        self.Strategy.STLogger.warning(f"SimMsg:{ts}, {symbol}, Close Price Na Warning")  # 这种情况下说明数据有问题
                        # 标的停牌，使用上次收盘价清算
                        self.Deal_Type98(symbol,LatestQuotes[symbol],nowTS)

            TradingStatus = (~cutted_future["close"].iloc[-1].isna()).to_dict()


            # step4 传输给策略回调TradingSignal
            if not((ts >= self.start_ts) and ((ts - self.start_ts)%(60000 * self.freq) ==0)):
                continue

            # 计算账户USDT净值，记录未进行交易前 (净值更加贴近实盘)
            Balance = self.Strategy.future_balance["USDT"]
            UnPnl = CalUnRealizedPnl(self.Strategy, LatestQuotes, "USDT")
            self.AccountNetValue_t0.append(Balance + UnPnl)

            self.Strategy.OnBacktestMinBar(cutted_spot,cutted_future,cutted_trading_signal)
            # 先处理卖出
            event_list = self.Strategy.get_event_list()
            for event in event_list:
                if (event['event_type'] == 99) and (event["side"] == "Close"):
                    self.Deal_Type99(event,LatestQuotes,TradingStatus, nowTS)
            # 再处理买入
            for event in event_list:
                if (event['event_type'] == 99) and (event["side"] == "Open"):
                    self.Deal_Type99(event,LatestQuotes,TradingStatus, nowTS)

            self.Strategy.reset_event_list()

            # 计算账户USDT净值
            Balance = self.Strategy.future_balance["USDT"]
            UnPnl = CalUnRealizedPnl(self.Strategy, LatestQuotes, "USDT")

            self.RecordingTS.append(nowTS)
            self.AccountNetValue.append(Balance + UnPnl)
            self.Strategy.AccountNetValue.append(Balance + UnPnl)

            LongVol = {x:self.Strategy.future_position[x]["LONG"]["pa"] for x in self.Strategy.future_position.keys()}
            LongVal = {x:self.Strategy.future_position[x]["LONG"]["pa"] * LatestQuotes[x] for x in self.Strategy.future_position.keys()}
            ShortVol ={x:self.Strategy.future_position[x]["SHORT"]["pa"] for x in self.Strategy.future_position.keys()}
            ShortVal ={x:self.Strategy.future_position[x]["SHORT"]["pa"] * LatestQuotes[x] for x in self.Strategy.future_position.keys()}
            self.HoldingVol["LONG"].append(LongVol)
            self.HoldingVol["SHORT"].append(ShortVol)
            self.HoldingVal["LONG"].append(LongVal)
            self.HoldingVal["SHORT"].append(ShortVal)

            # if (nowTS // 60000) % (24 * 60) % 10 == 2:
            self.Strategy.STLogger.critical(f"SimAccMsg#ts:{nowTS}#AccNetVal:{Balance + UnPnl}#LongVal:{np.nansum(list(LongVal.values()))}#LongMax:{np.nanmax(list(LongVal.values()))}#ShortVal:{np.nansum(list(ShortVal.values()))}#ShortMax:{np.nanmin(list(ShortVal.values()))}")

        t1 = time.time()
        print(t1 - t0)

    def Deal_FundingFee(self, FundingFeeRate, HoldVal, base_coin, nowTS, symbol):
        """
        根据FundingFee 持仓市值（正为做多，负为做空）
        :param FundingFeeRate:
        :param HoldVal:
        :return:
        """
        fundingfee = FundingFeeRate * HoldVal
        self.Strategy.future_balance[base_coin] -= fundingfee
        self.FundingFee[base_coin] += fundingfee
        self.EventHistory.append({"EvtType":"FundingFee","TS":nowTS,"FundingRate":FundingFeeRate,"Notional":HoldVal,"Amt":fundingfee,"Coin":base_coin,"Symbol":symbol})
        self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:FundingFee#TS:{nowTS}#FundingRate:{FundingFeeRate}#Notional:{HoldVal}#Amt:{fundingfee}#Coin:{base_coin}#Symbol:{symbol}")

    def Deal_Fee(self, FeeAmount, base_coin, nowTS):
        """
        产生交易费用，将交易费用进行记录，并且在Balance里面扣除
        :return:
        """
        # 扣除交易费用
        self.Strategy.future_balance[base_coin] -= FeeAmount
        self.ComissionFee[base_coin] += FeeAmount
        self.EventHistory.append({"EvtType":"ComissionFee","TS":nowTS,"Amt":FeeAmount,"Coin":base_coin})
        self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:ComissionFess#TS:{nowTS}#Amt:{FeeAmount}#Coin:{base_coin}")

    def Deal_Type98(self, symbol, ClearPrice, nowTS):
        """
        标的停牌清算
        :return:
        """
        price = ClearPrice
        base_coin = self.Strategy.ContractInfo[symbol]["quote"]
        og_long_position = copy.deepcopy(self.Strategy.future_position[symbol]["LONG"])
        og_long_quantity = og_long_position["pa"]
        og_short_position = copy.deepcopy(self.Strategy.future_position[symbol]["SHORT"])
        og_short_quantity = np.abs(og_short_position["pa"])

        if og_long_quantity > 0:
            # 存在多头，强制平多
            # 更新Balance
            realized_pnl = (price - og_long_position["ep"]) * og_long_quantity
            self.Strategy.future_balance[base_coin] += realized_pnl
            self.RealizedPnl[symbol] += realized_pnl

            # 更新仓位
            new_position = 0
            new_entry_price = 0
            self.Strategy.future_position[symbol]["LONG"] = {"pa": new_position, "ep": new_entry_price}
            self.EventHistory.append({"EvtType": "ForceClearLong", "TS": nowTS, "Symbol": symbol, "Price": price, "Quantity": og_long_quantity,"Profit": realized_pnl})
            self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:ForceClearLong#TS:{nowTS}#Symbol:{symbol}#Price:{price}#Quantity:{og_long_quantity}")

            # 更新手续费
            fee = og_long_quantity * price * self.FeeRate
            self.Deal_Fee(fee, base_coin, nowTS)  # 增加手续费，扣除Balance

        if og_short_quantity > 0:
            # 存在空头，强制平空
            # 更新Balance
            realized_pnl = -(price - og_short_position["ep"]) * og_short_quantity
            self.Strategy.future_balance[base_coin] += realized_pnl
            self.RealizedPnl[symbol] += realized_pnl

            # 更新仓位
            new_position = 0
            new_entry_price = 0
            self.Strategy.future_position[symbol]["SHORT"] = {"pa": new_position, "ep": new_entry_price}
            self.EventHistory.append({"EvtType": "ForceClearShort", "TS": nowTS, "Symbol": symbol, "Price": price, "Quantity": og_short_quantity,"Profit": realized_pnl})
            self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:ForceClearShort#TS:{nowTS}#Symbol:{symbol}#Price:{price}#Quantity:{og_short_quantity}")

            # 更新手续费
            fee = og_short_quantity * price * self.FeeRate
            self.Deal_Fee(fee, base_coin, nowTS)  # 增加手续费，扣除Balance


    def Deal_Type99(self, OrderInfo, LatestQuotes, TradingStatus, nowTS):
        """
        根据指定价格和交易量进行交易
        :param OrderInfo:
        :param LatestQuotes:
        :param TradingStatus:
        :return:
        """
        symbol = OrderInfo["symbol"]
        base_coin = self.Strategy.ContractInfo[symbol]["quote"]
        quantity = OrderInfo["quantity"]
        price = OrderInfo["price"]
        val = quantity * price

        # 首先确定目标可交易
        if not TradingStatus[symbol]:
            self.Strategy.STLogger.warning("SimEvtMsg#EvtType:OrderRej#Info:Close Price is Na Not Tradable")
            return 0

        # 确定给定价格是不是na
        if np.isnan(price) or price<0:
            self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:OrderRej#Info:Order Price is {price} Not Tradable")
            return 0

        # 开仓
        if OrderInfo["side"] == "Open":
            # 计算持仓市值，确认可用资金
            availableBal = CalAvailableBal(self.Strategy, LatestQuotes, base_coin)
            # TODO 这里的判断是估算的，应为真实开仓的时候还需要计算相对标记价格的开仓损益
            required_margin = val / self.Strategy.leverage
            # 判断是否能开仓位
            if required_margin > availableBal:
                self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:OrderRej#Info:Not Enough Available Balance, required = {required_margin}, available = {availableBal}")
                return 0

            # 如果可以开仓
            # 多头
            if OrderInfo["positionSide"] == "LONG":
                # 成功开多，1、更新仓位 2、更新累计手续费 更新Balance
                og_long_position = copy.deepcopy(self.Strategy.future_position[symbol]["LONG"])
                # 更新仓位
                new_position = og_long_position["pa"] + quantity
                new_entry_price = (og_long_position["pa"] * og_long_position["ep"] + val) / new_position
                self.Strategy.future_position[symbol]["LONG"] = {"pa":new_position,"ep":new_entry_price}
                self.EventHistory.append({"EvtType":"OpenLong","TS":nowTS,"Symbol":symbol,"Price":price,"Quantity":quantity})
                self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:OpenLong#TS:{nowTS}#Symbol:{symbol}#Price:{price}#Quantity:{quantity}")
                # 更新手续费
                fee = val * self.FeeRate
                self.Deal_Fee(fee,base_coin, nowTS) # 增加手续费，扣除Balance
                pass

            # 空头
            elif OrderInfo["positionSide"] == "SHORT":
                # 成功开空， 1、更新仓位 2、更新累计手续费 更新Balance
                og_short_position = copy.deepcopy(self.Strategy.future_position[symbol]["SHORT"])
                # 更新仓位
                new_position = og_short_position["pa"] - quantity
                new_entry_price = (og_short_position["pa"] * og_short_position["ep"] - val) / new_position
                self.Strategy.future_position[symbol]["SHORT"] = {"pa": new_position, "ep": new_entry_price}
                self.EventHistory.append({"EvtType":"OpenShort","TS":nowTS,"Symbol":symbol,"Price":price,"Quantity":quantity})
                self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:OpenShort#TS:{nowTS}#Symbol:{symbol}#Price:{price}#Quantity:{quantity}")
                # 更新手续费
                fee = val * self.FeeRate
                self.Deal_Fee(fee, base_coin, nowTS)  # 增加手续费，扣除Balance

                pass
            else:
                raise NotImplementedError("Wrong postionSide")

        # 平仓
        elif OrderInfo["side"] == "Close":
            # 多头
            if OrderInfo["positionSide"] == "LONG":
                og_long_position = copy.deepcopy(self.Strategy.future_position[symbol]["LONG"])
                # 是否可以平多
                if quantity > og_long_position["pa"]:
                    self.Strategy.STLogger.critical(f"SimMsg: Order Auto Changed, Not Enough Position to close long, symbol = {symbol}, required = {quantity}, postion = {og_long_position['pa']}")
                    quantity = og_long_position["pa"]

                # 更新Balance
                realized_pnl = (price - og_long_position["ep"]) * quantity
                self.Strategy.future_balance[base_coin] += realized_pnl
                self.RealizedPnl[symbol] += realized_pnl

                # 更新仓位
                new_position = og_long_position["pa"] - quantity
                # 如果剩余精度很小则归0
                if np.abs(new_position) < 1e-8:
                    new_position = 0
                if new_position <0:
                    raise Exception("Long Vol < 0 Problem")
                new_entry_price = og_long_position["ep"]
                if new_position == 0:
                    new_entry_price = 0
                self.Strategy.future_position[symbol]["LONG"] = {"pa": new_position, "ep": new_entry_price}
                self.EventHistory.append({"EvtType":"CloseLong","TS":nowTS,"Symbol":symbol,"Price":price,"Quantity":quantity, "Profit":realized_pnl})
                self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:CloseLong#TS:{nowTS}#Symbol:{symbol}#Price:{price}#Quantity:{quantity}")

                # 更新手续费
                fee = val * self.FeeRate
                self.Deal_Fee(fee, base_coin, nowTS)  # 增加手续费，扣除Balance

                pass

            # 空头
            elif OrderInfo["positionSide"] == "SHORT":
                og_short_position = copy.deepcopy(self.Strategy.future_position[symbol]["SHORT"])
                # 是否可以平空
                if quantity > np.abs(og_short_position["pa"]):
                    self.Strategy.STLogger.critical(f"SimMsg: Order Auto Changed, Not Enough Position to close short, symbol = {symbol}, required = {quantity}, postion = {np.abs(og_short_position['pa'])}")
                    quantity = np.abs(og_short_position["pa"])

                # 更新Balance
                realized_pnl = (og_short_position["ep"] - price) * quantity
                self.Strategy.future_balance[base_coin] += realized_pnl
                self.RealizedPnl[symbol] += realized_pnl

                # 更新仓位
                new_position = og_short_position["pa"] + quantity
                # 如果剩余精度很小则归0
                if np.abs(new_position) < 1e-8:
                    new_position = 0
                if new_position > 0:
                    raise Exception("Short Vol > 0 Problem")
                new_entry_price = og_short_position["ep"]
                if new_position ==0:
                    new_entry_price = 0
                self.Strategy.future_position[symbol]["SHORT"] = {"pa": new_position, "ep": new_entry_price}
                self.EventHistory.append({"EvtType":"CloseShort","TS":nowTS,"Symbol":symbol,"Price":price,"Quantity":quantity, "Profit":realized_pnl})
                self.Strategy.STLogger.warning(f"SimEvtMsg#EvtType:CloseShort#TS:{nowTS}#Symbol:{symbol}#Price:{price}#Quantity:{quantity}")

                # 更新手续费
                fee = val * self.FeeRate
                self.Deal_Fee(fee, base_coin, nowTS)  # 增加手续费，扣除Balance

                pass

            else:
                raise NotImplementedError("Wrong postionSide")

        else:
            raise NotImplementedError("Wrong Side")
