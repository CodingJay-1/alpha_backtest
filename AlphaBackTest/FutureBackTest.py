import json
import math
import os
import pickle
import copy

import pandas as pd

from .utils import *
from . import __config__ as cfg


class FutureBaseStrategy():
    def __init__(self):
        print("Init Future Strategy")
        # 是否回测交易
        self.local_test = True

        self.balance = {}
        self.future_balance = {}
        self.future_orders = {}
        self.exchangeinfo = {}
        self.pairinfo = {}
        self.future_exchangeinfo = {}
        self.size_dict = {}  # tick_size, step_size
        self.lot_dict = {}  # price_lot,size_lot,存放round精度
        self.pair_dict = {}  # 利用分词技术进行币对拆分{'ETHUSDT': ['ETH', 'USDT']}
        self.klines = {}  # 5分钟k线数据,总计6小时
        self.future_klines = {}  # 5分钟future k线数据,总计6小时
        self.mark_price = {}
        self.future_position = {}
        if self.local_test:
            self.event_cache_list = []

    # 分钟回测专用回调函数
    def creat_backtest_virtual_order(self, side, positionSide, symbol, quantity, price):
        if quantity ==0:
            print(side,positionSide,symbol,quantity,price)
            print("Order Error: Quantity == 0, Order Rej")
            raise Exception("Order Error: Quantity = 0, Order Rej")
            return
        elif quantity <0:
            raise Exception("Order Error: Quantity < 0, Order Rej")
        self.event_cache_list.append({"event_type":99,"side":side, "positionSide":positionSide, "symbol":symbol, "quantity":quantity, "price":price})

    def get_event_list(self):
        return self.event_cache_list

    def reset_event_list(self):
        self.event_cache_list = []