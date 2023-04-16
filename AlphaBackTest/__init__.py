import os
import sys

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
from .FutureBackTest import FutureBaseStrategy
# from .SpotBackTest import SpotBaseStrategy
from .TradingSimulator import TradingSimulatorConfigure, TradingSimulator
# from .SpotTradingSimulator import SpotTradingSimulatorConfigure, SpotTradingSimulator
from .utils import *
from . import __config__ as cfg