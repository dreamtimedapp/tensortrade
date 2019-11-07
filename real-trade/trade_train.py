import os
import sys
import warnings
import numpy
import ccxt
import pandas as pd
import numpy as np
from time import time
import math
import cProfile as cp
from tensortrade.rewards import RiskAdjustedReturnStrategy
from tensortrade.actions import DiscreteActionStrategy
from tensortrade.exchanges.simulated import SimulatedExchange
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features import FeaturePipeline
from tensortrade.environments import TradingEnvironment
from tensortrade.strategies import TensorforceTradingStrategy
from tensortrade.strategies import StableBaselinesTradingStrategy
from stable_baselines import PPO2

DATA_FILE_PATH ='./data/processed/binance/btc_usdt_1h.csv'

def load_data():
    df = pd.read_csv(DATA_FILE_PATH, index_col=[0])
    # number of days we want to pull from the dataframe
    days_of_data = 365
    # number of data frames (our DF is in 1h timesteps)
    frames = days_of_data * 24 * 5
    # frames = len(df)
    train_test_percentage = 0.4

    x_train = int(frames * (1 - train_test_percentage))
    x_test = int(frames - x_train)

    df_train = df[-frames:(-x_test - 1)]
    df_test = df[-x_test:]
    print("Friend Shape", df.shape)
    print('train shape', df_train.shape)
    print('test shape', df_test.shape)
    print('columns', df.columns)
    return x_train,x_test,df_train,df_test

def train_bot():
    normalize = MinMaxNormalizer(inplace=True)
    feature_pipeline = FeaturePipeline(steps=[normalize])
    reward_strategy = RiskAdjustedReturnStrategy()
    action_strategy = DiscreteActionStrategy(n_actions=20)
    x_train, x_test, df_train, df_test = load_data()
    exchange = SimulatedExchange(data_frame=df_train,
                        pretransform=True,
                        base_instrument='USDT',
                        commission_percent=0.75,
                        window_size=1,
                        max_allowed_slippage_percent=3.0,
                        min_order_amount=1E-4,
                        min_trade_amount=1E-4,
                        observation_columns=df_train.columns
                        )
    environment = TradingEnvironment(exchange=exchange,
                              action_strategy=action_strategy,
                              reward_strategy=reward_strategy,
                              feature_pipeline=feature_pipeline)

    strategy = StableBaselinesTradingStrategy(environment=environment,model=PPO2)
    print("Running through ", strategy, ' steps')
    strategy.restore_agent('./agent')
    strategy.run(episodes=2)
    strategy.save_agent('./agent')


train_bot()









