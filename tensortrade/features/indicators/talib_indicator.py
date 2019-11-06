# Copyright 2019 The TensorTrade Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import talib
import numpy as np
import pandas as pd

from gym import Space
from copy import copy
from abc import abstractmethod
from typing import Union, List, Callable

from tensortrade.features.feature_transformer import FeatureTransformer
"""
基于 open,high,close,low的价格
* indicators  一系列的的指标，比如rsi,macd,adx等，这里有个疑问，是如何将
* lows 
   - 观测空间的最低值， 
  highs
   - 观测空间的最高值
   
使用范例：
from tensortrade.features import FeaturePipeline  
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.indicators import TAlibIndicator

price_columns = ["open", "high", "low", "close"]
normalize_price = MinMaxNormalizer(price_columns)
moving_averages = TAlibIndicator(["EMA","RSI","BB"])
difference_all = FractionalDifference(difference_order=0.6)
feature_pipeline = FeaturePipeline(steps=[normalize_price,
        moving_averages,
            difference_all])
exchange.feature_pipeline = feature_pipeline
"""
class TAlibIndicator(FeatureTransformer):
    """Adds one or more TAlib indicators to a data frame, based on existing open, high, low, and close column values."""

    def __init__(self, indicators: List[str], lows: Union[List[float], List[int]] = None, highs: Union[List[float], List[int]] = None):
        self._indicator_names = indicators
        self._indicators = list(
            map(lambda indicator_name: self._str_to_indicator(indicator_name), indicators))

        self._lows = lows or np.zeros(len(indicators))
        self._highs = highs or np.ones(len(indicators))

    def _str_to_indicator(self, indicator_name: str):
        return getattr(talib, indicator_name.upper())

    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        output_space = copy(input_space)
        shape_x, *shape_y = input_space.shape

        output_space.shape = (shape_x + len(self._indicators), *shape_y)

        for i in range(len(self._indicators)):
            output_space.low = np.append(output_space.low, self._lows[i])
            output_space.high = np.append(output_space.high, self._highs[i])

        return output_space

    def transform(self, X: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        for i in range(len(self._indicators)):
            indicator_name = self._indicator_names[i]
            indicator = self._indicators[i]

            X[indicator_name.upper()] = indicator(X['open'], X['high'], X['low'], X['close'])

        return X
