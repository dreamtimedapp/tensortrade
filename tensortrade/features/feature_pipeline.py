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

import pandas as pd
import numpy as np

from gym import Space
from typing import List, Union, Callable

from tensortrade import Component
from .feature_transformer import FeatureTransformer


DTypeString = Union[type, str]
"""
Feature pipelines 将环境中观察到的数据转化为有意思的特征，从而让机器用于学习。
如果一个特征管道被添加到特定的交易所，那么在输出具体的buy, sell, hold交易命令前，观察到的特征将会传递给featurePipeline。
举个例子，一个特征管道在将观测到的数据生成动作之前，可以normalize所有的价格数据，并使得时间序列 stationary，添加移动平均列，移除掉不必要的列。

Feature pipelines 可以被任意数量的逗号分隔的transformers初始化。 每个 FeatureTransformer 需要被一些类数据初始化，如果什么都没有传进去，
那么所有的列都被转转化。

每个 feature transformer 有一个 transform方法，每个transform 转换一个单独的观测值从大数据中，同时保留必要的特征用于下一帧的转化。
由于这个原因，通常 featureTransformer 定期的执行。这个过程是被自动执行的，在FeaturePipeline或者InsrumentExchange被重置时。

示例如下：
from tensortrade.features import FeaturePipeline
from tensortrade.features.scalers import MinMaxNormalizer
from tensortrade.features.stationarity import FractionalDifference
from tensortrade.features.indicators import SimpleMovingAverage

price_columns = ["open", "high", "low", "close"]
normalize_price = MinMaxNormalizer(price_columns)
moving_averages = SimpleMovingAverage(price_columns)
difference_all = FractionalDifference(difference_order=0.6)
feature_pipeline = FeaturePipeline(steps=[normalize_price,
                                          moving_averages,
                                          difference_all])
exchange.feature_pipeline = feature_pipeline
"""

class FeaturePipeline(Component):
    """An pipeline for transforming observation data frames into features for learning."""
    registered_name = "features"

    def __init__(self, steps: List[FeatureTransformer], **kwargs):
        """
        Arguments:
            dtype: The `dtype` elements in the pipeline should be cast to.
        """
        self._steps = steps

        self._dtype: DTypeString = self.default('dtype', np.float16, kwargs)

    @property
    def steps(self) -> List[FeatureTransformer]:
        """A list of feature transformations to apply to observations."""
        return self._steps

    @steps.setter
    def steps(self, steps: List[FeatureTransformer]):
        self._steps = steps

    @property
    def dtype(self) -> DTypeString:
        """The `dtype` that elements in the pipeline should be input and output as."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        self._dtype = dtype

    def reset(self):
        """Reset all transformers within the feature pipeline."""
        for transformer in self._steps:
            transformer.reset()

    def transform_space(self, input_space: Space, column_names: List[str]) -> Space:
        """Get the transformed output space for a given input space.

        Args:
            input_space: A `gym.Space` matching the shape of the pipeline's input.
            column_names: A list of all column names in the input data frame.

        Returns:
            A `gym.Space` matching the shape of the pipeline's output.
        """
        output_space = input_space

        for transformer in self._steps:
            output_space = transformer.transform_space(output_space, column_names)

        return output_space

    def _transform(self, observations: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """Utility method for transforming observations via a list of `FeatureTransformer` objects."""
        for transformer in self._steps:
            observations = transformer.transform(observations, input_space)

        return observations

    def transform(self, observation: pd.DataFrame, input_space: Space) -> pd.DataFrame:
        """Apply the pipeline of feature transformations to an observation frame.

        Arguments:
            observation: A `pandas.DataFrame` corresponding to an observation within a `TradingEnvironment`.
            input_space: A `gym.Space` matching the shape of the pipeline's input.

        Returns:
            A `pandas.DataFrame` of features corresponding to an input oversvation.

        Raises:
            ValueError: In the case that an invalid observation frame has been input.
        """
        obs = observation.copy(deep=True)
        features = self._transform(obs, input_space)

        if not isinstance(features, pd.DataFrame):
            raise ValueError("A FeaturePipeline must transform a pandas.DataFrame into another pandas.DataFrame.\n \
                               Expected return type: {} `\n \
                               Actual return type: {}.".format(type(pd.DataFrame([])), type(features)))

        return features
