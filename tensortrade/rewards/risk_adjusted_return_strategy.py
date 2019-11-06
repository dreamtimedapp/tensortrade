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
# limitations under the License

import pandas as pd
import numpy as np

from typing import Callable

from tensortrade.rewards import RewardStrategy
from tensortrade.trades import TradeType, Trade

"""
奖励策略，奖励机器人增加其净资产的行为，同时惩罚更不良导致亏损的策略
在交易时，通常不仅查看模型的整体收益，也在查看与其他模型相比的收益，以及收益随着时间的波动性，比如最大回撤。
有两个主要的策略，sharpe比率和sortino比率。

* sharpe 比率着眼于投资组合的整体变动，并通过给巨额变动较低的分数的方式进行惩罚
* sortino ratio采用相同的方式，尽管它更侧重与上行空间。这意味着 sortino ratio会在价格上涨时给巨大的分数，
而在价格下降时只会给出负的得分。
这是RL算法一个很好的方向，我们不想承受沉重的下跌风险，但是想要承担较大的上涨空间。仅仅通过使用此指标就能够帮助
我们减轻负面影响和增加上行空间带来更大的进步。

"""
class RiskAdjustedReturnStrategy(RewardStrategy):
    """A reward strategy that rewards the agent for increasing its net worth, while penalizing more volatile strategies.
    """

    def __init__(self, return_algorithm: str = 'sharpe', risk_free_rate: float = 0., target_returns: float = 0.):
        """
        Args:
            return_algorithm (optional): The risk-adjusted return metric to use. Options are 'sharpe' and 'sortino'. Defaults to 'sharpe'.
            risk_free_rate (optional): The risk free rate of returns to use for calculating metrics. Defaults to 0.
            target_returns (optional): The target returns per period for use in calculating the sortino ratio. Default to 0.
        """
        self._return_algorithm = self._return_algorithm_from_str(return_algorithm)
        self._risk_free_rate = risk_free_rate
        self._target_returns = target_returns

    def _return_algorithm_from_str(self, algorithm_str: str) -> Callable[[pd.DataFrame], float]:
        if algorithm_str == 'sharpe':
            return self._sharpe_ratio
        elif algorithm_str == 'sortino':
            return self._sortino_ratio

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Return the sharpe ratio for a given series of a returns.

        https://en.wikipedia.org/wiki/Sharpe_ratio
        """
        return (returns.mean() - self._risk_free_rate) / (returns.std() + 1E-9)

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Return the sortino ratio for a given series of a returns.

        https://en.wikipedia.org/wiki/Sortino_ratio
        """
        downside_returns = pd.Series([0])

        returns[returns < self._target_returns] = returns ** 2

        expected_return = returns.mean()
        downside_std = np.sqrt(downside_returns.mean())

        return (expected_return - self._risk_free_rate) / (downside_std + 1E-9)

    def get_reward(self, current_step: int, trade: Trade) -> float:
        """Return the reward corresponding to the selected risk-adjusted return metric."""
        returns = self._exchange.performance['net_worth'].diff()

        risk_adjusted_return = self._return_algorithm(returns)

        return risk_adjusted_return
