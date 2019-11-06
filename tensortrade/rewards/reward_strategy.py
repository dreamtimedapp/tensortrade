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

from abc import abstractmethod

from tensortrade import Component
from tensortrade.trades import Trade

"""
Reward策略在每个时间步通过接受Trade策略，反回一个float类型，获取每个特定的动作的收益
举个例子，如果一个动作是卖然后产生了正向收益，那么奖励策略就产生一个正向的数字比如1 鼓励机器人的交易
另一方面，如果一个动作比如卖，导致了亏损，那么奖励策略则教会机器不要做类似的事情
每个奖励策略都有get_reward方法的具体实现，反回每个时间步的奖励策略。

"""
class RewardStrategy(Component):

    registered_name = "rewards"

    def __init__(self):
        pass

    @property
    def exchange(self) -> 'InstrumentExchange':
        """The exchange being used by the current trading environments. Setting the exchange causes the strategy to reset."""
        return self._exchange

    @exchange.setter
    def exchange(self, exchange: 'InstrumentExchange'):
        self._exchange = exchange
        self.reset()

    def reset(self):
        """Optionally implementable method for resetting stateful strategies."""
        pass

    @abstractmethod
    def get_reward(self, current_step: int, trade: Trade) -> float:
        """
        Arguments:
            current_step: The environments's current timestep.
            trade: The real-trade executed and filled this timestep.

        Returns:
            A float corresponding to the benefit earned by the action taken this timestep.
        """
        raise NotImplementedError()
