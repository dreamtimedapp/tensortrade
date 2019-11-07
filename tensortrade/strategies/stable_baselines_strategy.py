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

import os
import gym
import json

import pandas as pd
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Callable, List, Dict

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines import DQN

from tensortrade.environments.trading_environment import TradingEnvironment
from tensortrade.strategies import TradingStrategy

"""

700/5000
在此示例中，我们将使用“Baselines”库为我们的交易策略提供学习代理，
但是，TensorTrade框架与许多强化学习库（例如Tensorforce，
Ray的RLLib，OpenAI的Baselines，Intel的Coach或TensorFlow系列中的任何东西，例如TF Agents。
自定义TensorTrade学习代理可能会在将来添加到此框架中，尽管它将
框架的目标始终是与尽可能多的现有强化学习库互操作
可能的，因为该空间同时增长很多。 但就目前而言，“OpenAI的Baselines”既简单又强大
足以满足我们的需求。
代码示例如下：
from stable_baselines.common.policies import MlpLnLstmPolicy
from statble_baselines import PPO2

model = PPO2
policy = MlpLnLstmPolicy 
params = { "learning_rate": 1e-5 }
agent = model(policy, environment, model_kwargs=params)

在这个例子中我们使用GPU优化模型并带有一层 lstm神经网络。

"""

class StableBaselinesTradingStrategy(TradingStrategy):
    """A trading strategy capable of self tuning, training, and evaluating with stable-baselines.

    Parameters:
    ----------
    environments : `TradingEnvironment`
        An instance of a trading environments for the agent to real-trade within.
    model : BaseRLModel
        The RL model to create the agent with.
        Defaults to DQN.
    policy : Union[str, BasePolicy]
        The RL policy to train the agent's model with.
        Defaults to 'MlpPolicy'.
    model_kwargs : any
        Any additional keyword arguments to adjust the model.
    kwargs : dict
        Optional keyword arguments to adjust the strategy.
    """

    def __init__(self,
                 environment: TradingEnvironment,
                 model: BaseRLModel = DQN,
                 policy: Union[str, BasePolicy] = 'MlpPolicy',
                 model_kwargs: any = {},
                 **kwargs):
        self._model = model
        self._model_kwargs = model_kwargs

        self.environment = environment
        self._agent = self._model(policy, self._environment, **self._model_kwargs)

    @property
    def environment(self) -> 'TradingEnvironment':
        """A `TradingEnvironment` instance for the agent to real-trade within."""
        return self._environment

    @environment.setter
    def environment(self, environment: 'TradingEnvironment'):
        self._environment = DummyVecEnv([lambda: environment])

    def restore_agent(self, path: str):
        """Deserialize the strategy's learning agent from a file.

        Arguments:
            path: The `str` path of the file the agent specification is stored in.
        """
        self._agent = self._model.load(path, self._environment, self._model_kwargs)

    def save_agent(self, path: str):
        """Serialize the learning agent to a file for restoring later.

        Arguments:
            path: The `str` path of the file to store the agent specification in.
        """
        self._agent.save(path)

    def tune(self, steps: int = None, episodes: int = None, callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        raise NotImplementedError

    def run(self, steps: int = None, episodes: int = None, episode_callback: Callable[[pd.DataFrame], bool] = None) -> pd.DataFrame:
        if steps is None and episodes is None:
            raise ValueError(
                'You must set the number of `steps` or `episodes` to run the strategy.')

        steps_completed = 0
        episodes_completed = 0
        average_reward = 0

        obs, state, dones = self._environment.reset(), None, [False]

        performance = {}

        while (steps is not None and (steps == 0 or steps_completed < steps)) or (episodes is not None and episodes_completed < episodes):
            actions, state = self._agent.predict(obs, state=state, mask=dones)
            obs, rewards, dones, info = self._environment.step(actions)

            steps_completed += 1
            average_reward -= average_reward / steps_completed
            average_reward += rewards[0] / (steps_completed + 1)

            exchange_performance = info[0].get('exchange').performance
            performance = exchange_performance if len(exchange_performance) > 0 else performance


            if dones[0]:
                if episode_callback is not None and episode_callback(self._environment._exchange.performance):
                    break

                episodes_completed += 1
                obs = self._environment.reset()

                print("performance: {}, Average reward: {},steps: {}. ".format(performance,average_reward,episodes_completed))

        print("Finished running strategy.")
        print("Total episodes: {} ({} timesteps).".format(episodes_completed, steps_completed))
        print("Average reward: {}.".format(average_reward))

        return performance
