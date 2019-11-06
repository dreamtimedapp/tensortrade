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

import numpy as np

from typing import Union, List
from gym.spaces import Discrete

from tensortrade.actions import ActionStrategy, TradeActionUnion, DTypeString
from tensortrade.trades import Trade, TradeType


class MultiDiscreteActionStrategy(ActionStrategy):
    """Discrete strategy, which calculates the real-trade amount as a fraction of
    the total balance for each instrument provided.

    Parameters
    ----------
    actions_per_instrument : int
        The number of bins to divide the total balance by.
        Defaults to 20 (i.e. 1/20, 2/20, ..., 20/20).
    max_allowed_slippage_percent : float
        The maximum amount above the current price the strategy will pay for
        an instrument.
        Defaults to 1.0 (i.e. 1%).
    """

    def __init__(self, actions_per_instrument: int = 20, max_allowed_slippage_percent: float = 1.0):
        super().__init__(
            action_space=Discrete(len(self.context.products) * actions_per_instrument),
            dtype=np.int64
        )

        self._products = self.context.products

        self._actions_per_instrument = \
            self.context.get('actions_per_instrument', None) or \
            actions_per_instrument

        self._max_allowed_slippage_percent = \
            self.context.get('max_allowed_slippage_percent', None) or \
            max_allowed_slippage_percent

    @property
    def dtype(self) -> DTypeString:
        """A type or str corresponding to the dtype of the `action_space`."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DTypeString):
        raise ValueError(
            'Cannot change the dtype of a `SimpleDiscreteStrategy` due to '
            'the requirements of `gym.spaces.Discrete` spaces. ')

    def get_trade(self, action: TradeActionUnion) -> Trade:
        """The real-trade type is determined by `action % len(TradeType)`, and
        the real-trade amount is determined by the multiplicity of the action.

        For example:
            0 = HOLD
            1 = LIMIT_BUY|0.25
            2 = MARKET_BUY|0.25
            5 = HOLD
            6 = LIMIT_BUY|0.5
            7 = MARKET_BUY|0.5
            etc.
        """
        product_idx = int(action / self._actions_per_instrument)
        product = self._products[product_idx]

        n_splits = int(self._actions_per_instrument / len(TradeType))
        trade_type = TradeType(action % len(TradeType))
        trade_amount = int(action / len(TradeType)) * \
            float(1 / n_splits) + (1 / n_splits)
        trade_amount = trade_amount - product_idx

        current_price = self._exchange.current_price(symbol=product)
        base_precision = self._exchange.base_precision
        instrument_precision = self._exchange.instrument_precision

        amount = self._exchange.instrument_balance(product)
        price = current_price

        if trade_type is TradeType.MARKET_BUY or trade_type is TradeType.LIMIT_BUY:
            price_adjustment = 1 + (self._max_allowed_slippage_percent / 100)
            price = max(round(current_price * price_adjustment, base_precision), base_precision)
            amount = round(self._exchange.balance * 0.99 *
                           trade_amount / price, instrument_precision)

        elif trade_type is TradeType.MARKET_SELL or trade_type is TradeType.LIMIT_SELL:
            price_adjustment = 1 - (self._max_allowed_slippage_percent / 100)
            price = round(current_price * price_adjustment, base_precision)
            amount_held = self._exchange.portfolio.get(product, 0)
            amount = round(amount_held * trade_amount, instrument_precision)

        return Trade(product, trade_type, amount, price)
