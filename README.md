# backtest_RL_mmbot
backtest environment of market maker trading bot by reinforcement learning using Bitflyer's trading historical data.

Bot learns the range of appropriate limit order by Q-learning. The input is the current position and the output is the limit order position for the most recent trade price.

summary.csv can be obtained by running getdata.ipynb and processing trade history data.
