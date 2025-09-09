import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import scipy.stats
from Strategy_test import strategy_metrics
from Convex_opt import convex_optimization
import cvxpy as cvx # pip install cvxpy
from sklearn.covariance import LedoitWolf

def simple_optimal_weights(rets):
    """
    Calculate optimal weights for a portfolio based on the covariance matrix and mean returns.
    Uses the formula w = inv(sigma) * mu / sum(abs(inv(sigma) * mu))
    """
    def optimal_weights(sigma, mu):
        wgt = np.linalg.inv(sigma) @ mu
        wgt = wgt/np.abs(wgt).sum()
        return wgt
    sigma = rets.cov()
    mu = rets.mean()
    wgt = optimal_weights(sigma, mu)
    comb = (rets*wgt).sum(axis = 1)
    return comb


Daily_bar = pd.read_pickle("price_1d_20210101_20250817.pk")
strategy = strategy_metrics(Daily_bar)
sharpe_ratio = strategy.sharpe_ratio

ret = strategy.calculate_returns()
port_ret = strategy.portfolio

""" Volume Strategy """
# simple signal based on volume rank 0.95 sr
# sr 0.956
volume_signal = Daily_bar['Volume'].rank(axis=1, pct=True)
volume_weights = strategy.portfolio_weights(volume_signal)
# strategy.summary_stats(volume_weights)
v_port_ret = port_ret(volume_signal, ret)
sharpe_ratio(v_port_ret)

# import market cap data
import pickle
# info = {}
# for i in range(len(ret.columns)):
#     temp = yf.Ticker(ret.columns[i]).info
#     info[ret.columns[i]] = temp
# with open("sp500_generalinfo_all.pk", "wb") as file:
#     pickle.dump(info, file)
with open('sp500_generalinfo_all.pk', 'rb') as file:
    info = pickle.load(file)
market_cap = pd.Series({k: v['marketCap'] for k, v in info.items()})
pct_mcap_traded = (Daily_bar['Volume'] * Daily_bar['Adj Close']/market_cap)

# signal of volume greater than 20% of 20 days moving average volume
high_volume_signal_02 = (pct_mcap_traded > pct_mcap_traded.rolling(window=20).mean() * (1+0.2)).astype(int)
high_volume_weights_02 = high_volume_signal_02.div(high_volume_signal_02.abs().sum(axis=1), axis=0)
high_volume_ret_02 = (high_volume_weights_02.shift() * ret).sum(axis=1)
strategy.summary_stats(high_volume_weights_02)
sharpe_ratio(high_volume_ret_02)

high_volume_signal_045 = (pct_mcap_traded < pct_mcap_traded.rolling(window=20).mean() * (1 - 0.1)).astype(int)
high_volume_weights_045 = high_volume_signal_045.div(high_volume_signal_045.abs().sum(axis=1), axis=0)
high_volume_ret_045 = (high_volume_weights_045.shift() * ret).sum(axis=1)
sharpe_ratio(high_volume_ret_045)
""" Volume Strategy """

# create a ranked signal based on zscore of 5 day returns
# sr 0.738
z_score_5d_signal = (ret.rolling(window=5).mean() - ret.rolling(window=5).mean().rolling(window=20).mean()) / ret.rolling(window=5).std().rolling(window=20).std()
z_score_5d_signal = z_score_5d_signal.rank(axis=1, pct=True)
z_score_5d_weights = z_score_5d_signal.div(z_score_5d_signal.abs().sum(axis=1), axis=0)
z_score_5d_port_ret = port_ret(z_score_5d_signal, ret)
sharpe_ratio(z_score_5d_port_ret)

# create a signal entry when diff_ema_20_norm is above 0.02 and short when above 0.05
# sr 0.970
diff_ema_20_norm = Daily_bar['Adj Close'].ewm(span = 20, adjust=False).mean()
diff_ema_20_norm = (Daily_bar['Adj Close'] - diff_ema_20_norm) /diff_ema_20_norm

""" EMA 20 strategy | 1.24 SR"""
# grid searched for both above and below ema
# entry when 2 percent above 20 day ema and sell short when 5 percent above
ema_port_long_signal = (diff_ema_20_norm > 0.02).astype(int) - 2*(diff_ema_20_norm > 0.05).astype(int)
# short when below -0.03 and long when below -0.04
ema_port_short_signal = -(diff_ema_20_norm < -0.03).astype(int) + 2*(diff_ema_20_norm < -0.04).astype(int)

ema_combined_signal = ema_port_long_signal + ema_port_short_signal
ema_combined_weights = strategy.portfolio_weights(ema_combined_signal)
strategy.summary_stats(ema_combined_weights)
ema_combined_port_ret = port_ret(ema_combined_signal, ret)
sharpe_ratio(ema_combined_port_ret)
""" EMA 20 strategy | 1.24 SR"""

# rolling zscore 10 day/252 day strategy tanh and rank
# sr 0.774
zscore_10_252 = np.sqrt(10)*(ret.rolling(10,min_periods=1).mean() - ret.rolling(365,min_periods=10).mean())
zscore_10_252 = zscore_10_252 / ret.rolling(365,min_periods=10).std()
# tanh the zscore to avoid extreme values
zscore_10_252 = np.tanh(zscore_10_252)
# rank the zscore
zscore_signal = zscore_10_252.rank(axis=1, pct=True)
zscore_port_ret = port_ret(zscore_signal, ret)
sharpe_ratio(zscore_port_ret)

# rolling sharpe for the stock
rolling_sharpe_20 = ret.rolling(20).mean()/ret.rolling(20).std()* np.sqrt(252)

# enter when rolling sharpe is above 1
# sr 0.64
################### try rank the rolling sharpe ratio 
rolling_sharpe_20_signal = (rolling_sharpe_20 > 1) * 1
rolling_sharpe_20_port_ret = port_ret(rolling_sharpe_20_signal, ret)
# calculate the sharpe ratio of the portfolio
sharpe_ratio(rolling_sharpe_20_port_ret)

# Daytime and overnight return different sign long short strategy
# sr 0.795
Daily_bar_adjusted_open = Daily_bar['Open'] * (Daily_bar['Adj Close']/Daily_bar['Close'])
ret_daytime = Daily_bar['Adj Close'] / Daily_bar_adjusted_open - 1
ret_overnight = (1 + ret)/ (1 + ret_daytime) - 1
# signal overnight leads daytime
dt_on_signal = ((ret_overnight > 0) & (ret_daytime < 0)).astype(int) - ((ret_overnight < 0) & (ret_daytime > 0)).astype(int)
dt_on_port_ret = strategy.portfolio(dt_on_signal, ret)
dt_on_port_weights = strategy.portfolio_weights(dt_on_signal)
strategy.summary_stats(dt_on_port_weights)

# simple 5 days reversal strategy with rank
# 0.83 sr
reversal_5d_signal = (-1 * ret.rolling(5).sum()).rank(1)
reversal_5d_weights = strategy.portfolio_weights(reversal_5d_signal)
signal_5d_reversal_ret = (reversal_5d_weights.shift() * ret).sum(axis=1)
strategy.summary_stats(reversal_5d_weights)

""" ICA Alpha Strategy / prone to overfit"""

spy = strategy.benchmark
spy = spy.fillna(0)
spy = spy['SPY']
def pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=15, benchmark=spy, param_i = 0.4 , param_j = 0.5):
    """
    Focus purely on extracting non-market, non-Gaussian components
    """
    rets = ret.fillna(0)
    Strategy_weights = pd.DataFrame(0, index=ret.index, columns=ret.columns, dtype=float)
    
    for i in range(rolling_window, ret.shape[0]+1):
        train_data = rets[i-rolling_window:i]
        
        # Remove market factor explicitly
        market_factor = benchmark[i-rolling_window:i]
        market_betas = np.array([np.cov(train_data[col], market_factor)[0, 1] /
                               np.var(market_factor) for col in train_data.columns])
        
        market_neutral = train_data.values - np.outer(market_factor, market_betas)
        
        # Direct ICA on market-neutral returns (no PCA)
        ica = FastICA(n_components=min(ica_n_components, train_data.shape[1]), 
                     max_iter=2000, tol=1e-6, random_state=0)
        ica.fit(market_neutral)
        ica_mixing = ica.mixing_

        # Component analysis and weighting logic
        component_stats = {}
        
        for j in range(ica_n_components):
            w = ica_mixing[:, j]
            w = w / (np.abs(w).sum())

            # Calculate component returns
            port_ret = (market_neutral * w).sum(axis=1)

            # Alpha score based on - market corr, sigmoid kurtosis, and autocorr
            market_corr = np.corrcoef(port_ret, market_factor)[0, 1]
            kurtosis = 1 / (1 + np.exp(-scipy.stats.kurtosis(port_ret)))
            autocorr = np.corrcoef(port_ret[:-1], port_ret[1:])[0, 1]
            
            signal_strength = (-abs(market_corr) * param_i + 
                            abs(kurtosis) * param_j + 
                            max(0, autocorr) * (1 - param_i - param_j))

            # momentum signals
            recent_perf = np.mean(port_ret[-20:])  # Last 20 days performance
            signal_direction = 1 if recent_perf > 0 else -1

            # short_term_mom = np.mean(port_ret[-5:])   # 5-day momentum
            # medium_term_mom = np.mean(port_ret[-20:])  # 20-day momentum
            # signal_direction = 1 if short_term_mom * medium_term_mom > 0 else -1
            # stock_short_term_mom = np.mean(market_neutral[-5:, :], axis=0)   # shape: (n_stocks,)
            # stock_medium_term_mom = np.mean(market_neutral[-20:, :], axis=0) # shape: (n_stocks,)
            # stock_momentum = stock_short_term_mom * stock_medium_term_mom    # elementwise
            # stock_momentum = (stock_short_term_mom > stock_medium_term_mom) * 1

            component_stats[j] = {
                'signal_strength': signal_strength,
                # 'stock_momentum': stock_momentum,
                'signal_direction': signal_direction,
                'weights': w
            }
        # Aggregate signals across components
        final_signal = np.zeros(ret.shape[1])
        for stats in component_stats.values():
            final_signal += stats['signal_strength'] * stats['signal_direction'] * stats['weights']

        # final_weights = final_signal / (np.abs(final_signal).sum())
        Strategy_weights.iloc[i-1] = final_signal
    return Strategy_weights
ica_alpha_strat_signal = pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=15, benchmark=spy)
ica_alpha_strat_weights = ica_alpha_strat_signal.div(ica_alpha_strat_signal.abs().sum(axis=1), axis=0).fillna(0)
ica_alpha_strat_port_ret = (ica_alpha_strat_weights.shift() * ret).sum(axis=1)
strategy.summary_stats(ica_alpha_strat_weights)

# combine the strategies with simple optimal weights
Portfolio_comb = [zscore_port_ret, ema_combined_port_ret, v_port_ret, z_score_5d_port_ret, rolling_sharpe_20_port_ret, dt_on_port_ret, signal_5d_reversal_ret, high_volume_ret_02, ica_alpha_strat_port_ret]
Portfolio_comb = pd.concat(Portfolio_comb, axis=1)
Portfolio_comb.columns = ['zscore_port_ret', 'ema_combined_port_ret', 'v_port_ret', 'z_score_5d_port_ret', 'rolling_sharpe_20_port_ret', 'dt_on_port_ret', 'signal_5d_reversal_ret', 'high_volume_ret_02', 'ica_alpha_strat_port_ret']
for i in Portfolio_comb:
    strat = Portfolio_comb[i]
    print(f"{i} ir: {strategy.information_ratio(strat)}, sr: {strategy.sharpe_ratio(strat)}")

Portfolio_comb.corr()
sharpe_ratio(simple_optimal_weights(Portfolio_comb))


for i in range(Portfolio_comb.shape[1]):
    # exclude ith column in the portfolio
    temp_comb = Portfolio_comb.drop(Portfolio_comb.columns[i], axis=1)
    temp_comb_ret = simple_optimal_weights(temp_comb)
    sr = sharpe_ratio(temp_comb_ret)
    print(f"Sharpe ratio excluding {Portfolio_comb.columns[i]}: {sr}")

# optimizaed port with less strategies but still good sharpe ratio
# sr 2.35

signals = [ema_combined_signal, volume_signal, z_score_5d_signal, dt_on_signal, reversal_5d_signal, high_volume_signal_02]
for signal in signals:
    signal.fillna(0, inplace=True)

weight = strategy.portfolio_weights
for signal in signals:
    signal_weights = weight(signal)
    print(signal_weights.sum().sum())

def portfolio_comb_returns(signals):
    portfolio_comb_ret = []
    for signal in signals:
        signal_weights = strategy.portfolio_weights(signal)
        signal_ret = (signal_weights.shift() * ret).sum(axis=1)
        portfolio_comb_ret.append(signal_ret)
    portfolio_comb_ret = pd.concat(portfolio_comb_ret, axis=1)
    return portfolio_comb_ret
Portfolio_comb = portfolio_comb_returns(signals)
Portfolio_comb.columns = ['ema_combined_port_ret', 'v_port_ret', 'z_score_5d_port_ret', 'dt_on_port_ret', 'reversal_5d_port_ret', 'high_volume_port_ret_02']
Portfolio_comb.corr()


def signal_weights(signals):
    """
    Calculate the weights based on the signal.
    """
    def optimal_weights(sigma, mu):
        wgt = np.linalg.inv(sigma) @ mu
        wgt = wgt/np.abs(wgt).sum()
        return wgt
    rets = portfolio_comb_returns(signals)
    sigma = rets.cov()
    mu = rets.mean()
    wgts = optimal_weights(sigma, mu)
    print(wgts)

    signal_weights_list = []
    for signal, wgt in zip(signals, wgts):
        # test_w = strategy.portfolio_weights(signal)
        # test_ret = (test_w.shift() * ret).sum(axis=1)
        # sr = sharpe_ratio(test_ret)
        # print(f"Sharpe ratio of the signal: {sr}, weight: {wgt}")
        signal_weight = strategy.portfolio_weights(signal) * wgt
        # print(signal_weight.abs().sum(1).mean(), wgt)
        signal_weight = signal_weight.fillna(0)
        signal_weights_list.append(signal_weight)

    return sum(signal_weights_list)

port_weights = signal_weights(signals)
strategy.summary_stats(port_weights)

comb_ret_less_strat = (port_weights.shift() * ret).sum(axis=1)

# turnover and net returns with commission of 0.0035 per share converted to dollar
# in real world fractional share count as 1 share which are not yet incorporated
# slippage not included
port_tcost = strategy.tcost(port_weights, comm_bps_per_share=35, slippage_bps=0)
port_net_return = (port_weights.shift() * ret).sum(axis=1) - port_tcost.sum(axis=1)
sharpe_ratio(port_net_return)
port_net_return.cumsum().plot()


""" Convex optimization with long short fully invested constraints """
# reload Convex_opt
# import importlib
# import Convex_opt
# importlib.reload(Convex_opt)
# from Convex_opt import convex_optimization
convex_opt = convex_optimization(Daily_bar)

# opt with 1.0/100 tc_penalty
port_weights_optimized_4 = convex_opt.optimize_portfolio(port_weights, constraints=[cvx.sum(convex_opt.weights) == 1], comm_bps_per_share=35 * 1e-4, tc_penalty=1.0/100.)
port_weights_optimized_4 = pd.DataFrame(port_weights_optimized_4, index=port_weights.index)
strategy.summary_stats(port_weights_optimized_4)

port_opt_return_4 = (port_weights_optimized_4.shift() * ret).sum(axis=1)
port_opt_return_4.cumsum().plot()

# why not scale down the weights to sum to 1 over fit?
test = port_weights_optimized_4.apply(lambda x: x/port_weights_optimized_4.abs().sum(1))
strategy.summary_stats(test)

# dollar neutral test
port_weights_optimized_1 = convex_opt.optimize_portfolio(port_weights, constraints=[cvx.sum(convex_opt.weights) == 0], comm_bps_per_share=35 * 1e-4, tc_penalty=1.0/100.)
port_weights_optimized_1 = pd.DataFrame(port_weights_optimized_1, index=port_weights.index)
strategy.summary_stats(port_weights_optimized_1)

# long only fully invested test
port_weights_optimized_2 = convex_opt.optimize_portfolio(port_weights, constraints=[cvx.sum(convex_opt.weights) == 1, convex_opt.weights >= 0], comm_bps_per_share=35 * 1e-4, tc_penalty=1.0/100.)
port_weights_optimized_2 = pd.DataFrame(port_weights_optimized_2, index=port_weights.index)
strategy.summary_stats(port_weights_optimized_2)

# long short fully invested test
port_weights_optimized_3 = convex_opt.optimization_long_short_fully_invested(port_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=1.0/100)
port_weights_optimized_3 = pd.DataFrame(port_weights_optimized_3, index=port_weights.index)
strategy.summary_stats(port_weights_optimized_3)

# long short fully invested dollar neutral test
# weights are dollar amount not by share
port_weights_optimized_5 = convex_opt.optimization_long_short_fully_invested_dollar_neutral(port_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=1.0/100.)
port_weights_optimized_5 = pd.DataFrame(port_weights_optimized_5, index=port_weights.index)
strategy.summary_stats(port_weights_optimized_5)

# Grid search with long only constraint
grid_tc_longonly = []
for i in range(2, 22, 2):
    tc_penalty = i / 1000.
    constraints = [cvx.sum(convex_opt.weights) == 1, convex_opt.weights >= 0]
    port_weights_optimized_tc = convex_opt.optimize_portfolio(port_weights, constraints=constraints, comm_bps_per_share=35 * 1e-4, tc_penalty=tc_penalty)
    port_weights_optimized_tc = pd.DataFrame(port_weights_optimized_tc, index=port_weights.index)
    summary_stats = strategy.summary_stats(port_weights_optimized_tc)
    grid_tc_longonly.append(summary_stats)
    print(f"Summary stats for tc_penalty {tc_penalty}: {summary_stats}")
grid_tc_longonly = pd.concat(grid_tc_longonly, axis=1).T

# grid search for tc_penalty on long short fully invested
grid_tc_longshort = []
weights = {}
for i in range(2, 22, 2):
    tc_penalty = i / 1000.
    port_weights_optimized_tc = convex_opt.optimization_long_short_fully_invested(port_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=tc_penalty)
    port_weights_optimized_tc = pd.DataFrame(port_weights_optimized_tc, index=port_weights.index)
    weights[tc_penalty] = port_weights_optimized_tc
    summary_stats = strategy.summary_stats(port_weights_optimized_tc)
    grid_tc_longshort.append(summary_stats)
    print(f"Summary stats for tc_penalty {tc_penalty}: {summary_stats}")
grid_tc_longshort = pd.concat(grid_tc_longshort, axis=1).T

grid_tc_longshort_weights = weights.copy()
temp_012 = weights[0.012]
strategy.summary_stats(weights[0.012])

# grid search for tc_penalty on long short fully invested and dollar neutral
grid_tc_longshort_dollar_neutral = []
grid_tc_longshortdn_weights = {}
for i in range(2, 13, 2):
    tc_penalty = i / 1000.
    port_weights_optimized_tc = convex_opt.optimization_long_short_fully_invested_dollar_neutral(port_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=tc_penalty)
    port_weights_optimized_tc = pd.DataFrame(port_weights_optimized_tc, index=port_weights.index)
    grid_tc_longshortdn_weights[tc_penalty] = port_weights_optimized_tc
    summary_stats = strategy.summary_stats(port_weights_optimized_tc)
    grid_tc_longshort_dollar_neutral.append(summary_stats)
    print(f"Summary stats for tc_penalty {tc_penalty}: {summary_stats}")

# for some reason 0.014 not working
for i in range(14, 23, 2):
    tc_penalty = i / 1000.
    port_weights_optimized_tc = convex_opt.optimization_long_short_fully_invested_dollar_neutral(port_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=tc_penalty, solver= cvx.ECOS)
    port_weights_optimized_tc = pd.DataFrame(port_weights_optimized_tc, index=port_weights.index)
    grid_tc_longshortdn_weights[tc_penalty] = port_weights_optimized_tc
    summary_stats = strategy.summary_stats(port_weights_optimized_tc)
    grid_tc_longshort_dollar_neutral.append(summary_stats)
    print(f"Summary stats for tc_penalty {tc_penalty}: {summary_stats}")

grid_tc_longshort_dollar_neutral = pd.concat(grid_tc_longshort_dollar_neutral, axis=1).T

tc_penalty = 10 / 1000.
port_weights_optimized = convex_opt.optimization_long_short_fully_invested_dollar_neutral(port_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=tc_penalty)
port_weights_optimized = pd.DataFrame(port_weights_optimized, index=port_weights.index)
summary_stats = strategy.summary_stats(port_weights_optimized)

# dollar neutral really eats the sharpe
port_weights_optimized_6 = grid_tc_longshortdn_weights[0.014]
strategy.summary_stats(port_weights_optimized_6)



