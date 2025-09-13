import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import scipy.stats
from Strategy_test import strategy_metrics
from Convex_opt import convex_optimization
import cvxpy as cvx # pip install cvxpy
import pickle


Daily_bar = pd.read_pickle("price_1d_20210101_20250817.pk")
strategy = strategy_metrics(Daily_bar)
sharpe_ratio = strategy.sharpe_ratio
ret = strategy.calculate_returns()
port_ret = strategy.portfolio

""" Long Only Strategies """

""" 1. Volume Strategy 1 | SR 0.956"""
# simple signal based on volume rank 0.95 sr
# sr 0.956
volume_signal = Daily_bar['Volume'].rank(axis=1, pct=True)
volume_weights = strategy.portfolio_weights(volume_signal)
v_port_ret = port_ret(volume_signal, ret)
strategy.summary_stats(volume_weights)

"""2. Volume Strategy 2 | SR 1.21"""
# import market cap data
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

"""3. Zscore Strategy | SR 0.738 """
# create a ranked signal based on zscore of 5 day returns
z_score_5d_signal = (ret.rolling(window=5).mean() - ret.rolling(window=5).mean().rolling(window=20).mean()) / ret.rolling(window=5).std().rolling(window=20).std()
z_score_5d_signal = z_score_5d_signal.rank(axis=1, pct=True)
z_score_5d_weights = z_score_5d_signal.div(z_score_5d_signal.abs().sum(axis=1), axis=0)
z_score_5d_port_ret = port_ret(z_score_5d_signal, ret)
strategy.summary_stats(z_score_5d_weights)

""" 4. 5 days reversal strategy | 0.837 SR"""
# simple 5 days reversal strategy with rank
# 0.83 sr
reversal_5d_signal = (-1 * ret.rolling(5).sum()).rank(1, ascending = True)
reversal_5d_weights = strategy.portfolio_weights(reversal_5d_signal)
signal_5d_reversal_ret = (reversal_5d_weights.shift() * ret).sum(axis=1)
strategy.summary_stats(reversal_5d_weights)

""" Long Short Strategies with Dollar Neutral Weights """
""" 5. EMA 20 strategy | 1.03 SR"""
# create a signal entry when diff_ema_20_norm is above 0.02 and short when above 0.05
# sr 0.970
diff_ema_20_norm = Daily_bar['Adj Close'].ewm(span = 20, adjust=False).mean()
diff_ema_20_norm = (Daily_bar['Adj Close'] - diff_ema_20_norm) /diff_ema_20_norm


# grid searched for both above and below ema
# entry when 2 percent above 20 day ema and sell short when 5 percent above
ema_port_long_signal = (diff_ema_20_norm > 0.02).astype(int) - 2*(diff_ema_20_norm > 0.05).astype(int)
# short when below -0.03 and long when below -0.04
ema_port_short_signal = -(diff_ema_20_norm < -0.03).astype(int) + 2*(diff_ema_20_norm < -0.04).astype(int)

ema_combined_signal = ema_port_long_signal + ema_port_short_signal
ema_dn_weights = ema_combined_signal.sub(ema_combined_signal.mean(1), axis=0)
ema_dn_weights = ema_dn_weights.div(ema_dn_weights.abs().sum(axis=1), axis=0)

strategy.summary_stats(ema_dn_weights)

""" 6. Overnight return lead Daytime strategy | 0.825 SR"""
# Daytime and overnight return different sign long short strategy
# sr 0.825
Daily_bar_adjusted_open = Daily_bar['Open'] * (Daily_bar['Adj Close']/Daily_bar['Close'])
ret_daytime = Daily_bar['Adj Close'] / Daily_bar_adjusted_open - 1

ret_daytime_30d = ret_daytime.rolling(30).sum()
ret_overnight = (1 + ret) / (1 + ret_daytime) - 1
ret_overnight_30d = ret_overnight.rolling(30).sum()

thresh = 0
dt_on_signal = ((ret_daytime_30d > thresh) & (ret_overnight_30d < -thresh)).astype(int) - ((ret_daytime_30d < -thresh) & (ret_overnight_30d > thresh)).astype(int)
dt_on_port_weights = dt_on_signal.sub(dt_on_signal.mean(axis=1), axis=0)
dt_on_port_weights = dt_on_port_weights.div(dt_on_port_weights.abs().sum(axis=1), axis=0)
strategy.summary_stats(dt_on_port_weights)

""" Long Short Fully Invested Strategy portfolio weight combination"""
# look ahead bias when using future returns to calculate weights
long_short_port_weights_list = [ema_dn_weights, dt_on_port_weights]

def portfolio_comb_returns(weights_list):
    portfolio_comb_ret = []
    for weights in weights_list:
        strategy_ret = (weights.shift() * ret).sum(axis=1)
        portfolio_comb_ret.append(strategy_ret)
    portfolio_comb_ret = pd.concat(portfolio_comb_ret, axis=1)
    return portfolio_comb_ret

Portfolio_comb = portfolio_comb_returns(long_short_port_weights_list)
Portfolio_comb.columns = ['ema_combined_port_ret', 'dt_on_port_ret']
Portfolio_comb.corr()

def portfolio_weights_combined(weights_list):
    def optimal_weights(sigma, mu):
        wgt = np.linalg.inv(sigma) @ mu
        wgt = wgt/np.abs(wgt).sum()
        return wgt
    rets = portfolio_comb_returns(weights_list)
    sigma = rets.cov()
    mu = rets.mean()
    wgts = optimal_weights(sigma, mu)
    print("Optimal Weights for each strategy: ", wgts)
    signal_weights_list = []
    for weights, wgt in zip(weights_list, wgts):
        signal_weight = weights * wgt
        signal_weight = signal_weight.fillna(0)
        signal_weights_list.append(signal_weight)

    return sum(signal_weights_list)
long_short_port_weights = portfolio_weights_combined(long_short_port_weights_list)
print(strategy.summary_stats(long_short_port_weights))

def eqvol_weights(sigma):
    wgt = 1/np.sqrt(np.diag(sigma))
    wgt = wgt / np.abs(wgt).sum()
    return wgt
long_short_port_weights_eqvol = eqvol_weights(Portfolio_comb.cov())
long_short_port_weights_eqvol = sum(w.fillna(0) * wgt for w, wgt in zip(long_short_port_weights_list, long_short_port_weights_eqvol))
strategy.summary_stats(long_short_port_weights_eqvol)


# net return after t-costs before optimized | Sharpe 1.13
# turnover and net returns with commission of 0.0035 per share converted to dollar
# in real world fractional share count as 1 share which are not yet incorporated
# slippage not included
port_tcost = strategy.tcost(long_short_port_weights, comm_bps_per_share=35, slippage_bps=0)
port_net_return = (long_short_port_weights.shift() * ret).sum(axis=1) - port_tcost.sum(axis=1)
port_net_return.cumsum().plot()

strategy.summary_stats(long_short_port_weights, tcosts=port_tcost)

""" Long Short Fully Invested Strategy with Convex Optimization """
convex_opt = convex_optimization(Daily_bar)
long_short_port_optimized = convex_opt.optimization_long_short_fully_invested_dollar_neutral(long_short_port_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=1.4/100.)
long_short_port_optimized = pd.DataFrame(long_short_port_optimized, index=long_short_port_weights.index)
strategy.summary_stats(long_short_port_optimized)

grid_tc_longshort_dollar_neutral = []
grid_tc_longshortdn_weights = {}
for i in range(0, 21, 1):
    tc_penalty = i/1000
    port_weights_optimized_tc = convex_opt.optimization_long_short_fully_invested_dollar_neutral(long_short_port_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=tc_penalty)
    port_weights_optimized_tc = pd.DataFrame(port_weights_optimized_tc, index=long_short_port_weights.index)
    grid_tc_longshortdn_weights[tc_penalty] = port_weights_optimized_tc
    summary_stats = strategy.summary_stats(port_weights_optimized_tc)
    grid_tc_longshort_dollar_neutral.append(summary_stats)
    print(f"Summary stats for tc_penalty {tc_penalty}: {summary_stats}")

# # 0.014 gives the best sharpe ratio of 1.31
long_short_grid_df = pd.DataFrame(grid_tc_longshort_dollar_neutral, index=[i/1000 for i in range(21)])

# net return after t-costs with optimization | Sharpe 1.26
port_tcost_dn = strategy.tcost(long_short_port_optimized, comm_bps_per_share=35, slippage_bps=0)
port_net_return_dn = (long_short_port_optimized.shift() * ret) - port_tcost_dn.shift()
port_net_return_dn.cumsum().plot(title="Convex Optimized Dollar Neutral Strategy Cumulative Return")
plt.show()

strategy.summary_stats(long_short_port_optimized, tcosts=port_tcost_dn)


""" Long only Portfolio weights Combination """
# look ahead bias when using future returns to calculate weights
long_only_port_weights_list = [volume_weights, high_volume_weights_02, z_score_5d_weights, reversal_5d_weights]
long_only_port_comb = portfolio_comb_returns(long_only_port_weights_list)
long_only_port_comb.corr()
def eqvol_weights(sigma):
    wgt = 1/np.sqrt(np.diag(sigma))
    wgt = wgt / np.abs(wgt).sum()
    return wgt
long_only_weights_eqvol = eqvol_weights(long_only_port_comb.cov())
long_only_port_weights_eqvol = sum(w.fillna(0) * wgt for w, wgt in zip(long_only_port_weights_list, long_only_weights_eqvol))
strategy.summary_stats(long_only_port_weights_eqvol)

def sr_weights(sigma,mu):
    wgt = mu / np.diag(sigma) 
    wgt = wgt / np.abs(wgt).sum()
    return wgt
long_only_weights_sr = sr_weights(long_only_port_comb.cov(), long_only_port_comb.mean())
long_only_port_weights_sr = sum(w.fillna(0) * wgt for w, wgt in zip(long_only_port_weights_list, long_only_weights_sr))
strategy.summary_stats(long_only_port_weights_sr)

def optimal_weights(sigma, mu):
    wgt = np.linalg.inv(sigma) @ mu
    wgt = wgt/np.abs(wgt).sum()
    return wgt
long_only_weights_opt = optimal_weights(long_only_port_comb.cov(), long_only_port_comb.mean())
long_only_port_weights_opt = sum(w.fillna(0) * wgt for w, wgt in zip(long_only_port_weights_list, long_only_weights_opt))
strategy.summary_stats(long_only_port_weights_opt)

long_only_port_weights = long_only_port_weights_opt.sub(long_only_port_weights_opt.mean(1),0)
long_only_port_weights = long_only_port_weights.div(long_only_port_weights.abs().sum(1),0)
strategy.summary_stats(long_only_port_weights)



""" Long only Portfolio Optimization """
constraints = [cvx.sum(convex_opt.weights) == 1, convex_opt.weights >= 0]
long_only_port_optimized = convex_opt.optimize_portfolio(long_only_port_weights, constraints=constraints, comm_bps_per_share=35 * 1e-4, tc_penalty=0.0/100.)
long_only_port_optimized = pd.DataFrame(long_only_port_optimized, index=long_only_port_weights.index)
strategy.summary_stats(long_only_port_optimized)

# net return after t-costs before optimized | Sharpe 1.88
port_tcost_lo_bo = strategy.tcost(long_only_port_weights, comm_bps_per_share=35, slippage_bps=0)
strategy.summary_stats(long_only_port_weights, tcosts = port_tcost_lo_bo)

# net return before t-costs after optimized | Sharpe 1.5396
strategy.summary_stats(long_only_port_optimized)

# net return after t-costs after optimized | Sharpe 1.529
port_tcost_lo = strategy.tcost(long_only_port_optimized, comm_bps_per_share=35, slippage_bps=0)
port_net_return_lo = (long_only_port_optimized.shift() * ret).sum(axis=1) - port_tcost_lo.sum(axis=1)
strategy.summary_stats(long_only_port_optimized, tcosts=port_tcost_lo)
port_net_return_lo.cumsum().plot(title="Convex Optimized Long Only Strategy Cumulative Return")
plt.show()

grid_tc_longonly = []
grid_tc_longonly_weights = {}
for i in range(0, 31, 1):
    tc_penalty = i/10000
    port_weights_optimized_tc = convex_opt.optimize_portfolio(long_only_port_weights, constraints=constraints, comm_bps_per_share=35 * 1e-4, tc_penalty=tc_penalty)
    port_weights_optimized_tc = pd.DataFrame(port_weights_optimized_tc, index=long_only_port_weights.index)
    grid_tc_longonly_weights[tc_penalty] = port_weights_optimized_tc
    summary_stats = strategy.summary_stats(port_weights_optimized_tc)
    grid_tc_longonly.append(summary_stats)
    print(f"Summary stats for tc_penalty {tc_penalty}: {summary_stats}")

# # 0.003 gives the best sharpe ratio of 1.5396 gross
grid_tc_longonly_df = pd.DataFrame(grid_tc_longonly, index=[i/1000 for i in range(1, 31)])

""" Combined Portfolio long only and long short """
combined_port_weights_list = [long_only_port_optimized, long_short_port_optimized]
combined_port_weights = portfolio_weights_combined(combined_port_weights_list)
strategy.summary_stats(combined_port_weights)