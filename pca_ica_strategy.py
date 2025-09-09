import pandas as pd
import numpy as np
from Strategy_test import strategy_metrics
from sklearn.decomposition import FastICA, PCA
import scipy.stats
import matplotlib.pyplot as plt

## currently prone to overfitting
## future explore it on minute bar, less stock comps or 503 stocks with 503 ica

# ### 7. ICA alpha strategy
# **intuition is to use ICA to separate the return to linear combination of independent non-Gaussian components**
# **ICA are performed on a rolling basis from time T = t - windowsize to t, weights generate for t+1**
# **The mixing that maps the independent components to return are used to generate signals for next day return**
# **final Signal are computed by Signal Strength * Signal directions * ICA mixing**

Daily_bar = pd.read_pickle("price_1d_20210101_20250817.pk")
strategy = strategy_metrics(Daily_bar)
ret = strategy.calculate_returns()
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

        final_weights = final_signal / (np.abs(final_signal).sum())
        Strategy_weights.iloc[i-1] = final_signal
    return Strategy_weights
ica_alpha_strat_signal = pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=15, benchmark=spy)
ica_alpha_strat_weights = ica_alpha_strat_signal.div(ica_alpha_strat_signal.abs().sum(axis=1), axis=0).fillna(0)
strategy.summary_stats(ica_alpha_strat_weights)




grid_search_results = []
for i in range(1, 10, 1):
    for j in range(1, 10, 1):
        if i + j > 10:
            continue
        temp = pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=15, benchmark=spy, param_i=i/10, param_j=j/10)
        stats = strategy.summary_stats(temp)
        grid_search_results.append((i, j, stats))
        print(i/10,j/10, stats)

rows = []
for param1, param2, metrics in grid_search_results:
    row = {
        'param1': param1,
        'param2': param2
    }
    # Add all metrics to the row
    row.update(metrics.to_dict())
    rows.append(row)

df = pd.DataFrame(rows)
df['param3'] = 10 - df['param1'] - df['param2']
sorted_df = df.sort_values(by='sharpe', ascending=False)



grid_search_ica_n_components = []
for n_comp in range(4, 51, 2):
    temp = pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=n_comp, benchmark=spy)
    stats = strategy.summary_stats(temp)
    grid_search_ica_n_components.append((n_comp, stats))
    print(n_comp, stats)

for n_comp in range(3, 50, 2):
    temp = pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=n_comp, benchmark=spy)
    stats = strategy.summary_stats(temp)
    grid_search_ica_n_components.append((n_comp, stats))
    print(n_comp, stats)

for n_comp in range(50, 501, 5):
    temp = pure_alpha_ica_strategy(ret, rolling_window=504, ica_n_components=n_comp, benchmark=spy)
    stats = strategy.summary_stats(temp)
    grid_search_ica_n_components.append((n_comp, stats))
    print(n_comp, stats)


grid_search_rollingwindow = []
for window in range(504, 1009, 126):
    temp = pure_alpha_ica_strategy(ret, rolling_window=window, ica_n_components=15, benchmark=spy)
    stats = strategy.summary_stats(temp)
    grid_search_rollingwindow.append((window, stats))
    print(window, stats)

# plot the grid search sharpe by i and by j
rows = []
for param1, metrics in grid_search_ica_n_components:
    row = {
        'param1': param1
    }
    # Add all metrics to the row
    row.update(metrics.to_dict())
    rows.append(row)

df = pd.DataFrame(rows)
df['param3'] = 10 - df['param1'] - df['param2']
sorted_df = df.sort_values(by='param1', ascending=True)

sorted_df.set_index('param1', inplace=True)

# Plot param1 against both Sharpe and Information Ratio on the same figure
plt.figure(figsize=(10, 6))
plt.plot(sorted_df.index, sorted_df['sharpe'], label='Sharpe Ratio', marker='o')
plt.plot(sorted_df.index, sorted_df['information_ratio'], label='Information Ratio', marker='s')
plt.title('Sharpe Ratio and Information Ratio vs Param1 (ICA n_components)')
plt.xlabel('Param1 (ICA n_components)')
plt.ylabel('Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Daily_bar_7y = pd.read_pickle("price_1d_20180101_20250817.pk")
# strategy = strategy_metrics(Daily_bar_7y)
# ret = strategy.calculate_returns()


def pca_ica_strategy(ret, min_period = 252, pca_threshold = 0.95, ica_n_components = 10, top_n=5):
    rets = ret.fillna(0)
    Strategy_weights = pd.DataFrame(index=ret.index, columns=ret.columns)
    for i in range(min_period, ret.shape[0]+1):
        # clean training data by removing market_betas
        train_data = rets[:i]
        market_factor = spy[:i]
        market_betas = np.array([np.cov(train_data[col], market_factor)[0, 1] /
                               np.var(market_factor) for col in train_data.columns])
        
        market_neutral = train_data.values - np.outer(market_factor, market_betas)

        # reduce dimensionality performs PCA with variance captured threshold
        pca = PCA()
        pca.fit(market_neutral)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_pca = np.argmax(cumulative_variance >= pca_threshold) + 1

        X_reduced = pca.transform(market_neutral)[:, :n_components_pca]

        if n_components_pca < ica_n_components:
            n_components_pca = ica_n_components
            print(f"Warning: n_components_pca {n_components_pca} is less than ica_n_components {ica_n_components}. Setting n_components_pca to {ica_n_components}.")
        # perform ICA on the reduced data map back ica_mixing to original space
        ica = FastICA(n_components=ica_n_components, random_state=0, max_iter=2000)
        ica.fit_transform(X_reduced)
        ica_mixing = pca.components_[:n_components_pca, :].T @ ica.mixing_

        # Component analysis and weighting logic
        component_stats = {}
        
        for j in range(ica_n_components):
            w = ica_mixing[:, j]
            w = w / (np.abs(w).sum())

            # Calculate component returns
            port_ret = (market_neutral * w).sum(axis=1)

            # Alpha score based on market corr, kurtosis (tanh to bound -1,1), and autocorr
            market_corr = np.corrcoef(port_ret, market_factor)[0, 1]
            kurtosis = np.tanh(scipy.stats.kurtosis(port_ret))
            autocorr = np.corrcoef(port_ret[:-1], port_ret[1:])[0, 1]
            
            component_stats[j] = {
                'market_corr': market_corr,
                'kurtosis': kurtosis,
                'autocorr': autocorr,
                'weights': w
            }

        # Create lists of values
        market_corrs = [abs(comp['market_corr']) for comp in component_stats.values()] # Less is better
        kurtoses = [abs(comp['kurtosis']) for comp in component_stats.values()] # More is better
        autocorrs = [max(0, comp['autocorr']) for comp in component_stats.values()] # More is better
        rank_kurtosis = scipy.stats.rankdata(kurtoses) # High kurtosis gets high rank
        rank_autocorr = scipy.stats.rankdata(autocorrs) # High autocorr gets high rank
        rank_market_corr = scipy.stats.rankdata([-x for x in market_corrs]) # Low corr gets high rank

        # Now assign the combined rank score
        for idx, j in enumerate(component_stats.keys()):
            combined_rank_score = (rank_market_corr[idx] * 0.4 +
                                rank_kurtosis[idx] * 0.3 +
                                rank_autocorr[idx] * 0.3)
            
            component_stats[j]['alpha_score'] = combined_rank_score
        
        # Select components with best alpha characteristics
        sorted_components = sorted(component_stats.items(), 
                                 key=lambda x: x[1]['alpha_score'], 
                                 reverse=True)
        
        # Take top 2-3 alpha components
        top_components = sorted_components[:top_n]
        print(f"Top components at time {i}: {[comp[0] for comp in top_components]} with scores {[comp[1]['alpha_score'] for comp in top_components]}")

        if top_components:
            # Extract alpha scores and weights
            # alpha_scores = [comp[1]['alpha_score'] for comp in top_components]
            # component_weights = [comp[1]['weights'] for comp in top_components]
            
            # # Convert alpha scores to positive weights (softmax or normalized)
            # alpha_weights = np.array(alpha_scores)
            
            # # Normalize alpha weights to sum to 1
            # alpha_weights = alpha_weights / (alpha_weights.sum())
            
            # # Weight components by their alpha scores
            # final_weights = np.zeros_like(component_weights[0])
            # for alpha_w, comp_w in zip(alpha_weights, component_weights):
            #     final_weights += alpha_w * comp_w
            
            # Final normalization
            final_weights = np.mean([comp[1]['weights'] for comp in top_components], axis=0)
            final_weights = final_weights / (np.abs(final_weights).sum())
            Strategy_weights.iloc[i-1] = final_weights
    return Strategy_weights