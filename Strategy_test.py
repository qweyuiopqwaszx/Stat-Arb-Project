import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import statsmodels.api as sm


class strategy_metrics:
    def __init__(self, price_data):
        self.price_data = price_data
        self.rets = self.calculate_returns()
        self.benchmark = self.get_benchmark()

    def calculate_returns(self):
        ret = self.price_data['Adj Close'] / self.price_data['Adj Close'].shift() - 1
        return ret

    def get_benchmark(self):
        benchmark = yf.download("SPY", start=self.price_data.index[0], end=self.price_data.index[-1] + pd.Timedelta(days=1), auto_adjust=False)['Adj Close']
        benchmark = benchmark / benchmark.shift() - 1
        return benchmark

    def portfolio(self, signal, ret = None):
        if ret is None:
            ret = self.rets
        weights = signal.div(signal.abs().sum(axis=1), axis=0)
        port_ret = (weights.shift() * ret).sum(axis=1)
        return port_ret
    
    def portfolio_weights(self, signal):
        """
        Calculate the weights based on the signal with normalization.
        """
        weights = signal.div(signal.abs().sum(axis=1), axis=0)
        return weights

    def sharpe_ratio(self, portfolio_returns, risk_free_rate=0.0):
        excess_returns = portfolio_returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def signal_sr(self, signal, ret = None):
        if ret is None:
            ret = self.rets
        weights = signal.div(signal.abs().sum(axis=1), axis=0)
        port_ret = (weights.shift() * ret).sum(axis=1)
        sr = self.sharpe_ratio(port_ret)
        return sr

    def drawdown(self, portfolio_returns):
        """
        Calculate the maximum drawdown of a portfolio.
        """
        cum_ret = (1 + portfolio_returns).cumprod()
        drawdown = (cum_ret / cum_ret.expanding(min_periods=1).max() - 1)
        
        return drawdown
    
    def turnover(self, portfolio_weights):
        to = (portfolio_weights.fillna(0)-portfolio_weights.shift().fillna(0)).abs().sum(1)   
        return to

    def summary_stats(self, portfolio_weights, ret = None):
        """
        Calculate the annualized return, volatility, and Sharpe ratio of a portfolio.
        """
        if ret is None:
            ret = self.rets
        rets = (portfolio_weights.shift() * ret).sum(axis=1)

        stats = {}
        stats['avg']=rets.mean()*252
        stats['vol']=rets.std()*np.sqrt(252)
        stats['sharpe']=stats['avg']/stats['vol']
        stats['hit_rate']=(rets>0).sum()/rets.count()
        stats['turnover']=self.turnover(portfolio_weights).mean()
        stats['holding_period']=2/stats['turnover']
        stats['max_drawdown'], stats['max_duration']  = self.drawdown_duration(rets)
        stats['information_ratio'] = self.information_ratio(rets)
        try:
            stats = pd.DataFrame(stats)
        except:
            stats = pd.Series(stats)
        return stats

    def information_ratio(self, Strategy_return, benchmark = None):
        """
        Perform linear regression of strategy against benchmark to calculate the information ratio
        """
        if benchmark is None:
            benchmark = self.benchmark
        Y = Strategy_return[Strategy_return.ne(0).idxmax():]
        X = benchmark.loc[Y.index]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        alpha_contr = model.params['const'] + model.resid
        return alpha_contr.mean()/alpha_contr.std()*np.sqrt(252)

    def drawdown_duration(self, portfolio_returns):
        """
        Calculate the maximum drawdown of a portfolio.
        """
        cum_ret = (1 + portfolio_returns).cumprod()
        drawdown = (cum_ret / cum_ret.expanding(min_periods=1).max() - 1)
        max_drawdown = drawdown.min()
        try:
            max_duration = pd.Series()
            for col in drawdown.columns:
                duration = (drawdown[col] < 0).astype(int).groupby((drawdown[col] == 0).astype(int).cumsum()).cumsum()
                max_duration[col] = duration.max()
                
        except:
            duration = (drawdown != 0).astype(int).groupby((drawdown == 0).astype(int).cumsum()).cumsum()
            max_duration = duration.max()
        return max_drawdown, max_duration
    
    def net_returns(self, portfolio_weights, commissions_bps=1, slippage_bps=0, ret = None):
        """
        Calculate the net returns after transaction costs.
        """
        if ret is None:
            ret = self.rets
        tcost_bps = commissions_bps + slippage_bps
        rets = (portfolio_weights.shift() * ret).sum(axis=1)
        turnover = self.turnover(portfolio_weights)
        net_ret = rets.subtract(turnover*tcost_bps*1e-4, fill_value=0)
        return net_ret

    def tcost(self, portfolio_weights, comm_bps_per_share=35, slippage_bps=0):
        """
        Calculate the transaction costs.
        """
        weights_turnover = abs(portfolio_weights.shift() - portfolio_weights)
        comm_bps_per_dollar = comm_bps_per_share * 1e-4 / self.price_data['Adj Close']
        return (weights_turnover * comm_bps_per_dollar)

    def simple_optimal_weights(self, Strategy_comb_returns):
        sigma = Strategy_comb_returns.cov()
        mu = Strategy_comb_returns.mean()
        wgt = np.linalg.inv(sigma) @ mu
        wgt = wgt/np.abs(wgt).sum()
        return wgt
    
    
    
    