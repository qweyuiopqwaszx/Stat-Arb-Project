import pandas as pd
import numpy as np
import cvxpy as cvx
from sklearn.covariance import LedoitWolf

class convex_optimization:
    def __init__(self, Daily_bar):
        self.Daily_bar = Daily_bar
        self.returns = Daily_bar['Adj Close']/Daily_bar['Adj Close'].shift() - 1
        self.n_assets = self.returns.shape[1]
        self.observations = self.returns.shape[0]
        self.weights = cvx.Variable(self.n_assets)
        self.covariance = LedoitWolf().fit(self.returns.fillna(0).values).covariance_

    def get_tracking_error(self, w, ideal, sigma):
        param = cvx.Parameter(shape=sigma.shape, value=sigma, PSD=True)
        tracking_error = cvx.quad_form(w,param) - 2 * ideal @ sigma @ w
        return tracking_error
    
    # dpp compliant tracking error much faster with Positive definite assumption
    def get_tracking_error_v2(self, w, ideal, sigma):
        sigma_chol = np.linalg.cholesky(sigma)
        sigma_chol_constant = cvx.Constant(sigma_chol)
        tracking_error = cvx.sum_squares(sigma_chol_constant @ (w - ideal))
        return tracking_error

    
    def get_tcost(self, w,w_prev,comm_bps=35 * 1e-4,tc_penalty=1/100.):
        # slippage can be introduced in future
        tcost = tc_penalty*cvx.sum(cvx.abs(w - w_prev)*comm_bps) # only commission
        return tcost

    def constraints(self, constraints = None):
        if constraints is None:
            constraints = [cvx.sum(self.weights) == 1]
        return constraints

    def optimize_portfolio(self, portfolio_weights, constraints=None, comm_bps_per_share=35 * 1e-4, tc_penalty=1/100.):
        w_prev = portfolio_weights.iloc[0]
        w_cons_list = [w_prev]
        for i in range(1, self.observations):
            ideal = portfolio_weights.iloc[i]
            w = self.weights

            comm_bps = comm_bps_per_share/self.Daily_bar['Adj Close'].iloc[i]
            comm_bps = comm_bps.mean()
            tracking_error = self.get_tracking_error(w, ideal, self.covariance)
            tcost = self.get_tcost(w, w_prev, comm_bps, tc_penalty)
            
            objective = cvx.Minimize(tracking_error + tcost)
            constraints = self.constraints(constraints)
            prob = cvx.Problem(objective,constraints)
            prob.solve(warm_start=True)

            w_cons = pd.Series(w.value,index=ideal.index)
            w_prev = w_cons
            w_cons_list.append(w_cons)
            print(f"Optimized weights for time {portfolio_weights.index[i]}")

        return w_cons_list

    def one_iteration_optimization(self, w, ideal, w_prev, constraints, comm_bps, tc_penalty=1/100., solver=None):
        tracking_error = self.get_tracking_error(w, ideal, self.covariance)
        tcost = self.get_tcost(w, w_prev, comm_bps, tc_penalty)

        objective = cvx.Minimize(tracking_error + tcost)
        prob = cvx.Problem(objective, constraints)
        if solver is None:
            prob.solve(warm_start=True)
        else:
            prob.solve(solver=solver, warm_start=True)

        return pd.Series(w.value, index=ideal.index)

    def optimization_long_short_fully_invested(self, portfolio_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=1/100.):
        w_prev = portfolio_weights.iloc[0]
        w_cons_list = [w_prev]
        for i in range(1, self.observations):
            # one iteration of relaxed optimization with no constraints
            w = self.weights
            ideal = portfolio_weights.iloc[i]
            comm_bps = (comm_bps_per_share/self.Daily_bar['Adj Close'].iloc[i]).mean()
            w_relaxed = self.one_iteration_optimization(w, ideal, w_prev, None, comm_bps, tc_penalty)
            # another iteration with constraints
            sign = np.asarray(np.sign(w_relaxed))
            w = self.weights
            constraints = [cvx.multiply(sign, w) >= 0, sign @ w == 1]
            w_cons = self.one_iteration_optimization(w, ideal, w_prev, constraints, comm_bps, tc_penalty)
            w_cons_list.append(w_cons)
            w_prev = w_cons

        return w_cons_list
    
    
    def optimization_long_short_fully_invested_dollar_neutral(self, portfolio_weights, comm_bps_per_share=35 * 1e-4, tc_penalty=1/100., solver = cvx.SCS):
        w_prev = portfolio_weights.iloc[0]
        w_cons_list = [w_prev]
        for i in range(1, self.observations):
            # one iteration of relaxed optimization with no constraints
            w = self.weights
            ideal = portfolio_weights.iloc[i]
            comm_bps = (comm_bps_per_share/self.Daily_bar['Adj Close'].iloc[i]).mean()
            w_relaxed = self.one_iteration_optimization(w, ideal, w_prev, None, comm_bps, tc_penalty, solver=solver)
            # another iteration with constraints
            sign = np.asarray(np.sign(w_relaxed))
            w = self.weights
            constraints = [cvx.multiply(sign, w) >= 0, sign @ w == 1, cvx.sum(w) == 0]
            w_cons = self.one_iteration_optimization(w, ideal, w_prev, constraints, comm_bps, tc_penalty, solver=solver)
            w_cons_list.append(w_cons)
            w_prev = w_cons

        return w_cons_list