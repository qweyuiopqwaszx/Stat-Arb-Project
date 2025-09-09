def get_tracking_error(w,ideal,sigma):
    param = cvx.Parameter(shape=sigma.shape, value=sigma, PSD=True)
    tracking_error = cvx.quad_form(w,param) - 2 * ideal @ sigma @ w + ideal @ sigma @ ideal
    return tracking_error

def get_tracking_error_v2(w, ideal, sigma):
    sigma_chol = np.linalg.cholesky(sigma)
    sigma_chol_constant = cvx.Constant(sigma_chol)
    tracking_error = cvx.sum_squares(sigma_chol_constant @ (w - ideal))
    return tracking_error

w = cvx.Variable(ret.shape[1])
tracking_error = get_tracking_error(w,ideal,sigma_lw)
objective_tc = cvx.Minimize(tracking_error + tcost)
prob = cvx.Problem(objective_tc, constraints)
prob.solve(warm_start=True)
w_v1 = pd.Series(w.value,index=ideal.index)

w = cvx.Variable(ret.shape[1])
tracking_error_v2 = get_tracking_error_v2(w,ideal,sigma_lw)
objective_tc = cvx.Minimize(tracking_error_v2 + tcost)
prob2 = cvx.Problem(objective_tc, constraints)
prob2.solve(warm_start=True)
w_v2 = pd.Series(w.value,index=ideal.index)

sum(w_v1 - w_v2)


# convex optimization at each time step
def get_tracking_error(w,ideal,sigma):
    param = cvx.Parameter(shape=sigma.shape, value=sigma, PSD=True)
    tracking_error = cvx.quad_form(w,param) - 2 * ideal @ sigma @ w
    return tracking_error

def get_tcost(w,w_prev,comm_bps=1e-4,tc_penalty=1/100.):
    tcost = tc_penalty*cvx.sum(cvx.abs(w - w_prev)*comm_bps) # only commission of 1bp
    return tcost

# only fully_invested constraints 
# increase tc_penalty you trade less
comm_bps = 35 * 1e-4
tc_penalty = 1.2/100.
# static covariance matrix
sigma_lw = LedoitWolf().fit(ret.fillna(0).values).covariance_

w_prev = port_weights.iloc[0]
w_cons_list = [w_prev]
for i in range(1, port_weights.shape[0]):
    ideal = port_weights.iloc[i]
    # sigma_lw = LedoitWolf().fit(ret.iloc[:i].fillna(0).values).covariance_
    w = cvx.Variable(ret.shape[1])

    price_i = Daily_bar['Adj Close'].iloc[i]
    est_comm_bps = (comm_bps / price_i).mean()

    # estimates tcost
    tracking_error = get_tracking_error(w,ideal,sigma_lw)
    tcost = get_tcost(w, w_prev, est_comm_bps, tc_penalty)

    # fully_invested long only constraints
    # constraints = []
    # fully_invested = cvx.sum(w) == 1
    # constraints.append(fully_invested)
    # long_only = w >= 0
    # constraints.append(long_only)

    objective_tc = cvx.Minimize(tracking_error + tcost)
    prob = cvx.Problem(objective_tc)
    prob.solve(warm_start=True, solver = cvx.SCS)

    # inpost constraints for long short fully invested
    sign = np.sign(w.value)
    w = cvx.Variable(ret.shape[1])
    constraints = []
    # constraints = [cvx.sum(w) == 1, w >= 0]
    constraints.append(cvx.multiply(sign,w) >= 0)
    constraints.append(sign @ w == 1)
    constraints.append(cvx.sum(w) == 0)

    tracking_error = get_tracking_error(w,ideal,sigma_lw)
    tcost = get_tcost(w, w_prev, est_comm_bps, tc_penalty)

    objective_tc = cvx.Minimize(tracking_error + tcost)
    prob = cvx.Problem(objective_tc,constraints)
    prob.solve(warm_start=True, solver=cvx.SCS)
    w_cons = pd.Series(w.value,index=ideal.index)
    
    w_prev = w_cons
    print(f"Optimized weights for time {i}:")
    w_cons_list.append(w_cons)

port_weights_optimized = pd.DataFrame(w_cons_list, index=port_weights.index)
strategy.summary_stats(port_weights_optimized)

temp = port_weights_optimized.copy()