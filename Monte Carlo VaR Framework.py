import numpy as np
import pandas as pd
from scipy import stats
import time
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class MonteCarloVaRSimulator:
    """
    High-performance Monte Carlo simulation framework for VaR calculation
    with vectorized operations and statistical validation.
    """
    
    def __init__(self, confidence_level: float = 0.99, random_seed: int = 42):
        """
        Initialize the Monte Carlo VaR simulator.
        
        Args:
            confidence_level: Confidence level for VaR calculation (e.g., 0.99 for 99%)
            random_seed: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def _validate_inputs(self, returns: np.ndarray, num_simulations: int, time_horizon: int):
        """Validate input parameters."""
        if len(returns.shape) != 1:
            raise ValueError("Returns should be a 1D array")
        if num_simulations <= 0:
            raise ValueError("Number of simulations must be positive")
        if time_horizon <= 0:
            raise ValueError("Time horizon must be positive")
    
    def _estimate_parameters(self, returns: np.ndarray) -> Tuple[float, float]:
        """
        Estimate mean and volatility from historical returns.
        Uses vectorized operations for efficiency.
        """
        mu = np.mean(returns) * 252  # Annualized mean
        sigma = np.std(returns) * np.sqrt(252)  # Annualized volatility
        return mu, sigma
    
    def simulate_paths_vectorized(self, 
                                initial_price: float, 
                                returns: np.ndarray, 
                                num_simulations: int = 10000, 
                                time_horizon: int = 1) -> np.ndarray:
        """
        Perform vectorized Monte Carlo simulation using geometric Brownian motion.
        
        Args:
            initial_price: Initial asset price
            returns: Historical daily returns
            num_simulations: Number of simulation paths
            time_horizon: Time horizon in days
            
        Returns:
            Simulated price paths (num_simulations x time_horizon)
        """
        self._validate_inputs(returns, num_simulations, time_horizon)
        
        # Estimate parameters using vectorized operations
        mu, sigma = self._estimate_parameters(returns)
        
        # Daily parameters
        mu_daily = mu / 252
        sigma_daily = sigma / np.sqrt(252)
        
        # Generate random shocks using vectorized normal distribution
        # Shape: (num_simulations, time_horizon)
        random_shocks = np.random.normal(
            mu_daily - 0.5 * sigma_daily**2, 
            sigma_daily, 
            (num_simulations, time_horizon)
        )
        
        cumulative_returns = np.cumsum(random_shocks, axis=1)
        price_paths = initial_price * np.exp(cumulative_returns)
        
        return price_paths
    
    def calculate_var(self, 
                     initial_price: float, 
                     returns: np.ndarray, 
                     num_simulations: int = 10000, 
                     time_horizon: int = 1) -> Tuple[float, np.ndarray]:
        """
        Calculate Value at Risk using Monte Carlo simulation.
        
        Returns:
            VaR at specified confidence level and all simulated final prices
        """
        # Perform vectorized simulation
        price_paths = self.simulate_paths_vectorized(
            initial_price, returns, num_simulations, time_horizon
        )
        
        # Get final prices from all simulations
        final_prices = price_paths[:, -1]
        simulated_returns = (final_prices - initial_price) / initial_price
        var = -np.quantile(simulated_returns, 1 - self.confidence_level)
        
        return var, final_prices
    
    def backtest_var(self, 
                    historical_returns: np.ndarray, 
                    window_size: int = 252, 
                    num_simulations: int = 10000) -> Dict[str, float]:
        """
        Backtest VaR model using historical data.
        
        Args:
            historical_returns: Array of historical daily returns
            window_size: Rolling window size for estimation
            num_simulations: Number of Monte Carlo simulations per window
            
        Returns:
            Dictionary with backtesting results and performance metrics
        """
        violations = 0
        total_tests = len(historical_returns) - window_size
        
        # Vectorized backtesting
        for i in range(window_size, len(historical_returns)):
            # Use rolling window of returns
            window_returns = historical_returns[i-window_size:i]
            current_price = 100  # Arbitrary initial price for relative VaR
            
            # Calculate VaR for next period
            var, _ = self.calculate_var(
                current_price, window_returns, num_simulations, time_horizon=1
            )
            
            # Check if actual return violates VaR
            actual_return = historical_returns[i]
            if actual_return < -var:
                violations += 1
        
        # Calculate violation rate and confidence intervals
        violation_rate = violations / total_tests
        expected_violation_rate = 1 - self.confidence_level
        
        # Calculate confidence interval for binomial test
        se = np.sqrt(expected_violation_rate * (1 - expected_violation_rate) / total_tests)
        ci_lower = expected_violation_rate - 2.58 * se  # 99% CI
        ci_upper = expected_violation_rate + 2.58 * se
        
        # Perform binomial test
        p_value = stats.binomtest(
            violations, total_tests, expected_violation_rate, alternative='two-sided'
        ).pvalue
        
        return {
            'violations': violations,
            'total_tests': total_tests,
            'violation_rate': violation_rate,
            'expected_violation_rate': expected_violation_rate,
            'confidence_interval_lower': ci_lower,
            'confidence_interval_upper': ci_upper,
            'p_value': p_value,
            'model_valid': ci_lower <= violation_rate <= ci_upper
        }
    
    def performance_benchmark(self, 
                            returns: np.ndarray, 
                            initial_price: float = 100,
                            num_simulations: int = 10000) -> Dict[str, float]:
        """
        Benchmark performance against naive loop-based implementation.
        """
        # Benchmark vectorized implementation
        start_time = time.time()
        var_vectorized, _ = self.calculate_var(
            initial_price, returns, num_simulations
        )
        vectorized_time = time.time() - start_time
        
        # Benchmark naive implementation (for comparison)
        start_time = time.time()
        var_naive = self._naive_calculation(initial_price, returns, num_simulations)
        naive_time = time.time() - start_time
        
        # Calculate performance improvement
        speedup = naive_time / vectorized_time if vectorized_time > 0 else 0
        improvement_pct = ((naive_time - vectorized_time) / naive_time) * 100
        
        return {
            'vectorized_time_seconds': vectorized_time,
            'naive_time_seconds': naive_time,
            'speedup_factor': speedup,
            'improvement_percentage': improvement_pct,
            'var_vectorized': var_vectorized,
            'var_naive': var_naive,
            'results_match': np.isclose(var_vectorized, var_naive, rtol=1e-6)
        }
    
    def _naive_calculation(self, 
                         initial_price: float, 
                         returns: np.ndarray, 
                         num_simulations: int) -> float:
        """
        Naive loop-based implementation for performance comparison.
        """
        mu, sigma = self._estimate_parameters(returns)
        mu_daily = mu / 252
        sigma_daily = sigma / np.sqrt(252)
        
        final_prices = []
        
        for _ in range(num_simulations):
            price = initial_price
            for _ in range(1):  # Single period for simplicity
                drift = mu_daily - 0.5 * sigma_daily**2
                shock = np.random.normal(drift, sigma_daily)
                price *= np.exp(shock)
            final_prices.append(price)
        
        final_prices = np.array(final_prices)
        simulated_returns = (final_prices - initial_price) / initial_price
        return -np.quantile(simulated_returns, 1 - self.confidence_level)

# Example usage and demonstration
def main():
    """Demonstrate the Monte Carlo VaR simulator with performance benchmarks."""
    
    # Generate sample data
    np.random.seed(42)
    n_days = 1000
    sample_returns = np.random.normal(0.0005, 0.02, n_days)
    
    # Initialize simulator
    simulator = MonteCarloVaRSimulator(confidence_level=0.99)
    
    print("Monte Carlo VaR Simulation Framework")
    print("=" * 50)
    
    # Performance benchmark
    print("\n1. Performance Benchmark:")
    print("-" * 30)
    
    benchmark_results = simulator.performance_benchmark(
        sample_returns, num_simulations=100000
    )
    
    print(f"Vectorized implementation: {benchmark_results['vectorized_time_seconds']:.4f}s")
    print(f"Naive implementation: {benchmark_results['naive_time_seconds']:.4f}s")
    print(f"Speedup factor: {benchmark_results['speedup_factor']:.2f}x")
    print(f"Performance improvement: {benchmark_results['improvement_percentage']:.1f}%")
    print(f"Results match: {benchmark_results['results_match']}")
    
    # Calculate VaR
    print("\n2. VaR Calculation:")
    print("-" * 30)
    
    var, final_prices = simulator.calculate_var(
        100, sample_returns, num_simulations=100000
    )
    
    print(f"99% VaR: {var:.4%}")
    print(f"Number of simulations: 100,000")
    print(f"Final prices shape: {final_prices.shape}")
    
    # Backtesting
    print("\n3. VaR Backtesting Results:")
    print("-" * 30)
    
    backtest_results = simulator.backtest_var(
        sample_returns, window_size=252, num_simulations=10000
    )
    
    print(f"Violations: {backtest_results['violations']}/{backtest_results['total_tests']}")
    print(f"Violation rate: {backtest_results['violation_rate']:.4%}")
    print(f"Expected rate: {backtest_results['expected_violation_rate']:.4%}")
    print(f"99% Confidence interval: [{backtest_results['confidence_interval_lower']:.4%}, "
          f"{backtest_results['confidence_interval_upper']:.4%}]")
    print(f"Model valid: {backtest_results['model_valid']}")
    print(f"P-value: {backtest_results['p_value']:.4f}")

if __name__ == "__main__":
    main()