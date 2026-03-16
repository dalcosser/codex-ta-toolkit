import math
from dataclasses import dataclass
from scipy.stats import norm

@dataclass
class BSParams:
    S: float      # Spot price
    K: float      # Strike price
    T: float      # Time to expiration (in years)
    r: float      # Risk-free rate
    sigma: float  # Volatility (implied vol)
    q: float = 0.0  # Dividend yield

def _d1(params: BSParams) -> float:
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def _d2(params: BSParams) -> float:
    return _d1(params) - params.sigma * math.sqrt(params.T)

def greeks(params: BSParams, option_type: str = "call") -> dict:
    """
    Black-Scholes greeks for a call or put.
    Returns dict with delta, gamma, vega, theta, rho.
    """
    S, K, T, r, sigma, q = params.S, params.K, params.T, params.r, params.sigma, params.q
    d1 = _d1(params)
    d2 = _d2(params)
    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    if option_type.lower() == "call":
        delta = math.exp(-q * T) * Nd1
        rho = K * T * math.exp(-r * T) * Nd2 / 100
        theta = (- (S * nd1 * sigma * math.exp(-q * T)) / (2 * math.sqrt(T))
                 - r * K * math.exp(-r * T) * Nd2
                 + q * S * math.exp(-q * T) * Nd1) / 365
    else:
        delta = -math.exp(-q * T) * (1 - Nd1)
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
        theta = (- (S * nd1 * sigma * math.exp(-q * T)) / (2 * math.sqrt(T))
                 + r * K * math.exp(-r * T) * norm.cdf(-d2)
                 - q * S * math.exp(-q * T) * norm.cdf(-d1)) / 365

    gamma = (math.exp(-q * T) * nd1) / (S * sigma * math.sqrt(T))
    vega = S * math.exp(-q * T) * nd1 * math.sqrt(T) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
