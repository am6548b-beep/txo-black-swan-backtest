"""Black-Scholes helpers used only as conservative fallbacks."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["C", "P"]


def _safe_t(dte: float) -> float:
    return max(float(dte) / 365.0, 1e-8)


def bs_price(
    spot: float,
    strike: float,
    dte: float,
    rate: float,
    vol: float,
    cp: OptionType,
) -> float:
    """Return Black-Scholes option price in index points."""

    if spot <= 0 or strike <= 0 or vol <= 0:
        return float("nan")
    t = _safe_t(dte)
    sigma_sqrt_t = vol * math.sqrt(t)
    if sigma_sqrt_t <= 0:
        return float("nan")
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    if cp == "C":
        return spot * norm.cdf(d1) - strike * math.exp(-rate * t) * norm.cdf(d2)
    return strike * math.exp(-rate * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def bs_delta(
    spot: float,
    strike: float,
    dte: float,
    rate: float,
    vol: float,
    cp: OptionType,
) -> float:
    """Return Black-Scholes delta."""

    if spot <= 0 or strike <= 0 or vol <= 0:
        return float("nan")
    t = _safe_t(dte)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    if cp == "C":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    dte: float,
    rate: float,
    cp: OptionType,
) -> float | None:
    """Invert Black-Scholes IV. Return None if no sane solution exists."""

    if price <= 0 or spot <= 0 or strike <= 0 or dte <= 0:
        return None
    intrinsic = max(0.0, spot - strike) if cp == "C" else max(0.0, strike - spot)
    upper_bound = spot if cp == "C" else strike
    if price < intrinsic * 0.99 or price > upper_bound * 1.5:
        return None

    def objective(vol: float) -> float:
        return bs_price(spot, strike, dte, rate, vol, cp) - price

    try:
        return float(brentq(objective, 1e-4, 5.0, maxiter=100))
    except (ValueError, RuntimeError, OverflowError):
        return None


def valid_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def put_spread_expiry_payoff(
    expiry_price: float,
    long_put_strike: float,
    short_put_strike: float,
    point_value: float,
    quantity: int = 1,
) -> float:
    """Return gross long put-spread payoff at expiry in currency units."""

    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if short_put_strike >= long_put_strike:
        raise ValueError("short_put_strike must be below long_put_strike")
    long_payoff = max(long_put_strike - expiry_price, 0.0)
    short_payoff = max(short_put_strike - expiry_price, 0.0)
    return float((long_payoff - short_payoff) * point_value * quantity)
