"""Anytime-valid sequential test of a tier claim "WER <= tau".

Utterances are drawn with probability proportional to reference length. Under that
design the expected per-utterance error rate E[E/N] equals corpus WER, so after capping
the rate at `cap` the claim becomes a bounded-mean claim E[Y] <= tau/cap with
Y = min(E/N, cap)/cap in [0, 1].

Two betting capital processes run side by side (Waudby-Smith & Ramdas, 2024):
  certify: bets that E[Y] < tau/cap; by Ville's inequality it ever reaches 1/alpha with
           probability <= alpha if in fact WER >= tau.
  refute:  bets that E[Y] > tau/cap; same guarantee if in fact WER <= tau.
Bets use the predictable plug-in rule, so the audit may stop after any draw.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    certify_capital: np.ndarray
    refute_capital: np.ndarray
    decision: str           # "certified", "refuted" or "undecided"
    stop_index: int | None  # number of clusters seen when the decision was made


def _plugin_lambdas(y: np.ndarray, alpha: float, max_bet: float) -> np.ndarray:
    """Predictable plug-in bet sizes from running mean and variance (prior 1/4)."""
    t = np.arange(1, len(y) + 1)
    mean = (0.5 + np.cumsum(y)) / (t + 1)                 # running mean after t draws
    sq = np.concatenate([[0.0], np.cumsum((y - mean) ** 2)[:-1]])
    var_prev = (0.25 + sq) / t                            # uses draws 1..t-1 only
    lam = np.sqrt(2 * np.log(1 / alpha) / (var_prev * t * np.log(t + 1)))
    return np.minimum(lam, max_bet)


def sequential_rate_test(rates: np.ndarray, tau: float, alpha: float = 0.05,
                         cap: float = 2.0) -> Trajectory:
    """Tier test on per-utterance error rates drawn with probability proportional to
    reference length. Under that design E[min(E/N, cap)] equals corpus WER (up to the
    cap), so the claim is a bounded-mean claim E[r/cap] <= tau/cap with no frame-wide
    length bound, which makes the bets far more efficient than the cluster-sum form.
    """
    y = np.minimum(rates, cap) / cap
    y0 = tau / cap
    lam_c = _plugin_lambdas(y, alpha, max_bet=0.5 / (1 - y0))   # keeps 1 + lam*(y0 - y) > 0
    lam_r = _plugin_lambdas(y, alpha, max_bet=0.5 / y0)         # keeps 1 + lam*(y - y0) > 0
    k_cert = np.cumprod(1 + lam_c * (y0 - y))
    k_ref = np.cumprod(1 + lam_r * (y - y0))
    return _decide(k_cert, k_ref, alpha)


def _decide(k_cert: np.ndarray, k_ref: np.ndarray, alpha: float) -> Trajectory:
    thresh = 1 / alpha
    hit_c = np.flatnonzero(k_cert >= thresh)
    hit_r = np.flatnonzero(k_ref >= thresh)
    first_c = hit_c[0] if hit_c.size else np.inf
    first_r = hit_r[0] if hit_r.size else np.inf
    if first_c == first_r == np.inf:
        return Trajectory(k_cert, k_ref, "undecided", None)
    if first_c < first_r:
        return Trajectory(k_cert, k_ref, "certified", int(first_c) + 1)
    return Trajectory(k_cert, k_ref, "refuted", int(first_r) + 1)
