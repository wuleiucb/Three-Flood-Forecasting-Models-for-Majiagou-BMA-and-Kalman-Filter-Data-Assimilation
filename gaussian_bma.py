#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Bayesian Model Averaging (BMA) for three deterministic flood models.

Input  : one observation column + three model-output columns.
Output : BMA mean, predictive variance/SD, 95% predictive interval,
         CRPS, PICP, interval sharpness, NSE, RMSE, MAE and PBIAS.

Example
-------
python gaussian_bma.py 001.csv
# uses defaults: --obs "实测径流/(m³/s)"
#                --models "XAJ模拟/(m³/s)" "TANK模拟/(m³/s)" "TOPMODEL"
#                --output BMA_result.xlsx

Or override explicitly:
python gaussian_bma.py 001.csv --output result.xlsx --interval 0.9
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


# =========================
# BMA result object
# =========================
@dataclass
class GaussianBMAResult:
    model_names: Sequence[str]
    weights: np.ndarray
    sigma2: np.ndarray
    log_likelihood: float
    iterations: int
    converged: bool

    @property
    def sigma(self) -> np.ndarray:
        """Component residual standard deviations."""
        return np.sqrt(self.sigma2)


# =========================
# EM parameter estimation
# =========================
def fit_gaussian_bma(
    y: Sequence[float],
    predictions: np.ndarray,
    model_names: Optional[Sequence[str]] = None,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> GaussianBMAResult:
    """
    Estimate Gaussian BMA weights and component variances by EM.

    Model:
        y_t | M_k ~ Normal(f_kt, sigma_k^2)

    Parameters
    ----------
    y : array-like, shape (n,)
        Observations.
    predictions : array-like, shape (n, 3)
        Three model outputs. Each column corresponds to one model.
    model_names : sequence of str, optional
        Names of the three models.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Relative log-likelihood convergence tolerance.

    Returns
    -------
    GaussianBMAResult
    """
    y = np.asarray(y, dtype=float).ravel()
    P = np.asarray(predictions, dtype=float)

    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("predictions must have shape (n, 3): observation times x three models.")
    if len(y) != P.shape[0]:
        raise ValueError("Length of y must equal the number of prediction rows.")
    if len(y) < 10:
        raise ValueError("At least 10 valid samples are recommended for stable BMA estimation.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(P)):
        raise ValueError("y and predictions contain NaN or infinite values.")

    if model_names is None:
        model_names = ["Model_1", "Model_2", "Model_3"]
    if len(model_names) != 3:
        raise ValueError("Exactly three model names are required.")

    n, k = P.shape
    scale_var = max(float(np.var(y)), float(np.var(P)), 1e-12)
    var_floor = max(scale_var * 1e-10, 1e-12)

    # Equal initial weights; data-driven initial component variances.
    w = np.full(k, 1.0 / k)
    sigma2 = np.maximum(np.mean((P - y[:, None]) ** 2, axis=0), var_floor)

    prev_ll = -np.inf
    converged = False
    loglik = -np.inf

    for iteration in range(1, max_iter + 1):
        # log phi(y_t | f_kt, sigma_k^2)
        log_comp = -0.5 * (
            (P - y[:, None]) ** 2 / sigma2[None, :]
            + np.log(2.0 * np.pi * sigma2)[None, :]
        )
        a = log_comp + np.log(w)[None, :]
        a_max = np.max(a, axis=1, keepdims=True)
        exp_a = np.exp(a - a_max)
        z = exp_a / exp_a.sum(axis=1, keepdims=True)

        # M step
        w_new = z.mean(axis=0)
        sigma2_new = np.sum(z * (P - y[:, None]) ** 2, axis=0) / np.sum(z, axis=0)
        sigma2_new = np.maximum(sigma2_new, var_floor)

        # Observed-data log-likelihood
        loglik = float(np.sum(np.log(exp_a.sum(axis=1)) + a_max.ravel()))
        if abs(loglik - prev_ll) < tol * (1.0 + abs(prev_ll)):
            w, sigma2 = w_new, sigma2_new
            converged = True
            break

        w, sigma2, prev_ll = w_new, sigma2_new, loglik

    w = w / w.sum()
    return GaussianBMAResult(
        model_names=list(model_names),
        weights=w,
        sigma2=sigma2,
        log_likelihood=loglik,
        iterations=iteration,
        converged=converged,
    )


# =========================
# Prediction and uncertainty
# =========================
def _mixture_cdf(x: float, means: np.ndarray, weights: np.ndarray, sigmas: np.ndarray) -> float:
    return float(np.sum(weights * norm.cdf((x - means) / sigmas)))


def mixture_quantile(
    p: float,
    means: np.ndarray,
    result: GaussianBMAResult,
) -> float:
    """Quantile of a univariate Gaussian mixture."""
    means = np.asarray(means, dtype=float)
    weights = np.asarray(result.weights, dtype=float)
    sigmas = np.asarray(result.sigma, dtype=float)

    pad = max(1.0, 0.01 * np.max(np.abs(means)))
    lo = float(np.min(means - 12.0 * sigmas) - pad)
    hi = float(np.max(means + 12.0 * sigmas) + pad)

    return float(
        brentq(
            lambda x: _mixture_cdf(x, means, weights, sigmas) - p,
            lo,
            hi,
            xtol=1e-9,
            rtol=1e-12,
            maxiter=200,
        )
    )


def predict_gaussian_bma(
    predictions: np.ndarray,
    result: GaussianBMAResult,
    interval: float = 0.95,
) -> pd.DataFrame:
    """
    Predict BMA mean, total variance, standard deviation and predictive interval.

    BMA mean:
        mu_t = sum_k w_k f_kt

    Total BMA variance:
        v_t = sum_k w_k[(f_kt-mu_t)^2 + sigma_k^2]
    """
    P = np.asarray(predictions, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("predictions must have shape (n, 3).")
    if not np.all(np.isfinite(P)):
        raise ValueError("predictions contain NaN or infinite values.")

    if not (0.0 < interval < 1.0):
        raise ValueError("interval must be between 0 and 1.")

    w = result.weights
    s = result.sigma
    mean = P @ w
    variance = np.sum(w[None, :] * ((P - mean[:, None]) ** 2 + s[None, :] ** 2), axis=1)
    sd = np.sqrt(np.maximum(variance, 0.0))

    alpha = (1.0 - interval) / 2.0
    lower = np.empty(len(P), dtype=float)
    upper = np.empty(len(P), dtype=float)
    for i in range(len(P)):
        lower[i] = mixture_quantile(alpha, P[i], result)
        upper[i] = mixture_quantile(1.0 - alpha, P[i], result)

    return pd.DataFrame(
        {
            "bma_mean": mean,
            "bma_variance": variance,
            "bma_sd": sd,
            f"pi_{100 * alpha:g}%": lower,
            f"pi_{100 * (1 - alpha):g}%": upper,
            "interval_width": upper - lower,
        }
    )


# =========================
# CRPS and deterministic metrics
# =========================
def _eabs_normal(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """E|X| for X~N(mu, sigma^2), element-wise."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = mu / sigma
    return sigma * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z * z) + mu * (
        2.0 * norm.cdf(z) - 1.0
    )


def crps_gaussian_mixture(y: Sequence[float], predictions: np.ndarray, result: GaussianBMAResult) -> np.ndarray:
    """
    Exact CRPS for a Gaussian mixture using
        CRPS = E|X-y| - 0.5 E|X-X'|.
    """
    y = np.asarray(y, dtype=float)
    P = np.asarray(predictions, dtype=float)
    w = result.weights
    s = result.sigma

    first_term = _eabs_normal(P - y[:, None], s[None, :]) @ w

    pair = np.empty((len(w), len(w)), dtype=float)
    for j in range(len(w)):
        for k in range(len(w)):
            pair[j, k] = np.mean(
                _eabs_normal(
                    P[:, j] - P[:, k],
                    np.sqrt(s[j] ** 2 + s[k] ** 2),
                )
            )

    return first_term - 0.5 * (w @ pair @ w)


def deterministic_metrics(y: Sequence[float], pred: Sequence[float]) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "NSE": 1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2),
        "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
        "MAE": float(np.mean(np.abs(y - pred))),
        "PBIAS_%": float(np.sum(pred - y) / np.sum(y) * 100.0),
    }


def probabilistic_metrics(
    y: Sequence[float],
    predictions: np.ndarray,
    forecast: pd.DataFrame,
    result: GaussianBMAResult,
    interval: float = 0.95,
) -> dict:
    y = np.asarray(y, dtype=float)
    crps = crps_gaussian_mixture(y, predictions, result)
    alpha = (1.0 - interval) / 2.0
    lo_col = f"pi_{100 * alpha:g}%"
    hi_col = f"pi_{100 * (1 - alpha):g}%"
    inside = (forecast[lo_col].to_numpy() <= y) & (y <= forecast[hi_col].to_numpy())

    return {
        "CRPS_mean": float(np.mean(crps)),
        "PICP": float(np.mean(inside)),
        "Sharpness_mean_width": float(np.mean(forecast["interval_width"])),
        "Nominal_coverage": interval,
    }


# =========================
# One-step API
# =========================
def run_bma(
    data: pd.DataFrame,
    obs_col: str,
    model_cols: Sequence[str],
    interval: float = 0.95,
) -> Tuple[GaussianBMAResult, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run Gaussian BMA for a dataframe with exactly four required columns.

    Returns
    -------
    result : GaussianBMAResult
    details : DataFrame with model inputs, BMA mean and uncertainty interval
    parameters : DataFrame with weights and component residual variances
    metrics : DataFrame with deterministic and probabilistic metrics
    """
    if len(model_cols) != 3:
        raise ValueError("Exactly three model columns are required.")
    required = [obs_col, *model_cols]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in input data: {missing}")

    work = data[required].apply(pd.to_numeric, errors="coerce")
    valid = work.notna().all(axis=1)
    dropped = int((~valid).sum())
    if dropped:
        print(f"Warning: {dropped} rows with missing/non-numeric values were excluded.")
    work = work.loc[valid].copy()

    if len(work) < 10:
        raise ValueError("Too few valid samples for BMA.")

    y = work[obs_col].to_numpy(dtype=float)
    P = work[list(model_cols)].to_numpy(dtype=float)

    result = fit_gaussian_bma(y, P, model_names=model_cols)
    pred_df = predict_gaussian_bma(P, result, interval=interval)
    crps = crps_gaussian_mixture(y, P, result)

    details = pd.concat(
        [
            work.reset_index(drop=True),
            pred_df,
            pd.DataFrame(
                {
                    "inside_prediction_interval": (
                        (pred_df.iloc[:, 3].to_numpy() <= y)
                        & (y <= pred_df.iloc[:, 4].to_numpy())
                    ).astype(int),
                    "CRPS": crps,
                }
            ),
        ],
        axis=1,
    )

    parameters = pd.DataFrame(
        {
            "model": result.model_names,
            "weight": result.weights,
            "residual_variance": result.sigma2,
            "residual_sd": result.sigma,
        }
    )
    parameters.loc[len(parameters)] = [
        "EM_log_likelihood",
        result.log_likelihood,
        result.iterations,
        result.converged,
    ]

    metric_values = {
        **deterministic_metrics(y, pred_df["bma_mean"]),
        **probabilistic_metrics(y, P, pred_df, result, interval=interval),
        "n_samples": len(work),
        "EM_iterations": result.iterations,
        "EM_converged": result.converged,
        "EM_log_likelihood": result.log_likelihood,
    }
    metrics = pd.DataFrame(
        {
            "metric": list(metric_values.keys()),
            "value": list(metric_values.values()),
        }
    )

    return result, details, parameters, metrics


# =========================
# File input/output
# =========================
def read_table(
    path: str,
    sheet: Optional[str] = None,
    header_row: int = 0,
) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(
            path,
            sheet_name=sheet if sheet is not None else 0,
            header=header_row,
        )
    if ext == ".csv":
        return pd.read_csv(path, header=header_row)
    raise ValueError("Supported input formats: .xlsx, .xls and .csv")


def write_output(
    output_path: str,
    details: pd.DataFrame,
    parameters: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            metrics.to_excel(writer, sheet_name="metrics", index=False)
            parameters.to_excel(writer, sheet_name="bma_parameters", index=False)
            details.to_excel(writer, sheet_name="forecast_uncertainty", index=False)
    elif ext == ".csv":
        details.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("Output file must end with .xlsx, .xls or .csv")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gaussian BMA for one observation column and three model-output columns."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="001.csv",
        help="Input .xlsx/.xls/.csv file; default: 001.csv",
    )
    parser.add_argument("--sheet", default=None, help="Excel sheet name; default: first sheet")
    parser.add_argument("--header-row", type=int, default=0, help="Row number used as column names; default: 0 (first row)")
    parser.add_argument(
        "--obs",
        default="实测径流/(m³/s)",
        help="Observation column name; default: '实测径流/(m³/s)'",
    )
    parser.add_argument(
        "--models",
        nargs=3,
        default=["XAJ模拟/(m³/s)", "TANK模拟/(m³/s)", "TOPMODEL"],
        metavar=("MODEL1", "MODEL2", "MODEL3"),
        help="Exactly three model output column names; "
             "default: 'XAJ模拟/(m³/s)' 'TANK模拟/(m³/s)' 'TOPMODEL'",
    )
    parser.add_argument("--output", default="BMA_result.xlsx", help="Output file path; default: BMA_result.xlsx")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.95,
        help="Central predictive interval level, default: 0.95",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    data = read_table(args.input, sheet=args.sheet, header_row=args.header_row)
    result, details, parameters, metrics = run_bma(
        data=data,
        obs_col=args.obs,
        model_cols=args.models,
        interval=args.interval,
    )
    write_output(args.output, details, parameters, metrics)

    print("Gaussian BMA completed successfully.")
    print(parameters.to_string(index=False))
    print(metrics.to_string(index=False))
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
