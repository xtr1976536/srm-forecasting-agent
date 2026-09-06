"""Fair pooled HARM/GHARM benchmark estimators.

The four benchmark architectures are unchanged across estimation criteria.
MSE uses its closed-form pooled OLS solution.  QLIKE uses the same linear
predictor and ticker intercepts, optimized directly under QLIKE with a small
positive numerical floor applied only when evaluating the loss.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
import torch
from sklearn.covariance import GraphicalLassoCV

from .common import EPS, future_target, har_features


BASELINE_MODELS = ("HARM", "HARM+IV", "GHARM", "GHARM+IV")


@dataclass(frozen=True)
class BaselineFitConfig:
    qlike_learning_rate: float = 3e-3
    qlike_epochs: int = 800
    qlike_patience: int = 60
    qlike_min_delta: float = 1e-10
    seed: int = 42


def graph_weights(returns: np.ndarray) -> tuple[np.ndarray, dict]:
    """Estimate a training-information Graphical Lasso graph with no self loops."""
    x = np.asarray(returns, float)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + EPS)
    # sklearn can emit this warning when a *candidate* regularization value
    # has an invalid CV score.  It does not invalidate the selected fit; the
    # selected precision matrix is checked explicitly below and the warning is
    # recorded in the window audit.
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", message="invalid value encountered in subtract", category=RuntimeWarning)
        estimator = GraphicalLassoCV(cv=3, max_iter=300).fit(x)
    if not np.isfinite(estimator.precision_).all():
        raise FloatingPointError("Graphical Lasso produced a non-finite precision matrix")
    adjacency = (np.abs(estimator.precision_) > 1e-10).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(axis=1)
    weights = np.diag(1.0 / np.sqrt(degree + EPS)) @ adjacency @ np.diag(1.0 / np.sqrt(degree + EPS))
    audit = {
        "edges": int(adjacency.sum() / 2),
        "isolated_nodes": int((degree == 0).sum()),
        "iterations": int(getattr(estimator, "n_iter_", 0)),
        "converged": bool(getattr(estimator, "n_iter_", 0) < 300),
        "alpha": float(estimator.alpha_),
        "cv_runtime_warnings": [str(item.message) for item in caught],
    }
    return weights, audit


def pooled_ols(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ticker intercepts plus common slopes, with no ridge penalty."""
    n_time, n_assets, n_features = features.shape
    design = np.zeros((n_time * n_assets, n_assets + n_features), dtype=float)
    design[:, :n_assets] = np.tile(np.eye(n_assets), (n_time, 1))
    design[:, n_assets:] = features.reshape(n_time * n_assets, n_features)
    coefficients, *_ = np.linalg.lstsq(design, target.reshape(-1), rcond=None)
    return coefficients[:n_assets], coefficients[n_assets:]


def pooled_predict(features: np.ndarray, intercept: np.ndarray, slope: np.ndarray) -> np.ndarray:
    return intercept[None, :] + np.einsum("tnp,p->tn", features, slope)


def _qlike_torch(y: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(prediction, min=EPS)
    ratio = y / p
    return ratio - torch.log(ratio) - 1.0


def pooled_qlike(features: np.ndarray, target: np.ndarray, config: BaselineFitConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray, dict]:
    """Direct QLIKE optimization for the same pooled linear HARM predictor.

    OLS initializes the coefficients.  The initial intercepts are shifted by a
    common amount only when needed to keep all fitted values in the positive
    domain of QLIKE.  The fitted functional form remains a ticker intercept
    plus common linear slopes.
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    intercept0, slope0 = pooled_ols(features, target)
    initial = pooled_predict(features, intercept0, slope0)
    if initial.min() <= EPS:
        intercept0 = intercept0 + (EPS - initial.min()) + 1e-6

    x = torch.as_tensor(features, dtype=torch.float64, device=device)
    y = torch.as_tensor(target, dtype=torch.float64, device=device)
    intercept = torch.nn.Parameter(torch.as_tensor(intercept0, dtype=torch.float64, device=device))
    slope = torch.nn.Parameter(torch.as_tensor(slope0, dtype=torch.float64, device=device))
    optimizer = torch.optim.Adam((intercept, slope), lr=config.qlike_learning_rate)
    best_loss, best = float("inf"), None
    stale = 0
    for epoch in range(config.qlike_epochs):
        optimizer.zero_grad()
        prediction = intercept[None, :] + torch.einsum("tnp,p->tn", x, slope)
        loss = _qlike_torch(y, prediction).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite baseline QLIKE loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((intercept, slope), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss - config.qlike_min_delta:
            best_loss = value
            best = (intercept.detach().clone(), slope.detach().clone())
            stale = 0
        else:
            stale += 1
        if stale >= config.qlike_patience:
            break
    if best is None:
        raise RuntimeError("QLIKE baseline optimization did not produce a solution")
    intercept, slope = (item.detach().cpu().numpy() for item in best)
    return intercept, slope, {"epochs": epoch + 1, "objective": best_loss}


def feature_tensor(rv: np.ndarray, iv: np.ndarray, target_times: np.ndarray, horizon: int, graph: np.ndarray | None, use_iv: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([har_features(rv, iv, int(t), use_iv) for t in target_times])
    y = np.stack([future_target(rv, int(t), horizon) for t in target_times])
    if graph is not None:
        x = np.concatenate((x, np.einsum("ij,tjp->tip", graph, x)), axis=2)
    return x, y


def fit_and_predict_window(
    rv: np.ndarray,
    iv: np.ndarray,
    returns: np.ndarray,
    train_t: np.ndarray,
    validation_t: np.ndarray,
    test_t: np.ndarray,
    horizon: int,
    criterion: str,
    config: BaselineFitConfig,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit all four models on pre-test data and predict one test block.

    Validation outcomes are already available when the test block begins, so
    the final benchmark fit uses the non-overlapping union of training and
    validation origins.  No test origin or test outcome enters this fit.
    """
    if criterion not in {"mse", "qlike"}:
        raise ValueError("criterion must be mse or qlike")
    fit_t = np.concatenate((np.asarray(train_t, dtype=int), np.asarray(validation_t, dtype=int)))
    if len(np.unique(fit_t)) != len(fit_t):
        raise ValueError("training and validation origins overlap")
    # Graph and coefficients use only dates strictly before the test block.
    graph, graph_audit = graph_weights(returns[fit_t])
    predictions: dict[str, np.ndarray] = {}
    audit: dict = {"graph": graph_audit, "fits": {}}
    specifications = ((False, None, "HARM"), (True, None, "HARM+IV"), (False, graph, "GHARM"), (True, graph, "GHARM+IV"))
    for use_iv, graph_matrix, model in specifications:
        x_train, y_train = feature_tensor(rv, iv, fit_t, horizon, graph_matrix, use_iv)
        x_test, _ = feature_tensor(rv, iv, test_t, horizon, graph_matrix, use_iv)
        if criterion == "mse":
            intercept, slope = pooled_ols(x_train, y_train)
            fit_audit = {"objective": float(np.mean((pooled_predict(x_train, intercept, slope) - y_train) ** 2)), "epochs": 0}
        else:
            intercept, slope, fit_audit = pooled_qlike(x_train, y_train, config, device)
        predictions[model] = np.maximum(pooled_predict(x_test, intercept, slope), EPS)
        audit["fits"][model] = fit_audit
    return predictions, audit
