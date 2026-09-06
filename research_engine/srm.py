"""Cross-asset five-channel Shape Retrieval Model (SRM).

SRM imports only the shared target definition and the independent geometric
representation.  It never imports IV, returns, HAR features, GLASSO, or a
regression forecast head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np
import torch

from .common import EPS, future_target
from .geometry import CHANNELS, ChannelWeightTransformer, GeometryConfig, descriptors, query_state


RetrievalScope = Literal["cross_asset", "same_asset"]


@dataclass(frozen=True)
class SRMFitConfig:
    train_stride: int = 5
    training_neighbors: int = 100
    learning_rate: float = 1e-3
    epochs: int = 100
    patience: int = 10
    temperature_grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    candidate_chunk: int = 65536
    query_batch: int = 4
    seed: int = 42


def _loss(y: torch.Tensor, prediction: torch.Tensor, criterion: str) -> torch.Tensor:
    if criterion == "mse":
        return (y - prediction) ** 2
    if criterion == "qlike":
        ratio = y / (prediction + EPS)
        return ratio - torch.log(ratio) - 1.0
    raise ValueError("criterion must be mse or qlike")


class GeometryMemory:
    """Precomputed geometric descriptors for every ticker and valid endpoint."""
    def __init__(self, rv: np.ndarray, cfg: GeometryConfig, device: torch.device):
        self.rv = np.asarray(rv, float)
        self.cfg, self.device = cfg, device
        n_dates, n_assets = self.rv.shape
        endpoint_values = list(range(cfg.length - 1, n_dates))
        channel_rows = {key: [] for key in CHANNELS}
        assets, endpoints = [], []
        for asset in range(n_assets):
            for endpoint in endpoint_values:
                d = descriptors(self.rv[:, asset], endpoint, cfg)
                for key in CHANNELS:
                    channel_rows[key].append(d[key].reshape(-1))
                assets.append(asset)
                endpoints.append(endpoint)
        self.asset = np.asarray(assets, dtype=int)
        self.endpoint = np.asarray(endpoints, dtype=int)
        self.channels = {
            key: torch.as_tensor(np.asarray(rows, dtype=np.float32), device=device)
            for key, rows in channel_rows.items()
        }
        self._relative_cache: dict[int, np.ndarray] = {}

    def eligible(self, maximum_endpoint: int) -> np.ndarray:
        return np.flatnonzero(self.endpoint <= int(maximum_endpoint)).astype(np.int64)

    def eligible_for_asset(self, maximum_endpoint: int, asset: int, scope: RetrievalScope) -> np.ndarray:
        ids = self.eligible(maximum_endpoint)
        if scope == "same_asset":
            ids = ids[self.asset[ids] == int(asset)]
        elif scope != "cross_asset":
            raise ValueError("scope must be cross_asset or same_asset")
        return ids

    def relative_change(self, horizon: int) -> np.ndarray:
        if horizon in self._relative_cache:
            return self._relative_cache[horizon]
        values = np.empty(len(self.endpoint), dtype=np.float32)
        for row, (asset, endpoint) in enumerate(zip(self.asset, self.endpoint)):
            if endpoint + horizon >= len(self.rv):
                values[row] = np.nan
            else:
                future = self.rv[endpoint + 1 : endpoint + horizon + 1, asset].mean()
                values[row] = np.log(future / (self.rv[endpoint, asset] + EPS))
        self._relative_cache[horizon] = values
        return values

    def raw_distances(self, query: dict[str, np.ndarray], candidate_ids: np.ndarray) -> torch.Tensor:
        if not len(candidate_ids):
            return torch.empty((0, 5), dtype=torch.float32, device=self.device)
        idx = torch.as_tensor(candidate_ids, dtype=torch.long, device=self.device)
        out = []
        for key in CHANNELS:
            q = torch.as_tensor(np.asarray(query[key], dtype=np.float32).reshape(-1), device=self.device)
            values = self.channels[key].index_select(0, idx)
            out.append(torch.sqrt(torch.mean((values - q) ** 2, dim=1)))
        return torch.stack(out, dim=1)


def _nearest_curve(memory: GeometryMemory, query: dict[str, np.ndarray], candidate_ids: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k selection by the curve channel, streamed in chunks."""
    if not len(candidate_ids):
        return np.empty(0, dtype=np.int64)
    q = torch.as_tensor(np.asarray(query["C"], np.float32).reshape(-1), device=memory.device)
    keep_d = torch.empty(0, device=memory.device)
    keep_i = torch.empty(0, dtype=torch.long, device=memory.device)
    for start in range(0, len(candidate_ids), 65536):
        ids = torch.as_tensor(candidate_ids[start : start + 65536], dtype=torch.long, device=memory.device)
        values = memory.channels["C"].index_select(0, ids)
        distance = torch.sqrt(torch.mean((values - q) ** 2, dim=1))
        joined_d = torch.cat((keep_d, distance))
        joined_i = torch.cat((keep_i, ids))
        take = min(k, len(joined_d))
        keep_d, where = torch.topk(joined_d, take, largest=False, sorted=True)
        keep_i = joined_i.index_select(0, where)
    return keep_i.detach().cpu().numpy().astype(np.int64)


def _standardized_distances(memory: GeometryMemory, query: dict[str, np.ndarray], candidate_ids: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    raw = memory.raw_distances(query, candidate_ids).detach().cpu().numpy()
    return (raw - median) / (iqr + EPS)


def _scale_for_asset(memory: GeometryMemory, asset: int, reference_endpoint: int, candidate_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    query = descriptors(memory.rv[:, asset], reference_endpoint, memory.cfg)
    raw = memory.raw_distances(query, candidate_ids).detach().cpu().numpy()
    if not np.isfinite(raw).all():
        raise ValueError("non-finite training distance")
    return np.median(raw, axis=0), np.subtract(*np.percentile(raw, [75, 25], axis=0))


class SRMTrainer:
    """Pooled Transformer estimator; the output remains a retrieval forecast."""
    def __init__(self, geometry_config: GeometryConfig, fit_config: SRMFitConfig, horizon: int, criterion: str, device: torch.device):
        torch.manual_seed(fit_config.seed)
        np.random.seed(fit_config.seed)
        self.model = ChannelWeightTransformer(geometry_config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=fit_config.learning_rate)
        self.horizon, self.criterion, self.device = horizon, criterion, device
        self.fit_config = fit_config

    def _pack(self, examples: list[tuple]) -> tuple | None:
        if not examples:
            return None
        states, distances, relatives, targets, anchors = zip(*examples)
        width = max(len(x) for x in distances)
        n = len(examples)
        d = np.zeros((n, width, 5), np.float32)
        r = np.zeros((n, width), np.float32)
        valid = np.zeros((n, width), bool)
        for row, (distance, relative) in enumerate(zip(distances, relatives)):
            d[row, :len(distance)] = distance
            r[row, :len(relative)] = relative
            valid[row, :len(distance)] = True
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device),
            torch.as_tensor(d, dtype=torch.float32, device=self.device),
            torch.as_tensor(r, dtype=torch.float32, device=self.device),
            torch.as_tensor(valid, dtype=torch.bool, device=self.device),
            torch.as_tensor(np.asarray(targets), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.asarray(anchors), dtype=torch.float32, device=self.device),
        )

    def _predict(self, packed: tuple, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        states, distance, relative, valid, target, anchor = packed
        beta = self.model(states, self.horizon)
        composite = torch.sum(distance * beta[:, None, :], dim=2).masked_fill(~valid, float("inf"))
        weight = torch.softmax(-composite / temperature, dim=1)
        prediction = anchor * torch.exp(torch.sum(weight * relative, dim=1))
        return target, prediction

    def score(self, examples: list[tuple], temperature: float = 1.0) -> float:
        packed = self._pack(examples)
        if packed is None:
            return float("inf")
        self.model.eval()
        with torch.no_grad():
            target, prediction = self._predict(packed, temperature)
            return float(_loss(target, prediction, self.criterion).mean().cpu())

    def fit(self, train: list[tuple], validation: list[tuple]) -> dict:
        packed = self._pack(train)
        if packed is None:
            raise ValueError("SRM has no eligible training examples")
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(self.fit_config.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            target, prediction = self._predict(packed)
            loss = _loss(target, prediction, self.criterion).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite SRM training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            score = self.score(validation) if validation else float(loss.detach().cpu())
            if score < best - 1e-10:
                best = score
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= self.fit_config.patience:
                break
        if best_state is None:
            raise RuntimeError("SRM did not produce a valid model state")
        self.model.load_state_dict(best_state)
        return {"epochs": epoch + 1, "validation_objective": best}

    def select_temperature(self, validation: list[tuple]) -> tuple[float, float]:
        candidates = [(temperature, self.score(validation, temperature)) for temperature in self.fit_config.temperature_grid]
        return min(candidates, key=lambda item: item[1])

    def beta(self, state: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            result = self.model(torch.as_tensor(state[None], dtype=torch.float32, device=self.device), self.horizon)[0]
        beta = result.detach().cpu().numpy()
        if (beta < 0).any() or not np.isclose(beta.sum(), 1.0, atol=1e-6):
            raise ValueError("invalid Transformer weights")
        return beta


def _make_examples(memory: GeometryMemory, times: Iterable[int], candidate_limit: int, horizon: int, scales: dict[int, tuple[np.ndarray, np.ndarray]], config: SRMFitConfig, criterion: str, train_mode: bool, scope: RetrievalScope = "cross_asset") -> list[tuple]:
    """Create retrieval examples without using any later candidate endpoint."""
    relative = memory.relative_change(horizon)
    examples: list[tuple] = []
    for asset in range(memory.rv.shape[1]):
        series = memory.rv[:, asset]
        median, iqr = scales[asset]
        for t in list(times)[:: config.train_stride]:
            endpoint = int(t - 1)
            max_endpoint = min(candidate_limit, t - memory.cfg.length - horizon) if train_mode else candidate_limit
            candidate_ids = memory.eligible_for_asset(max_endpoint, asset, scope)
            if len(candidate_ids) < memory.cfg.neighbors:
                continue
            query = descriptors(series, endpoint, memory.cfg)
            selected = _nearest_curve(memory, query, candidate_ids, config.training_neighbors)
            distance = _standardized_distances(memory, query, selected, median, iqr)
            examples.append((query_state(query), distance.astype(np.float32), relative[selected].astype(np.float32), float(future_target(series, t, horizon)), float(series[t - 1])))
    if not examples:
        raise ValueError("no SRM examples; check the window lengths and candidate boundary")
    return examples


def _test_prediction(memory: GeometryMemory, trainer: SRMTrainer, asset: int, t: int, horizon: int, candidate_ids: np.ndarray, median: np.ndarray, iqr: np.ndarray, temperature: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    series = memory.rv[:, asset]
    query = descriptors(series, int(t - 1), memory.cfg)
    beta = trainer.beta(query_state(query))
    # Keep all candidate-distance arithmetic on the selected device.  This is
    # algebraically identical to the former NumPy implementation but avoids
    # moving the full cross-asset candidate matrix back to host memory for
    # every forecast origin.
    raw = memory.raw_distances(query, candidate_ids)
    median_t = torch.as_tensor(median, dtype=raw.dtype, device=raw.device)
    iqr_t = torch.as_tensor(iqr, dtype=raw.dtype, device=raw.device)
    beta_t = torch.as_tensor(beta, dtype=raw.dtype, device=raw.device)
    composite = torch.sum(((raw - median_t) / (iqr_t + EPS)) * beta_t, dim=1)
    take = min(memory.cfg.neighbors, len(candidate_ids))
    _, order_t = torch.topk(composite, take, largest=False, sorted=True)
    order = order_t.detach().cpu().numpy()
    selected = candidate_ids[order]
    relative = memory.relative_change(horizon)[selected]
    selected_composite = composite.index_select(0, order_t).detach().cpu().numpy()
    logits = -(selected_composite - selected_composite.min()) / temperature
    weight = np.exp(logits)
    weight /= weight.sum()
    prediction = float(series[t - 1] * np.exp(weight @ relative))
    if not np.isfinite(prediction) or prediction <= 0:
        raise FloatingPointError("invalid SRM prediction")
    return prediction, beta, selected_composite, selected


def fit_and_predict_window(rv: np.ndarray, train_t: np.ndarray, validation_t: np.ndarray, test_t: np.ndarray, horizon: int, criterion: str, geometry_config: GeometryConfig, fit_config: SRMFitConfig, device: torch.device, memory: GeometryMemory | None = None, scope: RetrievalScope = "cross_asset") -> tuple[np.ndarray, list[dict], dict]:
    """Select SRM settings before the test block, refit on all pre-test data.

    The first fit uses training examples and validation outcomes for early
    stopping and temperature selection.  Once selected, the Transformer is
    refit for the selected number of epochs on the union of training and
    validation origins.  Thus its final information set matches the four
    benchmark estimators while the test block remains untouched.
    """
    memory = memory or GeometryMemory(rv, geometry_config, device)
    fit_t = np.concatenate((np.asarray(train_t, dtype=int), np.asarray(validation_t, dtype=int)))
    if len(np.unique(fit_t)) != len(fit_t):
        raise ValueError("training and validation origins overlap")
    # Distance scales use only paths fully observed before the validation
    # endpoint.  They are fixed before any test observation is available.
    scales = {
        asset: _scale_for_asset(
            memory,
            asset,
            int(validation_t[-1] - 1),
            memory.eligible_for_asset(
                int(validation_t[-1] - geometry_config.length - horizon), asset, scope
            ),
        )
        for asset in range(rv.shape[1])
    }
    train = _make_examples(memory, train_t, int(train_t[-1] - horizon), horizon, scales, fit_config, criterion, train_mode=True, scope=scope)
    validation = _make_examples(memory, validation_t, int(train_t[-1] - horizon), horizon, scales, fit_config, criterion, train_mode=False, scope=scope)
    trainer = SRMTrainer(geometry_config, fit_config, horizon, criterion, device)
    selection_audit = trainer.fit(train, validation)
    temperature, validation_loss = trainer.select_temperature(validation)

    refit_epochs = max(1, int(selection_audit["epochs"]))
    refit_values = dict(fit_config.__dict__)
    refit_values["epochs"] = refit_epochs
    refit_values["patience"] = refit_epochs + 1
    refit_config = SRMFitConfig(**refit_values)
    refit_train = _make_examples(
        memory, fit_t, int(validation_t[-1] - horizon), horizon, scales,
        refit_config, criterion, train_mode=True, scope=scope,
    )
    trainer = SRMTrainer(geometry_config, refit_config, horizon, criterion, device)
    refit_audit = trainer.fit(refit_train, [])

    output = np.empty((len(test_t), rv.shape[1]), dtype=float)
    diagnostics: list[dict] = []
    candidate_maxima: list[int] = []
    for row, t in enumerate(test_t):
        for asset in range(rv.shape[1]):
            median, iqr = scales[asset]
            candidate_ids = memory.eligible_for_asset(int(validation_t[-1] - horizon), asset, scope)
            pred, beta, selected_distance, selected = _test_prediction(memory, trainer, asset, int(t), horizon, candidate_ids, median, iqr, temperature)
            output[row, asset] = pred
            candidate_max = int(memory.endpoint[candidate_ids].max())
            candidate_maxima.append(candidate_max)
            selected_assets = memory.asset[selected]
            diagnostics.append({
                "Asset_Index": asset, "Forecast_t": int(t),
                "Candidate_Count": int(len(candidate_ids)), "Neighbor_Count": int(len(selected_distance)),
                "Candidate_Max_Endpoint": candidate_max,
                "Validation_End_t": int(validation_t[-1]), "Horizon": horizon,
                "Retrieval_Scope": scope,
                "Cross_Asset_Neighbor_Share": float(np.mean(selected_assets != asset)),
                "Temperature": float(temperature), "Min_Distance": float(selected_distance.min()),
                "Mean_Neighbor_Distance": float(selected_distance.mean()),
                **{key: float(value) for key, value in zip(CHANNELS, beta)},
            })
    audit = {
        "selection_fit": selection_audit, "refit": refit_audit,
        "selected_temperature": float(temperature),
        "temperature_validation_loss": float(validation_loss),
        "training_examples": len(train), "validation_examples": len(validation),
        "refit_examples": len(refit_train),
        "test_candidate_max_endpoint": int(max(candidate_maxima)),
        "validation_end_t": int(validation_t[-1]),
        "retrieval_scope": scope,
    }
    return output, diagnostics, audit
