from __future__ import annotations

import math
import logging
from functools import partial
from typing import Any, Dict, List, Optional, Literal

import numpy as np
import torch
from tqdm import tqdm

from .metrics_wasserstein import wasserstein_1d_squared

SinkhornMethod = Literal["standard", "stabilized", "epsilon_scaling"]

logger = logging.getLogger(__name__)

_EPS = 1e-12


def _check_flat_array(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected numpy array or torch tensor, received {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"Array must be 2-D, received shape {x.shape}")
    return x


def _dft(x: torch.Tensor) -> torch.Tensor:
    max_len = x.size(1)
    dft_full = torch.fft.rfft(x, dim=1, norm="ortho")
    dft_re = dft_full.real
    dft_im = dft_full.imag

    zero_padding = torch.zeros_like(dft_im[:, 0:1, :])
    dft_im = dft_im[:, 1:, :]

    if max_len % 2 == 0:
        dft_im = dft_im[:, :-1, :]

    x_tilde = torch.cat((dft_re, dft_im), dim=1)
    return x_tilde.detach()


def _spectral_density(x: torch.Tensor) -> torch.Tensor:
    max_len = x.size(1)
    freq = _dft(x)

    n_real = math.ceil((max_len + 1) / 2)
    x_re = freq[:, :n_real, :]
    x_im = freq[:, n_real:, :]

    zero_padding = torch.zeros((x.size(0), 1, x.size(2)), device=x.device, dtype=x.dtype)
    x_im = torch.cat((zero_padding, x_im), dim=1)

    if max_len % 2 == 0:
        x_im = torch.cat((x_im, zero_padding), dim=1)

    x_dens = x_re**2 + x_im**2
    return x_dens


class WassersteinDistances:
    def __init__(
        self,
        original_data: np.ndarray,
        other_data: np.ndarray,
        normalisation: Optional[str] = "none",
        seed: Optional[int] = None,
        transport_backend: str = "sinkhorn",
        sinkhorn_method: str = "epsilon_scaling",
        sinkhorn_reg: float = 1e-2,
        sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.original_data = original_data
        self.other_data = other_data
        self.normalisation = normalisation
        self.rng = np.random.default_rng(seed)
        if transport_backend not in {"emd", "sinkhorn"}:
            raise ValueError("transport_backend must be 'emd' or 'sinkhorn'")
        self.transport_backend = transport_backend
        self.sinkhorn_method = sinkhorn_method
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_kwargs = dict(sinkhorn_kwargs or {})
    def _normalise(self, orig: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.normalisation == "none":
            return orig, other
        if self.normalisation == "standardise":
            sd = np.std(orig) + _EPS
            return orig / sd, other / sd
        raise ValueError(f"Unknown normalisation mode {self.normalisation}")

    def _project(self, data: np.ndarray, direction: np.ndarray) -> np.ndarray:
        return data @ direction

    def random_direction(self, dim: int) -> np.ndarray:
        vec = self.rng.normal(size=dim)
        norm = np.linalg.norm(vec) + _EPS
        return vec / norm

    def get_random_directions(self, n: int) -> List[np.ndarray]:
        dim = self.original_data.shape[1]
        return [self.random_direction(dim) for _ in range(n)]

    def get_marginal_directions(self) -> List[np.ndarray]:
        dim = self.original_data.shape[1]
        return [np.identity(dim)[i] for i in range(dim)]

    def _sinkhorn_distance(self, orig: np.ndarray, other: np.ndarray) -> float:
        return wasserstein_1d_squared(orig, other)

    def directional_distance(self, direction: np.ndarray) -> float:
        orig_proj = self._project(self.original_data, direction)
        other_proj = self._project(self.other_data, direction)
        orig_norm, other_norm = self._normalise(orig_proj, other_proj)
        distance = self._sinkhorn_distance(orig_norm, other_norm)
        return float(np.sqrt(max(distance, 0.0)))

    def feature_distance(self, idx: int) -> float:
        orig = self.original_data[:, idx]
        other = self.other_data[:, idx]
        orig_norm, other_norm = self._normalise(orig, other)
        distance = self._sinkhorn_distance(orig_norm, other_norm)
        return float(np.sqrt(max(distance, 0.0)))

    def sliced_distances(self, num_directions: int) -> np.ndarray:
        dirs = self.get_random_directions(num_directions)
        distances = []
        for direction in tqdm(dirs, desc="sliced_wasserstein", unit="dir", leave=False, disable=True):
            distances.append(self.directional_distance(direction))
        return np.array(distances)

    def marginal_distances(self) -> np.ndarray:
        if self.normalisation == "standardise":
            sd = np.std(self.original_data, axis=0) + _EPS
            orig = self.original_data / sd
            other = self.other_data / sd
        else:
            orig = self.original_data
            other = self.other_data

        num_features = orig.shape[1]
        distances = np.empty(num_features, dtype=np.float64)
        for idx in range(num_features):
            distances[idx] = np.sqrt(
                wasserstein_1d_squared(orig[:, idx], other[:, idx])
            )
        return distances


class Metric:
    def __init__(self, original_samples: np.ndarray | torch.Tensor) -> None:
        self.original_samples = _check_flat_array(original_samples)

    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> Dict[str, float]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def baseline_metrics(self) -> Dict[str, float]:
        return {}


class SlicedWasserstein(Metric):
    def __init__(
        self,
        original_samples: np.ndarray | torch.Tensor,
        random_seed: int,
        num_directions: int,
        save_all_distances: bool = False,
        transport_backend: str = "sinkhorn",
        sinkhorn_method: SinkhornMethod = "epsilon_scaling",
        sinkhorn_reg: float = 1e-2,
        sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(original_samples)
        self.random_seed = random_seed
        self.num_directions = num_directions
        self.save_all_distances = save_all_distances
        self.transport_backend = transport_backend
        self.sinkhorn_method = sinkhorn_method
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_kwargs = dict(sinkhorn_kwargs or {})

    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> Dict[str, float]:
        wd = WassersteinDistances(
            self.original_samples,
            _check_flat_array(other_samples),
            seed=self.random_seed,
            transport_backend=self.transport_backend,
            sinkhorn_method=self.sinkhorn_method,
            sinkhorn_reg=self.sinkhorn_reg,
            sinkhorn_kwargs=self.sinkhorn_kwargs,
        )
        distances = wd.sliced_distances(self.num_directions)
        metrics: Dict[str, float] = {
            "sliced_wasserstein_mean": float(np.mean(distances)),
            "sliced_wasserstein_max": float(np.max(distances)),
        }
        if self.save_all_distances:
            metrics["sliced_wasserstein_all"] = distances.tolist()  # type: ignore
        return metrics

    @property
    def baseline_metrics(self) -> Dict[str, float]:
        n = self.original_samples.shape[0]
        wd_self = WassersteinDistances(
            self.original_samples[: n // 2],
            self.original_samples[n // 2 :],
            seed=self.random_seed,
            transport_backend=self.transport_backend,
            sinkhorn_method=self.sinkhorn_method,
            sinkhorn_reg=self.sinkhorn_reg,
            sinkhorn_kwargs=self.sinkhorn_kwargs,
        )
        distances_self = wd_self.sliced_distances(self.num_directions)

        avg_sample = np.mean(self.original_samples, axis=0, keepdims=True)
        wd_dummy = WassersteinDistances(
            self.original_samples,
            avg_sample,
            seed=self.random_seed,
            transport_backend=self.transport_backend,
            sinkhorn_method=self.sinkhorn_method,
            sinkhorn_reg=self.sinkhorn_reg,
            sinkhorn_kwargs=self.sinkhorn_kwargs,
        )
        distances_dummy = wd_dummy.sliced_distances(self.num_directions)
        return {
            "sliced_wasserstein_mean_self": float(np.mean(distances_self)),
            "sliced_wasserstein_max_self": float(np.max(distances_self)),
            "sliced_wasserstein_mean_dummy": float(np.mean(distances_dummy)),
            "sliced_wasserstein_max_dummy": float(np.max(distances_dummy)),
        }

    @property
    def name(self) -> str:
        return "sliced_wasserstein"


class MarginalWasserstein(Metric):
    def __init__(
        self,
        original_samples: np.ndarray | torch.Tensor,
        random_seed: int,
        save_all_distances: bool = False,
        transport_backend: str = "sinkhorn",
        sinkhorn_method: SinkhornMethod = "epsilon_scaling",
        sinkhorn_reg: float = 1e-2,
        sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(original_samples)
        self.random_seed = random_seed
        self.save_all_distances = save_all_distances
        self.transport_backend = transport_backend
        self.sinkhorn_method = sinkhorn_method
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_kwargs = dict(sinkhorn_kwargs or {})

    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> Dict[str, float]:
        wd = WassersteinDistances(
            self.original_samples,
            _check_flat_array(other_samples),
            seed=self.random_seed,
            transport_backend=self.transport_backend,
            sinkhorn_method=self.sinkhorn_method,
            sinkhorn_reg=self.sinkhorn_reg,
            sinkhorn_kwargs=self.sinkhorn_kwargs,
        )
        distances = wd.marginal_distances()
        metrics: Dict[str, float] = {
            "marginal_wasserstein_mean": float(np.mean(distances)),
            "marginal_wasserstein_max": float(np.max(distances)),
        }
        if self.save_all_distances:
            metrics["marginal_wasserstein_all"] = distances.tolist()  # type: ignore
        return metrics

    @property
    def baseline_metrics(self) -> Dict[str, float]:
        n = self.original_samples.shape[0]
        wd_self = WassersteinDistances(
            self.original_samples[: n // 2],
            self.original_samples[n // 2 :],
            seed=self.random_seed,
            transport_backend=self.transport_backend,
            sinkhorn_method=self.sinkhorn_method,
            sinkhorn_reg=self.sinkhorn_reg,
            sinkhorn_kwargs=self.sinkhorn_kwargs,
        )
        distances_self = wd_self.marginal_distances()

        avg_sample = np.mean(self.original_samples, axis=0, keepdims=True)
        wd_dummy = WassersteinDistances(
            self.original_samples,
            avg_sample,
            seed=self.random_seed,
            transport_backend=self.transport_backend,
            sinkhorn_method=self.sinkhorn_method,
            sinkhorn_reg=self.sinkhorn_reg,
            sinkhorn_kwargs=self.sinkhorn_kwargs,
        )
        distances_dummy = wd_dummy.marginal_distances()
        return {
            "marginal_wasserstein_mean_self": float(np.mean(distances_self)),
            "marginal_wasserstein_max_self": float(np.max(distances_self)),
            "marginal_wasserstein_mean_dummy": float(np.mean(distances_dummy)),
            "marginal_wasserstein_max_dummy": float(np.max(distances_dummy)),
        }

    @property
    def name(self) -> str:
        return "marginal_wasserstein"


class CovarianceDifference(Metric):
    def __init__(
        self,
        original_samples: np.ndarray | torch.Tensor,
        ddof: int = 1,
        include_correlation: bool = True,
    ) -> None:
        super().__init__(original_samples)
        self.ddof = ddof
        self.include_correlation = include_correlation
        self._reference_cov = self._covariance(self.original_samples)
        self._reference_var = np.diag(self._reference_cov)
        if self.include_correlation:
            self._reference_corr = self._correlation(self.original_samples)

    def _covariance(self, samples: np.ndarray | torch.Tensor) -> np.ndarray:
        flat = _check_flat_array(samples)
        if flat.shape[0] <= 1:
            raise ValueError("Covariance requires at least two samples.")
        cov = np.cov(flat, rowvar=False, ddof=self.ddof)
        return cov

    def _correlation(self, samples: np.ndarray | torch.Tensor) -> np.ndarray:
        flat = _check_flat_array(samples)
        if flat.shape[0] <= 1:
            raise ValueError("Correlation requires at least two samples.")
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(flat, rowvar=False)
        return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    def __call__(self, other_samples: np.ndarray | torch.Tensor) -> Dict[str, float]:
        cov_other = self._covariance(other_samples)
        var_other = np.diag(cov_other)
        cov_diff = cov_other - self._reference_cov
        metrics: Dict[str, float] = {
            "covariance_frobenius": float(np.linalg.norm(cov_diff, ord="fro")),
            "covariance_mean_abs": float(np.mean(np.abs(cov_diff))),
            "variance_mean_abs": float(np.mean(np.abs(var_other - self._reference_var))),
        }
        if self.include_correlation:
            corr_other = self._correlation(other_samples)
            corr_diff = corr_other - self._reference_corr
            metrics |= {
                "correlation_frobenius": float(np.linalg.norm(corr_diff, ord="fro")),
                "correlation_mean_abs": float(np.mean(np.abs(corr_diff))),
            }
        return metrics

    @property
    def name(self) -> str:
        return "covariance_difference"


class MetricCollection:
    def __init__(
        self,
        metrics: List[Metric | partial],
        original_samples: Optional[torch.Tensor | np.ndarray] = None,
        include_baselines: bool = True,
        include_spectral_density: bool = True,
    ) -> None:
        if original_samples is None:
            raise ValueError("original_samples must be provided")
        logger.info(
            "MetricCollection init | samples shape=%s | metrics=%d | include_baselines=%s | include_spectral=%s",
            tuple(original_samples.shape),
            len(metrics),
            include_baselines,
            include_spectral_density,
        )
        self.include_baselines = include_baselines
        self.metrics_time: List[Metric] = []
        self.metrics_freq: List[Metric] = []

        original_samples_freq = _dft(original_samples if isinstance(original_samples, torch.Tensor) else torch.from_numpy(original_samples))

        for metric in metrics:
            if isinstance(metric, partial):
                self.metrics_time.append(metric(original_samples=original_samples))  # type: ignore[arg-type]
                self.metrics_freq.append(metric(original_samples=original_samples_freq))  # type: ignore[arg-type]
            else:
                raise TypeError("MetricCollection expects functools.partial metrics.")

        self.metric_spectral = None
        if include_spectral_density:
            self.metric_spectral = MarginalWasserstein(
                original_samples=_spectral_density(
                    original_samples if isinstance(original_samples, torch.Tensor) else torch.from_numpy(original_samples)
                ),
                random_seed=42,
                save_all_distances=True,
            )

    def __call__(self, other_samples: torch.Tensor | np.ndarray) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        other_freq = _dft(other_samples if isinstance(other_samples, torch.Tensor) else torch.from_numpy(other_samples))
        for metric_time, metric_freq in zip(self.metrics_time, self.metrics_freq):
            metrics.update({f"time_{k}": v for k, v in metric_time(other_samples).items()})
            metrics.update({f"freq_{k}": v for k, v in metric_freq(other_freq).items()})

        if self.include_baselines:
            metrics.update(self.baseline_metrics)

        if self.metric_spectral is not None:
            spectral_vals = self.metric_spectral(_spectral_density(other_samples if isinstance(other_samples, torch.Tensor) else torch.from_numpy(other_samples)))
            metrics.update({f"spectral_{k}": v for k, v in spectral_vals.items()})

        return dict(sorted(metrics.items(), key=lambda item: item[0]))

    @property
    def baseline_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for metric_time, metric_freq in zip(self.metrics_time, self.metrics_freq):
            out.update({f"time_{k}": v for k, v in metric_time.baseline_metrics.items()})
            out.update({f"freq_{k}": v for k, v in metric_freq.baseline_metrics.items()})
        return out


def compute_metric_collection(truth: torch.Tensor, preds: torch.Tensor) -> Dict[str, Any]:
    import os

    num_directions = int(os.environ.get("CFDIFF_SW_DIRECTIONS", 128))
    sample_limit = int(os.environ.get("CFDIFF_METRICS_SAMPLE_LIMIT", 0))
    if sample_limit > 0:
        limit = min(sample_limit, truth.size(0), preds.size(0))
        if limit < 2:
            raise ValueError(
                f"CFDIFF_METRICS_SAMPLE_LIMIT={sample_limit} results in fewer than two windows (limit={limit})."
            )
        if limit < truth.size(0):
            logger.info(
                "Limiting Fourier metrics to the first %d windows (requested=%d).",
                limit,
                sample_limit,
            )
            truth = truth[:limit]
            preds = preds[:limit]
    transport_backend = os.environ.get("CFDIFF_WASSERSTEIN_BACKEND", "sinkhorn").lower()
    sinkhorn_method_env = os.environ.get("CFDIFF_SINKHORN_METHOD", "epsilon_scaling").lower()
    if sinkhorn_method_env not in {"standard", "stabilized", "epsilon_scaling"}:
        logger.warning("Unknown CFDIFF_SINKHORN_METHOD=%s; defaulting to epsilon_scaling", sinkhorn_method_env)
        sinkhorn_method_env = "epsilon_scaling"
    sinkhorn_method: SinkhornMethod = sinkhorn_method_env  # type: ignore[assignment]
    sinkhorn_reg = float(os.environ.get("CFDIFF_SINKHORN_REG", 1e-2))
    sinkhorn_kwargs: Dict[str, Any] = {}
    if "CFDIFF_SINKHORN_NUMITER" in os.environ:
        sinkhorn_kwargs["numItermax"] = int(os.environ["CFDIFF_SINKHORN_NUMITER"])
    if "CFDIFF_SINKHORN_TAU" in os.environ:
        sinkhorn_kwargs["tau"] = float(os.environ["CFDIFF_SINKHORN_TAU"])
    if "CFDIFF_SINKHORN_STOP_THR" in os.environ:
        sinkhorn_kwargs["stopThr"] = float(os.environ["CFDIFF_SINKHORN_STOP_THR"])
    logger.info(
        "compute_metric_collection | truth shape=%s | preds shape=%s | directions=%d | backend=%s | method=%s | reg=%s",
        tuple(truth.shape),
        tuple(preds.shape),
        num_directions,
        transport_backend,
        sinkhorn_method,
        sinkhorn_reg,
    )
    metric_collection = MetricCollection(
        metrics=[
            partial(
                SlicedWasserstein,
                random_seed=42,
                num_directions=num_directions,
                save_all_distances=False,
                transport_backend=transport_backend,
                sinkhorn_method=sinkhorn_method,
                sinkhorn_reg=sinkhorn_reg,
                sinkhorn_kwargs=sinkhorn_kwargs,
            ),
            partial(
                MarginalWasserstein,
                random_seed=42,
                save_all_distances=False,
                transport_backend=transport_backend,
                sinkhorn_method=sinkhorn_method,
                sinkhorn_reg=sinkhorn_reg,
                sinkhorn_kwargs=sinkhorn_kwargs,
            ),
            partial(CovarianceDifference, include_correlation=True),
        ],
        original_samples=truth,
        include_baselines=True,
        include_spectral_density=True,
    )
    metrics = metric_collection(preds)
    logger.info("Fourier metric collection produced %d entries", len(metrics))
    return metrics
