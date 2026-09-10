"""The compute backends: exact on numpy, in agreement on a device, same either way."""

from __future__ import annotations

import numpy as np
import pytest

from drumsynth.backend import (
    STATS,
    Renderer,
    available_devices,
    benchmark_devices,
    describe_device,
    renderer,
)
from drumsynth.parallel import map_workers, resolve_jobs
from drumsynth.spectral import Candidate, encode

DEVICES = available_devices()
TORCH = [device for device in DEVICES if device != "numpy"]


@pytest.fixture(scope="module")
def fitted(tonal_hit, sr):
    model = encode(tonal_hit, sr, Candidate(512, 256, 32, "lowrank", 8))
    return model, tonal_hit


def make(fitted, device: str, jobs: int = 1) -> Renderer:
    model, reference = fitted
    return renderer(
        model.candidate.spec,
        model.bins,
        model.n_samples,
        reference[None, :],
        device=device,
        jobs=jobs,
    )


def batch(model, count: int = 6) -> np.ndarray:
    block = model.components()
    return np.stack([block * (1.0 + 0.02 * index) for index in range(count)])[:, None]


def test_the_numpy_renderer_is_the_model_itself(fitted):
    model, _ = fitted

    rendered = make(fitted, "numpy").render(model.components())

    assert np.array_equal(rendered, model.render())


def test_threads_do_not_change_a_measurement(fitted):
    model, _ = fitted
    blocks = batch(model)

    serial = make(fitted, "numpy", jobs=1).measure(blocks)
    threaded = make(fitted, "numpy", jobs=4).measure(blocks)

    assert np.array_equal(serial, threaded)


def test_a_measurement_says_what_the_metrics_say(fitted):
    from drumsynth.fitting.metrics import Quality

    model, reference = fitted
    stats = make(fitted, "numpy").measure(batch(model, 1))[0, 0]
    quality = Quality.measure(reference, model.render())

    assert stats[0] == pytest.approx(quality.relative_mse, rel=1e-12)
    assert stats[1] == pytest.approx(quality.correlation, rel=1e-12)
    assert stats[2] == pytest.approx(quality.estimate_rms, rel=1e-12)
    assert stats[3] == pytest.approx(quality.estimate_peak, rel=1e-12)


def test_the_batch_shape_survives(fitted):
    model, _ = fitted
    blocks = np.stack([batch(model, 3)[:, 0]] * 2).transpose(1, 0, 2, 3)  # (3, 2, k, T)

    stats = renderer(
        model.candidate.spec,
        model.bins,
        model.n_samples,
        np.stack([model.render(), model.render() * 0.5]),
    ).measure(blocks)

    assert stats.shape == (3, 2, len(STATS))


def test_a_silent_reference_is_not_a_division_by_zero(fitted):
    model, _ = fitted

    stats = renderer(
        model.candidate.spec, model.bins, model.n_samples, np.zeros((1, model.n_samples))
    ).measure(batch(model, 1))

    assert np.isfinite(stats).all()


@pytest.mark.skipif(not TORCH, reason="torch is not installed")
@pytest.mark.parametrize("device", TORCH)
def test_a_device_agrees_with_numpy(device, fitted):
    """float32 on a device, float64 on the host: the same answer to 1e-5."""
    model, _ = fitted
    blocks = batch(model, 6)

    exact = make(fitted, "numpy").measure(blocks)
    measured = make(fitted, device).measure(blocks)

    assert np.allclose(measured[..., 0], exact[..., 0], rtol=1e-4)
    assert np.allclose(measured[..., 1], exact[..., 1], rtol=1e-6)
    assert np.allclose(measured[..., 2], exact[..., 2], rtol=1e-5)


@pytest.mark.skipif(not TORCH, reason="torch is not installed")
def test_a_device_batches_in_pieces_that_fit(fitted):
    model, _ = fitted
    small = renderer(
        model.candidate.spec,
        model.bins,
        model.n_samples,
        model.render()[None, :],
        device=TORCH[0],
        budget=1024,
    )

    assert small.batch >= 1
    assert small.measure(batch(model, 3)).shape[0] == 3


def test_an_unavailable_device_says_what_there_is(fitted):
    with pytest.raises(ValueError, match="not available"):
        make(fitted, "quantum")


def test_auto_falls_back_to_numpy_without_a_gpu(fitted):
    engine = make(fitted, "auto")

    if "cuda" not in DEVICES:
        assert type(engine) is Renderer


def test_every_device_describes_itself():
    for device in DEVICES:
        assert describe_device(device)


def test_the_benchmark_times_what_is_there():
    timings = benchmark_devices(["numpy"], repeats=1)

    assert set(timings) == {"numpy"}
    assert timings["numpy"] > 0


# -- the thread helper --------------------------------------------------------


def test_workers_resolve_to_something_sane():
    assert resolve_jobs(0) >= 1
    assert resolve_jobs(-4) >= 1
    assert resolve_jobs(3) == 3


def test_mapping_keeps_the_order_it_was_given():
    items = list(range(17))

    assert map_workers(lambda n: n * n, items, jobs=4) == [n * n for n in items]
    assert map_workers(lambda n: n, [], jobs=4) == []
