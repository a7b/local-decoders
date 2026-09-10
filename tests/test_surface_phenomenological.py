"""Bit-exact tests for the JAX surface phenomenological-memory decoder."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phenomenological import surface_ca as jax_rule


SOURCE_REPO = Path(os.environ.get("LOCAL_MEMORY_REPO", "/home/a7b/local-memory"))
if not (SOURCE_REPO / "numerics/surface/reference.py").is_file():
    pytest.skip("set LOCAL_MEMORY_REPO to the research checkout", allow_module_level=True)
sys.path.insert(0, str(SOURCE_REPO))

from numerics.surface import reference as numpy_rule  # noqa: E402
from numerics.surface import rule as cuda_rule  # noqa: E402


FIELD_MAP = (
    ("s", "s"),
    ("tau", "tau"),
    ("b", "m00"),
    ("r", "m01"),
    ("g", "m10"),
    ("theta_b", "theta00"),
    ("theta_r", "theta01"),
    ("theta_g", "theta10"),
    ("right_cut", "right_cut"),
)


def _jax_state(bits, timers, right_cut):
    return jax_rule.MemoryState(
        *[jnp.asarray(value) for value in (
            bits[0], timers[0], bits[1], bits[2], bits[3],
            timers[1], timers[2], timers[3], right_cut,
        )]
    )


def _assert_state_matches_arrays(actual, bits, timers, right_cut, label=""):
    expected = {
        "s": bits[0], "tau": timers[0],
        "b": bits[1], "r": bits[2], "g": bits[3],
        "theta_b": timers[1], "theta_r": timers[2], "theta_g": timers[3],
        "right_cut": right_cut,
    }
    for name, want in expected.items():
        np.testing.assert_array_equal(np.asarray(getattr(actual, name)), want,
                                      err_msg=f"{label}: {name}")


def _assert_result_states(actual, expected, label=""):
    np.testing.assert_array_equal(np.asarray(actual.terminated), expected.terminated,
                                  err_msg=f"{label}: terminated")
    np.testing.assert_array_equal(np.asarray(actual.pred_flip), expected.pred_flip,
                                  err_msg=f"{label}: pred_flip")
    assert actual.cleanout == expected.cleanout
    assert actual.final_time == expected.final_time
    for jax_name, source_name in FIELD_MAP:
        np.testing.assert_array_equal(
            np.asarray(getattr(actual.state, jax_name)),
            getattr(expected.state, source_name),
            err_msg=f"{label}: {jax_name}",
        )


def _valid_random_state(rng, L, K):
    bits = rng.random((4, K, L - 1, L)) < 0.38
    timers = np.zeros(bits.shape, dtype=np.int32)
    for k in range(K - 1):
        threshold = 2 * 2**k
        for channel in range(4):
            values = rng.integers(0, threshold, (L - 1, L), dtype=np.int32)
            timers[channel, k] = np.where(bits[channel, k], values, 0)
    right_cut = rng.random((K, L)) < 0.5
    return bits, timers, right_cut


@pytest.mark.parametrize("L,K", [(3, 5), (4, 5), (3, 6), (5, 6)])
@pytest.mark.parametrize("ordinary", [False, True])
def test_arbitrary_update_is_bit_exact_in_every_state_field(L, K, ordinary):
    rng = np.random.default_rng(10_000 + 100 * L + 10 * K + ordinary)
    for trial in range(20):
        bits, timers, right_cut = _valid_random_state(rng, L, K)
        incoming = rng.random((L - 1, L)) < 0.35
        expected_bits, expected_timers = numpy_rule._update(
            bits.copy(), timers.copy(), incoming, ordinary)
        actual = jax_rule.step_ca(
            _jax_state(bits, timers, right_cut), incoming, ordinary=ordinary)
        _assert_state_matches_arrays(
            actual, expected_bits, expected_timers, right_cut,
            label=f"L={L},K={K},ordinary={ordinary},trial={trial}")


@pytest.mark.parametrize("L,K", [(3, 5), (4, 5), (5, 6), (6, 6)])
def test_arbitrary_split_is_bit_exact_in_every_state_field(L, K):
    rng = np.random.default_rng(20_000 + 100 * L + K)
    for trial in range(20):
        bits, timers, right_cut = _valid_random_state(rng, L, K)
        expected_cut = right_cut.copy()
        expected_bits, expected_timers = numpy_rule._split(
            bits.copy(), timers.copy(), expected_cut)
        actual = jax_rule.splitting_step(_jax_state(bits, timers, right_cut))
        _assert_state_matches_arrays(
            actual, expected_bits, expected_timers, expected_cut,
            label=f"L={L},K={K},trial={trial}")


@pytest.mark.parametrize("L,K", [(3, 5), (4, 5), (3, 6), (5, 6)])
def test_every_substep_of_random_trajectories_matches_numpy(L, K):
    """Compare every state field after every split, ordinary step and substep."""

    rng = np.random.default_rng(30_000 + 100 * L + K)
    for trial in range(3):
        phi = rng.integers(0, 2, (5, L - 1, L), dtype=np.uint8)
        bits, timers = numpy_rule._empty(K, L - 1, L)
        right_cut = np.zeros((K, L), dtype=bool)
        actual = jax_rule.init_state(L, K)

        bits, timers = numpy_rule._update(bits, timers, phi[0], True)
        actual = jax_rule.step_ca(actual, phi[0], ordinary=True)
        _assert_state_matches_arrays(actual, bits, timers, right_cut,
                                     label=f"trial={trial},initial")
        for substep in range(jax_rule.EXTRA_SUBSTEPS):
            bits, timers = numpy_rule._update(bits, timers, None, False)
            actual = jax_rule.step_ca(actual, ordinary=False)
            _assert_state_matches_arrays(actual, bits, timers, right_cut,
                                         label=f"trial={trial},initial-substep={substep}")

        supplied, cleanout = len(phi) - 1, 12
        for time in range(supplied + cleanout):
            if time % jax_rule.SPLITTING_PERIOD == 0:
                bits, timers = numpy_rule._split(bits, timers, right_cut)
                actual = jax_rule.splitting_step(actual)
                _assert_state_matches_arrays(actual, bits, timers, right_cut,
                                             label=f"trial={trial},time={time},split")
            incoming = phi[time + 1] if time < supplied else None
            bits, timers = numpy_rule._update(bits, timers, incoming, True)
            actual = jax_rule.step_ca(actual, incoming, ordinary=True)
            _assert_state_matches_arrays(actual, bits, timers, right_cut,
                                         label=f"trial={trial},time={time},ordinary")
            for substep in range(jax_rule.EXTRA_SUBSTEPS):
                bits, timers = numpy_rule._update(bits, timers, None, False)
                actual = jax_rule.step_ca(actual, ordinary=False)
                _assert_state_matches_arrays(
                    actual, bits, timers, right_cut,
                    label=f"trial={trial},time={time},substep={substep}")


@pytest.mark.parametrize(
    "L,K,rounds,cleanout,early_exit,shots",
    [
        (3, 5, 1, 0, False, 4),
        (3, 5, 4, 11, False, 3),
        (4, 5, 5, 13, True, 3),
        (3, 6, 4, 9, False, 3),
        (5, 6, 3, 12, True, 2),
    ],
)
def test_random_full_trajectories_match_numpy_and_cuda(
        L, K, rounds, cleanout, early_exit, shots):
    rng = np.random.default_rng(40_000 + 100 * L + rounds)
    phi = rng.integers(0, 2, (shots, rounds, L - 1, L), dtype=np.uint8)
    expected = numpy_rule.run_states(
        phi, K=K, cleanout=cleanout, early_exit=early_exit, return_state=True)
    actual = jax_rule.run_states(
        phi, K=K, cleanout=cleanout, early_exit=early_exit, return_state=True)
    _assert_result_states(actual, expected, label="jax-vs-numpy")

    cuda = cuda_rule.run_states(
        phi, K=K, cleanout=cleanout, early_exit=early_exit,
        return_state=True, threads_per_block=4)
    _assert_result_states(actual, cuda, label="jax-vs-cuda")


@pytest.mark.parametrize("L,p,shots,seed", [(3, 0.0, 8, 51), (4, 0.17, 12, 52), (5, 0.5, 9, 53)])
def test_complete_numpy_sampled_memory_experiment_is_bit_exact(L, p, shots, seed):
    phi, observed = numpy_rule.sample_phenom(L, L, p, shots, seed)
    expected = numpy_rule.run_states(phi, K=5, cleanout=15, early_exit=True, return_state=True)
    actual = jax_rule.run_states(phi, K=5, cleanout=15, early_exit=True, return_state=True)
    _assert_result_states(actual, expected, label="sampled-memory")
    np.testing.assert_array_equal(
        np.asarray((~actual.terminated) | (actual.pred_flip ^ observed)),
        (~expected.terminated) | (expected.pred_flip ^ observed),
    )


@pytest.mark.parametrize("L,shots", [(3, 7), (4, 5), (7, 3)])
def test_jax_noise_sampler_has_full_memory_shapes_and_noiseless_limits(L, shots):
    phi, observed = jax_rule.sample_phenom(
        jax.random.PRNGKey(60 + L), L, L, 0.0, shots=shots)
    assert phi.shape == (shots, L + 1, L - 1, L)
    assert observed.shape == (shots,)
    assert not bool(jnp.any(phi))
    assert not bool(jnp.any(observed))


@pytest.mark.parametrize("L,shots,p", [(3, 5, 0.0), (3, 5, 1.0), (4, 3, 1.0)])
def test_jax_noise_sampler_matches_oracle_for_deterministic_probabilities(L, shots, p):
    actual_phi, actual_observed = jax_rule.sample_phenom(
        jax.random.PRNGKey(65 + L), L, L, p, shots=shots)
    expected_phi, expected_observed = numpy_rule.sample_phenom(L, L, p, shots, 900 + L)
    np.testing.assert_array_equal(np.asarray(actual_phi), expected_phi)
    np.testing.assert_array_equal(np.asarray(actual_observed), expected_observed)


def test_calculate_syndrome_matches_numpy_oracle():
    rng = np.random.default_rng(70)
    for L in (3, 4, 7):
        ex_error = rng.integers(0, 2, (11, L, L), dtype=np.uint8).astype(bool)
        ey_error = rng.integers(0, 2, (11, L - 1, L - 1), dtype=np.uint8).astype(bool)
        np.testing.assert_array_equal(
            np.asarray(jax_rule.calculate_syndrome(ex_error, ey_error)),
            numpy_rule._z_syndrome(ex_error, ey_error),
        )


def test_run_monte_carlo_executes_full_memory_experiment(tmp_path):
    output = tmp_path / "surface-phenom-smoke.npz"
    jax_rule.run_monte_carlo(
        [3], [0.0], 5, filename=output, seed=81, batch_size=3, K_by_L={3: 5})
    data = np.load(output)
    np.testing.assert_array_equal(data["L"], [3])
    np.testing.assert_array_equal(data["p"], [0.0])
    np.testing.assert_array_equal(data["ler"], [0.0])
    np.testing.assert_array_equal(data["num_samples"], [5])
    np.testing.assert_array_equal(data["K"], [5])
    assert float(data["avg_steps"][0]) == 4.0  # L+1 detector words, then early exit.


def test_public_entry_points_and_registered_settings():
    assert callable(jax_rule.init_state)
    assert callable(jax_rule.step_ca)
    assert callable(jax_rule.splitting_step)
    assert callable(jax_rule.run_single_trajectory)
    assert callable(jax_rule.run_monte_carlo)
    assert callable(jax_rule.run_states)
    assert jax_rule.EXTRA_SUBSTEPS == 2
    assert jax_rule.SPLITTING_PERIOD == 10
    assert [jax_rule.K_for_L(L) for L in (7, 13, 19, 25, 31)] == [5, 5, 5, 5, 6]
    assert [jax_rule.cleanout_steps(L, jax_rule.K_for_L(L))
            for L in (7, 13, 19, 25, 31)] == [240, 420, 600, 780, 992]
