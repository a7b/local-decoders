"""
Surface code memory experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


MAX_BATCH_SIZE = 256
EXTRA_SUBSTEPS = 2
SPLITTING_PERIOD = 10
INFINITY = np.iinfo(np.int32).max


class MemoryState(NamedTuple):
    """One decoder trajectory; site fields have shape ``[K,L-1,L]``."""

    s: jax.Array
    tau: jax.Array
    b: jax.Array
    r: jax.Array
    g: jax.Array
    theta_b: jax.Array
    theta_r: jax.Array
    theta_g: jax.Array
    right_cut: jax.Array


@dataclass(frozen=True)
class MemoryResult:
    """Batched decoder result, matching the NumPy/CUDA result contract."""

    terminated: jax.Array
    pred_flip: jax.Array
    state: MemoryState | None
    cleanout: int
    final_time: int


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def K_for_L(L: int) -> int:
    """Use the registered depth convention, including the L=31 back wall."""

    length = _integer(L, "L", 3)
    if length > 31:
        raise ValueError("the registered surface winner supports L<=31")
    return 5 if length <= 25 else 6


def cleanout_steps(L: int, K: int) -> int:
    """Return the registered zero-input horizon."""

    length = _integer(L, "L", 3)
    depth = _integer(K, "K", 5)
    if depth not in (5, 6):
        raise ValueError("the winner uses only K=5 or K=6")
    return sum(2 * 2**k for k in range(depth - 1)) + 30 * length


def validate_stream(phi, K, cleanout):
    values = np.asarray(phi)
    if values.ndim != 4 or min(values.shape[:2]) < 1:
        raise ValueError("phi must have shape [shot,round,L-1,L] with positive shot/round counts")
    length = values.shape[3]
    if length < 3 or values.shape[2] != length - 1:
        raise ValueError("phi must have spatial shape [L-1,L] for L>=3")
    if np.any((values != 0) & (values != 1)):
        raise ValueError("phi must contain only zero and one")
    depth = _integer(K, "K", 5)
    default = cleanout_steps(length, depth)
    horizon = default if cleanout is None else _integer(cleanout, "cleanout", 0)
    if horizon + values.shape[1] - 1 > np.iinfo(np.int32).max:
        raise ValueError("final time must fit in int32")
    return np.ascontiguousarray(values, dtype=np.uint8), depth, horizon


def init_state(L: int, K: int) -> MemoryState:
    """Initialize the empty online decoder before the first detector word."""

    length = _integer(L, "L", 3)
    depth = _integer(K, "K", 5)
    if depth not in (5, 6):
        raise ValueError("the winner uses only K=5 or K=6")
    shape = (depth, length - 1, length)
    bits = jnp.zeros(shape, dtype=jnp.bool_)
    timers = jnp.zeros(shape, dtype=jnp.int32)
    right_cut = jnp.zeros((depth, length), dtype=jnp.bool_)
    return MemoryState(bits, timers, bits, bits, bits, timers, timers, timers, right_cut)


def _from_left(values):
    return jnp.concatenate((jnp.zeros_like(values[:, :1]), values[:, :-1]), axis=1)


def _from_right(values):
    return jnp.concatenate((values[:, 1:], jnp.zeros_like(values[:, :1])), axis=1)


def _from_down(values):
    return jnp.concatenate((jnp.zeros_like(values[:, :, :1]), values[:, :, :-1]), axis=2)


def _from_up(values):
    return jnp.concatenate((values[:, :, 1:], jnp.zeros_like(values[:, :, :1])), axis=2)


def _from_lower_slice(values):
    return jnp.concatenate((jnp.zeros_like(values[:1]), values[:-1]), axis=0)


def _minimum_source_age(first_bits, first_ages, second_bits, second_ages, cutoff):
    first = jnp.where(first_bits & (first_ages < cutoff), first_ages, INFINITY)
    second = jnp.where(second_bits & (second_ages < cutoff), second_ages, INFINITY)
    return jnp.minimum(first, second)


def _update_state(state: MemoryState, incoming: jax.Array, ordinary: bool) -> MemoryState:
    """One ordinary update or one zero-time movement/persistence substep."""

    s, tau, b, r, g, theta_b, theta_r, theta_g, right_cut = state
    depth = s.shape[0]
    timed = (jnp.arange(depth) < depth - 1)[:, None, None]
    threshold = (2 * (2 ** jnp.arange(depth, dtype=jnp.int32)))[:, None, None]
    ordinary_bit = jnp.asarray(ordinary, dtype=jnp.bool_)
    increment = ordinary_bit.astype(jnp.int32)

    b_left, r_left = _from_left(b), _from_left(r)
    b_down, g_down = _from_down(b), _from_down(g)
    left = b_left | r_left
    down = b_down | g_down

    left_age = jnp.minimum(
        jnp.where(b_left, _from_left(theta_b), INFINITY),
        jnp.where(r_left, _from_left(theta_r), INFINITY),
    )
    down_age = jnp.minimum(
        jnp.where(b_down, _from_down(theta_b), INFINITY),
        jnp.where(g_down, _from_down(theta_g), INFINITY),
    )

    promote = s & timed & ordinary_bit & (tau + 1 == threshold)
    prefer_left = left & ((~down) | (~timed) | (left_age <= down_age))
    move_left = s & (~promote) & prefer_left
    move_down = s & (~promote) & (~prefer_left) & down
    stay = s & (~promote) & (~move_left) & (~move_down)

    source_age = jnp.where(timed, tau + increment, 0)
    left_in = _from_right(move_left)
    down_in = _from_up(move_down)
    promote_in = _from_lower_slice(promote)
    environment = jnp.zeros_like(s).at[0].set(incoming.astype(jnp.bool_))

    next_s = stay ^ left_in ^ down_in ^ promote_in ^ environment
    arrival_age = jnp.where(stay, source_age, INFINITY)
    arrival_age = jnp.minimum(
        arrival_age, jnp.where(left_in, _from_right(source_age), INFINITY))
    arrival_age = jnp.minimum(
        arrival_age, jnp.where(down_in, _from_up(source_age), INFINITY))
    arrival_age = jnp.minimum(arrival_age, jnp.where(promote_in, 0, INFINITY))
    arrival_age = jnp.minimum(arrival_age, jnp.where(environment, 0, INFINITY))
    next_tau = jnp.where(next_s & timed, arrival_age, 0).astype(jnp.int32)

    r_up = _from_up(r)
    g_right = _from_right(g)
    theta_b_left, theta_b_down = _from_left(theta_b), _from_down(theta_b)
    theta_r_left, theta_r_up = _from_left(theta_r), _from_up(theta_r)
    theta_g_right, theta_g_down = _from_right(theta_g), _from_down(theta_g)

    vote_b = b.astype(jnp.int32) + b_left.astype(jnp.int32) + b_down.astype(jnp.int32) >= 2
    vote_r = r.astype(jnp.int32) + r_left.astype(jnp.int32) + r_up.astype(jnp.int32) >= 2
    vote_g = g.astype(jnp.int32) + g_right.astype(jnp.int32) + g_down.astype(jnp.int32) >= 2
    coupling = b & vote_b

    def update_message(message, theta, first_bits, first_theta,
                       second_bits, second_theta, vote):
        persistence = message & (vote | coupling)
        best = jnp.where(next_s, next_tau, INFINITY)
        persist_age = jnp.where(
            persistence & (theta < threshold - 1), theta + increment, INFINITY)
        best = jnp.minimum(best, persist_age)

        old_defect_age = jnp.where(
            ordinary_bit & s & (tau + 1 < threshold), tau + 1, INFINITY)
        best = jnp.minimum(best, old_defect_age)

        source = _minimum_source_age(
            first_bits, first_theta, second_bits, second_theta, threshold - 1)
        growth_age = jnp.where(ordinary_bit & (source != INFINITY), source + 1, INFINITY)
        best = jnp.minimum(best, growth_age)
        timed_message = best != INFINITY

        back_message = (next_s | persistence | (ordinary_bit & s)
                        | (ordinary_bit & (first_bits | second_bits)))
        next_message = jnp.where(timed, timed_message, back_message)
        next_theta = jnp.where(timed & next_message, best, 0).astype(jnp.int32)
        return next_message, next_theta

    next_b, next_theta_b = update_message(
        b, theta_b, b_left, theta_b_left, b_down, theta_b_down, vote_b)
    next_r, next_theta_r = update_message(
        r, theta_r, r_left, theta_r_left, r_up, theta_r_up, vote_r)
    next_g, next_theta_g = update_message(
        g, theta_g, g_right, theta_g_right, g_down, theta_g_down, vote_g)

    return MemoryState(next_s, next_tau, next_b, next_r, next_g,
                       next_theta_b, next_theta_r, next_theta_g, right_cut)


@partial(jax.jit, static_argnames=("ordinary",))
def _step_with_input(state: MemoryState, incoming: jax.Array, ordinary: bool) -> MemoryState:
    return _update_state(state, incoming, ordinary)


def step_ca(state: MemoryState, incoming=None, ordinary: bool = True) -> MemoryState:
    """Public CA step using the same ordinary/substep distinction as the oracle."""

    if incoming is None:
        incoming = jnp.zeros(state.s.shape[1:], dtype=jnp.bool_)
    return _step_with_input(state, jnp.asarray(incoming, dtype=jnp.bool_), ordinary)


def _split_state(state: MemoryState) -> MemoryState:
    s, tau, b, r, g, theta_b, theta_r, theta_g, right_cut = state
    depth, lx, _ = s.shape
    half = lx // 2
    positions = jnp.arange(lx)
    source_x = jnp.where(
        positions <= half - 2,
        positions + 1,
        jnp.where(positions >= half + 1, positions - 1, 0),
    )
    valid = ((positions <= half - 2) | (positions >= half + 1))[None, :, None]
    central = ((positions == half - 1) | (positions == half))[None, :, None]
    timed = (jnp.arange(depth) < depth - 1)[:, None, None]

    def gather(values):
        return jnp.take(values, source_x, axis=1)

    next_s = gather(s) & valid
    next_tau = jnp.where(next_s & timed, gather(tau), 0).astype(jnp.int32)

    def split_message(message, theta):
        translated = gather(message) & valid
        next_message = jnp.where(central, message, translated)
        translated_theta = gather(theta)
        next_theta = jnp.where(central, theta, translated_theta)
        next_theta = jnp.where(next_message & timed, next_theta, 0).astype(jnp.int32)
        return next_message, next_theta

    next_b, next_theta_b = split_message(b, theta_b)
    next_r, next_theta_r = split_message(r, theta_r)
    next_g, next_theta_g = split_message(g, theta_g)
    next_cut = right_cut ^ s[:, -1, :]
    return MemoryState(next_s, next_tau, next_b, next_r, next_g,
                       next_theta_b, next_theta_r, next_theta_g, next_cut)


@jax.jit
def splitting_step(state: MemoryState) -> MemoryState:
    """Apply the open-boundary outward split before an ordinary update."""

    return _split_state(state)


def _decode_lane(phi, K: int, cleanout: int, early_exit: bool):
    length = phi.shape[2]
    state = init_state(length, K)
    zero = jnp.zeros(phi.shape[1:], dtype=jnp.bool_)
    state = _update_state(state, phi[0], True)
    for _ in range(EXTRA_SUBSTEPS):
        state = _update_state(state, zero, False)

    supplied = phi.shape[0] - 1

    def body(time, carry):
        current, active, steps_used = carry
        empty = ~(jnp.any(current.s) | jnp.any(current.b)
                  | jnp.any(current.r) | jnp.any(current.g))
        stop = jnp.asarray(early_exit) & (time >= supplied) & empty
        run_step = active & (~stop)

        def advance(selected):
            selected = jax.lax.cond(
                time % SPLITTING_PERIOD == 0,
                _split_state,
                lambda value: value,
                selected,
            )
            index = jnp.minimum(time + 1, supplied)
            incoming = jnp.where(time < supplied, phi[index], zero)
            selected = _update_state(selected, incoming, True)
            for _ in range(EXTRA_SUBSTEPS):
                selected = _update_state(selected, zero, False)
            return selected

        current = jax.lax.cond(run_step, advance, lambda value: value, current)
        return current, run_step, steps_used + run_step.astype(jnp.int32)

    state, _, steps_used = jax.lax.fori_loop(
        0, supplied + cleanout, body, (state, jnp.asarray(True), jnp.int32(1)))
    terminated = ~jnp.any(state.s)
    prediction = jnp.bitwise_xor.reduce(state.right_cut.reshape(-1))
    return terminated, prediction, state, steps_used


@partial(jax.jit, static_argnames=("K", "cleanout", "early_exit"))
def _decode_batch(phi, K: int, cleanout: int, early_exit: bool):
    return jax.vmap(lambda lane: _decode_lane(lane, K, cleanout, early_exit))(phi)


def run_states(phi, *, K: int, cleanout: int | None = None,
               early_exit: bool = True, return_state: bool = False,
               threads_per_block: int = 256) -> MemoryResult:
    """Decode supplied detector words with the JAX winner implementation."""

    values, depth, horizon = validate_stream(phi, K, cleanout)
    _integer(threads_per_block, "threads_per_block", 1)  # API parity; unused by JAX.
    if values.shape[3] > 31:
        raise ValueError("the registered surface winner supports L<=31")
    terminated, prediction, state, _ = _decode_batch(
        jnp.asarray(values, dtype=jnp.bool_), K=depth, cleanout=horizon,
        early_exit=bool(early_exit))
    return MemoryResult(terminated, prediction, state if return_state else None,
                        horizon, values.shape[1] - 1 + horizon)


def calculate_syndrome(ex_error, ey_error):
    """Calculate the Z-check syndrome on an unrotated distance-L patch."""

    ex_error = jnp.asarray(ex_error, dtype=jnp.bool_)
    ey_error = jnp.asarray(ey_error, dtype=jnp.bool_)
    syndrome = ex_error[..., :-1, :] ^ ex_error[..., 1:, :]
    syndrome = syndrome.at[..., :, :-1].set(syndrome[..., :, :-1] ^ ey_error)
    syndrome = syndrome.at[..., :, 1:].set(syndrome[..., :, 1:] ^ ey_error)
    return syndrome


@partial(jax.jit, static_argnames=("L", "rounds", "shots"))
def sample_phenom(key, L: int, rounds: int, p, shots: int = 1):
    """Sample a full JAX phenomenological-noise memory history.

    The distribution and draw shapes match the NumPy oracle.  JAX's PRNG is
    intentionally used here, so an equal integer seed does not reproduce the
    oracle's PCG64 bit stream.
    """

    ex_error = jnp.zeros((shots, L, L), dtype=jnp.bool_)
    ey_error = jnp.zeros((shots, L - 1, L - 1), dtype=jnp.bool_)
    previous = jnp.zeros((shots, L - 1, L), dtype=jnp.bool_)

    def sample_round(carry, time):
        random_key, ex_current, ey_current, prior = carry
        random_key, ex_key, ey_key, measurement_key = jax.random.split(random_key, 4)
        ex_current = ex_current ^ jax.random.bernoulli(ex_key, p, ex_current.shape)
        ey_current = ey_current ^ jax.random.bernoulli(ey_key, p, ey_current.shape)
        measured = calculate_syndrome(ex_current, ey_current)
        measurement_flip = jax.random.bernoulli(measurement_key, p, measured.shape)
        measured = measured ^ (measurement_flip & (time < rounds))
        detector = measured ^ prior
        return (random_key, ex_current, ey_current, measured), detector

    (_, ex_error, _, _), detector_time_major = jax.lax.scan(
        sample_round, (key, ex_error, ey_error, previous), jnp.arange(rounds + 1))
    phi = jnp.swapaxes(detector_time_major, 0, 1)
    observed = jnp.bitwise_xor.reduce(ex_error[:, L - 1, :], axis=1)
    return phi, observed


def run_single_trajectory(key, L, p, K=None, rounds=None, cleanout=None):
    """Run one complete memory experiment and return failure and steps used."""

    length = _integer(L, "L", 3)
    depth = K_for_L(length) if K is None else _integer(K, "K", 5)
    if depth not in (5, 6):
        raise ValueError("the winner uses only K=5 or K=6")
    noisy_rounds = length if rounds is None else _integer(rounds, "rounds", 0)
    horizon = cleanout_steps(length, depth) if cleanout is None else _integer(cleanout, "cleanout", 0)
    phi, observed = sample_phenom(key, length, noisy_rounds, p, shots=1)
    terminated, prediction, _, steps_used = _decode_batch(
        phi, K=depth, cleanout=horizon, early_exit=True)
    failure = (~terminated[0]) | (prediction[0] ^ observed[0])
    return failure.astype(jnp.float32), steps_used[0].astype(jnp.float32)


def _samples_for(num_samples, L, p):
    if not isinstance(num_samples, dict):
        return _integer(num_samples, "num_samples", 1)
    if (L, p) in num_samples:
        value = num_samples[L, p]
    elif L in num_samples:
        value = num_samples[L]
    else:
        value = num_samples.get("default", 1_000)
    return _integer(value, "num_samples", 1)


def run_monte_carlo(L_values, p_values, num_samples,
                    filename="surface_phenom_data.npz", seed=42,
                    batch_size=MAX_BATCH_SIZE, K_by_L=None):
    """Run and incrementally save complete surface-code memory experiments."""

    width = _integer(batch_size, "batch_size", 1)
    print("Running JAX Surface Code Phenomenological Memory Evaluation...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")

    results = {}
    master_key = jax.random.PRNGKey(seed)
    for L in L_values:
        length = _integer(L, "L", 3)
        depth = K_for_L(length) if K_by_L is None else _integer(K_by_L[length], "K", 5)
        if depth not in (5, 6):
            raise ValueError("the winner uses only K=5 or K=6")
        horizon = cleanout_steps(length, depth)

        for p in p_values:
            probability = float(p)
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("p must be a finite probability")
            current_num_samples = _samples_for(num_samples, length, p)
            total_failures = 0
            total_success_steps = 0
            total_successes = 0
            samples_left = current_num_samples

            while samples_left:
                current_batch = min(samples_left, width)
                master_key, batch_key = jax.random.split(master_key)
                phi, observed = sample_phenom(
                    batch_key, length, length, probability, shots=current_batch)
                terminated, prediction, _, steps_used = _decode_batch(
                    phi, K=depth, cleanout=horizon, early_exit=True)
                failures = (~terminated) | (prediction ^ observed)
                successes = ~failures
                total_failures += int(jnp.sum(failures))
                total_successes += int(jnp.sum(successes))
                total_success_steps += int(jnp.sum(jnp.where(successes, steps_used, 0)))
                samples_left -= current_batch

            ler = total_failures / current_num_samples
            avg_steps = total_success_steps / total_successes if total_successes else 0.0
            results[length, probability] = (ler, avg_steps, current_num_samples, depth)
            print(f"  L={length}, p={probability:.4f} -> LER={ler:.9f}, "
                  f"AvgSteps (success)={avg_steps:.2f}, Samples={current_num_samples}")

            rows = list(results.items())
            np.savez(
                filename,
                L=np.asarray([cell[0] for cell, _ in rows]),
                p=np.asarray([cell[1] for cell, _ in rows]),
                ler=np.asarray([value[0] for _, value in rows]),
                avg_steps=np.asarray([value[1] for _, value in rows]),
                num_samples=np.asarray([value[2] for _, value in rows]),
                K=np.asarray([value[3] for _, value in rows]),
                seed=np.asarray(seed),
                extra_substeps=np.asarray(EXTRA_SUBSTEPS),
                splitting_period=np.asarray(SPLITTING_PERIOD),
            )
            print(f"Data saved to {filename}")


if __name__ == "__main__":
    run_monte_carlo(
        L_values=[7, 13, 19, 25, 31],
        p_values=[0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
                  0.009, 0.010, 0.011, 0.0115, 0.012, 0.013, 0.014],
        num_samples=100_000,
    )
