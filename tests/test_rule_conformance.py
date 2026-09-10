import jax
import jax.numpy as jnp
import numpy as np

from code_capacity import repetition_ca_jax as repetition
from code_capacity import repetition_ca_jax_uncoord as repetition_uncoord
from code_capacity import surface_ca_split
from code_capacity import toric_ca_rgb as toric
from code_capacity import toric_ca_rgb_uncoord as toric_uncoord


def _majority(a, b, c):
    return (a & b) | (a & c) | (b & c)


def test_repetition_step_matches_registered_rule():
    rng = np.random.default_rng(17)
    for length in (5, 8, 13):
        for step_count in range(4):
            qubits = rng.integers(0, 2, length).astype(bool)
            messages = rng.integers(0, 2, length).astype(bool)
            defect = qubits ^ np.roll(qubits, -1)
            left = np.roll(messages, 1)
            move = defect & left
            expected_qubits = qubits ^ move
            growth = ~defect & ~messages & (step_count % 2 == 0) & left
            persistence = ~defect & messages & left
            arrival = np.roll(defect, -1) & messages
            expected_messages = defect | growth | persistence | arrival

            actual = repetition.step_ca(
                (jnp.asarray(qubits), jnp.asarray(messages), jnp.asarray(step_count)))
            np.testing.assert_array_equal(actual[0], expected_qubits)
            np.testing.assert_array_equal(actual[1], expected_messages)
            assert int(actual[2]) == step_count + 1


def test_repetition_uncoordinated_site_step_matches_registered_rule():
    rng = np.random.default_rng(23)
    for length in (5, 9):
        for trial in range(12):
            qubits = rng.integers(0, 2, length).astype(bool)
            messages = rng.integers(0, 2, length).astype(bool)
            clocks = rng.integers(0, 2, length).astype(np.int32)
            key = jax.random.PRNGKey(1000 + trial + length)
            next_key, site_key = jax.random.split(key)
            selected = int(jax.random.randint(site_key, (), 0, length))
            left = (selected - 1) % length
            right = (selected + 1) % length
            defect = qubits[selected] ^ qubits[(selected + 1) % length]
            move = defect & messages[left]
            incoming = (qubits[right] ^ qubits[(right + 1) % length]) & messages[selected]
            next_message = defect | (
                incoming
                | (~defect
                   & ((~messages[selected] & (clocks[selected] == 0) & messages[left])
                      | (messages[selected] & messages[left]))))

            expected_qubits = qubits.copy()
            expected_messages = messages.copy()
            expected_clocks = clocks.copy()
            expected_messages[selected] = next_message
            expected_qubits[selected] ^= move
            expected_clocks[selected] = (expected_clocks[selected] + 1) % 2

            actual = repetition_uncoord._single_site_update_step((
                jnp.asarray(qubits), jnp.asarray(messages), jnp.asarray(clocks),
                jnp.asarray(0), key))
            np.testing.assert_array_equal(actual[0], expected_qubits)
            np.testing.assert_array_equal(actual[1], expected_messages)
            np.testing.assert_array_equal(actual[2], expected_clocks)
            np.testing.assert_array_equal(actual[4], next_key)


def test_repetition_uncoordinated_move_does_not_write_destination_message():
    length = 7
    key = jax.random.PRNGKey(71)
    _, site_key = jax.random.split(key)
    selected = int(jax.random.randint(site_key, (), 0, length))
    left = (selected - 1) % length

    qubits = jnp.zeros(length, dtype=bool).at[selected].set(True)
    messages = jnp.zeros(length, dtype=bool).at[left].set(True)
    clocks = jnp.zeros(length, dtype=jnp.int32)
    actual = repetition_uncoord._single_site_update_step(
        (qubits, messages, clocks, jnp.asarray(0), key))

    assert bool(actual[0][selected]) is False
    assert bool(actual[1][left]) is True
    assert int(jnp.sum(actual[1])) == 2


def test_repetition_uncoordinated_receiver_preserves_incoming_message():
    length = 7
    key = jax.random.PRNGKey(73)
    _, site_key = jax.random.split(key)
    selected = int(jax.random.randint(site_key, (), 0, length))
    right = (selected + 1) % length
    right_right = (selected + 2) % length

    qubits = jnp.zeros(length, dtype=bool).at[right_right].set(True)
    messages = jnp.zeros(length, dtype=bool).at[selected].set(True)
    clocks = jnp.zeros(length, dtype=jnp.int32)
    actual = repetition_uncoord._single_site_update_step(
        (qubits, messages, clocks, jnp.asarray(0), key))

    assert bool(repetition_uncoord.calculate_syndrome(qubits)[selected]) is False
    assert bool(repetition_uncoord.calculate_syndrome(qubits)[right]) is True
    assert bool(actual[1][selected]) is True


def _toric_sync_reference(h, v, b, r, g, clock, q):
    defect = v ^ np.roll(v, -1, axis=0) ^ h ^ np.roll(h, -1, axis=1)
    b_left, b_down = np.roll(b, 1, axis=0), np.roll(b, 1, axis=1)
    r_left, r_up = np.roll(r, 1, axis=0), np.roll(r, -1, axis=1)
    g_right, g_down = np.roll(g, -1, axis=0), np.roll(g, 1, axis=1)
    move_left = defect & (b_left | r_left)
    move_down = defect & ~(b_left | r_left) & (b_down | g_down)
    bm = _majority(b, b_left, b_down)
    rm = _majority(r, r_left, r_up)
    gm = _majority(g, g_right, g_down)
    growth = clock == 0
    next_b = np.where(defect, True, np.where(b, bm, growth & (b_left | b_down)))
    coupling = b & bm
    next_r = np.where(defect, True, np.where(r, rm | coupling, growth & (r_left | r_up)))
    next_g = np.where(defect, True, np.where(g, gm | coupling, growth & (g_right | g_down)))
    arrival = np.roll(move_left, -1, axis=0) | np.roll(move_down, -1, axis=1)
    return h ^ move_down, v ^ move_left, next_b | arrival, next_r | arrival, next_g | arrival, (clock + 1) % q


def test_toric_step_matches_registered_rule_and_keeps_rgb_channels():
    rng = np.random.default_rng(29)
    for length in (3, 5):
        for trial in range(10):
            arrays = [rng.integers(0, 2, (length, length)).astype(bool) for _ in range(5)]
            clock = trial % toric.CLOCK_PERIOD
            expected = _toric_sync_reference(*arrays, clock, toric.CLOCK_PERIOD)
            state = tuple(jnp.asarray(value) for value in arrays) + (
                clock, jnp.asarray(trial), toric.CLOCK_PERIOD)
            actual = toric.step_ca(state)
            for got, want in zip(actual[:6], expected):
                np.testing.assert_array_equal(got, want)


def test_toric_uncoordinated_step_is_strictly_site_local():
    rng = np.random.default_rng(31)
    length, q = 5, toric_uncoord.DEFAULT_CLOCK_PERIOD
    for trial in range(16):
        h, v, b, r, g = [rng.integers(0, 2, (length, length)).astype(bool) for _ in range(5)]
        clocks = rng.integers(0, q, (length, length)).astype(np.int32)
        key = jax.random.PRNGKey(2000 + trial)
        next_key, x_key, y_key = jax.random.split(key, 3)
        x = int(jax.random.randint(x_key, (), 0, length))
        y = int(jax.random.randint(y_key, (), 0, length))
        xl, xr, yd, yu = (x - 1) % length, (x + 1) % length, (y - 1) % length, (y + 1) % length
        defect = v[x, y] ^ v[xr, y] ^ h[x, y] ^ h[x, yu]
        defect_right = v[xr, y] ^ v[(xr + 1) % length, y] ^ h[xr, y] ^ h[xr, yu]
        defect_up = v[x, yu] ^ v[xr, yu] ^ h[x, yu] ^ h[x, (yu + 1) % length]
        move_left = defect & (b[xl, y] | r[xl, y])
        move_down = defect & ~move_left & (b[x, yd] | g[x, yd])
        incoming_left = defect_right & (b[x, y] | r[x, y])
        incoming_down = defect_up & (g[x, y] | b[x, y]) & ~r[xl, yu] & ~b[xl, yu]
        incoming = incoming_left | incoming_down
        bm = _majority(b[x, y], b[xl, y], b[x, yd])
        rm = _majority(r[x, y], r[xl, y], r[x, yu])
        gm = _majority(g[x, y], g[xr, y], g[x, yd])
        grow = clocks[x, y] == 0
        nb = True if defect or incoming else (bm if b[x, y] else grow & (b[xl, y] | b[x, yd]))
        coupling = b[x, y] & bm
        nr = True if defect or incoming else ((rm | coupling) if r[x, y] else grow & (r[xl, y] | r[x, yu]))
        ng = True if defect or incoming else ((gm | coupling) if g[x, y] else grow & (g[xr, y] | g[x, yd]))

        eh, ev, eb, er, eg, ec = [value.copy() for value in (h, v, b, r, g, clocks)]
        eb[x, y], er[x, y], eg[x, y] = nb, nr, ng
        eh[x, y] ^= move_down
        ev[x, y] ^= move_left
        ec[x, y] = (ec[x, y] + 1) % q

        actual = toric_uncoord._single_site_update_step((
            *map(jnp.asarray, (h, v, b, r, g, clocks)), jnp.asarray(0), q, key))
        for got, want in zip(actual[:6], (eh, ev, eb, er, eg, ec)):
            np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(actual[8], next_key)


def test_toric_uncoordinated_move_does_not_write_destination_messages():
    length, q = 5, toric_uncoord.DEFAULT_CLOCK_PERIOD
    key = jax.random.PRNGKey(91)
    _, x_key, y_key = jax.random.split(key, 3)
    x = int(jax.random.randint(x_key, (), 0, length))
    y = int(jax.random.randint(y_key, (), 0, length))
    left = ((x - 1) % length, y)

    h = jnp.zeros((length, length), dtype=bool)
    v = jnp.zeros((length, length), dtype=bool).at[x, y].set(True)
    b = jnp.zeros((length, length), dtype=bool).at[left].set(True)
    r = jnp.zeros_like(b)
    g = jnp.zeros_like(b)
    clocks = jnp.zeros((length, length), dtype=jnp.int32)

    actual = toric_uncoord._single_site_update_step(
        (h, v, b, r, g, clocks, jnp.asarray(0), q, key))

    assert bool(actual[1][x, y]) is False
    assert bool(actual[2][left]) is True
    assert bool(actual[3][left]) is False
    assert bool(actual[4][left]) is False


def test_surface_split_is_an_intermediate_operation():
    state = surface_ca_split.init_state(jax.random.PRNGKey(7), 6, 0.2)
    split = surface_ca_split.splitting_step(state, 6)
    assert surface_ca_split.CLOCK_PERIOD == 3
    assert int(split[5]) == int(state[5])
    assert int(split[6]) == int(state[6])


def test_surface_proof_clock_gates_movement_and_growth():
    length = 4
    tb = jnp.zeros((length - 1, length + 1), dtype=bool)
    lr = jnp.zeros((length, length), dtype=bool).at[1, 0].set(True)
    zeros = jnp.zeros((length + 1, length), dtype=bool)
    blue = zeros.at[1, 0].set(True)

    moving = surface_ca_split.step_ca(
        (tb, lr, blue, zeros, zeros, 0, jnp.asarray(0), 3), length)
    waiting = surface_ca_split.step_ca(
        (tb, lr, blue, zeros, zeros, 1, jnp.asarray(0), 3), length)
    assert not np.array_equal(moving[1], lr)
    np.testing.assert_array_equal(waiting[1], lr)

    empty_lr = jnp.zeros_like(lr)
    seed = zeros.at[1, 1].set(True)
    growing = surface_ca_split.step_ca(
        (tb, empty_lr, seed, zeros, zeros, 1, jnp.asarray(0), 3), length)
    resting = surface_ca_split.step_ca(
        (tb, empty_lr, seed, zeros, zeros, 2, jnp.asarray(0), 3), length)
    assert bool(growing[2][2, 1])
    assert not bool(resting[2][2, 1])


def test_original_simulation_entry_points_are_present():
    assert callable(repetition.init_state)
    assert callable(repetition.run_monte_carlo)
    assert callable(repetition.run_decoding_time_sweep)
    assert callable(toric.init_state)
    assert callable(toric.run_monte_carlo)
    assert callable(toric.run_decoding_time_sweep)
