"""
surface_ca.py

Surface code CA decoder with splitting dynamics.

Partition: left half Λ_L = {x ≤ ⌊L/2⌋}, right half Λ_R = {x > ⌊L/2⌋}.
x indexes axis-0 of the stabilizer grid (shape L+1, L), 0-indexed 0..L.

At multiples of q', first perform the splitting operation and then run the standard
RGB CA update. At other times, run only the standard RGB CA update.

The splitting operation translates both the syndrome and the messages (b, r, g)
one step outward, away from the cut between the two halves:
  - Left half (x ≤ mid): contents shift left by 1 (x → x-1).
  - Right half (x > mid): contents shift right by 1 (x → x+1).
  - Center x=mid (left): syndrome moves to mid-1; messages are COPIED, i.e. they
    appear at mid-1 AND are kept at mid.
  - Center x=mid+1 (right): syndrome moves to mid+2; messages are COPIED, i.e. they
    are kept at mid+1 AND appear at mid+2.
  - Boundary (x=0, x=L): content that would shift beyond the grid is deleted.
Hence after a splitting step the two center rows carry no syndrome
(s[mid] = s[mid+1] = 0), opening a defect-free gap along the cut, while messages
straddling the cut are duplicated so each half keeps a copy.

"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Circle, Patch, Wedge
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

CLOCK_PERIOD = 3
MAX_BATCH_SIZE = 2_000


@partial(jax.jit, static_argnames=['L', 'CLOCK_PERIOD'])
def init_state(key, L, p, CLOCK_PERIOD=CLOCK_PERIOD):
    """Initialize state."""
    k1, k2 = jax.random.split(key)
    tb_qubits = jax.random.bernoulli(k1, p, shape=(L-1, L-1)).astype(jnp.bool_)
    lr_qubits = jax.random.bernoulli(k2, p, shape=(L, L)).astype(jnp.bool_)
    tb_qubits = jnp.pad(tb_qubits, ((0, 0), (1, 1)))

    grid_shape = (L+1, L)
    b_grid = jnp.zeros(grid_shape, dtype=jnp.bool_)
    r_grid = jnp.zeros(grid_shape, dtype=jnp.bool_)
    g_grid = jnp.zeros(grid_shape, dtype=jnp.bool_)

    return (tb_qubits, lr_qubits, b_grid, r_grid, g_grid,
            0, jnp.array(0, dtype=jnp.int32), CLOCK_PERIOD)


@partial(jax.jit, static_argnames=['L'])
def calculate_syndrome(tb_qubits, lr_qubits, L):
    syndrome = (lr_qubits[:-1, :] ^ lr_qubits[1:, :]
                ^ tb_qubits[:, 1:] ^ tb_qubits[:, :-1])
    return jnp.pad(syndrome, ((1, 1), (0, 0)))


@partial(jax.jit, static_argnames=['L'])
def step_ca(state, L):
    """Standard RGB CA step with erasure at boundaries."""
    tb_qubits, lr_qubits, b, r, g, c, step_count, CLOCK_PERIOD = state

    s = calculate_syndrome(tb_qubits, lr_qubits, L)

    def left(x):  return jnp.concatenate([jnp.zeros((1, L), dtype=jnp.bool_), x[:-1, :]], axis=0)
    def right(x): return jnp.concatenate([x[1:, :], jnp.zeros((1, L), dtype=jnp.bool_)], axis=0)
    def up(x):    return jnp.concatenate([x[:, 1:], jnp.zeros((L+1, 1), dtype=jnp.bool_)], axis=1)
    def down(x):  return jnp.concatenate([jnp.zeros((L+1, 1), dtype=jnp.bool_), x[:, :-1]], axis=1)

    b_left = left(b);  b_down = down(b)
    r_left = left(r);  r_up   = up(r)
    g_right = right(g); g_down = down(g)
    s_right = right(s); s_up   = up(s)
    r_left_up = up(r_left)
    b_left_up = up(b_left)

    mask_s    = s
    mask_no_s = ~s

    next_b = jnp.where(mask_s, True, b)
    next_r = jnp.where(mask_s, True, r)
    next_g = jnp.where(mask_s, True, g)

    # Corrections: left if b_left|r_left; else down if g_down
    left_active = b_left | r_left
    movement_step = c == 0
    do_left = mask_s & movement_step & left_active
    cond_down = (g_down | b_down)
    do_down = mask_s & movement_step & (~do_left) & cond_down

    corrections_left = do_left[1:, :]
    corrections_down = jnp.concatenate([
        jnp.zeros((L-1, 1), dtype=jnp.bool_),
        do_down[1:-1, 1:],
        jnp.zeros((L-1, 1), dtype=jnp.bool_)
    ], axis=1)

    next_lr_qubits = lr_qubits ^ corrections_left
    next_tb_qubits = tb_qubits ^ corrections_down

    # Incoming syndrome → set all messages
    ils = movement_step & s_right & (b | r)
    ids = movement_step & s_up & (g | b) & (~r_left_up) & (~b_left_up)
    is_flag = ils | ids

    mask_is = mask_no_s & is_flag
    next_b = jnp.where(mask_is, True, next_b)
    next_r = jnp.where(mask_is, True, next_r)
    next_g = jnp.where(mask_is, True, next_g)

    # Growth and majority
    mask_else   = mask_no_s & (~is_flag)
    mask_growth = mask_else & (c != CLOCK_PERIOD - 1)

    next_b = jnp.where(mask_growth & (~b), b_down | b_left, next_b)
    next_r = jnp.where(mask_growth & (~r), r_up   | r_left, next_r)
    next_g = jnp.where(mask_growth & (~g), g_down  | g_right, next_g)

    bm = (b_left.astype(jnp.int32) + b_down.astype(jnp.int32)  + b.astype(jnp.int32)) >= 2
    rm = (r_left.astype(jnp.int32) + r_up.astype(jnp.int32)    + r.astype(jnp.int32)) >= 2
    gm = (g_right.astype(jnp.int32) + g_down.astype(jnp.int32) + g.astype(jnp.int32)) >= 2

    next_b = jnp.where(mask_else & b, bm,             next_b)
    next_r = jnp.where(mask_else & r, rm | (bm & b),  next_r)
    next_g = jnp.where(mask_else & g, gm | (bm & b),  next_g)

    return (next_tb_qubits, next_lr_qubits, next_b, next_r, next_g,
            (c + 1) % CLOCK_PERIOD, step_count + 1, CLOCK_PERIOD)


@partial(jax.jit, static_argnames=['L'])
def splitting_step(state, L):
    """
    Splitting step: translate the syndrome and the message grids one site outward
    from the center (mid = L//2), i.e. left half moves to smaller x and right half
    to larger x, leaving a defect-free gap along the cut.

    For each message channel (b, r, g):
      - Regular left-half sites  x=1..mid-1    →  shift to x-1.
      - Regular right-half sites x=mid+2..L-1  →  shift to x+1.
      - Center x=mid   (left):  messages COPIED to mid-1 AND kept at mid.
      - Center x=mid+1 (right): messages kept at mid+1 AND COPIED to mid+2.
      - Content at x=0 or x=L that would shift outside the grid is deleted.
    Copying at the two center sites (rather than moving) lets each half retain the
    message state that sat on the cut.

    The syndrome is not stored but recomputed from the qubits, so shifting it is a
    physical operation: lr_qubits are flipped so that calculate_syndrome returns the
    translated pattern on the next call. Flipping lr[i] toggles the padded syndrome
    at x=i and x=i+1, so with
      - Left half  (i=0..mid-1):   new_lr[i] = old_lr[i] XOR s[i+1]
      - Right half (i=mid+1..L-1): new_lr[i] = old_lr[i] XOR s[i]
      - Center     (i=mid):        unchanged
    the new syndrome s'[x] = s[x] XOR delta[x] XOR delta[x-1] works out to
      - s'[x] = s[x+1] for x=1..mid-1      (left half shifted outward)
      - s'[mid] = s'[mid+1] = 0            (the gap; no flip at i=mid)
      - s'[x] = s[x-1] for x=mid+2..L-1    (right half shifted outward)
    so the whole syndrome pattern, not just the two center rows, moves outward; what
    reaches the padded rows x=0 and x=L is absorbed.
    """
    tb_qubits, lr_qubits, b, r, g, c, step_count, CLOCK_PERIOD = state
    mid = L // 2  # split index in axis-0

    # --- Syndrome shift via lr_qubit flips ---
    s = calculate_syndrome(tb_qubits, lr_qubits, L)
    # Left half: flip lr[i] by s[i+1] for i = 0..mid-1
    # Right half: flip lr[i] by s[i]   for i = mid+1..L-1
    # Center (i=mid): no flip → new_syndrome[mid] = 0 and new_syndrome[mid+1] = 0 automatically
    new_lr_qubits = lr_qubits.at[0:mid, :].set(lr_qubits[0:mid, :] ^ s[1:mid+1, :])
    new_lr_qubits = new_lr_qubits.at[mid+1:L, :].set(lr_qubits[mid+1:L, :] ^ s[mid+1:L, :])

    # --- Message shift: translate outward from center, copy at center sites ---
    def shift_outward(msg):
        new_msg = jnp.zeros_like(msg)

        # Regular left-half: x=1..mid-1  →  0..mid-2
        # (x=0 deleted; x=mid is center)
        if mid >= 2:
            new_msg = new_msg.at[0:mid-1, :].set(msg[1:mid, :])

        # Center x=mid: copy outward to mid-1 AND keep at mid
        if mid >= 1:
            new_msg = new_msg.at[mid-1, :].set(new_msg[mid-1, :] | msg[mid, :])
        new_msg = new_msg.at[mid, :].set(msg[mid, :])

        # Center x=mid+1: keep at mid+1 AND copy outward to mid+2
        new_msg = new_msg.at[mid+1, :].set(msg[mid+1, :])
        if mid + 2 <= L:
            new_msg = new_msg.at[mid+2, :].set(new_msg[mid+2, :] | msg[mid+1, :])

        # Regular right-half: x=mid+2..L-1  →  mid+3..L
        # (x=L deleted; x=mid+1 is center)
        if mid + 2 <= L - 1:
            new_msg = new_msg.at[mid+3:L+1, :].set(msg[mid+2:L, :])

        return new_msg

    new_b = shift_outward(b)
    new_r = shift_outward(r)
    new_g = shift_outward(g)

    return (tb_qubits, new_lr_qubits, new_b, new_r, new_g,
            c, step_count, CLOCK_PERIOD)


@partial(jax.jit, static_argnames=['L'])
def check_logical_error(tb_qubits, lr_qubits, L):
    winding = jnp.sum(lr_qubits[L // 2, :]) % 2
    return winding != 0


@partial(jax.jit, static_argnames=['L', 'max_steps', 'CLOCK_PERIOD', 'q_prime'])
def run_single_trajectory(key, L, p, max_steps, q_prime, CLOCK_PERIOD=CLOCK_PERIOD):
    """
    Run one trajectory. Splitting step at t = 0, q', 2q', ...
    Returns (is_failure, step_count).
    """
    initial_state = init_state(key, L, p, CLOCK_PERIOD)

    def cond_fun(state):
        tb_qubits, lr_qubits, _, _, _, _, step_count, _ = state
        s = calculate_syndrome(tb_qubits, lr_qubits, L)
        return jnp.any(s) & (step_count < max_steps)

    def body_fun(state):
        _, _, _, _, _, _, step_count, _ = state
        prepared = jax.lax.cond(
            step_count % q_prime == 0,
            lambda s: splitting_step(s, L),
            lambda s: s,
            state
        )
        return step_ca(prepared, L)

    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    tb_qubits, lr_qubits, _, _, _, _, step_count, _ = final_state

    converged     = jnp.all(calculate_syndrome(tb_qubits, lr_qubits, L) == 0)
    logical_error = check_logical_error(tb_qubits, lr_qubits, L)
    is_failure    = (~converged) | logical_error

    return is_failure.astype(jnp.float32), step_count.astype(jnp.float32)


# --- Monte Carlo ---

def run_monte_carlo(L_values, p_values, num_samples, q_prime,
                    filename="surface_ca_split_data.npz"):
    print(f"Running Surface Code CA Splitting Decoder (q'={q_prime})")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")

    results = {}
    master_key = jax.random.PRNGKey(42)

    for L in L_values:
        max_steps = 5 * CLOCK_PERIOD * L * q_prime

        @partial(jax.jit, static_argnames=['CLOCK_PERIOD', 'q_prime'])
        def batch_run(keys, p_val, CLOCK_PERIOD=CLOCK_PERIOD, q_prime=q_prime):
            return jax.vmap(
                lambda k: run_single_trajectory(k, L, p_val, max_steps, q_prime, CLOCK_PERIOD)
            )(keys)

        for p in p_values:
            if isinstance(num_samples, dict):
                current_num_samples = num_samples.get(
                    (L, p), num_samples.get(L, num_samples.get('default', 1000))
                )
            else:
                current_num_samples = num_samples

            total_failures    = 0.0
            total_successes   = 0.0
            total_succ_steps  = 0.0
            samples_left      = current_num_samples
            master_key, loop_key = jax.random.split(master_key)

            while samples_left > 0:
                bs = min(samples_left, MAX_BATCH_SIZE)
                loop_key, sub = jax.random.split(loop_key)
                keys = jax.random.split(sub, bs)

                batch_fail, batch_steps = batch_run(keys, p)
                succ_mask = 1.0 - batch_fail
                total_failures   += float(jnp.sum(batch_fail))
                total_successes  += float(jnp.sum(succ_mask))
                total_succ_steps += float(jnp.sum(batch_steps * succ_mask))
                samples_left -= bs
                if samples_left % 1000 == 0 and samples_left > 0:
                    print(f"    ... {samples_left} remaining", end='\r')

            print()
            ler      = total_failures / current_num_samples
            avg_steps = (total_succ_steps / total_successes) if total_successes > 0 else 0.0
            results[(L, p)] = (ler, avg_steps, current_num_samples)
            print(f"  L={L}, p={p:.4f} -> LER={ler:.6f}, AvgSteps={avg_steps:.1f}, N={current_num_samples}")

            # Save incrementally
            Ls_o, ps_o, lers_o, steps_o, ns_o = [], [], [], [], []
            for (Lk, pk), (le, av, ns) in results.items():
                Ls_o.append(Lk); ps_o.append(pk); lers_o.append(le)
                steps_o.append(av); ns_o.append(ns)
            np.savez(filename,
                     L=np.array(Ls_o), p=np.array(ps_o),
                     ler=np.array(lers_o), avg_steps=np.array(steps_o),
                     num_samples=np.array(ns_o))
            print(f"  Saved to {filename}")


# --- Plotting ---

def plot_results(data_filename, plot_filename="surface_ca_split_results.png"):
    if not os.path.exists(data_filename):
        print(f"File {data_filename} not found.")
        return

    data = np.load(data_filename)
    Ls_raw, ps_raw, lers_raw = data['L'], data['p'], data['ler']
    results = {(int(L), float(p)): float(le) for L, p, le in zip(Ls_raw, ps_raw, lers_raw)}
    unique_L = sorted(set(int(L) for L in Ls_raw))
    unique_p = sorted(set(float(p) for p in ps_raw))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for L in unique_L:
        ps  = [p for p in unique_p if (L, p) in results]
        les = [results[(L, p)] for p in ps]
        if les:
            ax1.plot(ps, les, 'o-', label=f'L={L}', markersize=6)

    ax1.set_xlabel('Physical Error Rate (p)')
    ax1.set_ylabel('Logical Error Rate')
    ax1.set_yscale('log'); ax1.set_xscale('log')
    ax1.grid(True, which='both', ls='-', alpha=0.4)
    ax1.legend(); ax1.set_title('Surface Code CA (Splitting) Decoder Threshold')

    for p in unique_p:
        Ls  = [L for L in unique_L if (L, p) in results]
        les = [results[(L, p)] for L in Ls]
        if les:
            ax2.plot(Ls, les, 'x--', label=f'p={p:.4f}', markersize=8)

    ax2.set_xlabel('Code Distance (L)')
    ax2.set_ylabel('Logical Error Rate')
    ax2.set_yscale('log')
    ax2.grid(True, which='both', ls='-', alpha=0.4)
    ax2.legend(); ax2.set_title('Scaling with L')

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    print(f"Plot saved to {plot_filename}")
    plt.close(fig)


# --- Visualization ---

def save_simulation_gif(L, p, steps, q_prime,
                        filename="surface_ca_split.gif", channels=None):
    """Save GIF of splitting decoder dynamics."""
    if channels is None:
        channels = ['b', 'r', 'g']
    show_b = 'b' in channels
    show_r = 'r' in channels
    show_g = 'g' in channels

    key = jax.random.PRNGKey(np.random.randint(0, 10000))
    initial_state = init_state(key, L, p)
    tb0, lr0, b0, r0, g0, c0, sc0, cp0 = initial_state
    s0 = calculate_syndrome(tb0, lr0, L)
    done0 = jnp.all(s0 == 0) & jnp.all(b0 == 0) & jnp.all(r0 == 0) & jnp.all(g0 == 0)

    def scan_step(carry, _):
        tb, lr, b, r, g, c, sc, cp, done = carry
        state_in = (tb, lr, b, r, g, c, sc, cp)

        prepared = jax.lax.cond(
            sc % q_prime == 0,
            lambda s: splitting_step(s, L),
            lambda s: s,
            state_in
        )
        state_out = step_ca(prepared, L)
        tb_n, lr_n, b_n, r_n, g_n, c_n, sc_n, cp_n = state_out

        syn = calculate_syndrome(tb_n, lr_n, L)
        is_clean = jnp.all(syn == 0) & jnp.all(b_n == 0) & jnp.all(r_n == 0) & jnp.all(g_n == 0)
        new_done = done | is_clean

        new_carry = (
            jnp.where(done, tb, tb_n), jnp.where(done, lr, lr_n),
            jnp.where(done, b,  b_n),  jnp.where(done, r,  r_n),
            jnp.where(done, g,  g_n),
            c_n, sc_n, cp_n, new_done
        )
        output = (
            calculate_syndrome(new_carry[0], new_carry[1], L),
            new_carry[2], new_carry[3], new_carry[4], new_carry[5]
        )
        return new_carry, output

    carry_init = (tb0, lr0, b0, r0, g0, c0, sc0, cp0, done0)
    final_carry, history = jax.lax.scan(scan_step, carry_init, None, length=steps)
    s_hist, b_hist, r_hist, g_hist, c_hist = [np.array(h) for h in history]

    full_s = np.concatenate([[np.array(s0)], s_hist], axis=0)
    full_b = np.concatenate([[np.array(b0)], b_hist], axis=0)
    full_r = np.concatenate([[np.array(r0)], r_hist], axis=0)
    full_g = np.concatenate([[np.array(g0)], g_hist], axis=0)
    full_c = np.concatenate([[np.array(c0)], c_hist], axis=0)

    is_logical_fail = bool(check_logical_error(final_carry[0], final_carry[1], L))
    print("Logical error:", is_logical_fail)

    # --- Build figure ---
    fig, ax = plt.subplots(figsize=(10, 10))
    color_z   = '#97BBFF'
    color_x   = '#FFFFFF'
    color_b   = (0.3, 0.3, 1.0, 0.5)
    color_r   = (1.0, 0.3, 0.3, 0.5)
    color_g_m = (0.3, 0.8, 0.3, 0.5)
    color_sym = (0.9, 0.9, 0.3)
    mid = L // 2

    # Background
    bg = [Polygon([(i+0.5, j), (i+1, j+0.5), (i+1.5, j), (i+1, j-0.5)])
          for i in range(L) for j in range(L+1)]
    ax.add_collection(PatchCollection(bg, facecolor=color_x, alpha=0.3, zorder=0))

    # Z-stabilizer diamonds
    z_patches, z_centers = [], []
    for i in range(L+1):
        for j in range(L):
            cx, cy = i + 0.5, j + 0.5
            z_centers.append((cx, cy))
            if   j == 0:   verts = [(cx-0.5, cy), (cx+0.5, cy), (cx, cy+0.5)]
            elif j == L-1: verts = [(cx-0.5, cy), (cx+0.5, cy), (cx, cy-0.5)]
            else:          verts = [(cx-0.5, cy), (cx, cy-0.5), (cx+0.5, cy), (cx, cy+0.5)]
            z_patches.append(Polygon(verts))

    per_alpha = [0.2 if (i == 0 or i == L) else 1.0 for i in range(L+1) for _ in range(L)]
    z_colors = [(*plt.cm.colors.to_rgb(color_z), a) for a in per_alpha]
    z_coll = PatchCollection(z_patches, facecolors=z_colors,
                             edgecolors='black', linewidths=0.5, zorder=1)
    ax.add_collection(z_coll)

    # Center partition line
    ax.axvline(x=mid + 0.5, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.6, zorder=0)

    # Defect circles
    defect_patches = [Circle(c, 0.22) for c in z_centers]
    defect_coll = PatchCollection(defect_patches, zorder=2)
    defect_coll.set_facecolor('none'); defect_coll.set_edgecolor('none')
    ax.add_collection(defect_coll)

    # Message wedges (3 segments: B, R, G)
    n_seg = 3
    angle_per = 360 / n_seg
    q_patches = [Wedge(c, 0.28, i*angle_per, (i+1)*angle_per)
                 for c in z_centers for i in range(n_seg)]
    msg_coll = PatchCollection(q_patches, zorder=3)
    msg_coll.set_facecolor('none')
    ax.add_collection(msg_coll)

    # Qubits
    qb = ([Circle((i+1,   j+0.5), 0.05) for i in range(L)   for j in range(L)] +
          [Circle((i+1.5, j),     0.05) for i in range(L-1) for j in range(1, L)])
    ax.add_collection(PatchCollection(qb, facecolor='black', zorder=5))

    ax.set_xlim(-0.5, L + 1.5); ax.set_ylim(-0.5, L + 0.5)
    ax.set_aspect('equal'); ax.axis('off')

    legend_el = [
        Line2D([0], [0], marker='o', color='w', label='Defect',
               markerfacecolor=color_sym, markeredgecolor='black', markersize=10),
    ]
    if show_b: legend_el.append(Patch(facecolor=color_b,   label='Blue (b)'))
    if show_r: legend_el.append(Patch(facecolor=color_r,   label='Red (r)'))
    if show_g: legend_el.append(Patch(facecolor=color_g_m, label='Green (g)'))
    ax.legend(handles=legend_el, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=4, fontsize='small', framealpha=0.95)

    seg_colors = [color_b, color_r, color_g_m]

    def update(frame):
        s = full_s[frame]; bd = full_b[frame]; rd = full_r[frame]
        gd = full_g[frame]; cd = full_c[frame]
        N = s.size

        m_stack = np.stack([
            bd.flatten() if show_b else np.zeros(N, bool),
            rd.flatten() if show_r else np.zeros(N, bool),
            gd.flatten() if show_g else np.zeros(N, bool),
        ], axis=1)
        m_colors = [seg_colors[j] if m_stack[i, j] else (0,0,0,0)
                    for i in range(N) for j in range(n_seg)]
        msg_coll.set_facecolors(m_colors)

        fc = [((*plt.cm.colors.to_rgb(color_sym), 1.0) if v else (0,0,0,0)) for v in s.flatten()]
        ec = [('black' if v else (0,0,0,0)) for v in s.flatten()]
        defect_coll.set_facecolors(fc); defect_coll.set_edgecolors(ec)

        # Label whether this frame is a split step
        # frame 0 = initial; frame k = after step k-1 (step_count = k-1)
        is_split = (frame > 0) and ((frame - 1) % q_prime == 0)
        step_type = "SPLIT + CA" if is_split else "CA"
        ndefs = int(np.sum(s))
        title = f"T={frame} [{step_type}] | Defects={ndefs} | Clock={int(cd)}"
        if ndefs == 0 and np.sum(bd) == 0 and np.sum(rd) == 0 and np.sum(gd) == 0:
            title += " [LOGICAL ERROR]" if is_logical_fail else " [SUCCESS]"
        ax.set_title(title, fontsize=13)
        return msg_coll, defect_coll

    clean = [i for i in range(len(full_s))
             if np.all(full_s[i] == 0) and np.all(full_b[i] == 0)
             and np.all(full_r[i] == 0) and np.all(full_g[i] == 0)]
    cutoff = (clean[0] + 1) if clean else len(full_s)

    anim = FuncAnimation(fig, update, frames=cutoff, blit=True, repeat=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    try:
        anim.save(filename, writer=PillowWriter(fps=5))
        print(f"GIF saved to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
    plt.close(fig)


if __name__ == "__main__":
    q_prime = 19
    run_monte_carlo(
        L_values=[8, 14, 20],
        p_values=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        num_samples={'default': 10_000},
        q_prime=q_prime,
        filename=f"surface_ca_split_results_{q_prime}_>0.npz"
    )
    plot_results(f"surface_ca_split_results_{q_prime}_>0.npz", f"surface_ca_split_results_{q_prime}_>0.png")

    # save_simulation_gif(L=13, p=0.03, steps=100, q_prime=20, filename="surface_ca_split_example.gif")
