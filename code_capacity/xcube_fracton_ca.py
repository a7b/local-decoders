import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
import time

# --- Constants ---

CLOCK_PERIOD = 10
MAX_BATCH_SIZE = 2000


# --- Syndrome Evaluation ---

@jax.jit
def calculate_syndrome(links):
    """
    Compute X-cube cube stabilizers on an L×L×L cubic lattice with PBC.

    Each cube stabilizer is the XOR of its 12 incident edges.
    links : (3, L, L, L) bool  →  returns (L, L, L) bool
    """
    lx, ly, lz = links[0], links[1], links[2]

    # XOR the 4 x-edges, 4 y-edges, 4 z-edges on the cube
    ex = lx ^ jnp.roll(lx, -1, 1) ^ jnp.roll(lx, -1, 2) ^ jnp.roll(jnp.roll(lx, -1, 1), -1, 2)
    ey = ly ^ jnp.roll(ly, -1, 0) ^ jnp.roll(ly, -1, 2) ^ jnp.roll(jnp.roll(ly, -1, 0), -1, 2)
    ez = lz ^ jnp.roll(lz, -1, 0) ^ jnp.roll(lz, -1, 1) ^ jnp.roll(jnp.roll(lz, -1, 0), -1, 1)

    return ex ^ ey ^ ez


# --- CA Step ---

@jax.jit
def step_ca(state):
    """
    One synchronous CA step for the 3D X-cube model (new rule).

    state: (links, s, m, c, step_count, CLOCK_PERIOD)

    Parameters
    ----------
    links : (3, L, L, L) bool   – link variables  [x/y/z edges]
    s     : (L, L, L)    bool   – cube stabilizer syndrome
    m     : (2,2,2, L,L,L) bool – eight memory fields m_{ijk}
    c     : int32 scalar         – global clock  (0 … q-1)
    step_count : int32 scalar    – current step
    CLOCK_PERIOD : int (static)  – clock period

    Returns updated state tuple with same structure.
    """
    links, s, m, c, step_count, CLOCK_PERIOD = state
    q = CLOCK_PERIOD

    syn    = s
    no_syn = ~s
    clock0 = (c == 0)

    # ── Pre-compute all neighbor shifts ──────────────────────────────
    m_xp1 = jnp.roll(m, -1, axis=3)
    m_xm1 = jnp.roll(m, 1, axis=3)
    m_yp1 = jnp.roll(m, -1, axis=4)
    m_ym1 = jnp.roll(m, 1, axis=4)
    m_zp1 = jnp.roll(m, -1, axis=5)
    m_zm1 = jnp.roll(m, 1, axis=5)

    # Build neighbor arrays using broadcasting (pure JAX)
    i_idx = jnp.arange(2)[:, None, None, None, None, None]
    j_idx = jnp.arange(2)[None, :, None, None, None, None]
    k_idx = jnp.arange(2)[None, None, :, None, None, None]

    m_nb_x = jnp.where(i_idx == 0, m_xp1, m_xm1)
    m_nb_y = jnp.where(j_idx == 0, m_yp1, m_ym1)
    m_nb_z = jnp.where(k_idx == 0, m_zp1, m_zm1)

    # ── Initialize m_new ─────────────────────────────────────────────
    m_new = m

    # ═══ SYNDROME SITES: set all m_ijk = 1 ═══════════════════════════
    syn_bc = syn[None, None, None, :, :, :]
    m_new = jnp.where(syn_bc, True, m_new)

    # ═══ MOVES  (priority: z > y > x) ═══════════════════════════════
    m_xm1ym1 = jnp.roll(jnp.roll(m, 1, axis=3), 1, axis=4)
    z_trig = m_xm1ym1[1, 1, 0] | m_xm1ym1[1, 1, 1]
    m_xm1zm1 = jnp.roll(jnp.roll(m, 1, axis=3), 1, axis=5)
    y_trig = m_xm1zm1[1, 0, 1] | m_xm1zm1[1, 1, 1]
    m_ym1zm1 = jnp.roll(jnp.roll(m, 1, axis=4), 1, axis=5)
    x_trig = m_ym1zm1[0, 1, 1] | m_ym1zm1[1, 1, 1]

    do_z = syn & z_trig
    do_y = syn & ~z_trig & y_trig
    do_x = syn & ~z_trig & ~y_trig & x_trig

    flip_x = do_x
    flip_y = do_y
    flip_z = do_z

    # ═══ S_IN: neighbors of movers get all fields set to 1 ══════════
    dz_xp1     = jnp.roll(do_z, -1, axis=0)
    dz_yp1     = jnp.roll(do_z, -1, axis=1)
    dz_xp1yp1  = jnp.roll(dz_xp1, -1, axis=1)
    in_z = dz_xp1 | dz_yp1 | dz_xp1yp1

    dy_xp1     = jnp.roll(do_y, -1, axis=0)
    dy_zp1     = jnp.roll(do_y, -1, axis=2)
    dy_xp1zp1  = jnp.roll(dy_xp1, -1, axis=2)
    in_y = dy_xp1 | dy_zp1 | dy_xp1zp1

    dx_yp1     = jnp.roll(do_x, -1, axis=1)
    dx_zp1     = jnp.roll(do_x, -1, axis=2)
    dx_yp1zp1  = jnp.roll(dx_yp1, -1, axis=2)
    in_x = dx_yp1 | dx_zp1 | dx_yp1zp1

    is_incoming = (in_z | in_y | in_x) & no_syn
    in_bc = is_incoming[None, None, None, :, :, :]
    m_new = jnp.where(in_bc, True, m_new)

    # ═══ NON-SYNDROME, NON-INCOMING SITES ════════════════════════════
    rest = no_syn & ~is_incoming
    rest_bc = rest[None, None, None, :, :, :]

    # ── Spreading (clock == 0, m_ijk == 0): use (-1)^i x, (-1)^j y, (-1)^k z neighbors from m (old)
    m_zero = ~m
    m_spread_xp1 = jnp.roll(m, -1, axis=3)
    m_spread_xm1 = jnp.roll(m, 1, axis=3)
    m_spread_yp1 = jnp.roll(m, -1, axis=4)
    m_spread_ym1 = jnp.roll(m, 1, axis=4)
    m_spread_zp1 = jnp.roll(m, -1, axis=5)
    m_spread_zm1 = jnp.roll(m, 1, axis=5)
    m_spread_nb_x = jnp.where(i_idx == 0, m_spread_xp1, m_spread_xm1)
    m_spread_nb_y = jnp.where(j_idx == 0, m_spread_yp1, m_spread_ym1)
    m_spread_nb_z = jnp.where(k_idx == 0, m_spread_zp1, m_spread_zm1)
    any_neighbor = m_spread_nb_x | m_spread_nb_y | m_spread_nb_z
    spread_mask = rest_bc & m_zero & clock0 & any_neighbor
    m_new = jnp.where(spread_mask, True, m_new)

    # ── Rule: 0->1 if any neighbor; 1->0 if no neighbor ──
    # Neighbors: (-1)^i x, (-1)^j y, (-1)^k z (from m_nb_x, m_nb_y, m_nb_z)
    total = (m.astype(jnp.int32) + m_nb_x.astype(jnp.int32) + m_nb_y.astype(jnp.int32) + m_nb_z.astype(jnp.int32))
    has_neighbor = (total >= 2)  # self + at least one neighbor
    v_all = m & has_neighbor  # persist where m=1 and has neighbor
    growth = m_new & ~m  # 0->1 from spread/incoming

    # ── m_111 special handling (apply first, exclude from decay) ──
    m111_old = m[1, 1, 1]
    nbx_111 = m_xm1[1, 1, 1]
    nby_111 = m_ym1[1, 1, 1]
    nbz_111 = m_zm1[1, 1, 1]
    v111 = (m111_old.astype(jnp.int32) + nbx_111.astype(jnp.int32) + nby_111.astype(jnp.int32) + nbz_111.astype(jnp.int32)) >= 2
    growth_111 = m_new[1, 1, 1] & ~m111_old
    coupled = m111_old & v111
    m111_new = jnp.where(rest, growth_111 | (m111_old & v111), m_new[1, 1, 1])
    m_new = m_new.at[1, 1, 1].set(m111_new)

    # ── Apply decay with failsafe (m_111 coupled → boost other slices) ──
    exclude_111 = jnp.array([[[True, True], [True, True]],
                              [[True, True], [True, False]]], dtype=bool)
    exclude_111_bc = exclude_111[:, :, :, None, None, None]
    failsafe_ok = (coupled[None, None, None, :, :, :]) & (m | growth)
    boosted = growth | (v_all | failsafe_ok)
    decay_mask = rest_bc & exclude_111_bc  # all rest non-111
    m_new = jnp.where(decay_mask, boosted, m_new)

    # ═══ APPLY LINK FLIPS & RECOMPUTE SYNDROMES ═════════════════════
    links_new = jnp.stack([
        links[0] ^ flip_x,
        links[1] ^ flip_y,
        links[2] ^ flip_z,
    ]).astype(bool)
    s_new = calculate_syndrome(links_new).astype(bool)
    c_new = (c + 1) % q

    # Incoming overrides all other message updates (apply last)
    m_new = jnp.where(in_bc, True, m_new)

    return (links_new, s_new, m_new, c_new, step_count + 1, CLOCK_PERIOD)


# --- Logical Error Check ---

# @jax.jit
# def check_logical_error(links):
#     """
#     Check whether any Z-type logical operator of the X-cube model has
#     been flipped (odd parity).

#     The Z-logicals are non-contractible loops in each plane.
#     In each plane there are 2 loop directions, and loops at different
#     transverse positions can be independent (sub-extensive degeneracy).

#     We check all 3L² straight-line loops:
#       • x-loops:  sum x-edges along x for each (y₀, z₀)  →  (L, L)
#       • y-loops:  sum y-edges along y for each (x₀, z₀)  →  (L, L)
#       • z-loops:  sum z-edges along z for each (x₀, y₀)  →  (L, L)

#     Returns a boolean scalar: True ⟹ logical error.
#     """
#     lx, ly, lz = links[0], links[1], links[2]
#     # x-direction loops: parity along x for each (y, z)
#     x_par = jnp.logical_xor.reduce(lx, axis=0)  # (L, L)
#     y_par = jnp.logical_xor.reduce(ly, axis=1)  # (L, L)
#     z_par = jnp.logical_xor.reduce(lz, axis=2)  # (L, L)
#     return jnp.any(x_par) | jnp.any(y_par) | jnp.any(z_par)

@jax.jit
def check_logical_error(links):
    """
    Check whether any Z-type logical operator of the X-cube model has
    been flipped (odd parity).

    The X-cube model has 6L logical operators (on an L×L×L torus):
    one horizontal and one vertical non-contractible line per slice
    in each of the three plane orientations (xy, xz, yz).

    These logicals live on the DUAL lattice.  Each dual-lattice line
    consists of plaquettes of 4 edges per cross-section.  The parity
    of a dual line equals the 2D-plaquette of the single-edge parity
    array.  For example, for an x-directed dual line at position
    (y₀, z₀) on the dual lattice:

      Lx(y₀,z₀) = x_par[y₀,z₀] ^ x_par[y₀+1,z₀]
                 ^ x_par[y₀,z₀+1] ^ x_par[y₀+1,z₀+1]

    where x_par[y,z] = XOR_x lx[x,y,z].

    We check 6L representative logicals (one horiz + one vert per
    slice in each plane):
      xy-plane at z₀: horiz = Lx(0, z₀),  vert = Ly(0, z₀)
      xz-plane at y₀: horiz = Lx(y₀, 0),  vert = Lz(0, y₀)
      yz-plane at x₀: horiz = Ly(x₀, 0),  vert = Lz(x₀, 0)

    Returns a boolean scalar: True ⟹ logical error.
    """
    lx, ly, lz = links[0], links[1], links[2]

    # --- Single-edge parity arrays (XOR along the edge direction) ---
    # x_par[y, z] = XOR_x lx[x, y, z]   shape (L, L)
    # y_par[x, z] = XOR_y ly[x, y, z]   shape (L, L)
    # z_par[x, y] = XOR_z lz[x, y, z]   shape (L, L)
    xy_x = jnp.logical_xor.reduce(lx[:, 1:2, :], axis=0)   
    xy_y = jnp.logical_xor.reduce(ly[1:2, :, :], axis=1) 

    xz_x = jnp.logical_xor.reduce(lx[:, :, 1:2], axis=0)   
    xz_z = jnp.logical_xor.reduce(lz[1:2, :, :], axis=2)  

    yz_y = jnp.logical_xor.reduce(ly[:, :, 1:2], axis=1) 
    yz_z = jnp.logical_xor.reduce(lz[:, 1:2, :], axis=2)

    
    return (jnp.any(xy_x.squeeze()) | jnp.any(xy_y.squeeze()) |
            jnp.any(xz_x.squeeze()) | jnp.any(xz_z.squeeze()) |
            jnp.any(yz_y.squeeze()) | jnp.any(yz_z.squeeze()))


# --- State Initialization ---

@partial(jax.jit, static_argnames=['L', 'CLOCK_PERIOD'])
def init_state(key, L, p, CLOCK_PERIOD=CLOCK_PERIOD):
    """
    Initialize the state: links, s, m, c, step_count, CLOCK_PERIOD
    """
    subkey, _ = jax.random.split(key)
    links = jnp.zeros((3, L, L, L), dtype=bool)
    noise = jax.random.bernoulli(subkey, p, shape=(3, L, L, L))
    links = links ^ noise

    s = calculate_syndrome(links)
    m = jnp.zeros((2, 2, 2, L, L, L), dtype=bool)
    c = jnp.int32(0)
    step_count = 0

    return (links, s, m, c, step_count, CLOCK_PERIOD)


# --- Simulation (Monte Carlo) ---

@partial(jax.jit, static_argnames=['L', 'max_steps', 'CLOCK_PERIOD'])
def run_single_trajectory(key, L, p, max_steps, CLOCK_PERIOD=CLOCK_PERIOD):
    """
    Run one trajectory using while_loop for early termination.
    Returns (is_failure, step_count) where is_failure is 1.0 if logical error or not converged.
    """
    initial_state = init_state(key, L, p, CLOCK_PERIOD)

    def cond_fun(state):
        links, s, m, c, step, _ = state
        not_cleared = jnp.any(s) | jnp.any(m)
        under_limit = step < max_steps
        return not_cleared & under_limit

    final_state = jax.lax.while_loop(cond_fun, step_ca, initial_state)

    links_fin, s_fin, m_fin, _, steps, _ = final_state

    # Check convergence
    not_converged = jnp.any(s_fin) | jnp.any(m_fin)

    # Check logical error
    correction_failed = check_logical_error(links_fin)

    # Failure if logical error OR didn't converge
    is_failure = jnp.logical_or(correction_failed, not_converged).astype(jnp.float32)

    return is_failure, steps.astype(jnp.float32)


def run_monte_carlo(L_values, p_values, num_samples, filename="fracton_ca_newrule_data.npz"):
    """
    Run Monte Carlo simulation for multiple L and p values.
    Saves results incrementally to filename.
    """
    print(f"Running JAX Fracton (X-Cube) CA Decoder (New Rule) Evaluation...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")

    results = {}
    master_key = jax.random.PRNGKey(42)

    for L in L_values:
        max_steps = 20 * L

        @partial(jax.jit, static_argnames=['CLOCK_PERIOD'])
        def batch_run(keys, p_val, CLOCK_PERIOD=CLOCK_PERIOD):
            return jax.vmap(lambda k: run_single_trajectory(
                k, L, p_val, max_steps, CLOCK_PERIOD))(keys)

        for p in p_values:
            # Determine num_samples for this specific (L, p)
            if isinstance(num_samples, dict):
                if (L, p) in num_samples:
                    current_num_samples = num_samples[(L, p)]
                elif L in num_samples:
                    current_num_samples = num_samples[L]
                else:
                    current_num_samples = num_samples.get('default', 1000)
            else:
                current_num_samples = num_samples

            total_failures = 0.0
            total_success_steps = 0.0
            total_successes = 0.0
            samples_left = current_num_samples

            master_key, loop_key = jax.random.split(master_key)

            while samples_left > 0:
                current_batch_size = min(samples_left, MAX_BATCH_SIZE)

                loop_key, batch_subkey = jax.random.split(loop_key)
                keys = jax.random.split(batch_subkey, current_batch_size)

                batch_failures, batch_steps = batch_run(keys, p)
                total_failures += jnp.sum(batch_failures)

                success_mask = 1.0 - batch_failures
                total_successes += jnp.sum(success_mask)
                total_success_steps += jnp.sum(batch_steps * success_mask)

                samples_left -= current_batch_size
                print(f"    ... {samples_left} samples remaining (failures so far: {total_failures})", end='\r')

            print()
            ler = total_failures / current_num_samples

            if total_successes > 0:
                avg_steps = total_success_steps / total_successes
            else:
                avg_steps = 0.0

            ler_val = float(ler)
            avg_steps_val = float(avg_steps)
            results[(L, p)] = (ler_val, avg_steps_val, current_num_samples)
            print(f"  L={L}, p={p:.4f} -> LER={ler_val:.9f}, AvgSteps (success)={avg_steps_val:.2f}, Samples={current_num_samples}")

            # Save results immediately
            Ls_out = []
            ps_out = []
            lers_out = []
            avg_steps_out = []
            num_samples_out = []
            for (L_k, p_k), val in results.items():
                Ls_out.append(L_k)
                ps_out.append(p_k)
                lers_out.append(val[0])
                avg_steps_out.append(val[1])
                num_samples_out.append(val[2])

            np.savez(filename, L=np.array(Ls_out), p=np.array(ps_out),
                     ler=np.array(lers_out), avg_steps=np.array(avg_steps_out),
                     num_samples=np.array(num_samples_out))
            print(f"Data saved to {filename}")


# --- Plotting ---

def plot_results(data_filename, plot_filename="fracton_ca_newrule_results.png"):
    """Plot results from saved data file."""
    if not os.path.exists(data_filename):
        print(f"File {data_filename} not found.")
        return

    data = np.load(data_filename)
    Ls_raw = data['L']
    ps_raw = data['p']
    lers_raw = data['ler']

    results = {}
    for L, p, val in zip(Ls_raw, ps_raw, lers_raw):
        results[(L, p)] = val

    unique_L = sorted(list(set(Ls_raw)))
    unique_p = sorted(list(set(ps_raw)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: p vs LER for different L
    for L in unique_L:
        curr_lers = []
        curr_ps = []
        for p in unique_p:
            if (L, p) in results:
                curr_lers.append(results[(L, p)])
                curr_ps.append(p)
        if curr_lers:
            ax1.plot(curr_ps, curr_lers, 'o-', label=f'L={L}', markersize=6)

    ax1.set_xlabel('Physical Error Rate (p)')
    ax1.set_ylabel('Logical Error Rate')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.4)
    ax1.legend()
    ax1.set_title('Fracton (X-Cube) CA Decoder (New Rule) Threshold')

    # Plot 2: L vs LER for different p
    for p in unique_p:
        curr_lers = []
        curr_Ls = []
        for L in unique_L:
            if (L, p) in results:
                curr_lers.append(results[(L, p)])
                curr_Ls.append(L)
        if curr_lers:
            ax2.plot(curr_Ls, curr_lers, 'x--', label=f'p={p:.4f}', markersize=8)

    ax2.set_xlabel('Code Distance (L)')
    ax2.set_ylabel('Logical Error Rate')
    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.4)
    ax2.legend()
    ax2.set_title('Scaling with L')

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    print(f"Plot saved to {plot_filename}")
    plt.close(fig)


if __name__ == "__main__":
    run_monte_carlo(
        L_values=[8, 12, 16],
        p_values=np.linspace(0.001, 0.005, 7),
        num_samples=10_000,
        filename="xcube_fracton_ca_results.npz",
    )
    plot_results("xcube_fracton_ca_results.npz", "xcube_fracton_ca_results.png")
