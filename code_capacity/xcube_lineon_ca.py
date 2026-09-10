import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
import time

# --- Constants ---

CLOCK_PERIOD = 10
MAX_BATCH_SIZE = 50_000


# --- Syndrome Evaluation ---

@jax.jit
def calculate_syndrome(links):
    """
    Compute X-cube lineon sector vertex stabilizers on an L×L×L cubic lattice.

    Each vertex has 3 stabilizers (A_xy, A_xz, A_yz) — product of Z on 4 links in that plane.
    These are Z-type stabilizers: X errors (bit flips) on links violate them.

    Convention: ell_1 = x-lineon (A_yz violated), ell_2 = y-lineon (A_xz), ell_3 = z-lineon (A_xy).
    Single-third reduction: if two raw violations, represent as the third only.

    links : (3, L, L, L) bool  ->  (s, ell_1, ell_2, ell_3, raw_ell1, raw_ell2, raw_ell3)
    """
    lx, ly, lz = links[0], links[1], links[2]
    lx_m1 = jnp.roll(lx, 1, axis=0)
    ly_m1 = jnp.roll(ly, 1, axis=1)
    lz_m1 = jnp.roll(lz, 1, axis=2)

    raw_ell3 = lx ^ lx_m1 ^ ly ^ ly_m1
    raw_ell2 = lx ^ lx_m1 ^ lz ^ lz_m1
    raw_ell1 = ly ^ ly_m1 ^ lz ^ lz_m1

    pair_12 = raw_ell1 & raw_ell2
    pair_13 = raw_ell1 & raw_ell3
    pair_23 = raw_ell2 & raw_ell3

    ell_1 = pair_23
    ell_2 = pair_13
    ell_3 = pair_12

    s = ell_1 | ell_2 | ell_3
    return s, ell_1, ell_2, ell_3


# --- CA Step ---

@jax.jit
def step_ca(state):
    """
    One synchronous CA step for the X-cube lineon sector.

    Implements the algorithm:
      - Syndrome sites: set all m=1, then PrimaryThenFallback with DoMove.
      - Non-syndrome sites: spreading (c=0, m=0) then Toom-3D vote.
      - S_in override: force all m=1 at move targets.
      - Counter update.

    state: (links, s, ell_1, ell_2, ell_3, m, c, step_count, CLOCK_PERIOD)
    """
    links, s, ell_1, ell_2, ell_3, m, c, step_count, CLOCK_PERIOD = state
    q = CLOCK_PERIOD

    syn = s
    no_syn = ~s
    clock0 = (c == 0)

    # ── Pre-compute neighbor shifts for m  (2,2,2,L,L,L) ─────────────
    m_xp1 = jnp.roll(m, -1, axis=3)   # m(..., x+1, y, z)
    m_xm1 = jnp.roll(m,  1, axis=3)   # m(..., x-1, y, z)
    m_yp1 = jnp.roll(m, -1, axis=4)
    m_ym1 = jnp.roll(m,  1, axis=4)
    m_zp1 = jnp.roll(m, -1, axis=5)
    m_zm1 = jnp.roll(m,  1, axis=5)

    # m_nb_[axis][i,j,k] = m_{ijk}(r + (-1)^i * e_x)  etc.
    i_idx = jnp.arange(2)[:, None, None, None, None, None]
    j_idx = jnp.arange(2)[None, :, None, None, None, None]
    k_idx = jnp.arange(2)[None, None, :, None, None, None]
    m_nb_x = jnp.where(i_idx == 0, m_xp1, m_xm1)
    m_nb_y = jnp.where(j_idx == 0, m_yp1, m_ym1)
    m_nb_z = jnp.where(k_idx == 0, m_zp1, m_zm1)

    # ── Step 1: m_new ← m  (default: copy old values) ────────────────
    m_new = m

    # ══════════════════════════════════════════════════════════════════
    # SYNDROME SITES  (s(r,t) = 1)
    # ══════════════════════════════════════════════════════════════════

    # Set all m_ijk(r,t+1) = 1
    syn_bc = syn[None, None, None, :, :, :]
    m_new = jnp.where(syn_bc, True, m_new)

    # ── Trigger conditions for PrimaryThenFallback ────────────────────
    # a_trig(a, r) = ∃ α with α_a=1 s.t. m_α(r − α_a·ê_a, t) = 1
    #   For a=1 (x): α_1=1 ⇒ shift by -ê_x. The 4 channels with i=1:
    #     m[1,0,0], m[1,0,1], m[1,1,0], m[1,1,1]  all evaluated at (r-ê_x).
    x_trig = (m_xm1[1, 0, 0] | m_xm1[1, 0, 1] |
              m_xm1[1, 1, 0] | m_xm1[1, 1, 1])
    y_trig = (m_ym1[0, 1, 0] | m_ym1[0, 1, 1] |
              m_ym1[1, 1, 0] | m_ym1[1, 1, 1])
    z_trig = (m_zm1[0, 0, 1] | m_zm1[0, 1, 1] |
              m_zm1[1, 0, 1] | m_zm1[1, 1, 1])

    # ── PrimaryThenFallback(p, r, t) ──────────────────────────────────
    #   Primary fires:   p_trig ⇒ DoMove(p, r)
    #   Fallback (else):   for each a ∈ {1,2,3}\{p} independently:
    #                        a_trig ⇒ DoMove(a, r)
    #
    # Condition chain (if / elsif / elsif):
    #   ell_1 (= raw_ℓ₂ ∧ raw_ℓ₃)  →  PTF(1)  [primary = x]
    #   ell_2 (= raw_ℓ₁ ∧ raw_ℓ₃)  →  PTF(2)  [primary = y]
    #   ell_3 (= raw_ℓ₁ ∧ raw_ℓ₂)  →  PTF(3)  [primary = z]

    # PTF(1): primary = x;  fallback axes = {y, z} independently
    ptf1 = syn & ell_1
    do_x_ptf1 = ptf1 & x_trig
    do_y_ptf1 = ptf1 & ~x_trig & y_trig       # fallback y (independent)
    do_z_ptf1 = ptf1 & ~x_trig & z_trig       # fallback z (independent)

    # PTF(2): primary = y;  fallback axes = {x, z} independently
    #   (elsif: only fires when PTF(1) did not)
    ptf2 = syn & ell_2 & ~ell_1
    do_y_ptf2 = ptf2 & y_trig
    do_x_ptf2 = ptf2 & ~y_trig & x_trig       # fallback x
    do_z_ptf2 = ptf2 & ~y_trig & z_trig       # fallback z

    # PTF(3): primary = z;  fallback axes = {x, y} independently
    #   (elsif: only fires when neither PTF(1) nor PTF(2) fired)
    ptf3 = syn & ell_3 & ~ell_1 & ~ell_2
    do_z_ptf3 = ptf3 & z_trig
    do_x_ptf3 = ptf3 & ~z_trig & x_trig       # fallback x
    do_y_ptf3 = ptf3 & ~z_trig & y_trig       # fallback y

    # Combine per-axis DoMove flags across all PTF branches
    do_x = do_x_ptf1 | do_x_ptf2 | do_x_ptf3
    do_y = do_y_ptf1 | do_y_ptf2 | do_y_ptf3
    do_z = do_z_ptf1 | do_z_ptf2 | do_z_ptf3

    # DoMove(a, r): flip the link incident to r in the −x_a direction.
    #   The link connecting r to r−ê_a lives at position r−ê_a in the
    #   link array  ⟹  roll(do_a, −1, axis=a).
    #   S_in ← S_in ∪ {r − ê_a}.
    flip_x = jnp.roll(do_x, -1, axis=0)
    flip_y = jnp.roll(do_y, -1, axis=1)
    flip_z = jnp.roll(do_z, -1, axis=2)

    # ══════════════════════════════════════════════════════════════════
    # S_IN:  for each r' = r − ê_a where DoMove(a) fired at r,
    #        set all m_{ijk}(r', t+1) = 1
    # ══════════════════════════════════════════════════════════════════
    s_in = flip_x | flip_y | flip_z
    in_bc = s_in[None, None, None, :, :, :]
    m_new = jnp.where(in_bc, True, m_new)

    # ══════════════════════════════════════════════════════════════════
    # NON-SYNDROME SITES  (s(r,t) = 0)
    # ══════════════════════════════════════════════════════════════════
    rest = no_syn & ~s_in
    rest_bc = rest[None, None, None, :, :, :]

    # ── Spreading  (m_ijk = 0):
    #    m_ijk(r,t+1) = (nb_x OR nb_y OR nb_z) ∧ (c=0) ────────────────
    neighbor_or = m_nb_x | m_nb_y | m_nb_z
    m_zero = ~m
    spread_mask = rest_bc & m_zero & clock0 & neighbor_or
    m_new = jnp.where(spread_mask, True, m_new)

    # ── Toom-3D vote ──────────────────────────────────────────────────
    #   v_{ijk} = 𝟙[ m_{ijk}(r) + m_{ijk}(r+(-1)^i ex)
    #                            + m_{ijk}(r+(-1)^j ey)
    #                            + m_{ijk}(r+(-1)^k ez)  ≥ 2 ]
    total = (m.astype(jnp.int32) +
             m_nb_x.astype(jnp.int32) +
             m_nb_y.astype(jnp.int32) +
             m_nb_z.astype(jnp.int32))
    v_all = (total >= 2)

    # ── (1,1,1) channel first ─────────────────────────────────────────
    m111_old = m[1, 1, 1]                       # (L,L,L) bool
    v111 = v_all[1, 1, 1]                       # reuse pre-computed vote
    m111_new = jnp.where(rest,
                         jnp.where(m111_old, v111, m_new[1, 1, 1]),
                         m_new[1, 1, 1])
    m_new = m_new.at[1, 1, 1].set(m111_new)

    # ── Other (i,j,k) ≠ (1,1,1)  with  m_{ijk}=1 ────────────────────
    #    m_ijk(r,t+1) = v_ijk  ∨  (m_{111}(r,t) ∧ v_{111})
    coupled = m111_old & v111
    boost = coupled[None, None, None, :, :, :]
    exclude_111 = jnp.array([[[True, True],
                              [True, True]],
                             [[True, True],
                              [True, False]]], dtype=bool)
    exclude_111_bc = exclude_111[:, :, :, None, None, None]
    other_mask = rest_bc & exclude_111_bc & m
    m_new = jnp.where(other_mask, v_all | boost, m_new)

    # ══════════════════════════════════════════════════════════════════
    # APPLY LINK FLIPS & RECOMPUTE SYNDROMES
    # ══════════════════════════════════════════════════════════════════
    links_new = jnp.stack([links[0] ^ flip_x,
                           links[1] ^ flip_y,
                           links[2] ^ flip_z])
    out = calculate_syndrome(links_new)
    s_new, ell_1_new, ell_2_new, ell_3_new = out[0], out[1], out[2], out[3]

    # Counter update
    c_new = (c + 1) % q

    # ══════════════════════════════════════════════════════════════════
    # Final S_in override (unconditional, per algorithm last block)
    # ══════════════════════════════════════════════════════════════════
    m_new = jnp.where(in_bc, True, m_new)

    return (links_new, s_new, ell_1_new, ell_2_new, ell_3_new,
            m_new, c_new, step_count + 1, CLOCK_PERIOD)


# --- Logical Error Check ---
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
    xy_x = jnp.logical_xor.reduce(lx[1:2, :, :], axis=1)   
    xy_y = jnp.logical_xor.reduce(ly[:, 1:2, :], axis=0) 

    xz_x = jnp.logical_xor.reduce(lx[1:2, :, :], axis=2)   
    xz_z = jnp.logical_xor.reduce(lz[:, :, 1:2], axis=0)  

    yz_y = jnp.logical_xor.reduce(ly[:, 1:2, :], axis=2) 
    yz_z = jnp.logical_xor.reduce(lz[:, :, 1:2], axis=1)

    
    return (jnp.any(xy_x.squeeze()) | jnp.any(xy_y.squeeze()) |
            jnp.any(xz_x.squeeze()) | jnp.any(xz_z.squeeze()) |
            jnp.any(yz_y.squeeze()) | jnp.any(yz_z.squeeze()))


# --- State Initialization ---

@partial(jax.jit, static_argnames=['L', 'CLOCK_PERIOD'])
def init_state(key, L, p, CLOCK_PERIOD=CLOCK_PERIOD):
    """
    Initialize the state: links, s, raw_1, raw_2, raw_3, m, c, step_count, CLOCK_PERIOD
    """
    subkey, _ = jax.random.split(key)
    links = jnp.zeros((3, L, L, L), dtype=bool)
    noise = jax.random.bernoulli(subkey, p, shape=(3, L, L, L))
    links = links ^ noise

    out = calculate_syndrome(links)
    s = out[0]
    ell_1, ell_2, ell_3 = out[1], out[2], out[3]
    m = jnp.zeros((2, 2, 2, L, L, L), dtype=bool)
    c = jnp.int32(0)
    step_count = 0

    return (links, s, ell_1, ell_2, ell_3, m, c, step_count, CLOCK_PERIOD)


# --- Simulation (Monte Carlo) ---

@partial(jax.jit, static_argnames=['L', 'max_steps', 'CLOCK_PERIOD'])
def run_single_trajectory(key, L, p, max_steps, CLOCK_PERIOD=CLOCK_PERIOD):
    """
    Run one trajectory using while_loop for early termination.
    Returns (is_failure, step_count) where is_failure is 1.0 if logical error or not converged.
    """
    initial_state = init_state(key, L, p, CLOCK_PERIOD)

    def cond_fun(state):
        links, s, ell_1, ell_2, ell_3, m, c, step, _ = state
        not_cleared = jnp.any(s) | jnp.any(m)
        under_limit = step < max_steps
        return not_cleared & under_limit

    final_state = jax.lax.while_loop(cond_fun, step_ca, initial_state)

    links_fin, s_fin, _, _, _, m_fin, _, steps, _ = final_state

    # Check convergence
    not_converged = jnp.any(s_fin) | jnp.any(m_fin)

    # Check logical error
    correction_failed = check_logical_error(links_fin)

    # Failure if logical error OR didn't converge
    is_failure = jnp.logical_or(correction_failed, not_converged).astype(jnp.float32)

    return is_failure, steps.astype(jnp.float32)


def run_monte_carlo(L_values, p_values, num_samples, filename="lineon_ca_data.npz"):
    """
    Run Monte Carlo simulation for multiple L and p values.
    Saves results incrementally to filename.
    """
    print(f"Running JAX Lineon CA Decoder Evaluation...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")

    results = {}
    master_key = jax.random.PRNGKey(4322)

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

def plot_results(data_filename, plot_filename="lineon_ca_results.png"):
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
    ax1.set_title('Lineon CA Decoder Threshold')

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
    # run_monte_carlo(
    #     L_values=[5, 9, 13, 17, 21, 25],
    #     p_values=[0.007, 0.01, 0.015, 0.02, 0.025, 0.03],
    #     num_samples={
    #         'default': 1_000_000,
    #         (17, 0.007): 75_000_000,

    #     },
    #     filename="lineon_ca_results_no_wait.npz",
    # )

    run_monte_carlo(
        L_values=[4, 7, 13, 23, 43, 78, 141, 256],
        p_values=[0.001],
        num_samples={
            'default': 1_000_000,
            (9, 0.001): 100_000_000,
        },
        filename="lineon_ca_results_no_wait_9_001.npz",
    )

    # run_monte_carlo(
    #     L_values=np.floor(2**np.linspace(2, 8, 8)).astype(int),
    #     p_values=[0.007, 0.011, 0.014],
    #     num_samples={
    #         'default': 5_000,
    #     },
    #     filename="lineon_ca_avg_T.npz",
    # )
    # plot_results("lineon_ca_results.npz", "lineon_ca_results.png")
