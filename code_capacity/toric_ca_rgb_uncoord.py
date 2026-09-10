import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
import traceback

# --- Core CA Logic (JAX) ---

DEFAULT_CLOCK_PERIOD = 6
MAX_BATCH_SIZE = 2_000

@partial(jax.jit, static_argnames=['L', 'CLOCK_PERIOD'])
def init_state(key, L, p, CLOCK_PERIOD=DEFAULT_CLOCK_PERIOD):
    """
    Initialize state for Toric Code with RGB CA (uncoordinated).
    Returns: (h_qubits, v_qubits, b_grid, r_grid, g_grid, c, step_count, CLOCK_PERIOD, key)
    All grids are (L, L) boolean or int arrays.
    """
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Qubits store the X-error state
    h_qubits = jax.random.bernoulli(key1, p, shape=(L, L)).astype(jnp.bool_)
    v_qubits = jax.random.bernoulli(key2, p, shape=(L, L)).astype(jnp.bool_)
    
    # Message grids
    b_grid = jnp.zeros((L, L), dtype=jnp.bool_)  # Blue
    r_grid = jnp.zeros((L, L), dtype=jnp.bool_)  # Red
    g_grid = jnp.zeros((L, L), dtype=jnp.bool_)  # Green
    c_grid = jnp.zeros((L, L), dtype=jnp.int32)  # Per-site clock
    
    step_count = jnp.array(0, dtype=jnp.int32)
    
    return (h_qubits, v_qubits, b_grid, r_grid, g_grid, c_grid, step_count, CLOCK_PERIOD, key3)


@jax.jit
def calculate_syndrome(h_qubits, v_qubits):
    """
    Calculate syndrome (Z-checks on plaquettes).
    s(x,y) = v(x,y) XOR v(x+1,y) XOR h(x,y) XOR h(x,y+1) mod 2
    
    ROLL CONVENTION (matching toric_ca_shear.py):
    - x+1: roll -1 on axis 0
    - y+1: roll -1 on axis 1
    """
    v_right = jnp.roll(v_qubits, -1, axis=0)  # v(x+1, y)
    h_top = jnp.roll(h_qubits, -1, axis=1)    # h(x, y+1)
    syndrome = v_qubits ^ v_right ^ h_qubits ^ h_top
    return syndrome




# --- Single-site uncoordinated update ---

@jax.jit
def _single_site_update_step(state):
    """
    Single step of the RGB Toric CA: pick ONE random site (ix, iy) and only
    update that site's b, r, g, and the qubit corrections it triggers.
    state: (h_qubits, v_qubits, b_grid, r_grid, g_grid, c, step_count, CLOCK_PERIOD, key)
    """
    h_qubits, v_qubits, b, r, g, c, step_count, CLOCK_PERIOD, key = state
    L = h_qubits.shape[0]

    # --- Random site selection ---
    key, k1, k2 = jax.random.split(key, 3)
    ix = jax.random.randint(k1, shape=(), minval=0, maxval=L)
    iy = jax.random.randint(k2, shape=(), minval=0, maxval=L)

    # --- Helpers for periodic 2D indexing ---
    def get(arr, x, y):
        return arr[x % L, y % L]

    def local_s(hq, vq, x, y):
        # s(x,y) = v(x,y) ^ v(x+1,y) ^ h(x,y) ^ h(x,y+1)
        return get(vq, x, y) ^ get(vq, x+1, y) ^ get(hq, x, y) ^ get(hq, x, y+1)

    # --- Gather local neighborhood values ---
    s_xy     = local_s(h_qubits, v_qubits, ix, iy)
    s_right  = local_s(h_qubits, v_qubits, ix+1, iy)
    s_up     = local_s(h_qubits, v_qubits, ix, iy+1)

    b_xy     = get(b, ix, iy)
    b_left   = get(b, ix-1, iy)
    b_down   = get(b, ix, iy-1)
    b_left_up = get(b, ix-1, iy+1)

    r_xy     = get(r, ix, iy)
    r_left   = get(r, ix-1, iy)
    r_up     = get(r, ix, iy+1)
    r_left_up = get(r, ix-1, iy+1)

    g_xy     = get(g, ix, iy)
    g_right  = get(g, ix+1, iy)
    g_down   = get(g, ix, iy-1)

    mask_s   = s_xy.astype(jnp.bool_)
    mask_no_s = ~mask_s

    # --- Compute next_b, next_r, next_g for site (ix, iy) ---
    next_b_val = b_xy
    next_r_val = r_xy
    next_g_val = g_xy

    # 1. If s(ix,iy) == 1 -> set all to True
    next_b_val = jnp.where(mask_s, True, next_b_val)
    next_r_val = jnp.where(mask_s, True, next_r_val)
    next_g_val = jnp.where(mask_s, True, next_g_val)

    # Corrections (only when s==1)
    cond_left = b_left | r_left
    do_left = mask_s & cond_left
    cond_down = (g_down | b_down) 
    do_down = mask_s & (~do_left) & cond_down

    # 2. Else (s==0):

    # --- Incoming syndrome (is) ---
    # In a strict single-site update, only this site's messages are changed.
    # These predicates detect a neighboring defect enabled to move here.
    ils = s_right.astype(jnp.bool_) & (b_xy | r_xy)
    ids = s_up.astype(jnp.bool_) & (g_xy | b_xy) & (~r_left_up) & (~b_left_up)
    is_flag = ils | ids

    mask_is = mask_no_s & is_flag
    next_b_val = jnp.where(mask_is, True, next_b_val)
    next_r_val = jnp.where(mask_is, True, next_r_val)
    next_g_val = jnp.where(mask_is, True, next_g_val)

    # --- else (no incoming syndrome) ---
    mask_else = mask_no_s & (~is_flag)

    # Use the LOCAL clock value at this site
    c_local = get(c, ix, iy)
    is_c0 = (c_local == 0)
    mask_growth = mask_else & is_c0

    # b growth: b==0 -> b_down | b_left
    is_b0 = mask_growth & (~b_xy)
    b_grow = b_down | b_left
    next_b_val = jnp.where(is_b0, b_grow, next_b_val)

    # r growth: r==0 -> r_up | r_left
    is_r0 = mask_growth & (~r_xy)
    r_grow = r_up | r_left
    next_r_val = jnp.where(is_r0, r_grow, next_r_val)

    # g growth: g==0 -> g_down | g_right
    is_g0 = mask_growth & (~g_xy)
    g_grow = g_down | g_right
    next_g_val = jnp.where(is_g0, g_grow, next_g_val)

    # --- Majority ---
    sum_b = b_left.astype(jnp.int32) + b_xy.astype(jnp.int32) + b_down.astype(jnp.int32)
    bm = (sum_b >= 2)

    sum_r = r_left.astype(jnp.int32) + r_xy.astype(jnp.int32) + r_up.astype(jnp.int32)
    rm = (sum_r >= 2)

    sum_g = g_right.astype(jnp.int32) + g_xy.astype(jnp.int32) + g_down.astype(jnp.int32)
    gm = (sum_g >= 2)

    # b==1 -> bm
    is_b1 = mask_else & b_xy
    next_b_val = jnp.where(is_b1, bm, next_b_val)

    # r==1 -> rm | (bm & b)
    is_r1 = mask_else & r_xy
    next_r_val = jnp.where(is_r1, rm | (bm & b_xy), next_r_val)

    # g==1 -> gm | (bm & b)
    is_g1 = mask_else & g_xy
    next_g_val = jnp.where(is_g1, gm | (bm & b_xy), next_g_val)

    # --- Apply updates ---
    new_b = b.at[ix, iy].set(next_b_val.astype(jnp.bool_))
    new_r = r.at[ix, iy].set(next_r_val.astype(jnp.bool_))
    new_g = g.at[ix, iy].set(next_g_val.astype(jnp.bool_))

    # Apply qubit corrections
    # do_left -> flip v(ix, iy)
    v_idx_x = ix % L
    v_idx_y = iy % L
    new_v = v_qubits.at[v_idx_x, v_idx_y].set(
        v_qubits[v_idx_x, v_idx_y] ^ do_left
    )

    # do_down -> flip h(ix, iy)
    h_idx_x = ix % L
    h_idx_y = iy % L
    new_h = h_qubits.at[h_idx_x, h_idx_y].set(
        h_qubits[h_idx_x, h_idx_y] ^ do_down
    )

    # Increment the local clock at this site
    new_c = c.at[ix % L, iy % L].set((c_local + 1) % CLOCK_PERIOD)

    return (new_h, new_v, new_b, new_r, new_g, new_c, step_count + 1, CLOCK_PERIOD, key)


@jax.jit
def step_sweep(state):
    """
    Perform L*L single-site updates (one sweep).
    Clock is advanced per-site inside _single_site_update_step.
    This is the body function for while_loop in MC simulation.
    """
    h, v, b, r, g, c, step_count, CP, key = state
    L = h.shape[0]

    def scan_body(carry, _):
        next_carry = _single_site_update_step(carry)
        return next_carry, None

    final_sweep, _ = jax.lax.scan(scan_body, state, None, length=L * L)
    return final_sweep


@jax.jit
def check_logical_error(h_qubits, v_qubits):
    """Check for logical error (non-trivial loops)."""
    winding_v = jnp.sum(v_qubits[0, :]) % 2
    winding_h = jnp.sum(h_qubits[:, 0]) % 2
    return (winding_v != 0) | (winding_h != 0)


# --- Simulation (Monte Carlo) ---

@partial(jax.jit, static_argnames=['L', 'max_sweeps', 'CLOCK_PERIOD'])
def run_single_trajectory(key, L, p, max_sweeps, CLOCK_PERIOD=DEFAULT_CLOCK_PERIOD):
    """
    Run one trajectory using while_loop for early termination.
    Convergence is checked after every single-site update.
    Returns (is_failure, step_count) where is_failure is 1.0 if logical error or not converged.
    """
    initial_state = init_state(key, L, p, CLOCK_PERIOD)
    
    def cond_fun(state):
        h_qubits, v_qubits, b_grid, r_grid, g_grid, c, step_count, CP, _ = state
        syndrome = calculate_syndrome(h_qubits, v_qubits)
        not_cleared = jnp.any(syndrome)
        # step_count tracks individual updates; max_sweeps * L * L is total budget
        under_step_limit = step_count < (max_sweeps * L * L)
        return not_cleared & under_step_limit
    
    final_state = jax.lax.while_loop(cond_fun, _single_site_update_step, initial_state)
    
    h_qubits, v_qubits, b_grid, r_grid, g_grid, c, step_count, CP, _ = final_state
    
    # Check 1: Did we converge?
    final_syndrome = calculate_syndrome(h_qubits, v_qubits)
    converged = jnp.all(final_syndrome == 0)
    
    # Check 2: Logical error?
    logical_error = check_logical_error(h_qubits, v_qubits)
    
    # Failure condition: Not converged OR (Converged AND Logical Error)
    is_failure = (~converged) | logical_error
    
    return is_failure.astype(jnp.float32), step_count.astype(jnp.float32)


def run_monte_carlo(L_values, p_values, num_samples, filename="toric_ca_data.npz"):
    """
    Run Monte Carlo simulation for multiple L and p values.
    Saves results incrementally to filename.
    """
    print(f"Running JAX Toric CA RGB Uncoordinated Decoder Evaluation...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")
    
    results = {}
    master_key = jax.random.PRNGKey(20260921)
    
    for L in L_values:
        CLOCK_PERIOD = DEFAULT_CLOCK_PERIOD
        max_sweeps = 5*CLOCK_PERIOD * L
        
        @partial(jax.jit, static_argnames=['CLOCK_PERIOD'])
        def batch_run(keys, p_val, CLOCK_PERIOD=DEFAULT_CLOCK_PERIOD):
            return jax.vmap(lambda k: run_single_trajectory(k, L, p_val, max_sweeps, CLOCK_PERIOD))(keys)
        
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

def plot_results(data_filename, plot_filename="toric_ca_results.png"):
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
    ax1.set_title('Toric CA RGB Uncoord Decoder Threshold')
    
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


# --- Visualization (using jax.lax.scan) ---

def save_simulation_gif(L, p, steps, filename="toric_ca_shear.gif", channels=None, initial_errors=None):
    """
    Generate a GIF of the CA simulation using jax.lax.scan for history.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Patch, Wedge, Circle
    from matplotlib.lines import Line2D
    
    if channels is None:
        channels = ['b', 'r', 'g']
    
    show_b = 'b' in channels
    show_r = 'r' in channels
    show_g = 'g' in channels
    
    # Initialize state
    key = jax.random.PRNGKey(np.random.randint(0, 10000))
    
    if initial_errors is not None:
        h_err, v_err = initial_errors
        h_qubits = jnp.array(h_err, dtype=jnp.bool_)
        v_qubits = jnp.array(v_err, dtype=jnp.bool_)
        b_grid = jnp.zeros((L, L), dtype=jnp.bool_)
        r_grid = jnp.zeros((L, L), dtype=jnp.bool_)
        g_grid = jnp.zeros((L, L), dtype=jnp.bool_)
        c_grid = jnp.zeros((L, L), dtype=jnp.int32)
        step_count = jnp.array(0, dtype=jnp.int32)
        initial_state = (h_qubits, v_qubits, b_grid, r_grid, g_grid, c_grid, step_count, DEFAULT_CLOCK_PERIOD, key)
    else:
        initial_state = init_state(key, L, p, DEFAULT_CLOCK_PERIOD)
    
    # Scan step with done flag for early stopping visual
    def scan_step(carry, _):
        h, v, b, r, g, c, step_count, CP, rng_key, done = carry
        
        # Real step
        h_next, v_next, b_next, r_next, g_next, c_next, _, _, rng_key_next = step_ca((h, v, b, r, g, c, step_count, CP, rng_key))
        
        syndrome = calculate_syndrome(h, v)
        is_clean = jnp.all(syndrome == 0) & jnp.all(b == 0) & jnp.all(r == 0) & jnp.all(g == 0)
        
        new_done = done | is_clean
        
        # If already done, freeze state
        h_final = jnp.where(done, h, h_next)
        v_final = jnp.where(done, v, v_next)
        b_final = jnp.where(done, b, b_next)
        r_final = jnp.where(done, r, r_next)
        g_final = jnp.where(done, g, g_next)
        c_final = jnp.where(done, c, c_next)
        
        new_carry = (h_final, v_final, b_final, r_final, g_final, c_final, step_count + 1, CP, rng_key_next, new_done)
        
        # Output: syndrome, b, r, g, c
        output_s = calculate_syndrome(h_final, v_final)
        output = (output_s, b_final, r_final, g_final, c_final)
        return new_carry, output
    
    # Initial carry with done flag
    h0, v0, b0, r0, g0, c0, sc0, CP0, key0 = initial_state
    s0 = calculate_syndrome(h0, v0)
    done0 = jnp.all(s0 == 0) & jnp.all(b0 == 0) & jnp.all(r0 == 0) & jnp.all(g0 == 0)
    
    carry_init = (h0, v0, b0, r0, g0, c0, sc0, CP0, key0, done0)
    
    final_carry, history = jax.lax.scan(scan_step, carry_init, None, length=steps)
    
    # Extract history
    s_hist, b_hist, r_hist, g_hist, c_hist = history
    s_hist = np.array(s_hist)
    b_hist = np.array(b_hist)
    r_hist = np.array(r_hist)
    g_hist = np.array(g_hist)
    c_hist = np.array(c_hist)
    
    # Prepend initial state
    s0_np = np.array(s0)
    b0_np = np.array(b0)
    r0_np = np.array(r0)
    g0_np = np.array(g0)
    c0_np = np.array(c0)
    
    full_s = np.concatenate([[s0_np], s_hist], axis=0)
    full_b = np.concatenate([[b0_np], b_hist], axis=0)
    full_r = np.concatenate([[r0_np], r_hist], axis=0)
    full_g = np.concatenate([[g0_np], g_hist], axis=0)
    full_c = np.concatenate([[c0_np], c_hist], axis=0)
    
    # Final state for logical error check
    h_final, v_final = np.array(final_carry[0]), np.array(final_carry[1])
    is_logical_fail = bool(check_logical_error(jnp.array(h_final), jnp.array(v_final)))
    
    # --- Animation ---
    fig, ax = plt.subplots(figsize=(8, 8))
    
    color_b = (0.7, 0.7, 1.0)
    color_r = (1.0, 0.7, 0.7)
    color_g = (0.7, 1.0, 0.7)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Defect', markerfacecolor='black', markersize=10),
    ]
    if show_b:
        legend_elements.append(Patch(facecolor=color_b, label='Blue (b)'))
    if show_r:
        legend_elements.append(Patch(facecolor=color_r, label='Red (r)'))
    if show_g:
        legend_elements.append(Patch(facecolor=color_g, label='Green (g)'))
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_xticks(np.arange(L + 1))
        ax.set_yticks(np.arange(L + 1))
        ax.grid(True, color='gray', linestyle='-', linewidth=1.0, alpha=0.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax.set_aspect('equal')
        
        s = full_s[frame]
        b_data = full_b[frame]
        r_data = full_r[frame]
        g_data = full_g[frame]
        clock_state = full_c[frame]
        
        # Draw b, r, g
        for x in range(L):
            for y in range(L):
                has_b = show_b and b_data[x, y] == 1
                has_r = show_r and r_data[x, y] == 1
                has_g = show_g and g_data[x, y] == 1
                
                active_colors = []
                if has_b: active_colors.append(color_b)
                if has_r: active_colors.append(color_r)
                if has_g: active_colors.append(color_g)
                
                if not active_colors:
                    continue
                
                n_colors = len(active_colors)
                if n_colors == 1:
                    circ = Circle((x + 0.5, y + 0.5), 0.45, facecolor=active_colors[0], edgecolor='none', alpha=0.7)
                    ax.add_patch(circ)
                else:
                    # Split circle into wedges
                    angle_per_wedge = 360 / n_colors
                    start_angle = 90
                    for i, color in enumerate(active_colors):
                        theta1 = start_angle + i * angle_per_wedge
                        theta2 = start_angle + (i + 1) * angle_per_wedge
                        w = Wedge((x + 0.5, y + 0.5), 0.45, theta1, theta2, facecolor=color, edgecolor='none', alpha=0.7)
                        ax.add_patch(w)
        
        syn_x, syn_y = np.where(s == 1)
        if len(syn_x) > 0:
            ax.scatter(syn_x + 0.5, syn_y + 0.5, c='black', s=80, zorder=10)
        
        ax.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize='small', framealpha=0.95)
        
        defects = np.sum(s)
        status_text = f"T={frame} | Defects: {defects}"
        
        if defects == 0 and np.sum(b_data) == 0 and np.sum(r_data) == 0 and np.sum(g_data) == 0:
            if is_logical_fail:
                status_text += " [LOGICAL ERROR]"
                ax.text(0.5, 0.5, "LOGICAL ERROR", transform=ax.transAxes,
                        ha='center', va='center', color='red', fontsize=20, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
            else:
                status_text += " [SUCCESS]"
                ax.text(0.5, 0.5, "SUCCESS", transform=ax.transAxes,
                        ha='center', va='center', color='green', fontsize=20, weight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))
        
        ax.set_title(status_text)
    
    # Trim frames after first completion
    clean_indices = [i for i in range(len(full_s)) 
                     if np.all(full_s[i] == 0) and np.all(full_b[i] == 0) and np.all(full_r[i] == 0) and np.all(full_g[i] == 0)]
    if clean_indices:
        cutoff = clean_indices[0] + 1
        full_s = full_s[:cutoff]
        full_b = full_b[:cutoff]
        full_r = full_r[:cutoff]
        full_g = full_g[:cutoff]
        full_c = full_c[:cutoff]
    
    T = len(full_s)
    
    anim = FuncAnimation(fig, update, frames=T, repeat=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    try:
        anim.save(filename, writer=PillowWriter(fps=2))
        print(f"GIF saved to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        traceback.print_exc()
    plt.close(fig)


if __name__ == "__main__":
    # Example: Generate GIF
    L = 9
    h_err = np.zeros((L, L))
    v_err = np.zeros((L, L))
    h_err[2, 2] = 1
    v_err[3,2]=1
    h_err[3,3]=1

    # save_simulation_gif(L, 0.0, 20, "toric_ca_rgb_uncoord.gif", ['b', 'r', 'g'], initial_errors=(h_err, v_err))
    
    # Example: Run Monte Carlo
    run_monte_carlo(
        L_values= [7, 15, 21],
        p_values=[5e-3, 1e-2, 2e-2, 3e-2, 4e-2,],
        num_samples={                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
            'default': 10_000,
        },
        filename="tc_rgb_uncoord.npz"
    )
    
    # # # # Example: Plot results
    plot_results("tc_rgb_uncoord.npz", "tc_rgb_uncoord.png")
