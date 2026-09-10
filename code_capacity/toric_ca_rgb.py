import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os

# --- Core CA Logic (JAX) ---

CLOCK_PERIOD = 6
MAX_BATCH_SIZE = 2000

@partial(jax.jit, static_argnames=['L', 'CLOCK_PERIOD'])
def init_state(key, L, p, CLOCK_PERIOD=CLOCK_PERIOD):
    """
    Initialize the state: h, v, b, r, g, c, step_count
    """
    k1, k2 = jax.random.split(key)
    h_qubits = jax.random.bernoulli(k1, p, shape=(L, L)).astype(jnp.int32)
    v_qubits = jax.random.bernoulli(k2, p, shape=(L, L)).astype(jnp.int32)
    
    b_grid = jnp.zeros((L, L), dtype=jnp.bool_)
    r_grid = jnp.zeros((L, L), dtype=jnp.bool_)
    g_grid = jnp.zeros((L, L), dtype=jnp.bool_)
    c = 0 # Global clock
    step_count = 0
    
    return (h_qubits, v_qubits, b_grid, r_grid, g_grid, c, step_count, CLOCK_PERIOD)


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


@jax.jit
def step_ca(state):
    """
    Single step of the Toric CA with RGB rules (Algorithm 2).
    state: (h_qubits, v_qubits, b, r, g, c, step_count)
    """
    h_qubits, v_qubits, b, r, g, c, step_count, CLOCK_PERIOD = state
    
    s = calculate_syndrome(h_qubits, v_qubits)
    
    # --- Neighbors ---
    # Standard Rolls:
    # x-1: roll +1 axis 0
    # x+1: roll -1 axis 0
    # y-1: roll +1 axis 1
    # y+1: roll -1 axis 1
    
    # B Neighbors: Needs Up(y+1), Down(y-1), Left(x-1)
    b_left = jnp.roll(b, 1, axis=0)
    b_down = jnp.roll(b, 1, axis=1)

    
    # R Neighbors: Needs Left(x-1), Up(y+1), Right?(x+1 for MAJ?)
    r_left = jnp.roll(r, 1, axis=0)
    r_up = jnp.roll(r, -1, axis=1)
    r_right = jnp.roll(r, -1, axis=0) 

    # G Neighbors: Needs Down(y-1), Right(x+1)
    g_right = jnp.roll(g, -1, axis=0)
    g_down = jnp.roll(g, 1, axis=1)
    g_left_up = jnp.roll(jnp.roll(g, 1, axis=0), -1, axis=1) 
    
    # S Neighbors
    s_right = jnp.roll(s, -1, axis=0)   # s(x+1, y)
    s_up = jnp.roll(s, -1, axis=1)      # s(x, y+1)
    
    # --- Logic ---
    
    mask_s = (s == 1)
    mask_no_s = ~mask_s
    
    # Initialize next state
    next_b = b
    next_r = r
    next_g = g
    h_new = h_qubits
    v_new = v_qubits
    
    # === 1. if s(x,y,t) == 1 ===
    # next_b, next_r, next_g <- 1
    next_b = jnp.where(mask_s, True, next_b)
    next_r = jnp.where(mask_s, True, next_r)
    next_g = jnp.where(mask_s, True, next_g)
    
    is_c0 = (c == 0)
    
    # Line 4: if b(x-1) v r(x-1): move Left
    cond_left = (b_left | r_left)
    do_left = mask_s & cond_left 
    
    # Line 6: else if g(x, y-1): move Down
    cond_down = (g_down | b_down)
    do_down = mask_s & (~do_left) & cond_down 
    
    # Apply corrections
    v_new = v_new ^ do_left
    h_new = h_new ^ do_down
    
    # === 2. else (s=0) ===
    
    # Line 11: ils <- (s(x+1)=1) ^ (b v r) ^ (c=0)
    ils = s_right & (b | r) 
    
    # Line 12: ids <- (s(x,y+1)=1) ^ g ^ ~r(x-1, y+1) ^ ~b(x-1, y+1) ^ (c=0)
    # Need r(x-1, y+1) and b(x-1, y+1)
    r_left_up = jnp.roll(r_left, -1, axis=1)
    b_left_up = jnp.roll(b_left, -1, axis=1)
    
    ids = s_up & (g | b) & (~r_left_up) & (~b_left_up)
    
    # Line 13: is <- ils v ids
    is_flag = ils | ids
    
    # Line 14: if is
    mask_is = mask_no_s & is_flag
    next_b = jnp.where(mask_is, True, next_b)
    next_r = jnp.where(mask_is, True, next_r)
    next_g = jnp.where(mask_is, True, next_g)
    
    # Line 16: else (no incoming syndrome)
    mask_else = mask_no_s & (~is_flag)
    
    mask_growth = mask_else & is_c0
    

    # b update
    b_grow = b_down | b_left
    # Apply to (mask_growth & (b==0))
    # If b=0, next_b = b_grow
    next_b = jnp.where(mask_growth & (~b), b_grow, next_b)
    
    # r update: if r=0 -> r(x, y+1) v r(x-1, y)
    r_grow = r_up | r_left
    next_r = jnp.where(mask_growth & (~r), r_grow, next_r)
    
    # g update: if g=0 -> g(x, y-1) v g(x+1, y)
    g_grow = g_down | g_right
    next_g = jnp.where(mask_growth & (~g), g_grow, next_g)

    sum_r = r_left.astype(jnp.int32) + r_up.astype(jnp.int32) + r.astype(jnp.int32)
    rm = (sum_r >= 2)
    
    # bMAJ(lb, db, b)
    sum_b = b_left.astype(jnp.int32) + b_down.astype(jnp.int32) + b.astype(jnp.int32)
    bm = (sum_b >= 2)
    
    # gMAJ(rg, dg, g)
    sum_g = g_right.astype(jnp.int32) + g_down.astype(jnp.int32) + g.astype(jnp.int32)
    gm = (sum_g >= 2)
    
    # Line 46: if b(x, y, t) = 1
    # b <- bm
    mask_b1 = mask_else & b
    next_b = jnp.where(mask_b1, bm, next_b)

    mask_r1 = mask_else & r
    next_r = jnp.where(mask_r1, rm | (bm & b), next_r)
    
    mask_g1 = mask_else & g
    next_g = jnp.where(mask_g1, gm | (bm & b), next_g)
    
    # Clock Update
    next_c = (c + 1) % CLOCK_PERIOD
    
    return (h_new, v_new, next_b, next_r, next_g, next_c, step_count + 1, CLOCK_PERIOD)


@jax.jit
def check_logical_error(h_qubits, v_qubits):
    """Check for logical error (non-trivial loops)."""
    # Matches toric_ca_shear.py: 
    # v[0, :] (vary y) -> Vertical winding
    # h[:, 0] (vary x) -> Horizontal winding
    winding_v = jnp.sum(v_qubits[0, :]) % 2
    winding_h = jnp.sum(h_qubits[:, 0]) % 2
    return (winding_v != 0) | (winding_h != 0)


# --- Simulation (Monte Carlo) ---

@partial(jax.jit, static_argnames=['L', 'max_steps', 'CLOCK_PERIOD'])
def run_single_trajectory(key, L, p, max_steps, CLOCK_PERIOD=CLOCK_PERIOD):
    """
    Run one trajectory using while_loop for early termination.
    Returns (is_failure, step_count) where is_failure is 1.0 if logical error or not converged.
    """
    initial_state = init_state(key, L, p, CLOCK_PERIOD)
    
    def cond_fun(state):
        h, v, b, r, g, c, step, _ = state
        
        # Check if cleared
        syndrome = calculate_syndrome(h, v)
        not_cleared = jnp.any(syndrome)
        
        # Also check if under max_steps
        under_limit = step < max_steps
        
        return not_cleared & under_limit
    
    final_state = jax.lax.while_loop(cond_fun, step_ca, initial_state)
    
    h_fin, v_fin, b_fin, r_fin, g_fin, c_fin, steps, _ = final_state
    
    # Check logical error
    correction_failed = check_logical_error(h_fin, v_fin)
    
    # Check convergence failure (if steps reached max_steps and syndrome still not cleared)
    # Actually cond_fun returns false if not_cleared is false (cleared).
    # If loop terminated because step >= max_steps, it means not converged.
    # But wait, if it cleared at step N < max, it terminates.
    
    s_fin = calculate_syndrome(h_fin, v_fin)
    not_converged = jnp.any(s_fin)
    
    # Failure if logical error OR didn't converge
    is_failure = jnp.logical_or(correction_failed, not_converged).astype(jnp.float32)
    
    return is_failure, steps.astype(jnp.float32)


def run_monte_carlo(L_values, p_values, num_samples, filename="toric_ca_data.npz"):
    """
    Run Monte Carlo simulation for multiple L and p values.
    Saves results incrementally to filename.
    """
    print(f"Running JAX Toric CA Shear Decoder Evaluation...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")
    
    results = {}
    master_key = jax.random.PRNGKey(20260921)
    
    for L in L_values:
        max_steps = 5*CLOCK_PERIOD * L
        
        @partial(jax.jit, static_argnames=['CLOCK_PERIOD'])
        def batch_run(keys, p_val, CLOCK_PERIOD=CLOCK_PERIOD):
             return jax.vmap(lambda k: run_single_trajectory(k, L, p_val, max_steps, CLOCK_PERIOD))(keys)
        
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


def run_decoding_time_sweep(L_values, p_values, num_samples, filename="toric_ca_decoding_time.npz", CLOCK_PERIOD=CLOCK_PERIOD):
    """
    Run Monte Carlo simulation to measure average decoding time and its standard error
    for different L and p values. Saves results incrementally to filename.

    For each (L, p), runs num_samples trajectories and records the step count for
    successful trajectories (those that converge without logical error). Computes:
      - avg_decoding_time: mean of successful step counts
      - std_err_decoding_time: standard error = std(steps) / sqrt(N_success)

    Args:
        L_values: list of code distances
        p_values: list of physical error rates
        num_samples: int or dict mapping (L, p) / L / 'default' to sample counts
        filename: output .npz file
        CLOCK_PERIOD: clock period for the CA rule
    """
    print(f"Running Decoding Time Sweep...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")

    results = {}
    master_key = jax.random.PRNGKey(20260921)

    for L in L_values:
        max_steps = 5 * CLOCK_PERIOD * L

        @partial(jax.jit, static_argnames=['CLOCK_PERIOD'])
        def batch_run(keys, p_val, CLOCK_PERIOD=CLOCK_PERIOD):
            return jax.vmap(lambda k: run_single_trajectory(k, L, p_val, max_steps, CLOCK_PERIOD))(keys)

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

            # Collect per-trajectory step counts for successful runs
            all_success_steps = []
            total_failures = 0
            samples_left = current_num_samples

            master_key, loop_key = jax.random.split(master_key)

            while samples_left > 0:
                current_batch_size = min(samples_left, MAX_BATCH_SIZE)

                loop_key, batch_subkey = jax.random.split(loop_key)
                keys = jax.random.split(batch_subkey, current_batch_size)

                batch_failures, batch_steps = batch_run(keys, p, CLOCK_PERIOD=CLOCK_PERIOD)

                # Extract successful step counts
                batch_failures_np = np.array(batch_failures)
                batch_steps_np = np.array(batch_steps)
                success_mask = batch_failures_np == 0.0
                success_steps = batch_steps_np[success_mask]
                all_success_steps.append(success_steps)
                total_failures += int(np.sum(batch_failures_np))

                samples_left -= current_batch_size
                print(f"    L={L}, p={p:.4f}: {samples_left} samples remaining", end='\r')

            print()

            all_success_steps = np.concatenate(all_success_steps) if all_success_steps else np.array([])
            n_success = len(all_success_steps)

            if n_success > 0:
                avg_time = float(np.mean(all_success_steps))
                std_time = float(np.std(all_success_steps, ddof=1)) if n_success > 1 else 0.0
                std_err = std_time / np.sqrt(n_success)
            else:
                avg_time = np.nan
                std_err = np.nan

            results[(L, p)] = (avg_time, float(std_err), n_success, current_num_samples)
            print(f"  L={L}, p={p:.4f} -> avg_decoding_time={avg_time:.2f}, "
                  f"std_err={std_err:.4f}, successes={n_success}/{current_num_samples}")

            # Save results incrementally
            Ls_out, ps_out = [], []
            avg_time_out, std_err_out = [], []
            n_success_out, n_samples_out = [], []

            for (L_k, p_k), val in results.items():
                Ls_out.append(L_k)
                ps_out.append(p_k)
                avg_time_out.append(val[0])
                std_err_out.append(val[1])
                n_success_out.append(val[2])
                n_samples_out.append(val[3])

            np.savez(filename,
                     L=np.array(Ls_out),
                     p=np.array(ps_out),
                     avg_decoding_time=np.array(avg_time_out),
                     std_err_decoding_time=np.array(std_err_out),
                     n_success=np.array(n_success_out),
                     num_samples=np.array(n_samples_out))
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
    ax1.set_title('Toric CA Shear Decoder Threshold')
    
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
        c = 0
        step_count = jnp.array(0, dtype=jnp.int32)
        initial_state = (h_qubits, v_qubits, b_grid, r_grid, g_grid, c, step_count)
    else:
        initial_state = init_state(key, L, p)
    
    # Scan step with done flag for early stopping visual
    def scan_step(carry, _):
        h, v, b, r, g, c, step_count, done = carry
        
        # Real step
        h_next, v_next, b_next, r_next, g_next, c_next, _ = step_ca((h, v, b, r, g, c, step_count))
        
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
        
        new_carry = (h_final, v_final, b_final, r_final, g_final, c_final, step_count + 1, new_done)
        
        # Output: syndrome, b, r, g, c
        output_s = calculate_syndrome(h_final, v_final)
        output = (output_s, b_final, r_final, g_final, c_final)
        return new_carry, output
    
    # Initial carry with done flag
    h0, v0, b0, r0, g0, c0, sc0 = initial_state
    s0 = calculate_syndrome(h0, v0)
    done0 = jnp.all(s0 == 0) & jnp.all(b0 == 0) & jnp.all(r0 == 0) & jnp.all(g0 == 0)
    
    carry_init = (h0, v0, b0, r0, g0, c0, sc0, done0)
    
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
        # clock_state = full_c[frame][0, 0] # WRONG if c is scalar
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
        status_text = f"T={frame} | Defects: {defects} | Clock: {clock_state}"
        
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
        full_c = full_c[:cutoff]
        full_g = full_g[:cutoff]
    
    T = len(full_s)
    
    anim = FuncAnimation(fig, update, frames=T, repeat=False)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    try:
        anim.save(filename, writer=PillowWriter(fps=2))
        print(f"GIF saved to {filename}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error saving GIF: {e}")
    plt.close(fig)


# --- Legacy NumPy Interface (for compatibility) ---

class ToricCA:
    """NumPy-based wrapper for compatibility with existing code."""
    
    def __init__(self, L):
        self.L = L
        self.h_qubits = np.zeros((L, L), dtype=int)
        self.v_qubits = np.zeros((L, L), dtype=int)
        self.syndrome = np.zeros((L, L), dtype=int)
        self.b_grid = np.zeros((L, L), dtype=int)
        self.r_grid = np.zeros((L, L), dtype=int)
        self.c_grid = np.zeros((L, L), dtype=int)
    
    def reset(self):
        self.h_qubits.fill(0)
        self.v_qubits.fill(0)
        self.syndrome.fill(0)
        self.b_grid.fill(0)
        self.r_grid.fill(0)
        self.c_grid.fill(0)
    
    def apply_noise(self, p):
        noise_h = np.random.rand(self.L, self.L) < p
        noise_v = np.random.rand(self.L, self.L) < p
        self.h_qubits ^= noise_h.astype(int)
        self.v_qubits ^= noise_v.astype(int)
    
    def set_error_configuration(self, h_errors, v_errors):
        self.h_qubits = np.array(h_errors, dtype=int)
        self.v_qubits = np.array(v_errors, dtype=int)
    
    def calculate_syndrome(self):
        v_right = np.roll(self.v_qubits, -1, axis=0)
        h_top = np.roll(self.h_qubits, -1, axis=1)
        self.syndrome = (self.v_qubits + v_right + self.h_qubits + h_top) % 2
        return np.sum(self.syndrome)
    
    def check_logical_error(self):
        winding_v = np.sum(self.v_qubits[0, :]) % 2
        winding_h = np.sum(self.h_qubits[:, 0]) % 2
        return (winding_v != 0) or (winding_h != 0)


if __name__ == "__main__":
    # Example: Generate GIF
    # L = 16
    # h_err = np.zeros((L, L))
    # v_err = np.zeros((L, L))
    # h_err[7, 10] = 1
    # save_simulation_gif(L, 0.04, 50, "toric_ca_shear.gif", ['b', 'r'])
    
    # # Example: Run Monte Carlo
    # run_monte_carlo(
    #     L_values= [7, 13, 19, 25],
    #     p_values=[1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2],
    #     num_samples={
    #         'default': 10_000,
    #         (37, 1e-2): 100_000_000,
    #         (31, 1e-2): 100_000_000,
    #         (41, 1e-2): 100_000_000,
    #     },
    #     filename="tc_rgb_results_fully_fixed.npz"
    # )

    run_monte_carlo(
        L_values= [9, 14, 19],
        p_values=[0.04, 0.05, 0.05, 0.07, 0.08],
        num_samples={
            'default': 10_000
        },
        filename="tc_rgb_test_move_down.npz"
    )
    
    # # # Example: Plot results
    plot_results("tc_rgb_test_move_down.npz", "tc_rgb_test_move_down.png")
