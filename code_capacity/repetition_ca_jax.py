import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import argparse
import os

# --- Core CA Logic ---

@partial(jax.jit, static_argnames=['L'])
def init_state(key, L, p):
    """
    Initialize state: qubits (L,), m_grid (L,), step_count scalar.
    """
    key1, key2 = jax.random.split(key)
    # qubits on edges.
    qubits = jax.random.bernoulli(key1, p, shape=(L,)).astype(jnp.bool_)
    m_grid = jnp.zeros(L, dtype=jnp.bool_)
    step_count = jnp.array(0, dtype=jnp.int32)
    return (qubits, m_grid, step_count)

@jax.jit
def calculate_syndrome(qubits):
    """
    s[i] = qubits[i] XOR qubits[i+1]
    """
    right_q = jnp.roll(qubits, -1)
    syndrome = qubits ^ right_q
    return syndrome

@jax.jit
def step_ca(state):
    """
    Single step of the CA.
    state: (qubits, m_grid, step_count)
    """
    qubits, m_grid, step_count = state
    
    s = calculate_syndrome(qubits)
    
    # --- Repetition Code Rule (Algorithm 1, q = 2) ---
    m = m_grid
    left_message = jnp.roll(m, 1)

    # A defect moves one site left whenever the message immediately to its left
    # is present.  With s[i] = q[i] XOR q[i+1], this flips qubit i.
    move = s & left_message
    qubits = qubits ^ move

    growth = (~s) & (~m) & ((step_count % 2) == 0) & left_message
    persistence = (~s) & m & left_message
    arrival = jnp.roll(s, -1) & m
    next_m = s | growth | persistence | arrival
    
    return (qubits.astype(jnp.bool_), next_m.astype(jnp.bool_), step_count + 1)


# --- Simulation (Monte Carlo) ---

@partial(jax.jit, static_argnames=['L', 'max_steps'])
def run_single_trajectory_while(key, L, p, max_steps):
    """
    Run one trajectory using while_loop for early termination.
    Returns 1.0 if logical error, 0.0 otherwise.
    """
    initial_state = init_state(key, L, p)
    
    def cond_fun(state):
        qubits, m_grid, step_count = state
        syndrome = calculate_syndrome(qubits)
        # not_cleared = jnp.any(syndrome) | jnp.any(m_grid)
        not_cleared = jnp.any(syndrome)
        under_step_limit = step_count < max_steps
        return not_cleared & under_step_limit
        
    final_state = jax.lax.while_loop(cond_fun, step_ca, initial_state)
    
    qubits, m_grid, step_count = final_state
    
    # convergence check
    final_syndrome = calculate_syndrome(qubits)
    converged = jnp.all(final_syndrome == 0)
    
    # logical check
    logical_error = qubits[0] == 1
    
    # Failure condition: Not converged OR (Converged AND Logical Error)
    is_failure = (~converged) | logical_error
    return is_failure.astype(jnp.float32), step_count.astype(jnp.float32)

def run_monte_carlo(L_values, p_values, num_samples, filename="rep_ca_data.npz"):
    print(f"Running JAX Repetition CA Evaluation...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")
    
    results = {}
    
    # Prepare keys
    # We will loop over L and p (standard Python loop)
    # but vmap over samples.
    
    master_key = jax.random.PRNGKey(20260920)
    
    for L in L_values:
        max_steps = 2 * L - 1
        # JIT compile the vmapped function specifically for this L
        # vmap over keys
        runner = jax.jit(jax.vmap(partial(run_single_trajectory_while, L=L, max_steps=max_steps, p=None), in_axes=0))
        
        # Reformulate runner to take (key, p)
        # We need to map over keys, broadcast over p.
        @jax.jit
        def batch_run(keys, p_val):
             return jax.vmap(lambda k: run_single_trajectory_while(k, L, p_val, max_steps))(keys)
        
        MAX_BATCH_SIZE = 1_000
        
        if isinstance(num_samples, dict):
            current_num_samples = num_samples[L]
        else:
            current_num_samples = num_samples
            
        for p in p_values:
            total_failures = 0.0
            total_success_steps = 0.0
            total_successes = 0.0
            samples_left = current_num_samples
            
            # We need a subkey for the loop (re-split master key for independence if needed, or just flow)
            # Actually, using master_key split inside p loop is good, but if we vary samples, just ensure randomness.
            master_key, loop_key = jax.random.split(master_key)
            
            while samples_left > 0:
                current_batch_size = min(samples_left, MAX_BATCH_SIZE)
                
                # Generate keys for this batch
                loop_key, batch_subkey = jax.random.split(loop_key)
                keys = jax.random.split(batch_subkey, current_batch_size)
                
                batch_failures, batch_steps = batch_run(keys, p)
                total_failures += jnp.sum(batch_failures)
                
                # Mask steps by success
                # failure = 1.0, success = 0.0 (from run_single_trajectory_while logic? No wait check return)
                # run_single returns is_failure (1.0 if failed).
                success_mask = 1.0 - batch_failures
                total_successes += jnp.sum(success_mask)
                total_success_steps += jnp.sum(batch_steps * success_mask)
                
                samples_left -= current_batch_size
                print(f"    ... {samples_left} samples remaining (failures so far: {total_failures})", end='\r')
            
            print() # Newline after loop finishes
            ler = total_failures / current_num_samples
            
            if total_successes > 0:
                avg_steps = total_success_steps / total_successes
            else:
                avg_steps = 0.0
            
            # Unblock
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
                
            np.savez(filename, L=np.array(Ls_out), p=np.array(ps_out), ler=np.array(lers_out), avg_steps=np.array(avg_steps_out), num_samples=np.array(num_samples_out))
            print(f"Data saved to {filename}")

def run_decoding_time_sweep(L_values, p_values, num_samples, filename="rep_ca_decoding_time.npz"):
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
        num_samples: int or dict mapping L to sample counts
        filename: output .npz file
    """
    print(f"Running Repetition CA Decoding Time Sweep...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")

    results = {}
    master_key = jax.random.PRNGKey(20260920)

    MAX_BATCH_SIZE = 1_000

    for L in L_values:
        max_steps = 2 * L - 1

        @jax.jit
        def batch_run(keys, p_val):
            return jax.vmap(lambda k: run_single_trajectory_while(k, L, p_val, max_steps))(keys)

        if isinstance(num_samples, dict):
            current_num_samples = num_samples[L]
        else:
            current_num_samples = num_samples

        for p in p_values:
            # Collect per-trajectory step counts for successful runs
            all_success_steps = []
            total_failures = 0
            samples_left = current_num_samples

            master_key, loop_key = jax.random.split(master_key)

            while samples_left > 0:
                current_batch_size = min(samples_left, MAX_BATCH_SIZE)

                loop_key, batch_subkey = jax.random.split(loop_key)
                keys = jax.random.split(batch_subkey, current_batch_size)

                batch_failures, batch_steps = batch_run(keys, p)

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

def plot_results(data_filename, plot_filename="rep_ca_results.png"):
    if not os.path.exists(data_filename):
        print(f"File {data_filename} not found.")
        return

    data = np.load(data_filename)
    Ls_raw = data['L']
    ps_raw = data['p']
    lers_raw = data['ler']
    
    # Reconstruct dictionary
    results = {}
    for L, p, val in zip(Ls_raw, ps_raw, lers_raw):
        results[(L, p)] = val
        
    unique_L = sorted(list(set(Ls_raw)))
    unique_p = sorted(list(set(ps_raw)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: p vs LER for different L
    for L in unique_L:
        # Extract row
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
    ax1.set_title('Repetition CA Threshold')
    
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


# --- Visualisation (SCAN) ---

def save_simulation_gif(L, p, steps, filename="repetition_ca.gif"):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.lines import Line2D
    
    # Run one trajectory using scan (to catch history)
    # We still want early stopping behavior visually, 
    # but scan generates fixed length. We can implement a "done" mask in scan.
    
    key = jax.random.PRNGKey(np.random.randint(0, 10000))
    initial_state = init_state(key, L, p)
    # Augment state with a 'done' flag for visual consistency
    # (qubits, m_grid, step_count, done_mask)
    # done_mask = 0 (running), 1 (done)
        
    def scan_step(carry, _):
        qubits, m, done = carry
        
        # Calculate next
        # If done, just identity
        
        # Real step
        (q_next, m_next, _) = step_ca((qubits, m, 0)) # ignore step count in core logic
        
        syndrome = calculate_syndrome(qubits)
        syndrome = calculate_syndrome(qubits)
        is_clean = jnp.all(syndrome == 0) & jnp.all(m == 0)
        
        # If we are ALREADY done, keep old state
        # If we just became clean, we are now done (for next step?)
        # Let's say we update first, then check if we are done.
        
        new_done = done | is_clean # If it was done, it stays done. If it becomes clean, it becomes done.
        
        # Select
        # If ALREADY done before this step, we shouldn't have updated?
        # A clearer way: 
        #   If done, q_next = qubits, m_next = m
        #   Else, calculate q_next, m_next
        
        q_final = jnp.where(done, qubits, q_next)
        m_final = jnp.where(done, m, m_next) # Freeze m
        
        # Also, if we just finished (is_clean is true), we might want to clear m for visual clarity?
        # But let's stick to the rule evolution.
        
        new_carry = (q_final, m_final, new_done)
        
        # Output: syndrome, m, qubits
        output_s = calculate_syndrome(q_final)
        output = (output_s, m_final, q_final)
        return new_carry, output

    # Initial carry
    # Scan runs for 'steps'
    # Start: done=False (unless init syndrome is 0)
    s0 = calculate_syndrome(initial_state[0])
    done0 = jnp.all(s0 == 0) & jnp.all(initial_state[1] == 0)
    
    carry_init = (initial_state[0], initial_state[1], done0)
    
    final, history = jax.lax.scan(scan_step, carry_init, None, length=steps)
    
    # Process history to numpy
    s_hist, m_hist, q_hist = history
    s_hist = np.array(s_hist)
    m_hist = np.array(m_hist)
    q_hist = np.array(q_hist)
    
    # Prepend initial state? Optional. Scan output starts at t=1 usually? 
    # No, first output is result of first step. 
    # Usually we want t=0.
    # Let's add t=0 manually.
    current_s = np.array(s0)
    current_m = np.array(initial_state[1])
    current_q = np.array(initial_state[0])
    
    full_s = np.concatenate([[current_s], s_hist], axis=0)
    full_m = np.concatenate([[current_m], m_hist], axis=0)
    full_q = np.concatenate([[current_q], q_hist], axis=0)
    
    # Check logical error at end
    final_q = full_q[-1]
    is_logical_fail = (final_q[0] == 1)
    
    # --- Animation ---
    
    fig, ax = plt.subplots(figsize=(10, 3))
    color_m = (0.7, 0.7, 1.0)
    color_err = (1.0, 0.7, 0.7)
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Defect', markerfacecolor='black', markersize=10),
        Rectangle((0,0), 1, 1, facecolor=color_m, label='Move Left (m)'),
        Line2D([0], [0], color=color_err, linewidth=4, label='Error (Qubit)')
    ]
    
    def update(frame):
        ax.clear()
        ax.set_xlim(0, L)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(L+1))
        ax.set_yticks(np.arange(2))
        ax.grid(True, color='gray', linestyle='-', linewidth=1.0, alpha=0.5)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        
        s = full_s[frame]
        m = full_m[frame]
        q = full_q[frame]
        
        for i in range(L):
            if m[i] == 1:
                rect = Rectangle((i, 0), 1, 1, facecolor=color_m, edgecolor='none')
                ax.add_patch(rect)
                
            if s[i] == 1:
                circ = Circle((i + 0.5, 0.5), 0.15, facecolor='black', zorder=10)
                ax.add_patch(circ)
            
            if q[i] == 1:
                ax.plot([i+1, i+1], [0, 1], color=color_err, linewidth=4, zorder=5)
                
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')
        
        defects = np.sum(s)
        status = f"T={frame} | Defects: {defects}"
        
        # If effectively done (syndrome 0 and no messages)
        # Note: In our rule, m can persist briefly? 
        if defects == 0 and np.sum(m) == 0:
             res = "FAIL" if is_logical_fail else "SUCCESS"
             status += f" [{res}]"
             
        ax.set_title(status)

    # Trim frames after first completion
    clean_indices = [i for i in range(len(full_s)) if np.all(full_s[i] == 0) and np.all(full_m[i] == 0)]
    if clean_indices:
        # Keep up to the first clean frame (inclusive)
        cutoff = clean_indices[0] + 1
        full_s = full_s[:cutoff]
        full_m = full_m[:cutoff]
        full_q = full_q[:cutoff]

    frames = len(full_s)
    
    anim = FuncAnimation(fig, update, frames=frames, repeat=False)
    plt.tight_layout()
    try:
        anim.save(filename, writer=PillowWriter(fps=5))
        print(f"GIF saved to {filename}")
    except Exception as e:
        print(f"Error saving GIF: {e}")
    plt.close(fig)

# --- Spacetime Logic ---

def simulate_spacetime_data(key, L, p, steps):
    """
    Simulate the CA and return full history of qubits, messages, and syndromes.
    Returns: (q_hist, m_hist, s_hist) as numpy arrays of shape (steps+1, L)
    """
    initial_state = init_state(key, L, p)
    # state: (qubits, m_grid, step_count)
    
    def scan_step(carry, _):
        qubits, m, step_count = carry
        
        # We need the state BEFORE the step for the history
        # But scan usually outputs the result of the function.
        # Let's output the Current state, then update.
        # Wait, scan logic:
        #   y, carry = f(carry, x)
        # So we can output carry as y.
        
        current_s = calculate_syndrome(qubits)
        output = (qubits, m, current_s)
        
        # Update
        next_state = step_ca(carry)
        
        return next_state, output

    final_state, history = jax.lax.scan(scan_step, initial_state, None, length=steps)
    
    q_hist, m_hist, s_hist = history
    
    # The history contains steps 0 to steps-1.
    # We also want the final state? 
    # Or maybe scan_step should output the *next* state?
    # Usually we want t=0 to t=steps.
    # The above 'output = (qubits, m, current_s)' records the state *before* the update.
    # So history[0] is state at t=0.
    # history[steps-1] is state at t=steps-1.
    # We might want to append the final state from 'final_state'.
    
    final_q, final_m, _ = final_state
    final_s = calculate_syndrome(final_q)
    
    q_hist = jnp.concatenate([q_hist, final_q[None, :]], axis=0)
    m_hist = jnp.concatenate([m_hist, final_m[None, :]], axis=0)
    s_hist = jnp.concatenate([s_hist, final_s[None, :]], axis=0)
    
    # Check parity
    # Ensure numpy arrays for printing
    s_hist_np = np.array(s_hist)
    syndrome_sums = np.sum(s_hist_np, axis=1)
    
    # Check if any are odd
    odd_parity_indices = np.where(syndrome_sums % 2 != 0)[0]
    
    odd_counts = odd_parity_indices.size
    if odd_counts > 0:
        print(f"WARNING: Found {odd_counts} time steps with ODD syndrome parity!")
        # Print first few
        print(f"Indices: {odd_parity_indices[:5]}...")
    else:
        print("Parity check passed: All time steps have even syndrome parity.")

    return np.array(q_hist), np.array(m_hist), np.array(s_hist)


def plot_spacetime(data, filename="repetition_ca_spacetime.png", display_grid=True, plot_syndromes=True):
    """
    Plot spacetime diagram of qubits, messages, and optional syndromes.
    data: (q_hist, m_hist, s_hist)
    """
    q_hist, m_hist, s_hist = data
    T, L = q_hist.shape
    
    num_plots = 3 if plot_syndromes else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    
    # Common helper for grid
    def add_grid(ax, data):
        rows, cols = data.shape
        ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", size=0)
        ax.set_ylim(-0.5, rows - 0.5)
        ax.set_xlim(-0.5, cols - 0.5)

    # 1. Qubits (Errors)
    ax = axes[0]
    ax.set_title("Qubit Errors (Black=Error)")
    # cmap: gray_r (0=white, 1=black)
    ax.imshow(q_hist, origin='lower', cmap='gray_r', aspect='equal', interpolation='none', vmin=0, vmax=1)
    if display_grid:
        add_grid(ax, q_hist)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 2. Syndromes (Optional)
    current_ax_idx = 1
    if plot_syndromes:
        ax = axes[current_ax_idx]
        ax.set_title("Syndromes (Red=Active)")
        from matplotlib.colors import ListedColormap
        cmap_red = ListedColormap(['white', '#ff3030'])
        ax.imshow(s_hist, origin='lower', cmap=cmap_red, aspect='equal', interpolation='none', vmin=0, vmax=1)
        if display_grid:
            add_grid(ax, s_hist)
        ax.set_xlabel(r"$x$")
        ax.set_xticks([])
        ax.set_yticks([])
        current_ax_idx += 1
    
    # 3. Messages
    ax = axes[current_ax_idx]
    ax.set_title("Messages (Active)")
    # Use Indigo if syndromes are shown, Emerald if not (user preference)
    # msg_color = '#6366F1' if plot_syndromes else '#D1FAE5'
    msg_color = '#6366F1' if plot_syndromes else '#FFB6C1'

    cmap_msg = ListedColormap(['white', msg_color])
    ax.imshow(m_hist, origin='lower', cmap=cmap_msg, aspect='equal', interpolation='none', vmin=0, vmax=1)
    if display_grid:
        add_grid(ax, m_hist)
    ax.set_xlabel(r"$x$")
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Spacetime plot saved to {filename}")
    plt.close(fig)


def plot_spacetime_combined(data, filename="repetition_ca_combined.png", display_grid=False, plot_syndromes=True):
    """
    Plot integrated spacetime diagram by tripling horizontal resolution.
    Layout per bit: [Si (1px), Qi/Mi (2px)]
    This ensures qubit errors are physically surrounded by syndromes.
    """
    from matplotlib.patches import Patch
    q_hist, m_hist, s_hist = data
    T, L = q_hist.shape

    # color_m = np.array([186/255, 230/255, 253/255]) # Sky Blue 200 (#BAE6FD) - clearer background

    # Refined Contrast RGB colors
    # color_q = np.array([0.25, 0.25, 0.25])             # Dark Gray
    # color_q = np.array([0.25, 0.25, 0.25])
    color_q = np.array([0.329, 0.329, 0.329])
    color_s = np.array([255/255, 0/255, 0/255])     # Pure Bright Red
    
    if plot_syndromes:
        color_m = np.array([99/255, 102/255, 241/255])  # Electric Indigo (#6366F1)
    else:
        color_m = np.array([209/255, 250/255, 229/255]) # Light Emerald Green (#D1FAE5)
        # color_m = np.array([216/255, 191/255, 216/255]) # Light Purple (#D8BFD8)
        # color_m = np.array([203/255, 195/255, 227/255]) # Light Purple (#D8BFD8)
        # color_m = np.array([204/255, 204/255, 255/255]) # Light Purple (#D8BFD8)
        # color_m = np.array([188/255, 230/255, 188/255]) # Light Green (#9FE2BF)
        # color_m = np.array([1/255, 121/255, 111/255])
        # color_m = np.array([215/255, 252/255, 212/255])

    color_bg = np.array([1.0, 1.0, 1.0])           # White

    # Determine layout based on plot_syndromes
    if plot_syndromes:
        # Triadic layout: [S (1px), Q (2px)]
        img_width = 3 * L
        img = np.ones((T, img_width, 3))
        for t in range(T):
            for i in range(L):
                if m_hist[t, i]:
                    img[t, 3*i] = color_m
                if s_hist[t, i]:
                    img[t, 3*i] = color_s
                img[t, 3*i + 1 : 3*i + 3] = color_q if q_hist[t, i] else color_bg
    else:
        # Regular layout: Width L, i indexes both
        img_width = L
        img = np.ones((T, img_width, 3))
        for t in range(T):
            for i in range(L):
                # Overlay priority: Qubit > Message
                if m_hist[t, i]:
                    img[t, i] = color_m
                if q_hist[t, i]:
                    img[t, i] = color_q

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 28,
        "axes.titlesize": 28,
        "axes.linewidth": 1.5,
    })

    fig_height = 10
    # Wide enough to accommodate img_width pixels clearly
    aspect_ratio = img_width / T
    fig_width = fig_height * aspect_ratio
    fig_width = max(min(fig_width, 24), 10)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    # 'nearest' ensures sharp pixels for the large lattice
    ax.imshow(img, origin='lower', aspect='auto', interpolation='nearest')
    
    if display_grid and img_width <= 256: 
        ax.set_xticks(np.arange(-.5, img_width, 1), minor=True)
        ax.set_yticks(np.arange(-.5, T, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which="minor", size=0)
    
    ax.set_ylim(-0.5, T - 0.5)
    ax.set_xlim(-0.5, img_width - 0.5)

    ax.set_xlabel(r"$x$", labelpad=15, fontsize=32)
    ax.set_ylabel(r"$t$", labelpad=15, rotation='horizontal', fontsize=32)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove Spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Legend
    legend_elements = [Patch(facecolor=color_q, label='Qubit Error')]
    if plot_syndromes:
        legend_elements.append(Patch(facecolor=color_s, label='Syndrome'))
        legend_elements.append(Patch(facecolor=color_m, label='Message'))
    else:
        legend_elements.append(Patch(facecolor=color_m, label='Move-Left Message'))
    
    ax.legend(handles=legend_elements, loc='upper left', ncol=1, frameon=False, fontsize=28)

    plt.tight_layout()
    # High DPI for the 12k horizontal pixel density
    plt.savefig(filename, bbox_inches='tight', dpi=400)
    print(f"Refined combined spacetime plot saved to {filename}")
    plt.close(fig)


if __name__ == "__main__":
    # save_simulation_gif(64, 0.3, 100, "repetition_ca.gif")
    # samples_dict = {16: 10_000_000, 32: 10_000_000, 64: 10_000_000, 128: 100_000_000, 256: 100_000_000, 512: 100_000_000}
    # run_monte_carlo(list(map(int, 10**np.linspace(1.5, 5, 8))), [0.1, 0.2, 0.3], 10_000, filename="rep_ca_data_avg_steps.npz")
    run_decoding_time_sweep(list(map(int, 10**np.linspace(1.5, 5, 8))), [0.1, 0.2, 0.3], 10_000, filename="rep_ca_data_decoding_time.npz")

    # # --- Spacetime Plotting Example ---
    # # --- Spacetime Plotting Example ---
    # L = 256
    # p = 0.38
    # steps = 200
    # key = jax.random.PRNGKey(32)
    
    # print(f"Generating spacetime plots for L={L}, p={p}...")
    # q_hist, m_hist, s_hist = simulate_spacetime_data(key, L, p, steps)
    # # plot_spacetime((q_hist, m_hist, s_hist), "repetition_ca_spacetime.png", display_grid=False)
    # # Generate combined plot with syndromes
    # # plot_spacetime_combined((q_hist, m_hist, s_hist), "repetition_ca_combined.png", display_grid=False, plot_syndromes=True)
    # # Generate combined plot WITHOUT syndromes (emerald messages)
    # plot_spacetime_combined((q_hist, m_hist, s_hist), "repetition_ca_combined_no_syndromes.pdf", display_grid=False, plot_syndromes=False)
    # print("Done.")
