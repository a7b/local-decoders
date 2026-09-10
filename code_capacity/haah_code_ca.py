import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
import time
import numba

# --- Constants ---

CLOCK_PERIOD = 10
MAX_BATCH_SIZE = 4000


# --- JAX Helpers ---

def _roll(a, shift, axis):
    """jnp.roll wrapper — positive shift = shift towards +index."""
    return jnp.roll(a, shift, axis=axis)


def _shifted_3d(arr, dx, dy, dz):
    """Shift so that out[i,j,k] = arr[(i+dx)%L, (j+dy)%L, (k+dz)%L]."""
    a = jnp.roll(arr, -dx, axis=0)
    a = jnp.roll(a, -dy, axis=1)
    a = jnp.roll(a, -dz, axis=2)
    return a


# --- Syndrome Evaluation ---

@partial(jax.jit, static_argnums=(1,))
def calculate_syndrome(errors, L):
    """
    Vectorised Z-syndrome evaluation for Haah's cubic code.

    Parameters
    ----------
    errors : (L,L,L,2) bool JAX array
    L      : int (static)

    Returns
    -------
    syndrome : (L,L,L) bool JAX array
    """
    e0 = errors[:, :, :, 0]  # (L,L,L) qubit-0 errors
    e1 = errors[:, :, :, 1]  # (L,L,L) qubit-1 errors

    # q0 corners: 0(0,0,0), 2(1,1,0), 4(1,0,1), 6(0,1,1)
    # q1 corners: 0(0,0,0), 1(0,1,0), 3(1,0,0), 7(0,0,1)
    s = (
        _shifted_3d(e0, 0, 0, 0) ^  # corner 0, q0
        _shifted_3d(e0, 1, 1, 0) ^  # corner 2, q0
        _shifted_3d(e0, 1, 0, 1) ^  # corner 4, q0
        _shifted_3d(e0, 0, 1, 1) ^  # corner 6, q0
        _shifted_3d(e1, 0, 0, 0) ^  # corner 0, q1
        _shifted_3d(e1, 0, 1, 0) ^  # corner 1, q1
        _shifted_3d(e1, 1, 0, 0) ^  # corner 3, q1
        _shifted_3d(e1, 0, 0, 1)    # corner 7, q1
    )
    return s


# --- CA Step ---

@jax.jit
def step_ca(state):
    """
    Single step of the Haah code CA decoder.
    state: (errors, messages, counter, step_count, CLOCK_PERIOD)

    Implements the Check / TryPair / PairInCheck rule from 3d-haah-code.ipynb.

    Returns updated state tuple with same structure.
    """
    errors, m, counter, step_count, CLOCK_PERIOD = state
    e0 = errors[:, :, :, 0]
    e1 = errors[:, :, :, 1]

    # ══════════════════════════════════════════════════════════════
    # 1. Evaluate syndrome from error vector
    # ══════════════════════════════════════════════════════════════
    syndrome = (
        _shifted_3d(e0, 0, 0, 0) ^
        _shifted_3d(e0, 1, 1, 0) ^
        _shifted_3d(e0, 1, 0, 1) ^
        _shifted_3d(e0, 0, 1, 1) ^
        _shifted_3d(e1, 0, 0, 0) ^
        _shifted_3d(e1, 0, 1, 0) ^
        _shifted_3d(e1, 1, 0, 0) ^
        _shifted_3d(e1, 0, 0, 1)
    )

    s = syndrome
    sat = ~s

    # ══════════════════════════════════════════════════════════════
    # 2. HasM aggregates
    #    HasM(r', t, i) = ∃ α with α_i=1 s.t. m_α(r') ≠ 0
    # ══════════════════════════════════════════════════════════════
    any_m1jk = jnp.any(m[:, :, :, 1, :, :], axis=(-2, -1))  # HasM(·, ·, 1)
    any_mi1k = jnp.any(m[:, :, :, :, 1, :], axis=(-2, -1))  # HasM(·, ·, 2)
    any_mij1 = jnp.any(m[:, :, :, :, :, 1], axis=(-2, -1))  # HasM(·, ·, 3)

    # ══════════════════════════════════════════════════════════════
    # 3. Check conditions
    #    Check(r,t,i): HasM at r-e_i, r-e_i-e_a, r-e_i-e_b, r-e1-e2-e3
    # ══════════════════════════════════════════════════════════════
    # Common sub-expression: HasM at r-ex-ey-ez for each direction
    m1_xyz = _roll(_roll(_roll(any_m1jk, +1, 0), +1, 1), +1, 2)
    m2_xyz = _roll(_roll(_roll(any_mi1k, +1, 0), +1, 1), +1, 2)
    m3_xyz = _roll(_roll(_roll(any_mij1, +1, 0), +1, 1), +1, 2)

    # Check(r,t,1): {a,b}={2,3}, uses any_m1jk
    m1_xm1 = _roll(any_m1jk, +1, 0)
    check1 = (m1_xm1 |
              _roll(m1_xm1, +1, 1) |
              _roll(m1_xm1, +1, 2) |
              m1_xyz)

    # Check(r,t,2): {a,b}={1,3}, uses any_mi1k
    m2_ym1 = _roll(any_mi1k, +1, 1)
    check2 = (m2_ym1 |
              _roll(m2_ym1, +1, 0) |
              _roll(m2_ym1, +1, 2) |
              m2_xyz)

    # Check(r,t,3): {a,b}={1,2}, uses any_mij1
    m3_zm1 = _roll(any_mij1, +1, 2)
    check3 = (m3_zm1 |
              _roll(m3_zm1, +1, 0) |
              _roll(m3_zm1, +1, 1) |
              m3_xyz)

    # ══════════════════════════════════════════════════════════════
    # 4. TryPair raw conditions
    #    TryPair(r,t,c): HasM at distances 1,2 in the two non-c dirs
    # ══════════════════════════════════════════════════════════════
    # TryPair(r,t,1): {a,b}={2,3}
    tp1_raw = (_roll(any_mi1k, +1, 1) & _roll(any_mi1k, +2, 1) &
               _roll(any_mij1, +1, 2) & _roll(any_mij1, +2, 2))

    # TryPair(r,t,2): {a,b}={1,3}
    tp2_raw = (_roll(any_m1jk, +1, 0) & _roll(any_m1jk, +2, 0) &
               _roll(any_mij1, +1, 2) & _roll(any_mij1, +2, 2))

    # TryPair(r,t,3): {a,b}={1,2}
    tp3_raw = (_roll(any_m1jk, +1, 0) & _roll(any_m1jk, +2, 0) &
               _roll(any_mi1k, +1, 1) & _roll(any_mi1k, +2, 1))

    # ══════════════════════════════════════════════════════════════
    # 5. Exclusive condition chain (if / elseif / ... / else)
    # ══════════════════════════════════════════════════════════════
    cond_all = s & check1 & check2 & check3
    handled = cond_all

    tp1 = s & ~handled & tp1_raw
    handled = handled | tp1

    tp2 = s & ~handled & tp2_raw
    handled = handled | tp2

    tp3 = s & ~handled & tp3_raw
    handled = handled | tp3

    unhandled = s & ~handled  # set N

    # ══════════════════════════════════════════════════════════════
    # 6. PairInCheck fallback for unhandled syndromes
    #    IsIn(a) = s(a) & a ∈ N = unhandled(a)
    #    PairInCheck(r',i): IsIn(r'-e_a) & IsIn(r'-e_b) & Check(r',t,i)
    #    If any PairInCheck fires at satisfied r':
    #      Flip(XX, r'-ex-ey-ez)
    #      S_in ← S_in ∪ {r'-ex, r'-ey, r'-ez, r'-ex-ey, r'-ex-ez,
    #                      r'-ey-ez, r'-ex-ey-ez}
    # ══════════════════════════════════════════════════════════════
    isin_xm1 = _roll(unhandled, +1, 0)  # IsIn(r'-ex)
    isin_ym1 = _roll(unhandled, +1, 1)  # IsIn(r'-ey)
    isin_zm1 = _roll(unhandled, +1, 2)  # IsIn(r'-ez)

    # PairInCheck(r',1): {a,b}={2,3} → IsIn(r'-ey) & IsIn(r'-ez) & Check(r',t,1)
    pic1 = isin_ym1 & isin_zm1 & check1
    # PairInCheck(r',2): {a,b}={1,3} → IsIn(r'-ex) & IsIn(r'-ez) & Check(r',t,2)
    pic2 = isin_xm1 & isin_zm1 & check2
    # PairInCheck(r',3): {a,b}={1,2} → IsIn(r'-ex) & IsIn(r'-ey) & Check(r',t,3)
    pic3 = isin_xm1 & isin_ym1 & check3

    cond_nin = sat & (pic1 | pic2 | pic3)

    # ══════════════════════════════════════════════════════════════
    # 7. S_in accumulation
    # ══════════════════════════════════════════════════════════════
    s_in = jnp.zeros_like(s)

    # cond_all: S_in ← {r-(0,1,1), r-(1,1,0), r-(1,0,1)}
    s_in = s_in | _shifted_3d(cond_all, 0, 1, 1)
    s_in = s_in | _shifted_3d(cond_all, 1, 1, 0)
    s_in = s_in | _shifted_3d(cond_all, 1, 0, 1)

    # tp1 (c=1): S_in ← {r-(0,1,0), r-(0,2,0), r-(0,0,1), r-(0,0,2)}
    s_in = s_in | _shifted_3d(tp1, 0, 1, 0)
    s_in = s_in | _shifted_3d(tp1, 0, 2, 0)
    s_in = s_in | _shifted_3d(tp1, 0, 0, 1)
    s_in = s_in | _shifted_3d(tp1, 0, 0, 2)

    # tp2 (c=2): S_in ← {r-(1,0,0), r-(2,0,0), r-(0,0,1), r-(0,0,2)}
    s_in = s_in | _shifted_3d(tp2, 1, 0, 0)
    s_in = s_in | _shifted_3d(tp2, 2, 0, 0)
    s_in = s_in | _shifted_3d(tp2, 0, 0, 1)
    s_in = s_in | _shifted_3d(tp2, 0, 0, 2)

    # tp3 (c=3): S_in ← {r-(1,0,0), r-(2,0,0), r-(0,1,0), r-(0,2,0)}
    s_in = s_in | _shifted_3d(tp3, 1, 0, 0)
    s_in = s_in | _shifted_3d(tp3, 2, 0, 0)
    s_in = s_in | _shifted_3d(tp3, 0, 1, 0)
    s_in = s_in | _shifted_3d(tp3, 0, 2, 0)

    # cond_nin: S_in ← {r'-(1,0,0), r'-(0,1,0), r'-(0,0,1),
    #                    r'-(1,1,0), r'-(1,0,1), r'-(0,1,1), r'-(1,1,1)}
    s_in = s_in | _shifted_3d(cond_nin, 1, 0, 0)
    s_in = s_in | _shifted_3d(cond_nin, 0, 1, 0)
    s_in = s_in | _shifted_3d(cond_nin, 0, 0, 1)
    s_in = s_in | _shifted_3d(cond_nin, 1, 1, 0)
    s_in = s_in | _shifted_3d(cond_nin, 1, 0, 1)
    s_in = s_in | _shifted_3d(cond_nin, 0, 1, 1)
    s_in = s_in | _shifted_3d(cond_nin, 1, 1, 1)

    # ══════════════════════════════════════════════════════════════
    # 8. Message update (fully vectorized over all 8 channels)
    # ══════════════════════════════════════════════════════════════
    # Roll the full (L,L,L,2,2,2) message tensor in 6 spatial directions
    # once, then select the correct neighbor per channel via concatenate.
    #   channel (i,j,k): x-nb at r+(-1)^i*ex  →  i=0: roll -1; i=1: roll +1
    m_i = m.astype(jnp.int8)
    m_xp = jnp.roll(m_i, -1, axis=0)   # m[x+1,y,z,...]
    m_xm = jnp.roll(m_i, +1, axis=0)   # m[x-1,y,z,...]
    m_yp = jnp.roll(m_i, -1, axis=1)
    m_ym = jnp.roll(m_i, +1, axis=1)
    m_zp = jnp.roll(m_i, -1, axis=2)
    m_zm = jnp.roll(m_i, +1, axis=2)

    nb_x = jnp.concatenate([m_xp[:,:,:,0:1,:,:], m_xm[:,:,:,1:2,:,:]], axis=3)
    nb_y = jnp.concatenate([m_yp[:,:,:,:,0:1,:], m_ym[:,:,:,:,1:2,:]], axis=4)
    nb_z = jnp.concatenate([m_zp[:,:,:,:,:,0:1], m_zm[:,:,:,:,:,1:2]], axis=5)

    # Broadcast spatial masks to (L,L,L,1,1,1)
    s_6d = s[:,:,:,None,None,None]
    sat_6d = sat[:,:,:,None,None,None]
    spread_6d = (sat & (counter == 0))[:,:,:,None,None,None]

    # --- Spread: where counter=0 and m_old=0, absorb any neighbor ---
    any_nb = (nb_x | nb_y | nb_z).astype(bool)
    sat_msgs = jnp.where(spread_6d & ~m, any_nb, m)

    # --- Toom-3D vote: self + 3 neighbors >= 2 (all channels at once) ---
    vote = (m_i + nb_x + nb_y + nb_z) >= 2
    v111 = vote[:,:,:,1,1,1]
    m111_old = m[:,:,:,1,1,1]
    failsafe = (m111_old & v111)[:,:,:,None,None,None]
    toom_result = vote | failsafe

    # Apply Toom where m_old=1 (spread only touched m_old=0, so no conflict)
    sat_msgs = jnp.where(m, toom_result, sat_msgs)

    # Compose: syndrome sites → all 1s; satisfied sites → sat_msgs
    m_new = jnp.where(s_6d, True, sat_msgs)

    # --- S_in: force all messages on ---
    m_new = jnp.where(s_in[:,:,:,None,None,None], True, m_new)

    # Counter update
    counter_new = (counter + 1) % CLOCK_PERIOD

    # ══════════════════════════════════════════════════════════════
    # 9. Corrective flips on the error vector
    #    Flip(OP, site, corner) applies OP at the (-x,-y,-z) corner
    #    of the given site, i.e. at lattice position site + (0,0,0).
    # ══════════════════════════════════════════════════════════════
    flip_q0 = jnp.zeros_like(e0)
    flip_q1 = jnp.zeros_like(e1)

    # cond_all: Flip(XI, r, -ex-ey-ez) → q0 at r
    flip_q0 = flip_q0 ^ cond_all

    # tp1 (c=1): Flip(XI, r, ...) → q0 at r
    #            Flip(IX, r-ey, ...) → q1 at r-(0,1,0)
    #            Flip(IX, r-ez, ...) → q1 at r-(0,0,1)
    flip_q0 = flip_q0 ^ tp1
    flip_q1 = flip_q1 ^ _shifted_3d(tp1, 0, 1, 0)
    flip_q1 = flip_q1 ^ _shifted_3d(tp1, 0, 0, 1)

    # tp2 (c=2): Flip(XI, r, ...) → q0 at r
    #            Flip(IX, r-ex, ...) → q1 at r-(1,0,0)
    #            Flip(IX, r-ez, ...) → q1 at r-(0,0,1)
    flip_q0 = flip_q0 ^ tp2
    flip_q1 = flip_q1 ^ _shifted_3d(tp2, 1, 0, 0)
    flip_q1 = flip_q1 ^ _shifted_3d(tp2, 0, 0, 1)

    # tp3 (c=3): Flip(XI, r, ...) → q0 at r
    #            Flip(IX, r-ex, ...) → q1 at r-(1,0,0)
    #            Flip(IX, r-ey, ...) → q1 at r-(0,1,0)
    flip_q0 = flip_q0 ^ tp3
    flip_q1 = flip_q1 ^ _shifted_3d(tp3, 1, 0, 0)
    flip_q1 = flip_q1 ^ _shifted_3d(tp3, 0, 1, 0)

    # cond_nin: Flip(XX, r', -ex-ey-ez) → q0,q1 at r'
    flip_q0 = flip_q0 ^ cond_nin
    flip_q1 = flip_q1 ^ cond_nin

    # Apply flips to error vector
    errors_new = jnp.stack([e0 ^ flip_q0, e1 ^ flip_q1], axis=-1)

    return (errors_new, m_new, counter_new, step_count + 1, CLOCK_PERIOD)


# --- GF(2) Linear Algebra for Z-Logical Computation ---

_CORNER_OFFSETS = [
    (0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0),
    (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1),
]

_Z_STAB_Q0 = [True, False, True, False, True, False, True, False]
_Z_STAB_Q1 = [True, True, False, True, False, False, False, True]
_X_STAB_Q0 = [False, False, True, False, True, True, True, False]
_X_STAB_Q1 = [False, True, False, True, False, True, False, True]


def _build_parity_checks(L):
    """Build H_Z, H_X as (L^3, 2L^3) uint8 matrices."""
    n_stab = L ** 3
    n_qubits = 2 * L ** 3
    H_Z = np.zeros((n_stab, n_qubits), dtype=np.uint8)
    H_X = np.zeros((n_stab, n_qubits), dtype=np.uint8)
    row = 0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for ci, (dx, dy, dz) in enumerate(_CORNER_OFFSETS):
                    ix, iy, iz = (i+dx)%L, (j+dy)%L, (k+dz)%L
                    site = ix * L * L + iy * L + iz
                    q0, q1 = site, L**3 + site
                    if _Z_STAB_Q0[ci]: H_Z[row, q0] = 1
                    if _Z_STAB_Q1[ci]: H_Z[row, q1] = 1
                    if _X_STAB_Q0[ci]: H_X[row, q0] = 1
                    if _X_STAB_Q1[ci]: H_X[row, q1] = 1
                row += 1
    return H_Z, H_X


def _pack_matrix(M):
    """Pack (m, n) uint8 {0,1} -> (m, ceil(n/64)) uint64 via packbits."""
    m, n = M.shape
    pad = (64 - n % 64) % 64
    if pad:
        M = np.hstack([M, np.zeros((m, pad), dtype=np.uint8)])
    return np.packbits(M, axis=1, bitorder='little').view(np.uint64)


def _unpack_matrix(P, n):
    """Unpack (m, nwords) uint64 -> (m, n) uint8."""
    raw = P.view(np.uint8)
    unpacked = np.unpackbits(raw, axis=1, bitorder='little')
    return unpacked[:, :n]


@numba.njit
def _gf2_rref_packed(P, nrows, ncols):
    """
    GF(2) RREF on bit-packed uint64 matrix (in-place).
    Returns (pivot_col_array, rank).
    """
    nwords = P.shape[1]
    pivot_row = 0
    pivot_cols = np.empty(min(nrows, ncols), dtype=np.int64)
    n_pivots = 0

    for col in range(ncols):
        w = col >> 6
        bit = col & 63
        mask_bit = np.uint64(1) << np.uint64(bit)

        found = -1
        for r in range(pivot_row, nrows):
            if P[r, w] & mask_bit:
                found = r
                break
        if found == -1:
            continue

        if found != pivot_row:
            for j in range(nwords):
                P[pivot_row, j], P[found, j] = P[found, j], P[pivot_row, j]

        for r in range(nrows):
            if r != pivot_row and (P[r, w] & mask_bit):
                for j in range(nwords):
                    P[r, j] ^= P[pivot_row, j]

        pivot_cols[n_pivots] = col
        n_pivots += 1
        pivot_row += 1

    return pivot_cols[:n_pivots], n_pivots


# Trigger JIT compilation on import
_gf2_rref_packed(_pack_matrix(np.eye(4, dtype=np.uint8)), 4, 4)


def _gf2_nullspace(M_uint8):
    """Nullspace of (m, n) uint8 GF(2) matrix via bit-packed rref."""
    m, n = M_uint8.shape
    P = _pack_matrix(M_uint8)
    pivot_cols_arr, rank = _gf2_rref_packed(P, m, n)
    pivot_cols = list(pivot_cols_arr)

    A = _unpack_matrix(P, n)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    k = len(free_cols)
    if k == 0:
        return np.zeros((0, n), dtype=np.uint8)

    basis = np.zeros((k, n), dtype=np.uint8)
    for bi, fc in enumerate(free_cols):
        basis[bi, fc] = 1
        for pi, pc in enumerate(pivot_cols):
            if A[pi, fc]:
                basis[bi, pc] = 1
    return basis


_z_logicals_cache = {}


def compute_z_logicals(L, verbose=True):
    """
    Compute Z-logical operators for Haah's code on an L^3 torus.
    Uses bit-packed GF(2) Gaussian elimination with numba JIT.
    Results are cached per-L.
    """
    if L in _z_logicals_cache:
        logical_matrix = _z_logicals_cache[L]
        if verbose:
            print(f"L={L} (cached): {logical_matrix.shape[0]} Z-logicals")
        return logical_matrix

    n_qubits = 2 * L ** 3

    t0 = time.time()
    H_Z, H_X = _build_parity_checks(L)
    t_build = time.time() - t0

    t0 = time.time()
    K = _gf2_nullspace(H_X)
    t_kern = time.time() - t0

    t0 = time.time()
    P_hz = _pack_matrix(H_Z.copy())
    _, r_z = _gf2_rref_packed(P_hz, H_Z.shape[0], n_qubits)
    r_z = int(r_z)

    stacked = np.vstack([H_Z, K])
    P_st = _pack_matrix(stacked)
    _, total_rank = _gf2_rref_packed(P_st, stacked.shape[0], n_qubits)
    total_rank = int(total_rank)

    k_log = total_rank - r_z
    logicals = _unpack_matrix(P_st[r_z:r_z + k_log], n_qubits).astype(np.int8)
    t_quot = time.time() - t0

    _z_logicals_cache[L] = logicals

    if verbose:
        print(f"L={L}, n_qubits={n_qubits}")
        print(f"  Build H_Z, H_X: {t_build:.3f}s")
        print(f"  ker(H_X): dim={K.shape[0]}  ({t_kern:.3f}s)")
        print(f"  rank(H_Z)={r_z}, logicals={k_log}  ({t_quot:.3f}s)")
        print(f"  Total: {t_build + t_kern + t_quot:.3f}s")

    return logicals


# --- Logical Error Check ---

@jax.jit
def check_logical_error(errors, logicals_matrix):
    """Check for logical error using precomputed Z-logical matrix."""
    ef = errors.transpose(3, 0, 1, 2).reshape(-1).astype(jnp.int8)
    parities = (logicals_matrix @ ef) % 2
    return jnp.any(parities > 0)


# --- State Initialization ---

@partial(jax.jit, static_argnames=['L', 'CLOCK_PERIOD'])
def init_state(key, L, p, CLOCK_PERIOD=CLOCK_PERIOD):
    """
    Initialize the state: errors, messages, counter, step_count, CLOCK_PERIOD
    """
    subkey, _ = jax.random.split(key)
    errors = jax.random.uniform(subkey, (L, L, L, 2)) < p
    messages = jnp.zeros((L, L, L, 2, 2, 2), dtype=bool)
    counter = jnp.zeros((L, L, L), dtype=jnp.int8)
    step_count = 0
    return (errors, messages, counter, step_count, CLOCK_PERIOD)


# --- Simulation (Monte Carlo) ---

@partial(jax.jit, static_argnames=['L', 'max_steps', 'CLOCK_PERIOD'])
def run_single_trajectory(key, L, p, max_steps, CLOCK_PERIOD, logicals_matrix):
    """
    Run one trajectory using while_loop for early termination.
    Returns (is_failure, step_count) where is_failure is 1.0 if logical error or not converged.
    """
    initial_state = init_state(key, L, p, CLOCK_PERIOD)

    def cond_fun(state):
        errors, m, c, step, _ = state
        syndrome = calculate_syndrome(errors, L)
        not_cleared = jnp.any(syndrome)
        under_limit = step < max_steps
        return not_cleared & under_limit

    final_state = jax.lax.while_loop(cond_fun, step_ca, initial_state)

    errors_fin, _, _, steps, _ = final_state

    # Check convergence
    s_fin = calculate_syndrome(errors_fin, L)
    not_converged = jnp.any(s_fin)

    # Check logical error
    correction_failed = check_logical_error(errors_fin, logicals_matrix)

    # Failure if logical error OR didn't converge
    is_failure = jnp.logical_or(correction_failed, not_converged).astype(jnp.float32)

    return is_failure, steps.astype(jnp.float32)


def run_monte_carlo(L_values, p_values, num_samples, filename="haah_ca_data.npz"):
    """
    Run Monte Carlo simulation for multiple L and p values.
    Saves results incrementally to filename.
    """
    print(f"Running JAX Haah Code CA Decoder Evaluation...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")

    results = {}
    master_key = jax.random.PRNGKey(42)

    for L in L_values:
        max_steps = 5* CLOCK_PERIOD* L

        # Precompute Z-logicals once for this L
        logicals = compute_z_logicals(L, verbose=True)
        logicals_jax = jnp.array(logicals, dtype=jnp.int8)

        @partial(jax.jit, static_argnames=['CLOCK_PERIOD'])
        def batch_run(keys, p_val, CLOCK_PERIOD=CLOCK_PERIOD):
            return jax.vmap(lambda k: run_single_trajectory(
                k, L, p_val, max_steps, CLOCK_PERIOD, logicals_jax))(keys)

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


def run_decoding_time_sweep(L_values, p_values, num_samples, filename="haah_ca_decoding_time.npz", CLOCK_PERIOD=CLOCK_PERIOD):
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
    print(f"Running Haah Code Decoding Time Sweep...")
    print(f"L values: {L_values}")
    print(f"p values: {p_values}")
    print(f"Samples per point: {num_samples}")

    results = {}
    master_key = jax.random.PRNGKey(42)

    for L in L_values:
        max_steps = 10 * L

        # Precompute Z-logicals once for this L
        logicals = compute_z_logicals(L, verbose=True)
        logicals_jax = jnp.array(logicals, dtype=jnp.int8)

        @partial(jax.jit, static_argnames=['CLOCK_PERIOD'])
        def batch_run(keys, p_val, CLOCK_PERIOD=CLOCK_PERIOD):
            return jax.vmap(lambda k: run_single_trajectory(
                k, L, p_val, max_steps, CLOCK_PERIOD, logicals_jax))(keys)

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

def plot_results(data_filename, plot_filename="haah_ca_results.png"):
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
    ax1.set_title('Haah Code CA Decoder Threshold')

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
    #     L_values=[8, 12, 16],
    #     p_values=np.linspace(0.001, 0.01, 7),
    #     num_samples=10_000,
    #     filename="haah_ca_results_new.npz",
    # )
    # plot_results("haah_ca_results_new.npz", "haah_ca_results_new.png")
    compute_z_logicals(5, verbose=True)
