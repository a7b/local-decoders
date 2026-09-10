# local-decoders

JAX implementations of the cellular-automaton decoders used in *Local
Memory*. The repository is organized by noise model:

- [`code_capacity/`](code_capacity/) contains the original runnable JAX
  scripts, with only the rule and setting corrections listed in
  [`IMPLEMENTATION_DIFFS.md`](code_capacity/IMPLEMENTATION_DIFFS.md).
- [`phenomenological/`](phenomenological/) contains the JAX surface-code
  phenomenological-memory simulation.

The original APIs have been kept. In particular, the modules still expose
`init_state`, `step_ca` (or the original single-site update helper),
`run_monte_carlo`, plotting/GIF functions, and the decoding-time sweeps that
were present in the source files. Toric and surface messages are still named
`b`, `r`, and `g`.

## Install

Python 3.10 or newer is required. For a CPU installation:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

Install the appropriate `jax`/`jaxlib` build separately when using a GPU.

## Run the original drivers

Call the functions in the individual modules. For example:

```python
from code_capacity import repetition_ca_jax
from code_capacity import toric_ca_rgb

repetition_ca_jax.run_monte_carlo(
    [16, 32], [0.2, 0.3], 1_000, "rep_smoke.npz"
)
repetition_ca_jax.run_decoding_time_sweep(
    [31, 100], [0.1], 1_000, "rep_tdec_smoke.npz"
)

toric_ca_rgb.run_monte_carlo(
    [7, 13], [0.04, 0.06], 1_000, "toric_smoke.npz"
)
toric_ca_rgb.run_decoding_time_sweep(
    [8, 16], [0.01], 1_000, "toric_tdec_smoke.npz"
)
```

Publication grids, seeds, horizons, and sample totals copied from the
registered `numerics/` campaigns are in
[`numerical_settings.py`](code_capacity/numerical_settings.py),
[`repetition_campaign.py`](code_capacity/repetition_campaign.py), and
[`toric_campaign.py`](code_capacity/toric_campaign.py).

See [`DISCREPANCIES.md`](code_capacity/DISCREPANCIES.md) for the paper/code
audit and [`SOURCES.md`](code_capacity/SOURCES.md) for source hashes.
