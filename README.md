# local-decoders

Implementations of our local decoders in JAX.

An interactive visualization of the all the decoders can be found at
**<https://local-decoders.github.io/>**.

## Repository structure

```
local-decoders/
├── code_capacity/                 # code-capacity noise model (perfect syndromes)
│   ├── repetition_ca_jax.py       # 1D repetition code, synchronous global clock
│   ├── repetition_ca_jax_uncoord.py  # ... uncoordinated (asynchronous) variant
│   ├── toric_ca_rgb.py            # 2D toric code, three-colour (b/r/g) message CA
│   ├── toric_ca_rgb_uncoord.py    # ... uncoordinated variant
│   ├── surface_ca.py              # 2D planar surface code with boundaries + splitting
│   ├── haah_code_ca.py            # Haah's cubic code (3D, fracton)
│   ├── xcube_fracton_ca.py        # X-cube model, fracton (cube-stabilizer) sector
│   └── xcube_lineon_ca.py         # X-cube model, lineon (vertex-stabilizer) sector
├── phenomenological/              # phenomenological noise model (noisy syndromes)
│   └── surface_ca.py              # surface code memory experiment with splitting
└── pyproject.toml
```

### `code_capacity/`

Code-capacity simulations: uniform depolarizing noise is applied to the qubits, the
syndromes are measured perfectly, and the local cellular automaton (CA) decoder runs until either all nontrivial syndromes are cleared or a maximum number of steps is reached. The decoder is then checked for logical error.

This folder contains our hand-designed CA rules (as opposed to those that arise from our general construction for any translation invariant Pauli stabilizer code) for locally decoding

- the **1D repetition code** (`repetition_ca_jax.py`),
- the **2D toric code** (`toric_ca_rgb.py`),
- the **2D unrotated surface code**  (`surface_ca.py`),
- **Haah's cubic code** (`haah_code_ca.py`), and
- the **fractons** and **lineons** of the **X-cube model**
  (`xcube_fracton_ca.py`, `xcube_lineon_ca.py`),

The `*_uncoord.py` files are
asynchronous versions of the same rules: instead of every site updating in
lockstep on a shared clock, one randomly chosen site updates per step, in order to test the robustness of our rules to unsynchronised hardware.

### `phenomenological/`

Memory experiments with phenomenological noise on the surface code using our local streaming decoder. In each round, every data qubit undergoes a bit-flip error with probability `p`, and every syndrome measurement is independently flipped with probability `p`. The decoder processes the syndrome stream online as new data and measurement errors occur.
