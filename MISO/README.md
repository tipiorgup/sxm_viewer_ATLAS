# MISO

**Author:** CLGF  
**Journal:** Journal of Chemical Information and Modeling (JCIM)  
**Status:** Manuscript under review — citation to be added upon publication.

---

## Overview

This pipeline builds atomistic 3D models of glycolipids and glycopeptides
directly from scanning tunnelling microscopy (STM) experimental data.
Molecular positions are seeded from STM circle coordinates; conformer
generation, rotational alignment, glycosidic/phosphate/lipid bond geometry,
and force-field optimisation are then applied to produce energy-minimised
structures exported as SDF files.

The pipeline supports:

- Glycolipid structures (sugar chains + phosphate, PEtN, lipid tails)
- Glycopeptide structures (sugar chains + cyclic or linear peptides)
- Multiple independent polymer replicates per run
- Spatially-varying surface gravity from a pre-computed STM height grid

---

## Repository layout

```
Glycolipid/
├── module_B.py               Entry point — argument parsing, timer, polymer loop
├── pipeline.py               Thin facade over src/ called by module_B.py
├── src/
│   ├── constants.py          All numerical defaults (env-var overridable)
│   ├── monomer_building/
│   │   └── monomer_building.py   Conformer generation, Cremer-Pople analysis,
│   │                             rigid-body extraction, STM position translation
│   └── rotation_optimization/
│       ├── geometry/             Pure vector and rotation math (numpy only)
│       │   └── geometry_utils.py
│       ├── rotation/             QUEST rotation search, clash/ring filtering
│       │   └── rotation_search.py
│       ├── structure/            RDKit molecule construction and SDF export
│       │   ├── bonds_creation.py
│       │   ├── petn_building.py
│       │   ├── lipid_building.py
│       │   ├── building_chain.py
│       │   ├── peptide_building.py
│       │   └── saving_lps.py
│       └── md/                   Force-field optimisation and ring dynamics
│           ├── config.py         OptimizationConfig and MD dataclasses
│           ├── ring_functions.py
│           ├── force_field_optimization.py
│           └── utils.py
├── src_stm/                  STM image analysis utilities (not used by pipeline)
├── Inputs/                   YAML config files and coordinate CSVs
├── ARCHITECTURE.md           Data contracts, pipeline flow, YAML schema, known issues
├── MIGRATION.md              Import path changes from the flat to sub-package layout
└── README.md                 This file
```

---

## Dependencies

Tested on Python 3.10+ (Windows and Linux).  The full pinned environment is in
`requirements_cg_windows.txt`.  The minimum required packages are:

| Package | Purpose |
|---------|---------|
| `rdkit` | Molecule handling, conformer generation, SDF I/O |
| `openbabel-wheel` | Hydrogen addition |
| `numpy` | All numerical operations |
| `scipy` | Rotation (QUEST), interpolation (STM grid), optimisation |
| `pandas` | CSV loading |
| `PyYAML` | Input file parsing |
| `matplotlib` | Conformer RMSD plots |

> **Note:** `lipid_building.py` additionally requires the `IK/fabrikSolver`
> sub-package (FABRIK inverse kinematics). Ensure it is present in `src/IK/`
> before running inputs that include lipid chains.

---

## Installation

```bash
git clone <repository-url>
cd Glycolipid

# Create and activate a virtual environment (recommended)
python -m venv env
source env/bin/activate          # Linux / macOS
env\Scripts\activate             # Windows

pip install -r requirements_cg_windows.txt
```

---

## Usage

### Running a single input

```bash
python module_B.py \
    --input_file Inputs/glycopeptide_input_file.yaml \
    --iterations 100 \
    --n_polymers 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input_file` / `-i` | required | Path to the YAML configuration file |
| `--iterations` / `-n` | 100 | Rotation search samples per monomer during alignment |
| `--n_polymers` / `-p` | 5 | Number of independent polymer structures to generate |
| `--faces` | `enumerate` | `enumerate`: every polymer is built in a different face state (see below). `legacy`: previous behaviour |
| `--seed` | 0 | Seed for face state sampling and fallback rotations; same seed, same polymers |
| `--duplicate_rmsd` | 0.25 | Assembled polymers closer than this (A) to an earlier one are reported as near duplicates |
| `--no_screen` | off | Skip the per sugar face pre screen and try every face state |
| `--no_joint_rotation` | off | Skip the joint rotation of all sugars after assembly |
| `--max_gap` | 3.0 | Reject a face state if a linkage C...C stays above this after the joint rotation (A) |
| `--pool_factor` | 4 | Face states held in reserve per requested polymer, to replace rejected ones |

### Structural diversity: face states

A sugar ring deposited on a surface can lie with either face up.  A face
state assigns one face to every ring: `0` = oriented ring normal (O5, C1 to C5
order) pointing up, `1` = pointing down, one character per monomer in
positioning order (root first).  A glucose dimer has four states: 00, 01,
10, 11.

With `--faces enumerate` polymer *i* is built in state *i*: every monomer,
the root included, is rotated about its fixed experimental COM so that it
presents its face, with its linkage carbons aimed at their partners.  If
`--n_polymers` is at least 2^k (k rings) all states are built, in binary
order, and the run is capped at 2^k polymers; otherwise the all `0` state
plus a seeded random sample of distinct states.  The chosen states are
written to `<prefix>_face_states.csv`, and the log reports the requested and
assembled state of each polymer.

States that differ by turning the whole molecule over (00 and 11 for a
dimer) share the same internal conformation but present opposite faces to
the surface, so they are distinct adsorption structures.

#### Branched glycans: valid states only

In a branched glycan most face states cannot be built: a sugar with several
linkages must point C1 at its parent and C3/C4/C6 at its children with one
rigid orientation, and the order of those carbons around the ring, seen from
above, reverses with the face.  Three steps deal with this:

1. **Face pre screen** (`face_screening.screen_faces`, milliseconds): for
   every sugar and face, is there an orientation where all its linkage
   carbons are within reach of their partners?  Faces that fail are never
   used.  The log lists sugars whose face is fixed by the layout.  On a
   15 glucose test tree it fixed 4 faces, leaving 2048 of 32768 states.
2. **Joint rotation** (`joint_rotation.joint_rigid_rotation`): after the
   sequential assembly, all sugars rotate together about their fixed COMs
   (rings rigid) to close every linkage, keep the requested faces, bring the
   two replaced hydroxyl oxygens together and avoid clashes.  About 0.1 to
   1.5 s for 15 sugars.
3. **Rejection**: a state whose linkages stay open (`--max_gap`) or whose
   faces cannot be kept is skipped and the next state from the pool is
   tried.  `<prefix>_face_states.csv` lists every state tried, accepted or
   rejected, with the worst linkage distance and the reason.

### Running on a cluster (LoadLeveler)

Edit the `INPUT_PREFIXES` array and resource directives in
`module_B_scratch.job`, then submit:

```bash
llsubmit module_B_scratch.job

# Override optimisation constants without editing source:
DEFAULT_GRAVITY=0.055 POLYMERS=20 llsubmit module_B_scratch.job
```

Results are written to `${PROJECT_DIR}/results/`.

---

## Input files

Each run requires three files per prefix, placed in `Inputs/`:

| File | Description |
|------|-------------|
| `<prefix>_input_file.yaml` | Structure definition (see below) |
| `<prefix>_coordinates.csv` | STM circle positions — columns `X (Angstrom)`, `Y (Angstrom)`, `Height`, `Z (Angstrom)` |
| `<prefix>.npz` *(optional)* | Pre-computed STM height grid — keys `x[W]`, `y[H]`, `z[W,H]` in Å |

### YAML configuration

A minimal glycopeptide example:

```yaml
sxm_file: 'glycopeptide.sxm'
circle_input_path: 'glycopeptide_coordinates.csv'
stm_grid_path: 'glycopeptide.npz'         # omit or set null to use uniform gravity

sugars:
  Glc: 'C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O'

experimental_positions:
  Glc: [0, 1]                             # two circles → keys become Glc_0, Glc_1

conformer_parameters:
  num_conformers: 100
  max_keep: 1

conformer_selection:
  strategy: lowest_energy                 # or random

root_mol: 'Glc_1'                         # fixed anchor monomer

direction:
  - ['Glc_0', 'C1', 'Glc_1', 'C4']       # alignment directive (4-element short form)

glycosidic_bonds:
  - ['Glc_0', 'C1', 'Glc_1', 'C4', 'beta', 'Glc-Glc beta(1-4)']

peptide:
  sequence: 'A-F-N'
  residues:
    - aa: 'Asn'
      ca_position: 2                      # circle index for Cα
      functional_position: 3
    - aa: 'Phe'
      ca_position: 4
      functional_position: 5
    - aa: 'Ala'
      ca_position: 6
      functional_position: null

peptide_bonds:
  - sugar_mol: 'Glc_1'
    sugar_carbon: 'C1'
    aa_type: 'Asn'
    residue_index: 2
    glycosylation_type: 'N-glycosidic'
    anomeric_config: 'beta'
    use_spacer: false
```

The full YAML schema with all optional sections (phosphate, PEtN, lipid
chains, orientation constraints, cyclic peptides) is documented in
`ARCHITECTURE.md`.

---

## Output files

For each polymer replicate the following SDF files are written to the working
directory (or `results/` when running via the job script):

| File | Description |
|------|-------------|
| `<prefix>_p<N>_<strategy>_pre_opt.sdf` | Structure with hydrogens, before optimisation |
| `<prefix>_p<N>_<strategy>_optimized.sdf` | Final optimised structure |
| `<prefix>_p<N>_<strategy>_partial.sdf` | Last valid structure if optimisation did not fully converge |
| `<prefix>_p<N>_<strategy>_phase1_minimized.sdf` | After constrained minimisation (phase 1) |
| `<prefix>_p<N>_<strategy>_phase2_compressed.sdf` | After slab compression MD (phase 2) |
| `<prefix>_molecule_data.pkl` | Serialised monomer data dict; can be reused to re-optimise without repeating conformer generation |

`<strategy>` is the conformer selection strategy set in the YAML
(`lowest_energy` or `random`). `<N>` is the polymer index (1-based).

---

## Tuning optimisation constants

All numerical constants in `src/constants.py` can be overridden via
environment variables without touching Python source.  Commonly tuned values:

```bash
export DEFAULT_GRAVITY=0.055           # surface gravity scale
export DEFAULT_COMPRESSION_STEPS=1500  # MD steps in slab compression phase
export DEFAULT_SLAB_FORCE_SCALE=0.075  # slab force magnitude
export DEFAULT_TIMESTEP=0.05           # Langevin timestep
```

The full list with units and descriptions is on `src/constants.py`.

---

## Further documentation

`ARCHITECTURE.md` at the project root covers the complete pipeline execution
flow, the `molecule_data_dict` and `chain_dict` data contracts, the full YAML
schema, and a table of known issues.
