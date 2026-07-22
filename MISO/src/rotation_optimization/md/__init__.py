"""
md — force-field optimisation and ring constraint dynamics.

Sub-modules
-----------
config
    OptimizationConfig and all MD dataclasses.  Import from here to avoid
    pulling in the full MD machinery.
ring_functions
    Pyranose ring detection, integrity checking, and dual-mode ring-constraint
    dynamics (constrained CD-mode and free 3D-mode).
force_field_optimization
    Three-phase optimisation driver: constrained minimisation → Langevin MD
    with slab compression → free final minimisation.
utils
    Molecule-level utilities: geometry checks, overlap fixing, valence repair,
    file saving, convergence checking.

Typical import in pipeline.py
------------------------------
    from src.rotation_optimization.md.config import OptimizationConfig
    from src.rotation_optimization.md import ring_functions, force_field_optimization, utils
"""