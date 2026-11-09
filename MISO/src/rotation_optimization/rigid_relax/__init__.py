"""
rigid_relax — bonded-only, rotation-only "correct covalency" pass.

Run *before* the slab-compression MD (``optimize_with_slab_and_rings``) to
repair very deformed structures.  Every sugar monomer is treated as a RIGID
bead: its centre of mass is pinned at the STM position and only its orientation
may change.  The only energy is a bonded-only force field on the linkages
(no van der Waals, no electrostatics, no gravity, no slab):

    * a donor-C···acceptor-C distance term (linkage feasibility), which is
      equivalent to letting the free bridging oxygen relax its bond+angle
      springs to zero, and
    * a glycosidic dihedral φ restrained to an α/β reference read from the
      input ``glycosidic_bonds`` — this is the "correct the rotation" term.

Because the sugars are rigid the pyranose rings cannot deform; the pass only
rotates intact beads about their fixed COMs.

Sub-modules
-----------
config
    RigidRelaxConfig dataclass (defaults from src/constants.py).
rigid_relax
    Energy, deterministic bridging-oxygen placement, and the finite-difference
    rigid-body descent driver that operates on chain_dict.

Typical use (pipeline.py)
-------------------------
    from src.rotation_optimization.rigid_relax import rigid_relax
    from src.rotation_optimization.rigid_relax.config import RigidRelaxConfig
    chain_dict = rigid_relax.relax_chain(chain_dict, glycosidic_bonds)
"""
