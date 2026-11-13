"""
rigid_refine: REFINE with rigid sugar beads.

Each sugar unit keeps its template geometry exactly (rotation and height
only, xy restrained to the experimental position); bridging oxygens and
non sugar atoms are free; MMFF94 plus the slab compression act on all atoms
and are projected on the beads.

    from src.rotation_optimization.rigid_refine import refine_structure, RigidRefineConfig
    mol_out, report = refine_structure(mol_with_h, mol_no_h, chain_dict,
                                       config['glycosidic_bonds'])
"""
from .refine import refine_structure, RigidRefineConfig
from .beads import build_beads, bead_rigidity, template_deviation, find_linkages
