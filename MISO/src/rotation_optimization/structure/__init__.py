"""
structure — RDKit molecule construction and SDF export.

Sub-modules
-----------
bonds_creation
    Geometry constructors for glycosidic, phosphodiester and IK-based
    phosphate bonds.  Returns position data; does not modify RDKit molecules.
petn_building
    Geometry constructor for phosphoethanolamine (PEtN) modifications.
lipid_building
    Geometry constructor for lipid chains (ester / amide / ether).
    Uses FABRIK IK (src/IK/fabrikSolver) for the backbone.
building_chain
    High-level alignment pipeline: calls rotation_search algorithms, assembles
    chain_dict, creates glycosidic bonds, adds hydrogens via OpenBabel.
peptide_building
    RDKit-based peptide construction, Cα alignment, glycosylation site finding,
    and sugar-peptide linkage geometry.
saving_lps
    SDF export: combines all fragments, removes condensation atoms, adds
    bridging atoms, enforces trans dihedrals, restores experimental COMs.
"""