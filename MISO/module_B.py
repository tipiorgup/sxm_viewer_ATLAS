import os
import time
import argparse
import pickle
from contextlib import contextmanager
import pipeline as lpf
from rdkit.Geometry import rdGeometry
import numpy as np


# ============================================================================
# TIMING UTILITY
# ============================================================================

class PipelineTimer:
    def __init__(self):
        self.records = []  # List of (label, elapsed, level)
        self._stack = []

    @contextmanager
    def section(self, label, level=0):
        start = time.perf_counter()
        self._stack.append(label)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.records.append((label, elapsed, level))
            self._stack.pop()

    def _fmt(self, seconds):
        if seconds >= 3600:
            return f"{seconds/3600:.2f} hr"
        elif seconds >= 60:
            return f"{seconds/60:.2f} min"
        else:
            return f"{seconds:.3f} s"

    def report(self):
        print("\n" + "="*70)
        print("TIMING REPORT")
        print("="*70)

        # Top-level total (level==0 entries)
        top_level = [(l, e) for l, e, lv in self.records if lv == 0]
        total = sum(e for _, e in top_level)

        for label, elapsed, level in self.records:
            indent = "  " * level
            time_str = self._fmt(elapsed)

            if level == 0:
                pct = (elapsed / total * 100) if total > 0 else 0
                print(f"  {indent}{label:<50} {time_str:>12}  ({pct:5.1f}%)")
            else:
                print(f"  {indent}{label:<50} {time_str:>12}")

        print("-"*70)
        print(f"  {'TOTAL':<50} {self._fmt(total):>12}")
        print("="*70)

    def polymer_summary(self, n_polymers):
        """Print per-polymer timing breakdown."""
        print("\n" + "="*70)
        print("PER-POLYMER TIMING BREAKDOWN")
        print("="*70)

        for p in range(1, n_polymers + 1):
            tag = f"P{p}"
            polymer_records = [(l, e, lv) for l, e, lv in self.records
                               if l.startswith(tag + " ")]
            if not polymer_records:
                continue

            polymer_total = next((e for l, e, lv in polymer_records
                                  if lv == 1 and "total" in l.lower()), None)

            print(f"\n  Polymer {p}:")
            for label, elapsed, level in polymer_records:
                indent = "  " * (level - 1)
                time_str = self._fmt(elapsed)
                if polymer_total and level > 1:
                    pct = (elapsed / polymer_total * 100)
                    print(f"    {indent}{label:<48} {time_str:>12}  ({pct:5.1f}%)")
                else:
                    print(f"    {indent}{label:<48} {time_str:>12}")

        print("="*70)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build and optimize LPS structures from YAML configuration"
    )
    parser.add_argument(
        "--input_file", "-i",
        type=str,
        required=True,
        help="Path to the YAML configuration file"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=100,
        help="Number of iterations for molecule alignment (default: 10000)"
    )
    parser.add_argument(
        "--n_polymers", "-p",
        type=int,
        default=5,
        help="Number of independent polymers to generate (default: 5)"
    )
    parser.add_argument(
        "--phase1_kicks",
        action="store_true",
        default=False,
        help="Enable stochastic kicks during phase 1 minimization to escape torsional local minima"
    )
    parser.add_argument(
        "--debug_checkpoints",
        action="store_true",
        default=False,
        help="Save intermediate SDF structures (pre-opt, post-phase1, post-phase2 for atom "
             "runs; pre-refine for sugar only rigid runs) next to the final structure"
    )
    parser.add_argument(
        "--faces",
        choices=["enumerate", "legacy"],
        default="enumerate",
        help="enumerate: each polymer is built in a different face state "
             "(which face of every ring points up: 00, 01, 10, 11 for a dimer). "
             "legacy: previous behaviour, polymers differ only by random fallback sampling."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for face state sampling and fallback rotations (default: 0)"
    )
    parser.add_argument(
        "--duplicate_rmsd",
        type=float,
        default=0.25,
        help="Assembled polymers closer than this RMSD (Angstrom) to an earlier "
             "polymer are reported as duplicates (default: 0.25)"
    )
    parser.add_argument(
        "--no_screen",
        action="store_true",
        help="Do not pre screen faces per sugar (try every face state)"
    )
    parser.add_argument(
        "--no_joint_rotation",
        action="store_true",
        help="Skip the joint rotation of all sugars about their fixed COMs after assembly"
    )
    parser.add_argument(
        "--max_gap",
        type=float,
        default=3.0,
        help="Reject a face state if any linkage C...C stays above this after the joint "
             "rotation (Angstrom, default: 3.0)"
    )
    parser.add_argument(
        "--pool_factor",
        type=int,
        default=4,
        help="Face states kept in reserve per requested polymer, to replace rejected "
             "states (default: 4)"
    )
    parser.add_argument(
        "--refine",
        choices=["rigid", "atom", "auto"],
        default="auto",
        help="rigid: REFINE with rigid sugar beads (templates kept exactly, xy at the "
             "experimental position, z free, bridging oxygens free). atom: the previous "
             "atom level MMFF minimisation and Langevin MD. auto (default): rigid for "
             "sugar only systems, atom when a peptide or lipids are present."
    )
    parser.add_argument(
        "--r0_xy",
        type=float,
        default=0.5,
        help="Rigid REFINE: free radius (Angstrom) of each sugar COM around its "
             "experimental xy before the restraint acts (default: 0.5)"
    )
    parser.add_argument(
        "--ceiling_margin",
        type=float,
        default=1.0,
        help="Rigid REFINE: ceiling height above the thickest rigid sugar (Angstrom, default: 1.0)"
    )
    args = parser.parse_args()
    return args


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main pipeline execution - generates N independent optimized polymers."""

    timer = PipelineTimer()
    pipeline_start = time.perf_counter()

    # ========================================================================
    # SETUP
    # ========================================================================
    args = parse_arguments()
    config = lpf.load_config(args.input_file)

    # Extract configuration parameters
    sugars = config.get('sugars', {})
    num_conformers = config.get('conformer_parameters', {}).get('num_conformers', 20)
    max_keep = config.get('conformer_parameters', {}).get('max_keep', 3)
    conformers_selection = config.get('conformer_selection', {}).get('strategy', 'lowest_energy')
    base_name = os.path.splitext(os.path.basename(config['sxm_file']))[0]
    circles, brightness = lpf.load_circle_data(config['circle_input_path'])

    stm_npz_path = config.get('stm_grid_path', None)
    if stm_npz_path is not None and os.path.exists(stm_npz_path):
        print(f"\n✓ Found STM grid: {stm_npz_path}")
    else:
        if stm_npz_path is not None:
            print(f"\n⚠ STM grid path provided but not found: {stm_npz_path}")
        else:
            print(f"\n⚠ No STM grid found, using fallback gravity")
        stm_npz_path = None

    print("="*70)
    print("LPS STRUCTURE BUILDING AND OPTIMIZATION PIPELINE")
    print(f"GENERATING {args.n_polymers} INDEPENDENT POLYMERS")
    print("="*70)
    print(f"\nConfiguration:      {args.input_file}")
    print(f"Base output name:   {base_name}")
    print(f"Conformer strategy: {conformers_selection}")
    print(f"Alignment iters:    {args.iterations}")
    print(f"Number of polymers: {args.n_polymers}")
    print(f"STM grid:           {stm_npz_path or 'not provided (fallback gravity)'}")

    # ========================================================================
    # PHASE 1: MONOMER BUILDING (SHARED - done once)
    # ========================================================================

    print("\n" + "="*70)
    print("PHASE 1: MONOMER BUILDING (shared for all polymers)")
    print("="*70)

    if sugars:
        with timer.section("Phase 1: Monomer Building (shared)", level=0):
            with timer.section("  Conformer generation", level=1):
                conformers = lpf.generate_conformers(sugars, num_conformers, max_keep)
            with timer.section("  Conformer analysis", level=1):
                lpf.analyze_conformers(conformers)
            with timer.section("  Monomer data extraction", level=1):
                monomer_data = lpf.extract_monomer_data(conformers)
            with timer.section("  Translation to experimental positions", level=1):
                translated_conformers, carbon_vectors = lpf.translate_conformers_to_positions(
                    monomer_data, config['experimental_positions'], circles
                )
            with timer.section("  Build molecule dict", level=1):
                molecule_data_dict, mol_name_to_conformer = lpf.build_molecule_dict(
                    translated_conformers, config['experimental_positions'],
                    conformers_selection, brightness
                )
            with timer.section("  Pickle molecule data", level=1):
                output_file = f'{base_name}_molecule_data.pkl'
                with open(output_file, 'wb') as f:
                    pickle.dump(molecule_data_dict, f)
                print(f"\n✓ Exported molecule_data_dict to: {output_file}")
                print(f"  This file can be reused for re-optimization of any polymer")
    else:
        conformers = {}
        molecule_data_dict = {}
        mol_name_to_conformer = {}

    # ========================================================================
    # GENERATE N INDEPENDENT POLYMERS
    # ========================================================================

    all_polymer_results = []

    # ------------------------------------------------------------------------
    # Face states: one per polymer, so polymers differ structurally
    # ------------------------------------------------------------------------
    face_states = None
    face_mol_order = []
    assembled_coords = []          # (polymer_idx, label, coords) for duplicate checks
    state_log = []                 # (label, status, polymer_idx, worst_gap, reason)
    rigid_log = []                 # (polymer_idx, label, rigid REFINE report)
    if sugars and args.faces == "enumerate":
        face_mol_order, face_states, screening = lpf.plan_face_states(
            molecule_data_dict, config, args.n_polymers, seed=args.seed,
            pool_factor=args.pool_factor, screen=not args.no_screen)
        print("\n" + "="*70)
        print("FACE STATES")
        print("="*70)
        print(f"  Monomer order: {' '.join(face_mol_order)}   ('0' = face +1 up, '1' = face -1 up)")
        if screening is not None:
            n_valid = 1
            for m in face_mol_order:
                n_valid *= len(screening.get(m, {'valid': [1, -1]})['valid'])
            print(f"  Face pre screen: {n_valid} of {2 ** len(face_mol_order)} states possible")
            for m in face_mol_order:
                sc = screening.get(m)
                if sc and sc['note'] in ('face fixed by layout', 'neither face fits'):
                    face = ''.join('0' if f == 1 else '1' for f in sc['valid'])
                    print(f"    {m}: {sc['note']} (allowed {face}; slack face0 "
                          f"{sc['slack'][1]:+.2f} A, face1 {sc['slack'][-1]:+.2f} A)")
        if len(face_states) < args.n_polymers:
            print(f"  Only {len(face_states)} possible face states; "
                  f"generating at most {len(face_states)} polymers instead of {args.n_polymers}")
            args.n_polymers = len(face_states)
        print(f"  {len(face_states)} states queued for {args.n_polymers} polymers")

    attempts = face_states if face_states else [None] * args.n_polymers
    n_accepted = 0

    for face_state in attempts:
        if n_accepted >= args.n_polymers:
            break
        polymer_idx = n_accepted + 1
        counted = face_state is None       # legacy polymers always use their slot
        if counted:
            n_accepted += 1

        print("\n" + "#"*70)
        print(f"# POLYMER {polymer_idx}/{args.n_polymers}")
        print("#"*70)

        name = f"{base_name}_p{polymer_idx}"
        tag = f"P{polymer_idx}"
        face_label = (lpf.building_chain.face_state_label(face_state, face_mol_order)
                      if face_state is not None else '')

        try:
            with timer.section(f"{tag} total", level=1):

                # ============================================================
                # PHASE 2: STRUCTURE ASSEMBLY
                # ============================================================

                print("\n" + "="*70)
                print(f"PHASE 2: STRUCTURE ASSEMBLY - Polymer {polymer_idx}")
                print("="*70)

                with timer.section(f"{tag} Phase 2: Structure Assembly", level=2):
                    if sugars:
                        with timer.section(f"{tag}   Chain building (rotation/alignment)", level=3):
                            chain_dict, bonds_glyco, sorted_linkages, joint_report = lpf.build_glycan_chain(
                                molecule_data_dict, config, args.iterations,
                                face_state=face_state,
                                seed=args.seed * 1000 + len(state_log) + 1,
                                joint_rotation=not args.no_joint_rotation,
                                max_gap=args.max_gap,
                                return_report=True
                            )
                            if face_state is not None:
                                label = lpf.building_chain.face_state_label(face_state, face_mol_order)
                                if joint_report is not None and not joint_report['valid']:
                                    print(f"\n  Face state {label} REJECTED: {joint_report['reason']}"
                                          f" -- trying the next state")
                                    state_log.append((label, 'rejected', '', joint_report['worst_gap'],
                                                      joint_report['reason']))
                                    continue
                                state_log.append((label, 'accepted', polymer_idx,
                                                  joint_report['worst_gap'] if joint_report else '',
                                                  'ok'))
                                n_accepted += 1
                                counted = True
                                got = {m: lpf.building_chain.face_sign(chain_dict[m])
                                       for m in face_mol_order if m in chain_dict}
                                got_label = ''.join('1' if got.get(m, 1) == -1 else '0'
                                                    for m in face_mol_order)
                                print(f"\n  Face state requested {label}, assembled {got_label}"
                                      + ("" if got_label == label else "   <-- MISMATCH"))
                                coords = lpf.chain_coordinates(chain_dict, face_mol_order)
                                for prev_idx, prev_label, prev in assembled_coords:
                                    if prev.shape == coords.shape:
                                        rmsd = float(np.sqrt(((prev - coords) ** 2).sum(axis=1).mean()))
                                        if rmsd < args.duplicate_rmsd:
                                            print(f"  WARNING: polymer {polymer_idx} ({label}) is within "
                                                  f"{rmsd:.2f} A of polymer {prev_idx} ({prev_label}): "
                                                  f"near duplicate")
                                assembled_coords.append((polymer_idx, label, coords))
                        with timer.section(f"{tag}   Phosphate linkages", level=3):
                            phosphate_bonds_with_names, unbonded_monomers = lpf.add_phosphate_linkages(
                                config, molecule_data_dict, chain_dict
                            )
                        with timer.section(f"{tag}   PEtN modifications", level=3):
                            petn_linkages = lpf.add_petn_modifications(config, chain_dict, circles)
                        with timer.section(f"{tag}   Lipid chain building", level=3):
                            all_lipids = lpf.add_lipid_chains(config, molecule_data_dict, circles)
                    else:
                        chain_dict = {}
                        bonds_glyco = []
                        sorted_linkages = []
                        phosphate_bonds_with_names = []
                        unbonded_monomers = {}
                        petn_linkages = []
                        all_lipids = []

                    with timer.section(f"{tag}   Peptide building", level=3):
                        peptide_data = lpf.build_peptide(config, circles)
                        print(f"DEBUG peptide_data: {peptide_data}")

                    with timer.section(f"{tag}   Peptide linkages", level=3):
                        peptide_linkages = lpf.create_peptide_linkages(
                            config, peptide_data, chain_dict
                        )
                # ============================================================
                # PHASE 3: EXPORT
                # ============================================================

                print("\n" + "="*70)
                print(f"PHASE 3: EXPORT - Polymer {polymer_idx}")
                print("="*70)

                with timer.section(f"{tag} Phase 3: Export", level=2):

                    with timer.section(f"{tag}   Structure export (SDF)", level=3):
                        if sugars:
                            if peptide_data is not None:
                                final_no_h, enforced_atoms, residue_index = lpf.export_structure(
                                    chain_dict, mol_name_to_conformer, bonds_glyco,
                                    [tuple(link) for link in config.get('glycosidic_bonds', [])],
                                    phosphate_bonds_with_names, petn_linkages, all_lipids,
                                    unbonded_monomers, conformers, peptide_data, peptide_linkages,
                                    name, conformers_selection
                                )
                            else:
                                final_no_h, residue_index = lpf.export_structure(
                                    chain_dict, mol_name_to_conformer, bonds_glyco,
                                    [tuple(link) for link in config.get('glycosidic_bonds', [])],
                                    phosphate_bonds_with_names, petn_linkages, all_lipids,
                                    unbonded_monomers, conformers, peptide_data, peptide_linkages,
                                    name, conformers_selection
                                )
                                enforced_atoms = []
                        else:
                            # pure cyclic peptide, no sugars
                            from rdkit import Chem
                            final_no_h = peptide_data['rdkit_mol']
                            enforced_atoms = []
                            residue_index = None
                            Chem.MolToPDBFile(final_no_h, f"{name}_cyclic_peptide.pdb")
                            print(f"  Saved: {name}_cyclic_peptide.pdb")

                    with timer.section(f"{tag}   Lipid tail index extraction", level=3):
                        lipid_tail_indices = []
                        # lipid_tail_indices = lpf.extract_lipid_tail_indices(
                        #     final_no_h, molecule_data_dict)
                        petn_n_indices = lpf.extract_petn_nitrogen_indices(
                            final_no_h, petn_linkages)
                        if petn_n_indices:
                            lipid_tail_indices = list(set(lipid_tail_indices + petn_n_indices))


                # ============================================================
                # PHASE 4: OPTIMIZATION
                # ============================================================

                print("\n" + "="*70)
                print(f"PHASE 4: OPTIMIZATION - Polymer {polymer_idx}")
                print("="*70)

                with timer.section(f"{tag} Phase 4: Optimization", level=2):

                    with timer.section(f"{tag}   H addition + ring detection", level=3):
                        final_with_h, pyranose_rings_no_h = lpf.prepare_structure_for_optimization(
                            final_no_h, name, conformers_selection
                        )
                        # use stored linker ring indices directly — no position matching needed
                        if not sugars and peptide_data is not None and \
                           'linker_ring_indices' in peptide_data:
                            lipid_tail_indices = peptide_data['linker_ring_indices']
                            print(f"  Freezing {len(lipid_tail_indices)} linker ring atoms (phenyl + oxadiazole)")

                    with timer.section(f"{tag}   Gather fixed atoms", level=3):
                        fixed_atoms, torsion_constraints = lpf.gather_fixed_atoms(
                            peptide_data, enforced_atoms, lipid_tail_indices,
                            mol=final_with_h
                        )
                        # PEtN nitrogens stay fixed in ALL phases (linker freeze is
                        # handled inside gather_fixed_atoms; lipids remain free).
                        if petn_n_indices:
                            fixed_atoms = list(set(fixed_atoms + petn_n_indices))
                        reference_normals = None

                    refine_mode = args.refine
                    if refine_mode == "auto":
                        refine_mode = "rigid" if (sugars and peptide_data is None
                                                  and not all_lipids) else "atom"
                    print(f"\n  REFINE mode: {refine_mode}")

                    if refine_mode == "rigid":
                        # No check_and_fix_geometry here: fix_overlapping_atoms moves
                        # atoms, and the rigid path must start from the templates.
                        if args.debug_checkpoints:
                            from src.rotation_optimization.md.utils import save_molecule
                            save_molecule(final_with_h,
                                          f"{name}_{conformers_selection}_pre_refine",
                                          file_format='sdf')
                            print(f"  Saved: {name}_{conformers_selection}_pre_refine.sdf")
                        with timer.section(f"{tag}   Rigid body REFINE", level=3):
                            from src.rotation_optimization.rigid_refine import (
                                refine_structure, RigidRefineConfig)
                            rigid_cfg = RigidRefineConfig(
                                r0_xy=args.r0_xy, ceiling_margin=args.ceiling_margin)
                            optimized_mol, rigid_report = refine_structure(
                                final_with_h, final_no_h, chain_dict,
                                [tuple(link) for link in config.get('glycosidic_bonds', [])],
                                fixed_atoms=fixed_atoms, config=rigid_cfg)
                            success = (rigid_report['max_rigid_rmsd'] < 1e-6
                                       and rigid_report['faces_kept'])
                            rigid_log.append((polymer_idx, face_label, rigid_report))
                    else:
                        with timer.section(f"{tag}   Geometry check/fix", level=3):
                            final_with_h = lpf.check_and_fix_geometry(final_with_h)

                        with timer.section(f"{tag}   Ring COM extraction", level=3):
                            initial_ring_coms = lpf.extract_initial_ring_coms_from_monomers(
                                molecule_data_dict,
                                final_no_h,
                                pyranose_rings_no_h
                            )

                        with timer.section(f"{tag}   Force field optimization (total)", level=3):
                            optimized_mol, success = lpf.run_optimization(
                                final_with_h, fixed_atoms, reference_normals,
                                name, conformers_selection, molecule_data_dict,
                                lipid_tail_indices,
                                pyranose_rings_no_h=pyranose_rings_no_h,
                                initial_ring_coms=initial_ring_coms,
                                stm_npy_path=stm_npz_path,
                                enable_phase1_kicks=args.phase1_kicks,
                                torsion_constraints=torsion_constraints,
                                save_debug_checkpoints=args.debug_checkpoints
                            )

                    with timer.section(f"{tag}   Save results", level=3):
                        lpf.save_optimization_results(
                            final_with_h, optimized_mol, success,
                            name, conformers_selection, residue_index,
                            save_debug_checkpoints=args.debug_checkpoints
                        )

            # Store results
            all_polymer_results.append({
                'polymer_id': polymer_idx,
                'name': name,
                'success': success,
                'chain_dict': chain_dict,
                'final_no_h': final_no_h,
                'optimized_mol': optimized_mol
            })

            print(f"\n✓ Polymer {polymer_idx} complete: {'SUCCESS' if success else 'PARTIAL'}")

        except Exception as e:
            if not counted:                # a crashed build still uses its slot
                n_accepted += 1
            print(f"\n✗ ERROR in Polymer {polymer_idx}: {e}")
            import traceback
            traceback.print_exc()

            all_polymer_results.append({
                'polymer_id': polymer_idx,
                'name': name,
                'success': False,
                'error': str(e)
            })

    if rigid_log:
        path = f"{base_name}_rigid_refine.csv"
        with open(path, "w") as fh:
            fh.write("polymer,face_state,rigid_ok,faces_kept,mmff_energy_kcal,mmff_energy_before,"
                     "worst_linkage_stretch_A,max_xy_drift_A,thickness_A,seconds\n")
            for pidx, label, rep in rigid_log:
                stretch = max((r['stretch'] for r in rep['linkages']), default=0.0)
                drift = max((r['xy_drift'] for r in rep['beads']), default=0.0)
                fh.write(f"{pidx},{label},{int(rep['max_rigid_rmsd'] < 1e-6)},"
                         f"{int(rep['faces_kept'])},{rep['energy_mmff']:.2f},"
                         f"{rep['energy_mmff_before']:.2f},{stretch:.3f},{drift:.3f},"
                         f"{max(rep['thickness']):.2f},{rep['run']['seconds']:.1f}\n")
        state_path = f"{base_name}_rigid_state.csv"
        with open(state_path, "w") as fh:
            fh.write("polymer,face_state,bead,com_x,com_y,com_z,qx,qy,qz,qw,"
                     "refine_qx,refine_qy,refine_qz,refine_qw\n")
            for pidx, label, rep in rigid_log:
                for r in rep.get('state', []):
                    c, q, rq = r['com'], r['quaternion'], r['refine_quaternion']
                    fh.write(f"{pidx},{label},{r['bead']},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},"
                             f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f},"
                             f"{rq[0]:.6f},{rq[1]:.6f},{rq[2]:.6f},{rq[3]:.6f}\n")
        print(f"\nRigid REFINE summary -> {path}")
        print(f"Rigid body state (COM + quaternion per sugar) -> {state_path}")
        for pidx, label, rep in rigid_log:
            print(f"  polymer {pidx} {label}: MMFF {rep['energy_mmff']:.1f} kcal/mol, "
                  f"faces kept {rep['faces_kept']}, worst stretch "
                  f"{max((r['stretch'] for r in rep['linkages']), default=0.0):+.3f} A")

    if state_log:
        with open(f"{base_name}_face_states.csv", "w") as fh:
            fh.write("try,label,status,polymer,worst_linkage_CC_A,reason\n")
            for i, (label, status, pidx, gap, reason) in enumerate(state_log, 1):
                gap_s = f"{gap:.3f}" if isinstance(gap, float) else gap
                fh.write(f"{i},{label},{status},{pidx},{gap_s},{reason}\n")
        n_rej = sum(1 for r in state_log if r[1] == 'rejected')
        print(f"\nFace states: {len(state_log) - n_rej} accepted, {n_rej} rejected "
              f"-> {base_name}_face_states.csv")
        if n_accepted < args.n_polymers:
            print(f"  WARNING: only {n_accepted} of {args.n_polymers} requested polymers found a "
                  f"valid face state (raise --pool_factor or check the layout)")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*70)
    print("ALL POLYMERS COMPLETE!")
    print("="*70)

    successful = [p for p in all_polymer_results if p.get('success', False)]
    partial = [p for p in all_polymer_results if not p.get('success', False) and 'error' not in p]
    failed = [p for p in all_polymer_results if 'error' in p]

    print(f"\nResults:")
    print(f"  Successful: {len(successful)}/{args.n_polymers}")
    print(f"  Partial:    {len(partial)}/{args.n_polymers}")
    print(f"  Failed:     {len(failed)}/{args.n_polymers}")

    print(f"\nOutput files generated:")
    for result in all_polymer_results:
        if 'error' not in result:
            name = result['name']
            success = result['success']
            print(f"\n  Polymer {result['polymer_id']}:")
            print(f"    - {name}_{conformers_selection}_pre_opt.sdf (with H)")
            if success:
                print(f"    - {name}_{conformers_selection}_optimized.sdf (OPTIMIZED ✓)")
            else:
                print(f"    - {name}_{conformers_selection}_partial.sdf (partial)")
        else:
            print(f"\n  Polymer {result['polymer_id']}: FAILED - {result['error']}")

    # ========================================================================
    # TIMING REPORTS
    # ========================================================================

    timer.report()
    timer.polymer_summary(args.n_polymers)

    total_elapsed = time.perf_counter() - pipeline_start
    print(f"\n  Wall clock total: {timer._fmt(total_elapsed)}")

    print("\n" + "="*70)
    print(f"PIPELINE COMPLETE - {len(successful)} optimized polymers generated")
    print("="*70)


if __name__ == "__main__":
    main()