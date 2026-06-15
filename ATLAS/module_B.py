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

    for polymer_idx in range(1, args.n_polymers + 1):

        print("\n" + "#"*70)
        print(f"# POLYMER {polymer_idx}/{args.n_polymers}")
        print("#"*70)

        name = f"{base_name}_p{polymer_idx}"
        tag = f"P{polymer_idx}"

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
                            chain_dict, bonds_glyco, sorted_linkages = lpf.build_glycan_chain(
                                molecule_data_dict, config, args.iterations
                            )
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
                                final_no_h, enforced_atoms = lpf.export_structure(
                                    chain_dict, mol_name_to_conformer, bonds_glyco,
                                    [tuple(link) for link in config.get('glycosidic_bonds', [])],
                                    phosphate_bonds_with_names, petn_linkages, all_lipids,
                                    unbonded_monomers, conformers, peptide_data, peptide_linkages,
                                    name, conformers_selection
                                )
                            else:
                                final_no_h = lpf.export_structure(
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
                            torsion_constraints=torsion_constraints
                        )

                    with timer.section(f"{tag}   Save results", level=3):
                        lpf.save_optimization_results(
                            final_with_h, optimized_mol, success,
                            name, conformers_selection
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
            print(f"\n✗ ERROR in Polymer {polymer_idx}: {e}")
            import traceback
            traceback.print_exc()

            all_polymer_results.append({
                'polymer_id': polymer_idx,
                'name': name,
                'success': False,
                'error': str(e)
            })

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