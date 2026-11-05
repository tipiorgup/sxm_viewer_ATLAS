"""Build MISO sugar-only configurations from explicitly connected placed units."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


class MISOExportError(ValueError):
    """Placed units or connection choices cannot form a valid MISO input."""


@dataclass(frozen=True)
class SugarConnection:
    donor: int
    donor_carbon: str
    acceptor: int
    acceptor_carbon: str
    anomer: str


@dataclass(frozen=True)
class PeptideBond:
    """A sugar carbon bonded to the glycosylation site of one amino acid unit."""
    sugar: int
    sugar_carbon: str
    residue: int
    anomer: str


# Residues MISO can glycosylate (find_glycosylation_site / create_peptide_sugar_linkage).
# Trp (C-linked) is deliberately absent: MISO raises "not yet implemented" for it.
GLYCOSYLATION_TYPES = {"Asn": "N-glycosidic", "Ser": "O-glycosidic",
                       "Thr": "O-glycosidic", "Tyr": "O-glycosidic"}

# Every residue MISO can place in a peptide sequence.
AA_ONE_LETTER = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}


def is_sugar(instance: dict) -> bool:
    return instance.get("kind", "sugar") == "sugar"


def is_amino_acid(instance: dict) -> bool:
    return instance.get("kind") == "aa"


def default_peptide_order(instances: list[dict]) -> list[int]:
    """Amino acid units in chain order: ascending point index."""
    return sorted((i for i, inst in enumerate(instances) if is_amino_acid(inst)),
                  key=lambda i: (instances[i].get("position", i), i))


def validate_peptide_order(instances: list[dict], order) -> list[int]:
    expected = {i for i, inst in enumerate(instances) if is_amino_acid(inst)}
    if (not isinstance(order, (list, tuple)) or any(type(i) is not int for i in order)
            or len(order) != len(set(order)) or set(order) != expected):
        raise MISOExportError(
            "The peptide order must list every amino acid unit exactly once.")
    return list(order)


def linkage_carbons(instance: dict) -> list[str]:
    rigid = instance.get("rigid") or {}
    data = rigid.get(instance["conf_name"], {})
    carbon_map = data.get("carbon_map", {})
    oh_map = data.get("oh_map", {})
    return sorted(
        (name for name in carbon_map
         if re.fullmatch(r"C\d+", name) and name in oh_map),
        key=lambda name: int(name[1:]))


def validate_connections(instances: list[dict],
                         connections: list[SugarConnection],
                         peptide_bonds=()) -> None:
    used_sites = set()
    for number, link in enumerate(connections, 1):
        if not (0 <= link.donor < len(instances) and
                0 <= link.acceptor < len(instances)):
            raise MISOExportError(f"Connection {number} refers to a unit that no longer exists.")
        if link.donor == link.acceptor:
            raise MISOExportError(f"Connection {number} links a unit to itself.")
        if link.anomer not in ("alpha", "beta"):
            raise MISOExportError(f"Choose alpha or beta for connection {number}.")
        for index, carbon in ((link.donor, link.donor_carbon),
                              (link.acceptor, link.acceptor_carbon)):
            instance = instances[index]
            if not is_sugar(instance):
                raise MISOExportError(
                    f"{instance['label']} is not a sugar. Sugar connections join sugars only.")
            if carbon not in linkage_carbons(instance):
                raise MISOExportError(
                    f"{instance['label']}: {carbon} is not a mapped hydroxyl-bearing carbon.")
            site = (index, carbon)
            if site in used_sites:
                raise MISOExportError(
                    f"{instance['label']}: {carbon} is already used by another connection.")
            used_sites.add(site)
    used_residues = set()
    for number, bond in enumerate(peptide_bonds, 1):
        if not (type(bond.sugar) is int and type(bond.residue) is int
                and 0 <= bond.sugar < len(instances) and 0 <= bond.residue < len(instances)):
            raise MISOExportError(
                f"Peptide bond {number} refers to a unit that no longer exists.")
        sugar, residue = instances[bond.sugar], instances[bond.residue]
        if not is_sugar(sugar):
            raise MISOExportError(f"Peptide bond {number}: {sugar['label']} is not a sugar.")
        if not is_amino_acid(residue):
            raise MISOExportError(
                f"Peptide bond {number}: {residue['label']} is not an amino acid.")
        if residue["name"] == "Trp":
            raise MISOExportError(
                f"{residue['label']}: C-linked glycosylation of Trp is not implemented in MISO yet.")
        if residue["name"] not in GLYCOSYLATION_TYPES:
            raise MISOExportError(
                f"{residue['label']}: MISO can attach a sugar only to Asn, Ser, Thr or Tyr.")
        if bond.anomer not in ("alpha", "beta"):
            raise MISOExportError(f"Choose alpha or beta for peptide bond {number}.")
        if bond.sugar_carbon not in linkage_carbons(sugar):
            raise MISOExportError(
                f"{sugar['label']}: {bond.sugar_carbon} is not a mapped hydroxyl-bearing carbon.")
        site = (bond.sugar, bond.sugar_carbon)
        if site in used_sites:
            raise MISOExportError(
                f"{sugar['label']}: {bond.sugar_carbon} is already used by another connection.")
        used_sites.add(site)
        if bond.residue in used_residues:
            raise MISOExportError(
                f"{residue['label']} already carries a sugar. MISO attaches one sugar per residue.")
        used_residues.add(bond.residue)


def build_miso_config(instances: list[dict], connections: list[SugarConnection],
                      root: int | None, fixed_orientation: bool,
                      out_path: str, sxm_path: Path | None = None,
                      has_lipids: bool = False,
                      positions_csv: str | None = None,
                      peptide_bonds=(), peptide_order=None,
                      functional_points=None) -> dict:
    """MISO input for the sugar and amino acid ``instances``.

    Amino acid units become the ``peptide`` section (chain order ``peptide_order``,
    default ascending point index) and ``peptide_bonds`` join sugars to residues.
    ``functional_points`` optionally maps an amino acid unit index to the circle
    index of its side-chain point (``functional_position``); default None.

    Each instance occupies point ``inst["position"]`` of the positions CSV
    (default: its list index). ``positions_csv`` overrides the default
    ``<stem>_positions.csv`` circle input.
    """
    if has_lipids or any(not (is_sugar(inst) or is_amino_acid(inst)) for inst in instances):
        raise MISOExportError(
            "MISO YAML export supports sugars and amino acids only. Remove lipid rows "
            "before exporting.")
    sugar_ids = [i for i, inst in enumerate(instances) if is_sugar(inst)]
    aa_ids = [i for i, inst in enumerate(instances) if is_amino_acid(inst)]
    peptide_bonds = list(peptide_bonds)
    if not sugar_ids:
        raise MISOExportError("Build and position at least one sugar before exporting.")
    if root is None or not 0 <= root < len(instances):
        raise MISOExportError("Choose the root monomer explicitly.")
    if root not in sugar_ids:
        raise MISOExportError("The root must be a sugar, not an amino acid.")

    sugars = {}
    positions = {}
    geometries = {}
    molecule_ids = {}
    used_points = {}
    for index, inst in enumerate(instances):
        point = inst.get("position", index)
        if type(point) is not int or point < 0:
            raise MISOExportError(f"{inst['label']}: invalid point index {point!r}.")
        if point in used_points:
            raise MISOExportError(
                f"{inst['label']} and {used_points[point]} use the same point {point}.")
        used_points[point] = inst["label"]
        if index not in sugar_ids:
            continue
        name = inst["name"]
        if not isinstance(name, str) or not name.strip():
            raise MISOExportError(f"{inst['label']}: a sugar Name is required.")
        rigid = inst.get("rigid")
        if not rigid or inst["conf_name"] not in rigid:
            raise MISOExportError(
                f"{inst['label']}: exact monomer geometry is missing. Rebuild this sugar.")
        data = rigid[inst["conf_name"]]
        required = ("relative_coordinates", "atom_types", "carbon_map", "quaternion", "COM")
        if any(key not in data for key in required):
            raise MISOExportError(f"{inst['label']}: monomer geometry is incomplete. Rebuild it.")
        signature = (inst["smiles"], inst["conf_name"], rigid)
        if name in geometries and signature != geometries[name]:
            raise MISOExportError(
                f"Rows named '{name}' have different SMILES or conformers. "
                "Give them distinct Names in the subunit table and rebuild.")
        geometries[name] = signature
        sugars[name] = inst["smiles"]
        positions.setdefault(name, []).append(point)
        molecule_ids[index] = f"{name}_{point}"

    validate_connections(instances, connections, peptide_bonds)
    if aa_ids and not peptide_bonds:
        raise MISOExportError(
            "Amino acids are placed but no sugar is bonded to them. Add a sugar to "
            "residue bond (Asn, Ser, Thr or Tyr), or remove the amino acid units.")
    order = (validate_peptide_order(instances, peptide_order) if peptide_order is not None
             else default_peptide_order(instances))
    functional_points = dict(functional_points or {})
    for key, value in functional_points.items():
        if key not in aa_ids or type(value) is not int or value < 0:
            raise MISOExportError("A side-chain point refers to an invalid unit or index.")
    residue_names = [instances[i]["name"] for i in order]
    unknown = [n for n in residue_names if n not in AA_ONE_LETTER]
    if unknown:
        raise MISOExportError(f"Unknown amino acid '{unknown[0]}' in the peptide.")

    adjacent = {i: set() for i in sugar_ids}
    for link in connections:
        adjacent[link.donor].add(link.acceptor)
        adjacent[link.acceptor].add(link.donor)
    reachable = {i: set(n) for i, n in adjacent.items()}
    if fixed_orientation:
        # With fixed poses the peptide joins every glycan that is attached to it.
        attached = sorted({bond.sugar for bond in peptide_bonds})
        for a in attached:
            reachable[a].update(b for b in attached if b != a)
    reached = {root}
    pending = [root]
    while pending:
        for neighbor in reachable[pending.pop()] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    if len(reached) != len(sugar_ids):
        missing = ", ".join(instances[i]["label"] for i in sugar_ids if i not in reached)
        hint = (" With optimized orientations all sugars must form one glycan joined by "
                "sugar-to-sugar bonds; separate glycans on different residues need "
                "'Keep positioned geometry and rotations'."
                if peptide_bonds and not fixed_orientation else "")
        raise MISOExportError(f"All sugars must connect to the root. Unconnected: {missing}.{hint}")
    if not fixed_orientation:
        reached = {root}
        while True:
            donors = {link.donor for link in connections if link.acceptor in reached}
            if donors <= reached:
                break
            reached.update(donors)
        if len(reached) != len(sugar_ids):
            raise MISOExportError(
                "For optimized orientation, every donor must lead toward the root through "
                "donor-to-acceptor links. Choose the reducing-end root or correct the links; "
                "chemical bonds will not be reversed automatically.")

    stem = Path(out_path).resolve().with_suffix("")
    bonds = []
    for link in connections:
        donor = molecule_ids[link.donor]
        acceptor = molecule_ids[link.acceptor]
        description = (f"{instances[link.donor]['label']} {link.donor_carbon} -> "
                       f"{instances[link.acceptor]['label']} {link.acceptor_carbon} "
                       f"({link.anomer})")
        bonds.append([donor, link.donor_carbon, acceptor, link.acceptor_carbon,
                      link.anomer, description])
    config = {
        "sxm_file": str(sxm_path.resolve()) if sxm_path else str(stem.with_suffix(".sxm")),
        "circle_input_path": (str(Path(positions_csv).resolve()) if positions_csv
                              else f"{stem}_positions.csv"),
        "monomer_data_path": f"{stem}_monomer_data.pkl",
        "use_fixed_orientation": fixed_orientation,
        "sugars": sugars,
        "experimental_positions": positions,
        "conformer_parameters": {"num_conformers": 20, "max_keep": 1},
        "conformer_selection": {"strategy": "lowest_energy"},
        "root_mol": molecule_ids[root],
        "direction": [bond[:] for bond in bonds],
        "glycosidic_bonds": bonds,
    }
    if aa_ids:
        config["peptide"] = {
            "sequence": "-".join(AA_ONE_LETTER[n] for n in residue_names),
            "residues": [{"aa": instances[i]["name"],
                          "ca_position": instances[i].get("position", i),
                          "functional_position": functional_points.get(i)} for i in order],
        }
        config["peptide_bonds"] = [{
            "sugar_mol": molecule_ids[bond.sugar],
            "sugar_carbon": bond.sugar_carbon,
            "aa_type": instances[bond.residue]["name"],
            "residue_index": order.index(bond.residue),
            "glycosylation_type": GLYCOSYLATION_TYPES[instances[bond.residue]["name"]],
            "anomeric_config": bond.anomer,
            "use_spacer": False,
        } for bond in peptide_bonds]
    if fixed_orientation:
        config["orientation_csv_path"] = f"{stem}_orientations.csv"
    return config
