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
                         connections: list[SugarConnection]) -> None:
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
            if carbon not in linkage_carbons(instance):
                raise MISOExportError(
                    f"{instance['label']}: {carbon} is not a mapped hydroxyl-bearing carbon.")
            site = (index, carbon)
            if site in used_sites:
                raise MISOExportError(
                    f"{instance['label']}: {carbon} is already used by another connection.")
            used_sites.add(site)


def build_miso_config(instances: list[dict], connections: list[SugarConnection],
                      root: int | None, fixed_orientation: bool,
                      out_path: str, sxm_path: Path | None = None,
                      has_lipids: bool = False,
                      positions_csv: str | None = None) -> dict:
    """MISO input for the sugar ``instances``.

    Each instance occupies point ``inst["position"]`` of the positions CSV
    (default: its list index). ``positions_csv`` overrides the default
    ``<stem>_positions.csv`` circle input.
    """
    if has_lipids or any(inst.get("kind") != "sugar" for inst in instances):
        raise MISOExportError(
            "MISO YAML export currently supports sugars only. "
            "Remove amino-acid units and lipid rows before exporting.")
    if not instances:
        raise MISOExportError("Build and position at least one sugar before exporting.")
    if root is None or not 0 <= root < len(instances):
        raise MISOExportError("Choose the root monomer explicitly.")

    sugars = {}
    positions = {}
    geometries = {}
    molecule_ids = []
    used_points = {}
    for index, inst in enumerate(instances):
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
        point = inst.get("position", index)
        if type(point) is not int or point < 0:
            raise MISOExportError(f"{inst['label']}: invalid point index {point!r}.")
        if point in used_points:
            raise MISOExportError(
                f"{inst['label']} and {used_points[point]} use the same point {point}.")
        used_points[point] = inst["label"]
        sugars[name] = inst["smiles"]
        positions.setdefault(name, []).append(point)
        molecule_ids.append(f"{name}_{point}")

    validate_connections(instances, connections)
    adjacent = {i: set() for i in range(len(instances))}
    for link in connections:
        adjacent[link.donor].add(link.acceptor)
        adjacent[link.acceptor].add(link.donor)
    reached = {root}
    pending = [root]
    while pending:
        for neighbor in adjacent[pending.pop()] - reached:
            reached.add(neighbor)
            pending.append(neighbor)
    if len(reached) != len(instances):
        missing = ", ".join(instances[i]["label"] for i in adjacent if i not in reached)
        raise MISOExportError(f"All sugars must connect to the root. Unconnected: {missing}.")
    if not fixed_orientation:
        reached = {root}
        while True:
            donors = {link.donor for link in connections if link.acceptor in reached}
            if donors <= reached:
                break
            reached.update(donors)
        if len(reached) != len(instances):
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
    if fixed_orientation:
        config["orientation_csv_path"] = f"{stem}_orientations.csv"
    return config
