"""Build MISO sugar-only configurations from explicitly connected placed units."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


class MISOExportError(ValueError):
    """Placed units or connection choices cannot form a valid MISO input."""


# "glycosidic" joins two sugars directly. "phosphate" joins them through a phosphate
# bridge: MISO rotates the donor sugar toward the acceptor (the target, which must sit
# in the glycosidic chain) and builds sugar carbon, O, P, O, sugar carbon.
BOND_TYPES = ("glycosidic", "phosphate")


@dataclass(frozen=True)
class SugarConnection:
    donor: int
    donor_carbon: str
    acceptor: int
    acceptor_carbon: str
    anomer: str
    bond: str = "glycosidic"


@dataclass(frozen=True)
class PEtNGroup:
    """A phosphoethanolamine on a sugar carbon; ``point`` is the picked image point
    (its index in the positions CSV) where the terminal N atom should end up."""
    sugar: int
    sugar_carbon: str
    point: int


@dataclass(frozen=True)
class PeptideBond:
    """A sugar carbon bonded to the glycosylation site of one amino acid unit."""
    sugar: int
    sugar_carbon: str
    residue: int
    anomer: str


# Linkages MISO's build_linkage_headgroup accepts for lipid chains.
LIPID_LINKAGES = ("ester", "amide", "ether")


@dataclass(frozen=True)
class LipidAttachment:
    """Where one lipid row of the GUI joins the assembly (chosen by the user or assistant).

    ``lipid`` is the row number (0 based). ``anchor`` is "sugar" (a carbon of a sugar
    unit) or "lipid" (a branch off an existing lipid chain). ``linkage`` None keeps
    the row's own linkage.
    """
    lipid: int
    anchor: str
    sugar: int | None = None
    sugar_carbon: str | None = None
    linkage: str | None = None


@dataclass(frozen=True)
class LipidBond:
    """One lipid chain as MISO needs it. Points are (x, y) in Angstrom on the image.

    ``start`` is the free tip of the chain and ``end`` is where the lipid is bonded
    (to a sugar carbon or to another lipid chain). MISO grows the chain from the
    bond toward the tip, so ``start`` becomes its ``direction`` and, for a branch,
    ``end`` becomes its ``connection_blob``.
    """
    label: str
    anchor: str
    sugar: int | None
    sugar_carbon: str | None
    start: tuple | None        # free tip of the chain (MISO direction)
    end: tuple | None          # bond point; for a "lipid" anchor the blob on the parent chain
    mid: tuple | None
    carbons: int | None
    linkage: str


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


def lipid_carbons(instance: dict) -> list[str]:
    """Mapped ring carbons a lipid may attach to (a missing OH or NH2 only warns in MISO)."""
    rigid = instance.get("rigid") or {}
    carbon_map = rigid.get(instance.get("conf_name"), {}).get("carbon_map", {})
    return sorted((name for name in carbon_map if re.fullmatch(r"C\d+", name)),
                  key=lambda name: int(name[1:]))


def _point_ok(point) -> bool:
    try:
        return (len(point) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                        and v == v and abs(v) != float("inf") for v in point))
    except TypeError:
        return False


def validate_lipids(instances: list[dict], lipids, used_sites=()) -> None:
    """Check lipid attachments against MISO's rules (see lipid_building / pipeline)."""
    used = set(used_sites)
    anchored_sugars = set()
    branched = []
    for lipid in lipids:
        name = lipid.label
        if lipid.linkage not in LIPID_LINKAGES:
            raise MISOExportError(
                f"{name}: choose ester, amide or ether. MISO has no 'none' linkage.")
        for what, point in (("Start", lipid.start), ("End", lipid.end), ("Mid", lipid.mid)):
            if point is None:
                if what != "Mid":
                    raise MISOExportError(
                        f"{name} has no {what} point. Pick the Start (free tip) and the "
                        "End (bond point) first.")
                continue
            if not _point_ok(point):
                raise MISOExportError(f"{name}: the {what} point is not a valid coordinate.")
        if lipid.carbons is not None and (
                type(lipid.carbons) is not int or not 2 <= lipid.carbons <= 200):
            raise MISOExportError(f"{name}: the carbon count must be between 2 and 200 (or auto).")
        if lipid.anchor == "sugar":
            index = lipid.sugar
            if type(index) is not int or not 0 <= index < len(instances):
                raise MISOExportError(f"{name}: choose the sugar this lipid is attached to.")
            unit = instances[index]
            if not is_sugar(unit):
                raise MISOExportError(f"{name}: {unit['label']} is not a sugar.")
            if lipid.sugar_carbon not in lipid_carbons(unit):
                raise MISOExportError(
                    f"{name}: {lipid.sugar_carbon} is not a mapped carbon of {unit['label']}.")
            site = (index, lipid.sugar_carbon)
            if site in used:
                raise MISOExportError(
                    f"{name}: {unit['label']} {lipid.sugar_carbon} is already used by another "
                    "bond or lipid. A carbon takes only one.")
            used.add(site)
            anchored_sugars.add(index)
        elif lipid.anchor == "lipid":
            if lipid.linkage != "amide":
                raise MISOExportError(
                    f"{name}: a branch off a lipid chain must use amide. MISO builds ester "
                    "and ether branches without any chain, so they would silently be missing.")
            branched.append(lipid)
        else:
            raise MISOExportError(f"{name}: the anchor must be 'sugar' or 'lipid'.")
    if branched:
        if not anchored_sugars:
            raise MISOExportError(
                f"{branched[0].label} branches off a lipid chain, but no lipid is attached to "
                "a sugar yet. Attach another lipid to a sugar carbon first.")
        if len(anchored_sugars) > 1:
            raise MISOExportError(
                "Branched lipids are only supported when all sugar attached lipids sit on ONE "
                "sugar (MISO branches from the first sugar that carries lipids).")


def _distance_to_polyline(point, polyline) -> float:
    import math
    best = float("inf")
    for a, b in zip(polyline, polyline[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = 0.0 if length2 == 0 else max(0.0, min(1.0, (
            (point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2))
        best = min(best, math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy))
    return best


def lipid_branch_parent(lipids):
    """Index (in ``lipids``) of the sugar attached lipid the branches grow from, or None.

    MISO bonds every branch to carbon 2 of the FIRST sugar attached lipid it built, so
    all branches must sit on one lipid. The parent is the sugar attached lipid whose
    drawn line is closest to the branch's End (bond point).
    """
    lipids = list(lipids)
    sugar_rows = [i for i, lip in enumerate(lipids) if lip.anchor == "sugar"]
    parents = set()
    for lip in lipids:
        if lip.anchor != "lipid" or lip.end is None:
            continue
        def line_of(i):
            points = [p for p in (lipids[i].start, lipids[i].mid, lipids[i].end) if p is not None]
            return points * 2 if len(points) == 1 else points

        best = min(sugar_rows, key=lambda i: _distance_to_polyline(lip.end, line_of(i)),
                   default=None)
        if best is not None:
            parents.add(best)
    if len(parents) > 1:
        names = ", ".join(lipids[i].label for i in sorted(parents))
        raise MISOExportError(
            f"The branches sit on different lipids ({names}). MISO can only branch from one "
            "lipid (the first one it builds): put every branch End point on the same lipid.")
    return next(iter(parents)) if parents else None


def _validate_phosphate_bridges(instances, connections, peptide_bonds=()):
    """MISO rotates the donor of a phosphate bridge as a separate unit, so it must be a leaf."""
    bridges = [link for link in connections if link.bond == "phosphate"]
    if not bridges:
        return
    glycosidic_units = {unit for link in connections if link.bond == "glycosidic"
                        for unit in (link.donor, link.acceptor)}
    peptide_units = {bond.sugar for bond in peptide_bonds}
    donors = set()
    for link in bridges:
        donor, target = instances[link.donor]["label"], instances[link.acceptor]["label"]
        if link.donor in donors:
            raise MISOExportError(
                f"{donor} is the rotated sugar of two phosphate bridges. MISO allows one per sugar.")
        donors.add(link.donor)
        if link.donor in glycosidic_units or link.donor in peptide_units:
            raise MISOExportError(
                f"{donor} has a phosphate bridge to {target} and another bond. MISO rotates the "
                "donor of a phosphate bridge as a separate sugar, so it can have no glycosidic "
                "or peptide bond. Swap donor and acceptor, or use a glycosidic bond.")
    for link in bridges:
        if link.acceptor in donors:
            raise MISOExportError(
                f"{instances[link.acceptor]['label']} is the rotated sugar of another phosphate "
                "bridge, so it cannot be the target of this one. The target must sit in the "
                "glycosidic chain.")


def validate_petn(instances: list[dict], petn, used_sites=(), point_count=None,
                  connections=(), reserved_points=()) -> None:
    """Check phosphoethanolamine groups against MISO's rules (pipeline.add_petn_modifications)."""
    used = set(used_sites)
    reserved = {inst.get("position", i) for i, inst in enumerate(instances)} | set(reserved_points)
    rotated = {link.donor for link in connections if link.bond == "phosphate"}
    seen_points = set()
    for number, group in enumerate(petn, 1):
        if not (type(group.sugar) is int and 0 <= group.sugar < len(instances)):
            raise MISOExportError(f"PEtN {number} refers to a unit that no longer exists.")
        unit = instances[group.sugar]
        if not is_sugar(unit):
            raise MISOExportError(f"PEtN {number}: {unit['label']} is not a sugar.")
        if group.sugar in rotated:
            raise MISOExportError(
                f"{unit['label']} is rotated into place by a phosphate bridge. MISO adds PEtN "
                "only to sugars of the glycosidic chain: use another sugar or carbon.")
        if group.sugar_carbon not in linkage_carbons(unit):
            raise MISOExportError(
                f"{unit['label']}: {group.sugar_carbon} is not a mapped hydroxyl-bearing carbon.")
        site = (group.sugar, group.sugar_carbon)
        if site in used:
            raise MISOExportError(
                f"{unit['label']} {group.sugar_carbon} is already used by another bond, lipid "
                "or PEtN. A carbon takes only one.")
        used.add(site)
        if type(group.point) is not int or group.point < 0:
            raise MISOExportError(f"PEtN {number}: choose the point where the N atom goes.")
        if point_count is not None and group.point >= point_count:
            raise MISOExportError(
                f"PEtN {number}: point {group.point} does not exist (the image has "
                f"{point_count} points).")
        if group.point in reserved:
            raise MISOExportError(
                f"PEtN {number}: point {group.point} is already used by a sugar or amino acid. "
                "Pick a separate point for the N atom.")
        if group.point in seen_points:
            raise MISOExportError(f"Point {group.point} is used by two PEtN groups.")
        seen_points.add(group.point)


def validate_connections(instances: list[dict],
                         connections: list[SugarConnection],
                         peptide_bonds=(), lipid_sites=()) -> None:
    used_sites = set(lipid_sites)
    for number, link in enumerate(connections, 1):
        if not (0 <= link.donor < len(instances) and
                0 <= link.acceptor < len(instances)):
            raise MISOExportError(f"Connection {number} refers to a unit that no longer exists.")
        if link.donor == link.acceptor:
            raise MISOExportError(f"Connection {number} links a unit to itself.")
        if link.bond not in BOND_TYPES:
            raise MISOExportError(
                f"Connection {number}: choose a glycosidic bond or a phosphate bridge.")
        if link.bond == "glycosidic" and link.anomer not in ("alpha", "beta"):
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
    _validate_phosphate_bridges(instances, connections, peptide_bonds)
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
                      functional_points=None, lipids=(), petn=(), point_count=None) -> dict:
    """MISO input for the sugar and amino acid ``instances``.

    Amino acid units become the ``peptide`` section (chain order ``peptide_order``,
    default ascending point index) and ``peptide_bonds`` join sugars to residues.
    ``functional_points`` optionally maps an amino acid unit index to the circle
    index of its side-chain point (``functional_position``); default None.

    ``petn`` are phosphoethanolamine groups (``PEtNGroup``); ``point_count`` is the number
    of picked points, so a PEtN point can be checked against the positions CSV.
    Connections with ``bond == "phosphate"`` become the ``Phosphate`` section.

    Each instance occupies point ``inst["position"]`` of the positions CSV
    (default: its list index). ``positions_csv`` overrides the default
    ``<stem>_positions.csv`` circle input.
    """
    lipids = list(lipids)
    petn = list(petn)
    if (has_lipids and not lipids) or any(
            not (is_sugar(inst) or is_amino_acid(inst)) for inst in instances):
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

    petn_sites = {(group.sugar, group.sugar_carbon) for group in petn}
    validate_connections(instances, connections, peptide_bonds, petn_sites)
    connection_sites = {
        (link.donor, link.donor_carbon) for link in connections} | {
        (link.acceptor, link.acceptor_carbon) for link in connections} | {
        (bond.sugar, bond.sugar_carbon) for bond in peptide_bonds}
    validate_lipids(instances, lipids, connection_sites | petn_sites)
    validate_petn(instances, petn, connection_sites | {
        (lip.sugar, lip.sugar_carbon) for lip in lipids if lip.anchor == "sugar"},
        point_count, connections, set((functional_points or {}).values()))
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

    glycosidic = [link for link in connections if link.bond == "glycosidic"]
    bridges = [link for link in connections if link.bond == "phosphate"]
    if root in {link.donor for link in bridges}:
        raise MISOExportError(
            f"{instances[root]['label']} is the root, but it is the sugar rotated by a phosphate "
            "bridge. Choose another root, or swap donor and acceptor of the bridge.")
    adjacent = {i: set() for i in sugar_ids}
    for link in glycosidic:
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
    for link in bridges:
        if link.acceptor not in reached:
            raise MISOExportError(
                f"Phosphate bridge to {instances[link.acceptor]['label']}: the target sugar must "
                "be connected to the root by glycosidic bonds, because MISO builds the bridge "
                "from it.")
    reached.update(link.donor for link in bridges)
    if len(reached) != len(sugar_ids):
        missing = ", ".join(instances[i]["label"] for i in sugar_ids if i not in reached)
        hint = (" With optimized orientations all sugars must form one glycan joined by "
                "sugar-to-sugar bonds; separate glycans on different residues need "
                "'Keep positioned geometry and rotations'."
                if peptide_bonds and not fixed_orientation else "")
        raise MISOExportError(f"All sugars must connect to the root. Unconnected: {missing}.{hint}")
    if not fixed_orientation:
        reached = {root} | {link.donor for link in bridges}
        while True:
            donors = {link.donor for link in glycosidic if link.acceptor in reached}
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
    for link in glycosidic:
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
    if bridges:
        config["Phosphate"] = [
            [molecule_ids[link.donor], link.donor_carbon,
             molecule_ids[link.acceptor], link.acceptor_carbon] for link in bridges]
    if petn:
        config["PEtN"] = [{"connection": [molecule_ids[group.sugar], group.sugar_carbon],
                           "direction": [group.point]} for group in petn]
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
    if lipids:
        # MISO branches from the first sugar attached lipid it builds: put the parent first.
        parent = lipid_branch_parent(lipids)
        sugar_first = [lip for lip in lipids if lip.anchor == "sugar"]
        if parent is not None:
            parent_lipid = lipids[parent]
            sugar_first.remove(parent_lipid)
            sugar_first.insert(0, parent_lipid)
        lipids = sugar_first + [lip for lip in lipids if lip.anchor == "lipid"]

        def xyz(point):
            return [round(float(point[0]), 6), round(float(point[1]), 6), 0.0]

        entries = []
        for lip in lipids:
            entry = {"connection_anchor": lip.anchor}
            if lip.anchor == "sugar":
                entry["connection"] = [molecule_ids[lip.sugar], lip.sugar_carbon]
            else:
                entry["connection_blob"] = xyz(lip.end)
            entry["direction"] = xyz(lip.start)
            if lip.mid is not None:
                entry["midpoint"] = xyz(lip.mid)
            if lip.carbons:
                entry["n_carbons"] = int(lip.carbons)
            entry["linkage"] = lip.linkage
            entries.append(entry)
        config["Lipids"] = entries
    if fixed_orientation:
        config["orientation_csv_path"] = f"{stem}_orientations.csv"
    return config
