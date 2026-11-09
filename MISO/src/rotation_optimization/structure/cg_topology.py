"""Merge per-segment polyply .itp topologies into one CG [ moleculetype ].

Hybrid strategy: polyply generates a correct Martini 3 topology for each
segment it covers (the sugar chain, the peptide, each lipid tail = PE:N, each
PO4/PEN block).  This module concatenates those segment topologies into a
single molecule - renumbering every atom index - and then adds the
inter-segment ATTACHMENT bonds (sugar<->peptide/lipid/PO4/PEtN) detected from
our assembled all-atom bond graph.

Within-segment topology (bonds, angles, dihedrals, ring constraints, virtual
sites) is polyply's, untouched.  Only the cross-segment attachment bonds are
ours, as simple Martini bonds.

Segment order MUST match the order used by the .gro emitter (cg_published_map),
so the merged .itp and the .gro agree particle-for-particle.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

# 1-based column positions (within a data line's whitespace tokens) that hold
# atom indices, per section.  'atoms' and 'virtual_sitesn' are handled specially.
_INDEX_COLS = {
    "bonds": [0, 1],
    "constraints": [0, 1],
    "pairs": [0, 1],
    "angles": [0, 1, 2],
    "dihedrals": [0, 1, 2, 3],
    "impropers": [0, 1, 2, 3],
    "dihedral_restraints": [0, 1, 2, 3],
    "exclusions": "all",
}

_SECTION_RE = re.compile(r"^\s*\[\s*(.+?)\s*\]\s*$")


def parse_itp(text: str) -> Dict[str, List[str]]:
    """Split an .itp into {section: [data lines]} (comments/blanks dropped)."""
    sections: Dict[str, List[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].rstrip()
        if not line.strip():
            continue
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1).lower()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return sections


def _shift_tokens(line: str, cols, atom_off: int) -> str:
    toks = line.split()
    idxs = range(len(toks)) if cols == "all" else cols
    for c in idxs:
        if c < len(toks) and toks[c].lstrip("-").isdigit():
            toks[c] = str(int(toks[c]) + atom_off)
    return " ".join(toks)


def _shift_atoms(line: str, atom_off: int, res_off: int) -> str:
    # atoms: nr type resnr residue atom cgnr charge mass
    toks = line.split()
    toks[0] = str(int(toks[0]) + atom_off)          # nr
    if len(toks) > 2 and toks[2].isdigit():
        toks[2] = str(int(toks[2]) + res_off)       # resnr
    if len(toks) > 5 and toks[5].isdigit():
        toks[5] = str(int(toks[5]) + atom_off)      # cgnr
    return " ".join(toks)


def _shift_vsiten(line: str, atom_off: int) -> str:
    # "site funct a1 a2 ..." — shift every index except the funct (token 1).
    toks = line.split()
    for c in range(len(toks)):
        if c == 1:
            continue
        if toks[c].lstrip("-").isdigit():
            toks[c] = str(int(toks[c]) + atom_off)
    return " ".join(toks)


def _count_atoms(sections: Dict[str, List[str]]) -> int:
    return len(sections.get("atoms", []))


def _count_residues(sections: Dict[str, List[str]]) -> int:
    res = set()
    for line in sections.get("atoms", []):
        toks = line.split()
        if len(toks) > 2 and toks[2].isdigit():
            res.add(int(toks[2]))
    return len(res)


def merge_segments(segment_itps: List[str], molname: str = "CG") -> Tuple[str, List[int]]:
    """Merge segment .itp texts into one [ moleculetype ].

    Returns (merged_itp_text, atom_offsets) where atom_offsets[i] is the global
    index (0-based) of segment i's first atom — used to translate a segment-
    local bead index to a global one when adding attachment bonds.
    """
    order = ["atoms", "bonds", "pairs", "constraints", "angles", "dihedrals",
             "impropers", "dihedral_restraints", "exclusions", "virtual_sitesn"]
    merged: Dict[str, List[str]] = {k: [] for k in order}

    atom_off = 0
    res_off = 0
    atom_offsets: List[int] = []

    for text in segment_itps:
        sec = parse_itp(text)
        atom_offsets.append(atom_off)
        for name, lines in sec.items():
            if name == "moleculetype":
                continue
            if name == "atoms":
                merged["atoms"].extend(_shift_atoms(l, atom_off, res_off) for l in lines)
            elif name == "virtual_sitesn":
                merged["virtual_sitesn"].extend(_shift_vsiten(l, atom_off) for l in lines)
            elif name in _INDEX_COLS:
                merged.setdefault(name, []).extend(
                    _shift_tokens(l, _INDEX_COLS[name], atom_off) for l in lines)
            # unknown sections are dropped (e.g. position_restraints)
        atom_off += _count_atoms(sec)
        res_off += _count_residues(sec)

    return _render(molname, merged, order), atom_offsets


def add_attachment_bonds(merged_itp: str,
                         bonds: List[Tuple[int, int]],
                         b0: float = 0.40, k: float = 5000.0) -> str:
    """Append inter-segment attachment bonds (1-based global bead indices)."""
    if not bonds:
        return merged_itp
    extra = ["; attachment bonds (cross-segment, added from assembled graph)"]
    for i, j in bonds:
        extra.append(f"{i:5d} {j:5d} {1:5d} {b0:8.4f} {k:9.1f}")
    # Insert into the existing [ bonds ] section (or create it after atoms).
    lines = merged_itp.splitlines()
    out, injected = [], False
    for idx, line in enumerate(lines):
        out.append(line)
        if line.strip().lower().startswith("[ bonds ]") and not injected:
            out.extend(extra)
            injected = True
    if not injected:
        out.extend(["", "[ bonds ]", *extra])
    return "\n".join(out) + "\n"


def _render(molname: str, merged: Dict[str, List[str]], order) -> str:
    out = ["[ moleculetype ]", "; name  nrexcl", f"{molname[:15]:15s} 1", ""]
    for name in order:
        lines = merged.get(name)
        if not lines:
            continue
        out.append(f"[ {name} ]")
        out.extend(lines)
        out.append("")
    return "\n".join(out) + "\n"
