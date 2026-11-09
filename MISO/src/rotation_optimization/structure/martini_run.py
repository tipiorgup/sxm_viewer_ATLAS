"""Run a Martini 3 GROMACS optimization (vacuum) on an exported CG model.

Takes the coarse-grained ``.gro`` + molecule ``.itp`` from :mod:`cg_export`,
writes the missing GROMACS inputs (system ``.top`` and ``.mdp`` files), and
drives ``gmx grompp`` + ``gmx mdrun`` for an energy minimisation and an
(optional) short vacuum MD.  The minimised/relaxed ``.gro`` is the input to
:mod:`cg_backmap`.

Requires:
  * ``gmx`` on PATH (GROMACS).
  * A Martini 3 force-field include (``martini_v3.0.0.itp``) supplied by the
    caller — its path is ``#include``-d into the generated ``.top``.

Vacuum only: no solvent/ions.  Standard Martini 3 non-bonded settings
(reaction-field, epsilon_r=15, rcoulomb=rvdw=1.1, potential-shift-verlet).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# .mdp templates (Martini 3, vacuum)
# ---------------------------------------------------------------------------

_NONBONDED = """\
cutoff-scheme            = Verlet
nstlist                  = 20
coulombtype              = reaction-field
rcoulomb                 = 1.1
epsilon_r                = 15
epsilon_rf               = 0
vdw_type                 = cutoff
vdw-modifier             = potential-shift-verlet
rvdw                     = 1.1
verlet-buffer-tolerance  = 0.005
pbc                      = xyz
"""

EM_MDP = f"""\
; Martini 3 vacuum energy minimisation
integrator               = steep
nsteps                   = 10000
emtol                    = 100.0
emstep                   = 0.01
{_NONBONDED}"""

# dt is intentionally conservative (10 fs): the CG topology from cg_export is
# bonds-only (no angles/constraints yet), so the usual 20 fs Martini step can be
# unstable.  Raise it once angles/constraints are added.
MD_MDP = f"""\
; Martini 3 vacuum short MD (fast relaxation)
integrator               = md
dt                       = 0.010
nsteps                   = 50000        ; 0.5 ns
nstxout-compressed       = 1000
nstlog                   = 1000
nstenergy                = 1000
{_NONBONDED}tcoupl                   = v-rescale
tc-grps                  = System
tau-t                    = 1.0
ref-t                    = 310
gen-vel                  = yes
gen-temp                 = 310
"""

# Flattening ("sandwich") MD — recreates src/md calculate_slab_forces + gravity
# in GROMACS: gravity via a constant acceleration in -z; floor+ceiling via walls
# at z=0 and z=box_z (repulsive 9-3).  The "rising slab / press" is done by
# shrinking box_z between short MD segments (see run_martini_flatten), which
# lowers the top wall onto the structure.
#
# Non-bonded here uses pbc = xy (required for walls), overriding the pbc = xyz in
# _NONBONDED.  GOTCHAS (validate on Linux/gmx):
#   * comm-mode = none is MANDATORY: a uniform `accelerate` is cancelled exactly
#     by the default linear COM-motion removal, so gravity would do nothing.
#   * wall-atomtype must name atomtypes defined by the Martini FF include; tune
#     wall-density/ref-t to taste. Two entries = bottom + top wall.
#   * -maxwarn may be needed (walls + bonds-only topology raise notes).
_FLATTEN_NONBONDED = _NONBONDED.replace("pbc                      = xyz",
                                        "pbc                      = xy")

FLATTEN_MDP = """\
; Martini 3 flattening MD (gravity + sandwich walls)
integrator               = md
dt                       = {dt}
nsteps                   = {nsteps}
nstxout-compressed       = 1000
nstlog                   = 1000
nstenergy                = 1000
{nonbonded}tcoupl                   = v-rescale
tc-grps                  = System
tau-t                    = 1.0
ref-t                    = {ref_t}
gen-vel                  = yes
gen-temp                 = {ref_t}
; --- gravity: constant downward acceleration (nm/ps^2), COM removal OFF ---
comm-mode                = none
acc-grps                 = System
accelerate               = 0 0 {accel_z}
; --- sandwich: repulsive floor (z=0) + ceiling (z=box_z) ---
nwall                    = 2
wall-type                = 9-3
wall-atomtype            = {wall_atomtype} {wall_atomtype}
wall-density             = {wall_density} {wall_density}
wall-r-linpot            = 1.0
"""


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def read_moleculetype_name(itp_path: str) -> str:
    """First token of the line under [ moleculetype ] (the molecule name)."""
    lines = Path(itp_path).read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("[ moleculetype"):
            for follow in lines[i + 1:]:
                s = follow.strip()
                if not s or s.startswith(";"):
                    continue
                return s.split()[0]
    raise ValueError(f"no [ moleculetype ] name found in {itp_path}")


def _parse_itp_atom_types(itp_path: str) -> list:
    """Bead type of every atom in a molecule .itp's [ atoms ] block."""
    types = []
    in_atoms = False
    for line in Path(itp_path).read_text().splitlines():
        s = line.strip()
        if s.startswith("["):
            in_atoms = s.lower().startswith("[ atoms")
            continue
        if in_atoms and s and not s.startswith(";"):
            cols = s.split()
            if len(cols) >= 2:
                types.append(cols[1])
    return types


def _parse_ff_atomtypes(ff_itp: str) -> set:
    """Defined atomtype names from a Martini FF [ atomtypes ] block."""
    names = set()
    in_at = False
    for line in Path(ff_itp).read_text().splitlines():
        s = line.strip()
        if s.startswith("["):
            in_at = s.lower().startswith("[ atomtypes")
            continue
        if in_at and s and not s.startswith(";"):
            names.add(s.split()[0])
    return names


def validate_itp_against_ff(itp_path: str, ff_itp: str) -> list:
    """Return the sorted set of bead types used in ``itp`` but NOT defined in the FF.

    An empty list means every bead type resolves against the force field, so
    ``gmx grompp`` will not choke on an unknown atomtype.  Prints a summary.
    """
    used = _parse_itp_atom_types(itp_path)
    defined = _parse_ff_atomtypes(ff_itp)
    missing = sorted({t for t in used if t not in defined})
    if missing:
        print(f"  ! {len(missing)} bead type(s) NOT in FF: {', '.join(missing)}")
    else:
        print(f"  [martini] all {len(set(used))} bead types resolve against the FF")
    return missing


def write_top(itp_path: str, ff_itp: str, out_top: str,
              molname: Optional[str] = None) -> str:
    """Write a vacuum system .top including the FF and molecule itp."""
    molname = molname or read_moleculetype_name(itp_path)
    # Relative include keeps the .top portable if the files sit together.
    text = (
        f'#include "{ff_itp}"\n'
        f'#include "{Path(itp_path).name}"\n\n'
        "[ system ]\n"
        "CG structure (vacuum)\n\n"
        "[ molecules ]\n"
        f"{molname} 1\n"
    )
    Path(out_top).write_text(text)
    return out_top


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _gmx(gmx_exe: str, args, cwd: Path, log_stem: str):
    """Run one gmx command, tee-ing stdout/stderr to a log file."""
    out = cwd / f"{log_stem}.out.log"
    err = cwd / f"{log_stem}.err.log"
    with open(out, "w") as o, open(err, "w") as e:
        res = subprocess.run([gmx_exe, *args], cwd=cwd, stdout=o, stderr=e, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"gmx {args[0]} failed (rc={res.returncode}). See:\n  {out}\n  {err}")


def run_martini(
    gro: str,
    itp: str,
    ff_itp: str,
    workdir: str = "martini_run",
    *,
    do_md: bool = True,
    gmx_exe: Optional[str] = None,
    maxwarn: int = 2,
) -> dict:
    """Energy-minimise (and optionally short-MD) the CG model in vacuum.

    Returns paths of the produced structures, notably ``min_gro`` (and
    ``md_gro`` if ``do_md``) — feed the last one to cg_backmap.
    """
    gmx_exe = gmx_exe or shutil.which("gmx")
    if gmx_exe is None:
        raise RuntimeError("gmx (GROMACS) not found on PATH.")

    wd = Path(workdir).resolve()
    wd.mkdir(parents=True, exist_ok=True)

    # Stage inputs into the workdir so the .top's relative include resolves.
    local_gro = wd / Path(gro).name
    local_itp = wd / Path(itp).name
    local_gro.write_text(Path(gro).read_text())
    local_itp.write_text(Path(itp).read_text())

    (wd / "em.mdp").write_text(EM_MDP)
    top = write_top(str(local_itp), ff_itp, str(wd / "system.top"))

    # --- Energy minimisation ---
    _gmx(gmx_exe, ["grompp", "-f", "em.mdp", "-c", local_gro.name,
                   "-p", "system.top", "-o", "em.tpr", "-maxwarn", str(maxwarn)],
         wd, "grompp_em")
    _gmx(gmx_exe, ["mdrun", "-deffnm", "em", "-v"], wd, "mdrun_em")
    result = {"workdir": str(wd), "top": top, "min_gro": str(wd / "em.gro")}

    # --- Optional short vacuum MD ---
    if do_md:
        (wd / "md.mdp").write_text(MD_MDP)
        _gmx(gmx_exe, ["grompp", "-f", "md.mdp", "-c", "em.gro",
                       "-p", "system.top", "-o", "md.tpr", "-maxwarn", str(maxwarn)],
             wd, "grompp_md")
        _gmx(gmx_exe, ["mdrun", "-deffnm", "md", "-v"], wd, "mdrun_md")
        result["md_gro"] = str(wd / "md.gro")

    return result


def _set_gro_box_z(gro_in: str, gro_out: str, box_z: float) -> None:
    """Copy a .gro, overwriting the z component of the box vector (nm).

    The last line of a .gro is the box: at minimum ``bx by bz``.  We keep bx/by
    and replace bz, which (with two walls) lowers the top wall to box_z.
    """
    lines = Path(gro_in).read_text().splitlines()
    parts = lines[-1].split()
    bx, by = float(parts[0]), float(parts[1])
    lines[-1] = f"{bx:10.5f}{by:10.5f}{box_z:10.5f}"
    Path(gro_out).write_text("\n".join(lines) + "\n")


def run_martini_flatten(
    gro: str,
    itp: str,
    ff_itp: str,
    workdir: str = "martini_flatten",
    *,
    gravity_accel: float = -2.0,      # nm/ps^2 downward (uniform gravity)
    wall_atomtype: str = "C1",        # must be a Martini FF atomtype
    wall_density: float = 1.0,
    ref_t: float = 310.0,
    dt: float = 0.010,
    box_z_final: float = 1.0,         # nm — target layer thickness ("press")
    n_segments: int = 20,             # compression segments (~slab steps)
    steps_per_segment: int = 5000,
    gmx_exe: Optional[str] = None,
    maxwarn: int = 3,
) -> dict:
    """Recreate the src/md two-phase optimise+flatten in GROMACS Martini.

    Phase 1 (optimise): energy minimisation.
    Phase 2 (flatten):  ``n_segments`` short MD runs; before each, the box z is
    shrunk one step toward ``box_z_final`` so the top wall presses the structure
    onto the floor while uniform gravity pulls it down.  Returns the final
    flattened ``.gro`` as ``flat_gro`` — feed it to cg_backmap.
    """
    gmx_exe = gmx_exe or shutil.which("gmx")
    if gmx_exe is None:
        raise RuntimeError("gmx (GROMACS) not found on PATH.")

    wd = Path(workdir).resolve()
    wd.mkdir(parents=True, exist_ok=True)

    local_gro = wd / Path(gro).name
    local_itp = wd / Path(itp).name
    local_gro.write_text(Path(gro).read_text())
    local_itp.write_text(Path(itp).read_text())
    (wd / "em.mdp").write_text(EM_MDP)
    write_top(str(local_itp), ff_itp, str(wd / "system.top"))

    # Phase 1: minimise.
    _gmx(gmx_exe, ["grompp", "-f", "em.mdp", "-c", local_gro.name,
                   "-p", "system.top", "-o", "em.tpr", "-maxwarn", str(maxwarn)],
         wd, "grompp_em")
    _gmx(gmx_exe, ["mdrun", "-deffnm", "em", "-v"], wd, "mdrun_em")

    # Starting box z (from the minimised structure).
    box_z_start = float(Path(wd / "em.gro").read_text().splitlines()[-1].split()[2])

    # Phase 2: segmented compression.
    current = "em.gro"
    for seg in range(1, n_segments + 1):
        frac = seg / n_segments
        box_z = box_z_start + (box_z_final - box_z_start) * frac
        seg_in = wd / f"seg{seg:02d}_in.gro"
        _set_gro_box_z(str(wd / current), str(seg_in), box_z)

        (wd / "flatten.mdp").write_text(FLATTEN_MDP.format(
            dt=dt, nsteps=steps_per_segment, nonbonded=_FLATTEN_NONBONDED,
            ref_t=ref_t, accel_z=gravity_accel,
            wall_atomtype=wall_atomtype, wall_density=wall_density,
        ))
        _gmx(gmx_exe, ["grompp", "-f", "flatten.mdp", "-c", seg_in.name,
                       "-p", "system.top", "-o", f"seg{seg:02d}.tpr",
                       "-maxwarn", str(maxwarn)], wd, f"grompp_seg{seg:02d}")
        _gmx(gmx_exe, ["mdrun", "-deffnm", f"seg{seg:02d}", "-v"],
             wd, f"mdrun_seg{seg:02d}")
        current = f"seg{seg:02d}.gro"
        print(f"  [martini] compression segment {seg}/{n_segments}: box_z={box_z:.2f} nm")

    return {
        "workdir": str(wd),
        "min_gro": str(wd / "em.gro"),
        "flat_gro": str(wd / current),
        "box_z_start": box_z_start,
        "box_z_final": box_z_final,
    }
