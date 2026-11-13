"""
refine.py: rigid body REFINE.

Replaces the atom level minimization and Langevin MD of REFINE for sugar
systems.  Every sugar unit is a rigid bead (see beads.py) with four degrees
of freedom that matter: three rotations about its pivot and the height of
that pivot.  Its xy stays at the experimental position through a flat
bottom restraint (free inside r0_xy, stiff outside).  Bridging oxygens and
any non sugar atoms are free particles.  Fixed atoms stay where they are.

Energy
------
    E = E_MMFF(all atoms)                      bonds, angles, torsions, vdW,
                                               electrostatics of the whole
                                               molecule; the intra bead part
                                               is a constant (rigid motion)
      + E_slab(z)                              gravity (brightness weighted),
                                               floor, rising slab, ceiling,
                                               per atom, same functional
                                               forms as force_field_optimization
      + k_xy  * sum_beads max(0, |dxy| - r0)^2     experimental xy
      + k_face * sum_beads max(0, margin - face * n_z)^2   keep the assembled face
      + k_site * sum_linkages (|O - S_donor|^2 + |O - S_acceptor|^2)
                                               the bridging O is tied to the two
                                               hydroxyl sites it replaced; each
                                               site is a body fixed point of its
                                               bead (fourth tetrahedral direction
                                               at the linkage carbon), so the O
                                               cannot cross to the H side of a
                                               linkage carbon and invert it, and
                                               a stretched linkage shares its
                                               stretch between the two C-O bonds
      + phi restraint per linkage              flat bottom torsion restraint
                                               added to the MMFF force field

Forces on atoms are projected on each bead as a net force on the pivot and
a torque about it; the face term enters as a torque only.  The structure is
relaxed by damped descent with backtracking on the total energy while the
slab rises in increments, then relaxed once more with the slab held.

Because atoms inside a bead only ever move through the bead's rotation and
translation, the internal geometry is identical to the start to machine
precision.  bead_rigidity() in beads.py verifies this after the run.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation as Rot

from .beads import (BeadSet, bead_rigidity, build_beads, find_linkages,
                    oriented_normal, ring_pucker, template_deviation)
from ..structure import pdb_export
from ..geometry.geometry_utils import rotate_molecule_around_com


@dataclass
class RigidRefineConfig:
    # External potential (same forms as force_field_optimization)
    gravity: float = 2.0
    use_brightness: bool = True
    slab_force_scale: float = 50.0
    floor_z: float = 0.0
    ceiling_z: Optional[float] = None     # None: thickest bead + ceiling_margin
    ceiling_margin: float = 1.0
    slab_start_offset: float = 2.0         # slab starts this far below the lowest atom,
                                           # ceiling this far above the highest
    start_clearance: float = 1.0           # whole structure lifted so its lowest atom sits here above the floor
    slab_step: float = 0.05
    slab_interval: int = 10                # descent iterations per slab increment
    max_slab_increments: int = 400
    slab_stop_margin: float = 0.5          # stop rising when slab + thickest bead + margin reaches ceiling

    # Restraints
    k_xy: float = 50.0
    r0_xy: float = 0.5
    # Face hold.  The assembled face is the hypothesis under test, so it is held
    # stiffly through the press; a state whose faces cannot be kept flat shows
    # up as strain (MMFF energy, linkage stretch, xy drift) in the report and
    # the score decides.  With k_face around 200 the press flips a sugar instead
    # (cellobiose 00 and 11 collapse onto 10).
    k_face: float = 5000.0
    face_margin: float = 0.5
    k_phi: float = 0.02                    # kcal/mol/deg^2 outside the window; 0 disables
    phi_window: float = 25.0
    # Bridging O to hydroxyl site springs (kcal/mol/A^2 per site); 0 disables.
    # MMFF bond terms are about 10x stiffer, so the lengths stay MMFF's and the
    # springs fix the direction (stereo at both linkage carbons) and share the
    # stretch.  k_site 50 at a 0.3 A site offset costs 4.5 kcal/mol.
    k_site: float = 50.0
    site_bond: float = 1.43                # C-O distance at which the site sits
    phi_ref_alpha: float = 70.0
    phi_ref_beta: float = -70.0

    # Descent (L-BFGS per slab increment, then a final run)
    final_iters: int = 2000
    grad_tol: float = 1e-3
    # Release: after the press, relax once more with slab and ceiling removed
    # (floor and gravity kept), as the paper's final unconstrained minimisation,
    # so the structure keeps only the strain it needs to lie on the surface.
    release: bool = True
    release_iters: int = 2000

    verbose: bool = True
    trajectory_path: Optional[str] = None
    trajectory_interval: int = 20


# ---------------------------------------------------------------------------
# External potential: energies and z forces per atom
# ---------------------------------------------------------------------------

def external_energy_forces(z, masses, weights, slab_z, ceiling_z, cfg: RigidRefineConfig):
    """Return (energy, fz) for gravity + floor + slab + ceiling.  The forces
    are the ones in force_field_optimization.calculate_slab_forces and
    calculate_gravity_forces; the energies are their integrals, continuous
    across the piecewise boundaries, so a line search can use them."""
    S = cfg.slab_force_scale
    g = cfg.gravity * masses * weights

    above = z > 0
    e_grav = (g * np.where(above, z, 0.0)).sum()
    f_grav = np.where(above, -g, 0.0)

    zf = z - cfg.floor_z
    ef = np.exp(np.minimum(-15.0 * zf, 60.0))
    e_floor = S * 200.0 / 15.0 * ef
    f_floor = S * 200.0 * ef

    zc = z - ceiling_z
    ec = np.exp(np.minimum(10.0 * zc, 60.0))
    e_ceil = S * 10.0 * ec
    f_ceil = -S * 100.0 * ec

    d = z - slab_z
    below = d < 0
    trans = (d >= 0) & (d < 1.5)
    es = np.exp(np.minimum(-5.0 * d, 60.0))
    e_slab = np.where(below, S * 2.0 * es - 2.0 * S + S * 1.125,
                      np.where(trans, S * (1.5 - d) ** 3 / 3.0, 0.0))
    f_slab = np.where(below, S * 10.0 * es,
                      np.where(trans, S * (1.5 - d) ** 2, 0.0))

    energy = e_grav + e_floor.sum() + e_ceil.sum() + e_slab.sum()
    fz = f_grav + f_floor + f_ceil + f_slab
    return float(energy), fz


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class _State:
    """Positions of every atom as a function of the bead variables."""

    def __init__(self, pos0: np.ndarray, beadset: BeadSet):
        self.beadset = beadset
        self.pos = pos0.copy()
        self.pivots = np.array([b.pivot for b in beadset.beads], dtype=float)
        self.rots = np.array([np.eye(3) for _ in beadset.beads])
        self.free = np.array(beadset.free_atoms, dtype=int)
        self.sync()

    def copy(self):
        s = _State.__new__(_State)
        s.beadset = self.beadset
        s.pos = self.pos.copy()
        s.pivots = self.pivots.copy()
        s.rots = self.rots.copy()
        s.free = self.free
        return s

    def sync(self):
        for k, b in enumerate(self.beadset.beads):
            self.pos[b.atoms] = self.pivots[k] + b.body @ self.rots[k].T

    def move(self, dp, dtheta, dfree):
        """Apply bead translations dp (nb,3), rotation vectors dtheta (nb,3)
        and free atom displacements dfree (nf,3).  Returns a new state."""
        s = self.copy()
        s.pivots = s.pivots + dp
        for k in range(len(s.rots)):
            if np.any(dtheta[k]):
                s.rots[k] = Rot.from_rotvec(dtheta[k]).as_matrix() @ s.rots[k]
        if len(s.free):
            s.pos[s.free] = s.pos[s.free] + dfree
        s.sync()
        return s


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RigidRefine:
    def __init__(self, mol, beadset: BeadSet, linkages, cfg: RigidRefineConfig):
        self.mol = mol
        self.bs = beadset
        self.links = linkages
        self.cfg = cfg
        self.n = mol.GetNumAtoms()
        self.masses = np.array([a.GetMass() for a in mol.GetAtoms()])
        self.weights = np.ones(self.n)
        if cfg.use_brightness:
            for b in beadset.beads:
                self.weights[b.atoms] = max(1.0, b.brightness + 1.0)

        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            raise RuntimeError("MMFF94 could not be set up for this molecule")
        self.ff = AllChem.MMFFGetMoleculeForceField(mol, props)
        if cfg.k_phi > 0:
            for l in linkages:
                if l['o5'] is None or l['anomer'] not in ('alpha', 'beta'):
                    continue
                ref = cfg.phi_ref_alpha if l['anomer'] == 'alpha' else cfg.phi_ref_beta
                self.ff.MMFFAddTorsionConstraint(
                    l['o5'], l['cd'], l['o'], l['ca'], False,
                    ref - cfg.phi_window, ref + cfg.phi_window, cfg.k_phi)
        self.ff.Initialize()
        self.ff_plain = AllChem.MMFFGetMoleculeForceField(mol, props)   # no restraints, for the report
        self.ff_plain.Initialize()

        self.target_xy = np.array([b.pivot[:2] for b in beadset.beads], dtype=float)
        self.faces = np.array([b.face for b in beadset.beads], dtype=float)
        self.bead_mass = np.array([b.masses.sum() for b in beadset.beads])
        self.bead_inertia = np.array([max(b.inertia, 1.0) for b in beadset.beads])
        self.fixed = np.array(beadset.fixed_atoms, dtype=int)

        # Thickness of each bead in its flattest orientation (extent along the
        # ring normal, atom centres).  Sets the ceiling and the slab stop.
        self.thickness = []
        for b in beadset.beads:
            if b.ring is not None:
                nrm = oriented_normal(b.body[[b.atoms.index(i) for i in b.ring]])
            else:
                nrm = None
            if nrm is None:
                ext = b.body[:, 2]
            else:
                ext = b.body @ nrm
            self.thickness.append(float(ext.max() - ext.min()))
        self.t_max = max(self.thickness) if self.thickness else 2.0
        self.ceiling = cfg.ceiling_z if cfg.ceiling_z is not None else self.t_max + cfg.ceiling_margin

        # Hydroxyl sites: for each linkage, the point where the replaced
        # hydroxyl oxygen sat at the donor and at the acceptor carbon, as body
        # fixed vectors (relative to the bead pivot, in the start orientation).
        # site = C + site_bond * u, u opposite to the mean unit vector towards
        # the carbon's three remaining neighbours (fourth tetrahedral position).
        conf = mol.GetConformer()
        pos0 = np.array([list(conf.GetAtomPosition(i)) for i in range(self.n)])
        self.sites = []                     # (link index, bead index, body vector, carbon atom)
        for li, l in enumerate(linkages):
            for c, bk in ((l['cd'], l['donor']), (l['ca'], l['acceptor'])):
                b = beadset.beads[bk]
                nbrs = [a.GetIdx() for a in mol.GetAtomWithIdx(c).GetNeighbors()
                        if a.GetIdx() != l['o'] and a.GetIdx() in b.atoms]
                if len(nbrs) < 2:
                    continue
                u = np.zeros(3)
                for nb_ in nbrs:
                    v = pos0[nb_] - pos0[c]
                    u += v / np.linalg.norm(v)
                nu = np.linalg.norm(u)
                if nu < 1e-6:
                    continue
                site = pos0[c] - cfg.site_bond * u / nu
                self.sites.append((li, bk, site - b.pivot, c))

    def _site_world(self, st: _State, bk: int, body: np.ndarray):
        return st.pivots[bk] + st.rots[bk] @ body

    # -- energy -------------------------------------------------------------

    def energy(self, st: _State, slab_z: float, ceiling_z: float):
        pos = st.pos
        e_ff = self.ff.CalcEnergy(pos.ravel().tolist())
        e_ext, _ = external_energy_forces(pos[:, 2], self.masses, self.weights,
                                          slab_z, ceiling_z, self.cfg)
        dxy = st.pivots[:, :2] - self.target_xy
        r = np.linalg.norm(dxy, axis=1)
        e_xy = self.cfg.k_xy * (np.maximum(0.0, r - self.cfg.r0_xy) ** 2).sum()
        e_face = 0.0
        for k, b in enumerate(self.bs.beads):
            if b.ring is None or self.faces[k] == 0:
                continue
            nrm = oriented_normal(pos[b.ring])
            if nrm is None:
                continue
            e_face += self.cfg.k_face * max(0.0, self.cfg.face_margin - self.faces[k] * nrm[2]) ** 2
        e_site = 0.0
        if self.cfg.k_site > 0:
            for li, bk, body, _ in self.sites:
                d = pos[self.links[li]['o']] - self._site_world(st, bk, body)
                e_site += self.cfg.k_site * float(d @ d)
        return (e_ff + e_ext + e_xy + e_face + e_site,
                {'ff': e_ff, 'ext': e_ext, 'xy': e_xy, 'face': e_face, 'site': e_site})

    # -- generalized forces -------------------------------------------------

    def forces(self, st: _State, slab_z: float, ceiling_z: float):
        """Returns (F_bead (nb,3), tau_bead (nb,3), F_free (nf,3), gmax)."""
        pos = st.pos
        grad = np.array(self.ff.CalcGrad(pos.ravel().tolist())).reshape(-1, 3)
        f = -grad
        _, fz = external_energy_forces(pos[:, 2], self.masses, self.weights,
                                       slab_z, ceiling_z, self.cfg)
        f[:, 2] += fz

        nb = len(self.bs.beads)
        F = np.zeros((nb, 3))
        T = np.zeros((nb, 3))
        for k, b in enumerate(self.bs.beads):
            fa = f[b.atoms]
            F[k] = fa.sum(axis=0)
            r = pos[b.atoms] - st.pivots[k]
            T[k] = np.cross(r, fa).sum(axis=0)
            if b.ring is not None and self.faces[k] != 0:
                nrm = oriented_normal(pos[b.ring])
                if nrm is not None:
                    viol = self.cfg.face_margin - self.faces[k] * nrm[2]
                    if viol > 0:
                        dE_dnz = -2.0 * self.cfg.k_face * viol * self.faces[k]
                        T[k] -= dE_dnz * np.cross(nrm, np.array([0.0, 0.0, 1.0]))
        if self.cfg.k_site > 0:
            for li, bk, body, _ in self.sites:
                o = self.links[li]['o']
                sw = self._site_world(st, bk, body)
                fs = -2.0 * self.cfg.k_site * (pos[o] - sw)   # force on the oxygen
                f[o] += fs
                F[bk] -= fs                                     # reaction on the bead
                T[bk] += np.cross(sw - st.pivots[bk], -fs)
        dxy = st.pivots[:, :2] - self.target_xy
        r = np.linalg.norm(dxy, axis=1)
        out = r > self.cfg.r0_xy
        if out.any():
            F[out, :2] -= (2.0 * self.cfg.k_xy * (r[out] - self.cfg.r0_xy) / r[out])[:, None] * dxy[out]
        Ff = f[st.free] if len(st.free) else np.zeros((0, 3))
        gmax = 0.0
        if nb:
            gmax = max(gmax, float(np.abs(F / self.bead_mass[:, None]).max()),
                       float(np.abs(T / self.bead_inertia[:, None]).max()))
        if len(Ff):
            gmax = max(gmax, float((np.linalg.norm(Ff, axis=1) / self.masses[st.free]).max()))
        return F, T, Ff, gmax

    # -- descent --------------------------------------------------------------

    @staticmethod
    def _left_jacobian(w):
        """Left Jacobian of SO(3): dR = [J(w) dw]x R for R = exp([w]x) R0."""
        th = np.linalg.norm(w)
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        if th < 1e-6:
            return np.eye(3) + 0.5 * K
        return (np.eye(3) + (1 - np.cos(th)) / th ** 2 * K
                + (th - np.sin(th)) / th ** 3 * K @ K)

    def _unpack(self, x, base: _State):
        nb = len(self.bs.beads)
        nf = len(base.free)
        dp = x[:3 * nb].reshape(nb, 3)
        w = x[3 * nb:6 * nb].reshape(nb, 3)
        df = x[6 * nb:6 * nb + 3 * nf].reshape(nf, 3)
        return base.move(dp, w, df), w

    def descend(self, st: _State, slab_z: float, ceiling_z: float, n_iter: int, scale=None):
        """Minimise over bead translations, bead rotation vectors (about the
        current orientation) and free atom positions with L-BFGS.  Returns
        (state, E, None, converged)."""
        from scipy.optimize import minimize
        nb = len(self.bs.beads)
        nf = len(st.free)
        base = st

        def fun(x):
            s, w = self._unpack(x, base)
            E, _ = self.energy(s, slab_z, ceiling_z)
            F, T, Ff, _ = self.forces(s, slab_z, ceiling_z)
            gT = np.array([-(self._left_jacobian(w[k]).T @ T[k]) for k in range(nb)])
            g = np.concatenate([(-F).ravel(), gT.ravel(), (-Ff).ravel()])
            return E, g

        x0 = np.zeros(6 * nb + 3 * nf)
        res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': n_iter, 'gtol': self.cfg.grad_tol, 'ftol': 1e-10})
        s, _ = self._unpack(res.x, base)
        E, _ = self.energy(s, slab_z, ceiling_z)
        return s, E, None, bool(res.success)

    # -- driver -----------------------------------------------------------------

    def run(self, st: _State):
        """Press protocol: the structure is lifted clear of the floor, the slab
        starts below it and the ceiling above it, and both close in at
        slab_step per increment until the slab reaches its stop and the
        ceiling its final height.  Then a last relaxation with both held."""
        cfg = self.cfg
        t0 = time.perf_counter()
        pos_start = None            # set after the lift: stages measure the press
        stages = []

        def _stage(name, s, slab, ceil):
            rig = bead_rigidity(s.pos, self.bs, pos_start)
            heavy = [i for b in self.bs.beads for i in b.heavy]
            moved = float(np.sqrt(((s.pos[heavy] - pos_start[heavy]) ** 2).sum(axis=1).mean()))
            faces_ok = True
            for k, b in enumerate(self.bs.beads):
                if b.ring is None or self.faces[k] == 0:
                    continue
                nrm = oriented_normal(s.pos[b.ring])
                if nrm is not None and np.sign(nrm[2]) != self.faces[k]:
                    faces_ok = False
            worst = 0.0
            for l in self.links:
                r1 = np.linalg.norm(s.pos[l['cd']] - s.pos[l['o']])
                r2 = np.linalg.norm(s.pos[l['o']] - s.pos[l['ca']])
                worst = max(worst, float(max(r1, r2) - cfg.site_bond))
            E, parts = self.energy(s, slab, ceil)
            row = {'stage': name, 'max_rigid_rmsd': max((r['rmsd'] for r in rig), default=0.0),
                   'mirror': any(r['mirror'] for r in rig), 'moved_rmsd': moved,
                   'faces_ok': faces_ok, 'worst_stretch': worst,
                   'max_xy_drift': float(np.linalg.norm(s.pivots[:, :2] - self.target_xy, axis=1).max()),
                   'z_min': float(s.pos[:, 2].min()), 'z_max': float(s.pos[:, 2].max()),
                   'E': E, 'ff': parts['ff'], 'ext': parts['ext'], 'xy': parts['xy'],
                   'face': parts['face'], 'site': parts['site'], 't': time.perf_counter() - t0}
            stages.append(row)
            if cfg.verbose:
                print(f"  [{name:<8}] rigid rmsd {row['max_rigid_rmsd']:.1e}  moved {moved:5.2f} A  "
                      f"faces {'ok' if faces_ok else 'LOST'}  stretch {worst:+.3f}  "
                      f"xy {row['max_xy_drift']:.2f}  z {row['z_min']:.2f}..{row['z_max']:.2f}  "
                      f"ff {parts['ff']:.1f}  site {parts['site']:.1f}  {row['t']:.1f} s")

        lift = cfg.floor_z + cfg.start_clearance - float(st.pos[:, 2].min())
        if lift > 0:
            st = st.move(np.tile([0.0, 0.0, lift], (len(self.bs.beads), 1)),
                         np.zeros((len(self.bs.beads), 3)),
                         np.tile([0.0, 0.0, lift], (len(st.free), 1)))
            if len(self.fixed):
                st.pos[self.fixed] -= [0.0, 0.0, lift]   # fixed atoms stay put

        slab_z = float(st.pos[:, 2].min() - cfg.slab_start_offset)
        ceiling = max(self.ceiling, float(st.pos[:, 2].max() + cfg.slab_start_offset))
        slab_stop = self.ceiling - self.t_max - cfg.slab_stop_margin
        E0, parts0 = self.energy(st, slab_z, ceiling)
        pos_start = st.pos.copy()
        _stage('start', st, slab_z, ceiling)
        if cfg.verbose:
            print(f"  lifted by {max(lift, 0):.2f} A; z range {st.pos[:, 2].min():.2f} .. "
                  f"{st.pos[:, 2].max():.2f} A")
            print(f"  press: slab {slab_z:.2f} -> {slab_stop:.2f} A, ceiling {ceiling:.2f} -> "
                  f"{self.ceiling:.2f} A (thickest bead {self.t_max:.2f} A)")
            print(f"  start: E = {E0:.3f} (ff {parts0['ff']:.2f}, ext {parts0['ext']:.2f}, "
                  f"xy {parts0['xy']:.3f}, face {parts0['face']:.3f})")
        n_inc = 0
        frames = []
        while (slab_z < slab_stop or ceiling > self.ceiling) and n_inc < cfg.max_slab_increments:
            st, E, _, _ = self.descend(st, slab_z, ceiling, cfg.slab_interval)
            slab_z = min(slab_z + cfg.slab_step, slab_stop)
            ceiling = max(ceiling - cfg.slab_step, self.ceiling)
            n_inc += 1
            if cfg.trajectory_path and n_inc % cfg.trajectory_interval == 0:
                frames.append(st.pos.copy())
            if cfg.verbose and n_inc % 40 == 0:
                print(f"    slab {slab_z:6.2f} A  ceiling {ceiling:5.2f} A  E = {E:10.3f}  "
                      f"z range {st.pos[:, 2].min():.2f} .. {st.pos[:, 2].max():.2f}")
        _stage('press', st, slab_z, ceiling)
        st, E, _, conv = self.descend(st, slab_z, ceiling, cfg.final_iters)
        E, parts = self.energy(st, slab_z, ceiling)
        _stage('final', st, slab_z, ceiling)
        if cfg.verbose:
            print(f"  pressed: E = {E:.3f} (ff {parts['ff']:.2f}, ext {parts['ext']:.2f}, "
                  f"xy {parts['xy']:.3f}, face {parts['face']:.3f}), "
                  f"{'converged' if conv else 'iteration limit'}, "
                  f"{n_inc} increments, z range {st.pos[:, 2].min():.2f} .. "
                  f"{st.pos[:, 2].max():.2f}, {time.perf_counter() - t0:.1f} s")
        if cfg.release:
            slab_z, ceiling = -1e3, 1e3          # slab and ceiling removed, floor and gravity stay
            st, E, _, conv = self.descend(st, slab_z, ceiling, cfg.release_iters)
            E, parts = self.energy(st, slab_z, ceiling)
            _stage('release', st, slab_z, ceiling)
            if cfg.verbose:
                print(f"  released: E = {E:.3f} (ff {parts['ff']:.2f}, ext {parts['ext']:.2f}, "
                      f"xy {parts['xy']:.3f}, face {parts['face']:.3f}), "
                      f"{'converged' if conv else 'iteration limit'}, z range "
                      f"{st.pos[:, 2].min():.2f} .. {st.pos[:, 2].max():.2f}, "
                      f"{time.perf_counter() - t0:.1f} s")
        if cfg.trajectory_path and frames:
            _write_frames(self.mol, frames, cfg.trajectory_path)
        return st, {'energy_initial': E0, 'energy_final': E, 'parts': parts, 'stages': stages,
                    'slab_z': slab_z, 'ceiling': ceiling, 'converged': conv,
                    'slab_increments': n_inc, 'seconds': time.perf_counter() - t0}


def _write_frames(mol, frames, path):
    w = Chem.SDWriter(path)
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for k, pos in enumerate(frames):
        for i in range(m.GetNumAtoms()):
            conf.SetAtomPosition(i, pos[i].tolist())
        m.SetProp('_Name', f'frame {k}')
        w.write(m)
    w.close()


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))))


def linkage_report(pos, links, beadset: BeadSet, co_ideal=1.43):
    rows = []
    for l in links:
        o, cd, ca = pos[l['o']], pos[l['cd']], pos[l['ca']]
        r1, r2 = np.linalg.norm(cd - o), np.linalg.norm(o - ca)
        v1, v2 = cd - o, ca - o
        ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (r1 * r2), -1, 1)))
        phi = _dihedral(pos[l['o5']], cd, o, ca) if l['o5'] is not None else None
        rows.append({'name': l['name'], 'r_donor': float(r1), 'r_acceptor': float(r2),
                     'stretch': float(max(r1, r2) - co_ideal), 'angle': float(ang),
                     'phi': phi, 'anomer': l['anomer']})
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def refine_structure(mol_h, mol_no_h=None, chain_dict=None, linkage_definitions=None,
                     fixed_atoms=(), config: Optional[RigidRefineConfig] = None,
                     residue_index=None):
    """
    Rigid body REFINE of an exported structure.

    mol_h : RDKit molecule with hydrogens (output of prepare_structure_for_optimization).
    mol_no_h : the heavy atom molecule it was made from, carrying pdb_export residue
        info.  Used to propagate residue names to mol_h.  Not needed when mol_h
        already carries residue info or residue_index is given.
    chain_dict : builder unit dicts (experimental COM, brightness, template coords).
    linkage_definitions : config['glycosidic_bonds'] entries, for the anomer per linkage.
    fixed_atoms : atom indices held fixed (linker rings and the like).

    Returns (mol_out, report).  mol_out is a copy of mol_h with new coordinates.
    """
    cfg = config or RigidRefineConfig()
    mol = Chem.Mol(mol_h)
    if mol_no_h is not None and all(a.GetMonomerInfo() is None for a in mol.GetAtoms()):
        pdb_export.propagate_residue_info_to_h(mol_no_h, mol)
    if residue_index is None:
        residue_index = pdb_export.build_residue_index(mol)

    if cfg.verbose:
        print("\n" + "=" * 60)
        print("RIGID BODY REFINE (sugars rigid, xy restrained, z free)")
        print("=" * 60)
    bs = build_beads(mol, residue_index, chain_dict=chain_dict, fixed_atoms=fixed_atoms,
                     verbose=cfg.verbose)
    if not bs.beads:
        raise RuntimeError("no sugar beads found: residue info missing on the molecule")
    links = find_linkages(mol, bs, linkage_definitions, verbose=cfg.verbose)

    conf = mol.GetConformer()
    pos0 = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    pre = template_deviation(mol, bs, chain_dict, verbose=cfg.verbose) if chain_dict else []

    engine = RigidRefine(mol, bs, links, cfg)
    st0 = _State(pos0, bs)
    st, run = engine.run(st0)

    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, st.pos[i].tolist())

    rig = bead_rigidity(st.pos, bs, pos0)

    # Rigid body state per bead: the rotation accumulated during REFINE (the
    # start orientation is the identity) and the final pivot.  Written back to
    # chain_dict so the unit's quaternion, COM and coordinates describe the
    # refined structure: (template, COM, quaternion) per sugar is the complete
    # description of a polymer and what a score should consume.
    state_rows = []
    for k, b in enumerate(bs.beads):
        q = Rot.from_matrix(st.rots[k]).as_quat()        # x, y, z, w (scipy order)
        new_com = st.pivots[k]
        if chain_dict is not None and b.name in chain_dict:
            md = chain_dict[b.name]
            if 'quaternion' not in md:
                md['quaternion'] = [0.0, 0.0, 0.0, 1.0]
            if md.get('COM') is None:
                md['COM'] = b.pivot.tolist()
            upd = rotate_molecule_around_com(md, st.rots[k])    # about the old COM
            shift = new_com - np.asarray(md['COM'], dtype=float)
            upd['absolute_coordinates'] = (np.asarray(upd['absolute_coordinates']) + shift).tolist()
            upd['COM'] = new_com.tolist()
            upd['refine_rotation_quaternion'] = q.tolist()   # REFINE's own rotation, start = identity
            chain_dict[b.name] = upd
            q_total = upd['quaternion']
        else:
            q_total = q.tolist()
        state_rows.append({'bead': b.name, 'com': new_com.tolist(), 'com_start': b.pivot.tolist(),
                           'refine_quaternion': q.tolist(), 'quaternion': list(q_total)})
    faces_now = []
    for k, b in enumerate(bs.beads):
        n = oriented_normal(st.pos[b.ring]) if b.ring is not None else None
        faces_now.append(0 if n is None else (1 if n[2] > 0 else -1))
    beads_rows = []
    for k, b in enumerate(bs.beads):
        Q, th = ring_pucker(st.pos, b.ring) if b.ring is not None else (None, None)
        beads_rows.append({'bead': b.name, 'rigid_rmsd': rig[k]['rmsd'],
                           'face_start': int(b.face), 'face_end': int(faces_now[k]),
                           'face_kept': int(b.face) == int(faces_now[k]) or b.face == 0,
                           'xy_drift': float(np.linalg.norm(st.pivots[k][:2] - b.pivot[:2])),
                           'z': float(st.pivots[k][2]), 'z_start': float(b.pivot[2]),
                           'Q': Q, 'theta': th})
    link_rows = linkage_report(st.pos, links, bs)
    # Site angles: angle at each linkage carbon between the C-O bond and the
    # direction to the hydroxyl site it replaced.  Near 0 is right; above 90
    # means the oxygen crossed to the hydrogen side and the carbon is inverted.
    for li, bk, body, c in engine.sites:
        sw = engine._site_world(st, bk, body)
        o = links[li]['o']
        v1, v2 = st.pos[o] - st.pos[c], sw - st.pos[c]
        ang = float(np.degrees(np.arccos(np.clip(
            v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))))
        key = 'site_angle_donor' if c == links[li]['cd'] else 'site_angle_acceptor'
        link_rows[li][key] = ang
    for r in link_rows:
        r['stereo_ok'] = max(r.get('site_angle_donor', 0.0), r.get('site_angle_acceptor', 0.0)) < 90.0
    e_mmff = float(engine.ff_plain.CalcEnergy(st.pos.ravel().tolist()))
    e_mmff0 = float(engine.ff_plain.CalcEnergy(pos0.ravel().tolist()))
    report = {'run': run, 'beads': beads_rows, 'linkages': link_rows, 'state': state_rows,
              'energy_mmff': e_mmff, 'energy_mmff_before': e_mmff0,
              'template_deviation_before': pre,
              'max_rigid_rmsd': max((r['rmsd'] for r in rig), default=0.0),
              'faces_kept': all(r['face_kept'] for r in beads_rows),
              'stereo_ok': all(r['stereo_ok'] for r in link_rows),
              'ceiling': engine.ceiling, 'thickness': engine.thickness}
    if cfg.verbose:
        print(f"  MMFF energy (no restraints): {e_mmff0:.2f} -> {e_mmff:.2f} kcal/mol")
        print("  Beads after REFINE:")
        for r in beads_rows:
            print(f"    {r['bead']:<10} rigid rmsd={r['rigid_rmsd']:.2e} A  face "
                  f"{'+' if r['face_start'] > 0 else '-'}->{'+' if r['face_end'] > 0 else '-'}"
                  f"{'' if r['face_kept'] else ' LOST'}  xy drift={r['xy_drift']:.2f} A  "
                  f"z {r['z_start']:.2f}->{r['z']:.2f} A"
                  + (f"  Q={r['Q']:.2f} theta={r['theta']:.0f}" if r['Q'] is not None else ''))
        print("  Linkages after REFINE:")
        for r in link_rows:
            print(f"    {r['name']:<28} C-O {r['r_donor']:.3f}/{r['r_acceptor']:.3f} A "
                  f"(stretch {r['stretch']:+.3f}) C-O-C {r['angle']:.1f}  "
                  + (f"phi {r['phi']:.0f} ({r['anomer']})  " if r['phi'] is not None else '')
                  + f"site {r.get('site_angle_donor', 0):.0f}/{r.get('site_angle_acceptor', 0):.0f}"
                  + ('' if r['stereo_ok'] else ' INVERTED'))
    return mol, report
