"""Shared helpers to load the demo STM image for coordinate/monomer tools.

These mirror ``PositionCoordinatesDialog._load_demo_image`` exactly so any tool
that overlays on the preview image produces numerically identical Angstrom
coordinates (same ``load_sxm_file`` + Pixelsize convention). Extracted as plain
functions so several dialogs can reuse them without subclassing the god class.
"""
from __future__ import annotations
import sys
from pathlib import Path


def demo_loader_dir():
    """Directory holding the MISO demo ``sxm_loader.py``."""
    repo = Path(__file__).resolve().parents[3]
    for rel in (("MISO_demo", "app", "src"), ("MISO", "src", "src_stm")):
        cand = repo.joinpath(*rel)
        if (cand / "sxm_loader.py").exists():
            return cand
    return None


def resolve_sxm_path(viewer):
    """Find the original .sxm for the currently previewed image."""
    file_key = None
    try:
        view = viewer.preview_canvas.views[0]
        file_key = view.get("path") or (view.get("meta") or {}).get("file_path")
    except Exception:
        file_key = None
    header = None
    if file_key:
        header, _ = viewer.headers.get(str(file_key), (None, None))
    # 1) explicit ConvertedSource recorded by the nanonis adapter
    if header:
        for k, v in header.items():
            if str(k).strip().lower().replace("_", "") == "convertedsource" and v:
                p = Path(str(v))
                if p.exists():
                    return p
    # 2) a loaded .sxm whose stem matches the previewed file
    stem = Path(str(file_key)).stem if file_key else ""
    sxm_files = [Path(f) for f in (getattr(viewer, "files", []) or [])
                 if str(f).lower().endswith(".sxm")]
    for f in sxm_files:
        if f.exists() and f.stem == stem:
            return f
    # 3) sibling <stem>.sxm next to the header
    if file_key:
        cand = Path(str(file_key)).with_suffix(".sxm")
        if cand.exists():
            return cand
    # 4) any single loaded .sxm
    for f in sxm_files:
        if f.exists():
            return f
    return None


def load_demo_image(viewer):
    """Load the previewed image via the MISO demo ``load_sxm_file``.

    Returns a dict with keys:
        img       -- originalimg (Angstrom, plane-corrected) numpy array or None
        px        -- (Pixelsize_x, Pixelsize_y) in nm/px
        scan_dir  -- 'up' / 'down' / ''
        sxm_path  -- pathlib.Path of the source .sxm or None
        error     -- error string ('' when img is valid)
    """
    import numpy as np
    out = {"img": None, "px": (1.0, 1.0), "scan_dir": "", "sxm_path": None, "error": ""}

    # Ensure the vendored nanonispy2 is importable (demo loader needs it).
    try:
        from ...providers.nanonis.adapter import _ensure_nanonis_reader
        _ensure_nanonis_reader()
    except Exception:
        pass

    sxm_path = resolve_sxm_path(viewer)
    if not sxm_path:
        out["error"] = "Could not locate the source .sxm file."
        return out
    loader_dir = demo_loader_dir()
    if loader_dir is None:
        out["error"] = "MISO demo sxm_loader.py not found."
        return out
    if str(loader_dir) not in sys.path:
        sys.path.insert(0, str(loader_dir))
    try:
        from sxm_loader import load_sxm_file  # demo loader
    except Exception as exc:
        out["error"] = f"Import of demo loader failed: {exc}"
        return out
    try:
        data = load_sxm_file(str(sxm_path))
    except Exception as exc:
        out["error"] = f"load_sxm_file failed: {exc}"
        return out
    if not data:
        out["error"] = "load_sxm_file returned no data."
        return out
    try:
        out["img"] = np.asarray(data["originalimg"], dtype=float)
        out["px"] = (float(data["Pixelsize"][0]), float(data["Pixelsize"][1]))
        out["scan_dir"] = str(data["header"].get("scan_dir", "")).strip()
        out["sxm_path"] = sxm_path
    except Exception as exc:
        out["img"] = None
        out["error"] = f"Unexpected load_sxm_file output: {exc}"
    return out
