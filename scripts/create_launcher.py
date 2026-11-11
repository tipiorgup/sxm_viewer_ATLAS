#!/usr/bin/env python3
"""Create a click to start icon for the MISO GUI on Windows or Linux.

Run it ONCE on each computer, with the Python of the environment you use there
(an activated venv, an activated conda environment, or plain python):

    python scripts/create_launcher.py

The environment that runs this script is the one the icon will start, so nothing
has to be activated by hand afterwards. Run it again after moving the repository
or recreating the environment.

What it makes
  Windows, venv or plain Python : a Desktop shortcut to pythonw.exe (no console window).
  Windows, conda                : a hidden launcher (launchers/launch_miso_gui.vbs) that
                                  activates the conda environment, plus a Desktop shortcut.
  Linux, venv or plain Python   : launchers/launch_miso_gui.sh and a menu entry
                                  (~/.local/share/applications/miso-gui.desktop) and a
                                  copy on the Desktop.
  Linux, conda                  : the same, with the script activating the conda environment.

The generated files hold paths of THIS computer, so the launchers folder is not meant
to be committed; the folder ignores itself in git. Only the standard library is used.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

APP_NAME = "MISO GUI"
APP_COMMENT = "MISO GUI for STM image analysis"
LAUNCHERS_DIRNAME = "launchers"


@dataclass
class Plan:
    """Everything needed to create the icon, as data so it can be checked without a desktop."""
    system: str                                  # "windows" or "linux"
    kind: str                                    # "venv", "conda" or "system"
    files: dict[str, str] = field(default_factory=dict)    # launcher files, by name
    target: str = ""                             # Windows shortcut target
    arguments: str = ""                          # Windows shortcut arguments
    exec_path: str = ""                          # Linux script started by the menu entry
    notes: list[str] = field(default_factory=list)


def detect_kind(prefix: Path) -> str:
    """conda when the prefix is a conda environment, venv when it has pyvenv.cfg."""
    prefix = Path(prefix)
    if (prefix / "conda-meta").is_dir():
        return "conda"
    if (prefix / "pyvenv.cfg").is_file():
        return "venv"
    return "system"


def conda_base(prefix: Path, conda_exe: str | None = None) -> Path:
    """Folder of the conda installation that owns this environment."""
    prefix = Path(prefix)
    if conda_exe:
        exe = Path(conda_exe)
        # <base>/bin/conda, <base>/condabin/conda or <base>\Scripts\conda.exe
        if exe.parent.name.lower() in ("bin", "scripts", "condabin") and exe.parent.parent.is_dir():
            return exe.parent.parent
    if prefix.parent.name == "envs":
        return prefix.parent.parent
    return prefix


def _pure(system: str, path) -> "PurePosixPath | PureWindowsPath":
    """The path written the way the TARGET system spells it, whatever system this runs on."""
    text = str(path)
    if system == "windows":
        return PureWindowsPath(text.replace("/", "\\"))
    return PurePosixPath(text.replace("\\", "/"))


def _vbs_quote(text: str) -> str:
    return text.replace('"', '""')


def make_plan(system: str, kind: str, python, prefix, repo, base=None, launchers=None) -> Plan:
    p = lambda value: _pure(system, value)          # noqa: E731
    python, prefix, repo = p(python), p(prefix), p(repo)
    launchers = p(launchers) if launchers else repo / LAUNCHERS_DIRNAME
    plan = Plan(system=system, kind=kind)
    if system == "windows":
        pythonw = python.with_name("pythonw.exe")
        if kind == "conda":
            base = p(base) if base else p(conda_base(Path(str(prefix))))
            activate = base / "Scripts" / "activate.bat"
            if os.name == "nt" and not Path(str(activate)).is_file():
                plan.notes.append(f"Could not find {activate}; check the conda install.")
            command = (f'cmd /c call "{activate}" "{prefix}" && '
                       f'"{prefix / "pythonw.exe"}" -m sxm_viewer')
            vbs = (
                'Set sh = CreateObject("WScript.Shell")\r\n'
                f'sh.CurrentDirectory = "{_vbs_quote(str(repo))}"\r\n'
                f'sh.Run "{_vbs_quote(command)}", 0, False\r\n')
            plan.files["launch_miso_gui.vbs"] = vbs
            plan.target = "wscript.exe"
            plan.arguments = f'"{launchers / "launch_miso_gui.vbs"}"'
        else:
            plan.target = str(pythonw)
            plan.arguments = "-m sxm_viewer"
            if os.name == "nt" and not Path(str(pythonw)).is_file():
                plan.notes.append(f"{pythonw} not found; the shortcut will use python.exe.")
                plan.target = str(python)
        return plan

    # Linux
    q = shlex.quote
    lines = ["#!/bin/sh", f"cd {q(str(repo))} || exit 1"]
    if kind == "conda":
        base = p(base) if base else p(conda_base(Path(str(prefix))))
        lines += [f". {q(str(base / 'etc' / 'profile.d' / 'conda.sh'))}",
                  f"conda activate {q(str(prefix))}",
                  'exec python -m sxm_viewer "$@"']
    else:
        lines.append(f'exec {q(str(python))} -m sxm_viewer "$@"')
    plan.files["launch_miso_gui.sh"] = "\n".join(lines) + "\n"
    plan.exec_path = str(launchers / "launch_miso_gui.sh")
    return plan


def desktop_entry(script: Path, repo: Path) -> str:
    """Contents of the Linux .desktop file."""
    script, repo = _pure("linux", script), _pure("linux", repo)
    icon = repo / "miso.png"
    exec_value = '"' + str(script).replace("\\", "\\\\").replace('"', '\\"') + '"'
    return ("[Desktop Entry]\n"
            "Type=Application\n"
            f"Name={APP_NAME}\n"
            f"Comment={APP_COMMENT}\n"
            f"Exec={exec_value}\n"
            f"Path={repo}\n"
            f"Icon={icon}\n"
            "Terminal=false\n"
            "Categories=Science;Education;\n")


# --------------------------------------------------------------------------- writing
def _desktop_dir_linux(home: Path) -> Path | None:
    try:
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True,
                             timeout=5, check=False).stdout.strip()
        if out and Path(out).is_dir() and Path(out) != home:
            return Path(out)
    except (OSError, subprocess.SubprocessError):
        pass
    fallback = home / "Desktop"
    return fallback if fallback.is_dir() else None


def _prepare_launchers_dir(launchers: Path):
    """Create the folder with a .gitignore of its own, so git never sees this computer's paths."""
    launchers.mkdir(parents=True, exist_ok=True)
    (launchers / ".gitignore").write_text("*\n", encoding="utf-8", newline="\n")


def write_linux(plan: Plan, repo: Path, launchers: Path, home: Path,
                desktop: Path | None = None) -> list[Path]:
    _prepare_launchers_dir(launchers)
    written = []
    for name, text in plan.files.items():
        path = launchers / name
        path.write_text(text, encoding="utf-8", newline="\n")
        path.chmod(0o755)
        written.append(path)
    entry = desktop_entry(Path(plan.exec_path), repo)
    menu = home / ".local" / "share" / "applications"
    menu.mkdir(parents=True, exist_ok=True)
    targets = [menu / "miso-gui.desktop"]
    desktop = desktop if desktop is not None else _desktop_dir_linux(home)
    if desktop is not None:
        Path(desktop).mkdir(parents=True, exist_ok=True)
        targets.append(Path(desktop) / "MISO GUI.desktop")
    for path in targets:
        path.write_text(entry, encoding="utf-8", newline="\n")
        path.chmod(0o755)
        if shutil.which("gio"):       # lets file managers launch it without asking
            subprocess.run(["gio", "set", str(path), "metadata::trusted", "true"],
                           capture_output=True, check=False)
        written.append(path)
    return written


_PS_SHORTCUT = r"""
$shell = New-Object -ComObject WScript.Shell
$s = $shell.CreateShortcut($env:LNK_PATH)
$s.TargetPath = $env:LNK_TARGET
$s.Arguments = $env:LNK_ARGS
$s.WorkingDirectory = $env:LNK_WORKDIR
$s.Description = $env:LNK_DESC
if (Test-Path $env:LNK_ICON) { $s.IconLocation = $env:LNK_ICON + ",0" }
$s.Save()
"""


def write_windows(plan: Plan, repo: Path, launchers: Path, desktop: Path | None = None) -> list[Path]:
    written = []
    if plan.files:
        _prepare_launchers_dir(launchers)
    for name, text in plan.files.items():
        path = launchers / name
        path.write_text(text, encoding="utf-8", newline="")
        written.append(path)
    if desktop is None:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "[Environment]::GetFolderPath('Desktop')"],
            capture_output=True, text=True, check=True).stdout.strip()
        desktop = Path(out)
    link = Path(desktop) / f"{APP_NAME}.lnk"
    env = dict(os.environ, LNK_PATH=str(link), LNK_TARGET=plan.target, LNK_ARGS=plan.arguments,
               LNK_WORKDIR=str(repo), LNK_DESC=APP_COMMENT, LNK_ICON=str(repo / "miso.ico"))
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", _PS_SHORTCUT],
                   env=env, check=True)
    written.append(link)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Create a click to start icon for the MISO GUI.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created.")
    parser.add_argument("--desktop", help="Desktop folder to use (default: detected).")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    launchers = repo / LAUNCHERS_DIRNAME
    if sys.platform.startswith("win"):
        system = "windows"
    elif sys.platform.startswith("linux"):
        system = "linux"
    else:
        print("This script supports Windows and Linux only.")
        return 1
    prefix = Path(sys.prefix)
    kind = detect_kind(prefix)
    plan = make_plan(system, kind, Path(sys.executable), prefix, repo,
                     conda_base(prefix, os.environ.get("CONDA_EXE")))
    print(f"System: {system}   Environment: {kind}   Python: {sys.executable}")
    for note in plan.notes:
        print("Note:", note)
    if args.dry_run:
        for name, text in plan.files.items():
            print(f"--- {launchers / name}\n{text}")
        print("Dry run: nothing written.")
        return 0
    desktop = Path(args.desktop) if args.desktop else None
    if system == "windows":
        written = write_windows(plan, repo, launchers, desktop)
    else:
        written = write_linux(plan, repo, launchers, Path.home(), desktop)
    print("Created:")
    for path in written:
        print("  ", path)
    print("Double click the icon to start. If nothing opens, start the program once from a "
          "terminal (python -m sxm_viewer) to see the error message.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
