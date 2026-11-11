"""Compatibility layer between the GUI and whatever MISO engine folder is installed.

The MISO engine (module_B.py and its src folder) is updated on its own schedule, so
the GUI must not assume a fixed command line. This module reads the installed engine
and answers three questions without running a calculation:

1. Which command line options does module_B.py declare? (``EngineInfo.options``)
2. Which settings and YAML sections does the engine source still know about?
3. What should the GUI send, and what must it skip, so the run starts?

Only the standard library is used. Nothing here edits the engine or the YAML.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Options the GUI knows how to send. Anything else the engine declares is only reported.
GUI_OPTIONS = ("--input_file", "--iterations", "--n_polymers", "--phase1_kicks",
               "--debug_checkpoints")
# Used when the engine cannot be read at all: the options of the engine this GUI was built on.
LEGACY_OPTIONS = frozenset(GUI_OPTIONS)
# Environment variables the GUI sets for the engine (read in the engine's constants).
GUI_ENV_VARS = ("DEFAULT_COMPRESSION_STEPS", "DEFAULT_GRAVITY")
# YAML keys the GUI writes, with the meaning shown when the engine no longer reads one.
YAML_KEYS = {
    "glycosidic_bonds": "sugar to sugar bonds",
    "Phosphate": "phosphate bridges",
    "PEtN": "phosphoethanolamine groups",
    "Lipids": "lipid attachments",
    "peptide": "peptide chain",
    "peptide_bonds": "sugar to amino acid bonds",
    "orientation_csv_path": "fixed orientations",
    "monomer_data_path": "reuse of the built monomer geometry",
    "circle_input_path": "the positions CSV",
}
MAX_SOURCE_BYTES = 4_000_000


@dataclass(frozen=True)
class EngineOption:
    name: str
    required: bool = False
    takes_value: bool = True


@dataclass
class EngineInfo:
    script: Path
    found: bool = False
    options: dict[str, EngineOption] = field(default_factory=dict)
    how: str = "source"                 # source, help or assumed
    env_vars: set[str] = field(default_factory=set)
    yaml_keys: set[str] = field(default_factory=set)

    def supports(self, flag: str) -> bool:
        return flag in self.options

    def extra_options(self) -> list[str]:
        """Options the engine has and this GUI does not send."""
        return sorted(name for name in self.options if name not in GUI_OPTIONS)

    def required_missing(self) -> list[str]:
        """Required engine options the GUI does not know how to fill."""
        return sorted(name for name, opt in self.options.items()
                      if opt.required and name not in GUI_OPTIONS)


def console_python() -> str:
    """Interpreter for child processes. A shortcut that starts the GUI with pythonw.exe has
    no console, and a pythonw child would swallow all output, so use python.exe beside it."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        sibling = exe.with_name("python.exe")
        if sibling.is_file():
            return str(sibling)
    return sys.executable


# --------------------------------------------------------------------------- reading
def _options_from_source(text: str) -> dict[str, EngineOption]:
    """Options declared by ``add_argument`` calls, read from the syntax tree."""
    options: dict[str, EngineOption] = {}
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return options
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        names = [arg.value for arg in node.args
                 if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                 and arg.value.startswith("--")]
        if not names:
            continue
        required, takes_value = False, True
        for keyword in node.keywords:
            if keyword.arg == "required" and isinstance(keyword.value, ast.Constant):
                required = bool(keyword.value.value)
            if (keyword.arg == "action" and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value in ("store_true", "store_false", "count")):
                takes_value = False
        for name in names:
            options[name] = EngineOption(name, required, takes_value)
    return options


def _options_from_help(script: Path, miso_dir: Path, timeout: float) -> dict[str, EngineOption]:
    """Fallback: ask the engine itself (``--help`` runs nothing else)."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(miso_dir)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        done = subprocess.run([console_python(), "-X", "utf8", str(script), "--help"],
                              cwd=str(miso_dir), env=env, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return {}
    names = re.findall(r"(?<![\w-])(--[A-Za-z][\w-]*)", done.stdout + done.stderr)
    return {name: EngineOption(name) for name in dict.fromkeys(names) if name != "--help"}


def _engine_sources(miso_dir: Path):
    budget = MAX_SOURCE_BYTES
    candidates = [miso_dir / "module_B.py", miso_dir / "pipeline.py"]
    src = miso_dir / "src"
    if src.is_dir():
        candidates += sorted(p for p in src.rglob("*.py") if "__pycache__" not in p.parts)
    for path in candidates:
        try:
            if path.stat().st_size > budget:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        budget -= len(text)
        yield text


def inspect_engine(miso_dir, use_help_fallback: bool = True, help_timeout: float = 30.0) -> EngineInfo:
    """Read the installed engine. Never raises: problems become an ``assumed`` result."""
    miso_dir = Path(miso_dir)
    script = miso_dir / "module_B.py"
    info = EngineInfo(script=script)
    try:
        main_text = script.read_text(encoding="utf-8", errors="replace")
        info.found = True
    except OSError:
        info.options = {name: EngineOption(name) for name in LEGACY_OPTIONS}
        info.how = "assumed"
        return info
    info.options = _options_from_source(main_text)
    if not info.options and use_help_fallback:
        info.options = _options_from_help(script, miso_dir, help_timeout)
        info.how = "help"
    if not info.options:
        info.options = {name: EngineOption(name) for name in LEGACY_OPTIONS}
        info.how = "assumed"
    combined = main_text + "\n" + "\n".join(_engine_sources(miso_dir))
    info.env_vars = {name for name in GUI_ENV_VARS if name in combined}
    info.yaml_keys = {key for key in YAML_KEYS
                      if re.search(rf"""["']{re.escape(key)}["']""", combined)}
    return info


# --------------------------------------------------------------------------- using it
def build_arguments(info: EngineInfo, input_file: str, iterations: int, n_polymers: int,
                    phase1_kicks: bool = True, debug_checkpoints: bool = False):
    """(argument list after the script path, notes). Options the engine lacks are skipped.

    Raises ``EngineError`` when the engine cannot be started from this GUI at all.
    """
    if not info.found:
        raise EngineError(f"MISO engine not found: {info.script}")
    if not info.supports("--input_file"):
        raise EngineError("This MISO version has no --input_file option, so the GUI cannot "
                          "pass the configuration to it. The GUI needs an update for it.")
    missing = info.required_missing()
    if missing:
        raise EngineError("This MISO version requires options the GUI does not provide: "
                          + ", ".join(missing) + ". The GUI needs an update for it.")
    args = ["--input_file", str(input_file)]
    notes = []
    for flag, value in (("--iterations", iterations), ("--n_polymers", n_polymers)):
        if info.supports(flag):
            args += [flag, str(value)]
        else:
            notes.append(f"This MISO version has no {flag} option; its own default is used.")
    for flag, wanted in (("--phase1_kicks", phase1_kicks),
                         ("--debug_checkpoints", debug_checkpoints)):
        if not wanted:
            continue
        if info.supports(flag):
            args.append(flag)
        else:
            notes.append(f"This MISO version has no {flag} option; skipped.")
    return args, notes


def compatibility_notes(info: EngineInfo, config: dict | None = None,
                        compression_steps_set: bool = True, gravity_set: bool = True) -> list[str]:
    """Plain warnings about things the engine may silently ignore."""
    notes = []
    if info.how == "assumed" and info.found:
        notes.append("The options of this MISO version could not be read; the options of the "
                     "previous version are assumed.")
    for var, label, wanted in (("DEFAULT_COMPRESSION_STEPS", "Compression steps", compression_steps_set),
                               ("DEFAULT_GRAVITY", "Gravity", gravity_set)):
        if wanted and info.found and var not in info.env_vars:
            notes.append(f"{label} from this window may be ignored: this MISO version no "
                         f"longer reads {var}.")
    for key in sorted(config or {}):
        if key in YAML_KEYS and key not in info.yaml_keys and info.found:
            notes.append(f"The YAML has '{key}' ({YAML_KEYS[key]}), but this MISO version's "
                         "source never mentions it, so it may be ignored.")
    extra = info.extra_options()
    if extra:
        notes.append("New MISO options not used by this window (MISO defaults apply): "
                     + ", ".join(extra) + ".")
    return notes


class EngineError(Exception):
    """The installed engine cannot be driven by this GUI."""
