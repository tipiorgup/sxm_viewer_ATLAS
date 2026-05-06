# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**SXM Viewer** is a PyQt5 desktop GUI for visualizing and analyzing SPM (Scanning Probe Microscopy) data — Anfatec/Omicron `.sxm` files and Nanonis formats. This branch (**ATLAS**) adds in-house tools that bridge the viewer with the ATLAS polymer-conformation analysis pipeline.

## Running the app

```powershell
# From repo root (activate your conda env first)
python -m sxm_viewer
```

Requirements (Python 3.11, conda env recommended):
```powershell
pip install -r scripts/requirements.txt
```

There are no automated tests or linting configs in this repo.

## Architecture

### Entry flow
`__main__.py` → `cli.py::main()` → creates `QApplication` → instantiates `SXMGridViewer` → `show()`

### `SXMGridViewer` (god class in `gui/main_window.py`)
The central widget. It is very large (~15k+ lines split across several mixin files). All feature logic ultimately connects back here. Key references it holds:
- `self.files` — list of loaded file paths
- `self.headers` / `self.header_cache` — parsed SXM metadata dicts
- `self.preview_canvas` — the active matplotlib `FigureCanvasQTAgg` for the detail view
- `self._channel_data_cache` / `self._filtered_channel_cache` — LRU-capped numpy array caches
- `self.matrix_datasets` — loaded Matrix spectroscopy datasets

`ViewerState` (`gui/viewer/state.py`) is a thin dataclass that bundles all these caches so helper modules can receive them without importing the whole viewer.

### Main window is split across files
The `SXMGridViewer` class is assembled from multiple files acting as logical sections:
| File | Content |
|---|---|
| `gui/main_window.py` | Core class, event handlers, most business logic |
| `gui/main_window_layout.py` | `create_lower_controls()` — mode buttons, toolbar widgets |
| `gui/main_window_toolbar.py` | Top toolbar, molecule SVG icon rendering |
| `gui/main_window_spectro.py` | Spectroscopy mode integration |

### Three viewer modes
Toggled via `Ctrl+B` / `Ctrl+M` / `Ctrl+Alt+S` or the toolbar buttons:
- **Browse** — thumbnail grid + detail preview
- **Measure** — profile lines, angle tools, crop
- **Spectro** — spectroscopy curve browser and overlays

### Canvas system (`gui/canvases/`)
`ExperimentalCanvasWindow` is a separate floating Qt window for free-form figure composition. It contains a `CanvasGraphicsView` with drag-and-drop `CanvasImageItem` objects. State is managed via `canvas_state.py` (undo/redo snapshots). This is distinct from the main `preview_canvas`.

### Data pipeline
1. `data/io.py` — parses Omicron/Anfatec `.txt` headers (`parse_header`) and reads binary/ASCII channel files into numpy arrays (`read_channel_file`)
2. `providers/nanonis/` — wraps nanonispy2 for `.sxm` Nanonis format
3. `data/spectroscopy.py` — spectroscopy curve parsing
4. `data/matrix.py` — Matrix format spectroscopy datasets
5. `processing/filters.py` — image filters (flatten, plane subtract, gaussian, highpass, laplacian)
6. `processing/detection.py` — auto-detects topography channel from loaded data

### Controllers pattern
Heavy UI interactions are extracted from the god class into controller objects stored on `self`:
- `self.collection_controller` (`controllers/collection.py`) — collection tray management
- `self.session_controller` (`controllers/session.py`) — save/restore sessions
- `self.quick_crop_controller` (`controllers/quick_crop.py`)
- `self.spectro_compare_controller` (`controllers/spectro_compare.py`)
- `self.thumbnail_controller` (`controllers/thumbnail_controller.py`)

### Dialogs (`gui/dialogs/`)
All are `QDialog` subclasses launched from menu actions. They receive `viewer` as parent and access `viewer.preview_canvas` directly for canvas interaction.

## Key conventions

- All Qt imports come from `sxm_viewer._shared` (re-exports `QtCore`, `QtWidgets`, `QtGui`, `np`, `sys`) — don't import PyQt5 directly in new files unless adding to `_shared`.
- New dialogs go in `gui/dialogs/`, new viewer-level logic goes in `gui/viewer/` or a new controller in `gui/controllers/`.
- Cache dicts on the viewer are size-capped — see `CHANNEL_DATA_CACHE_LIMIT` and `FILTERED_CACHE_LIMIT` in `config_defaults.py`.
- Matplotlib canvases use `draw_idle()` not `draw()` for redraws to avoid blocking the Qt event loop.

## Communication between Claude Code and user

- Promote technical/ scientific questions before executing or writing anycode
- Keep language short and consice 

## Target of project

- Implement ATLAS code into the GUI, first by getting coordiantes from **Tools → Position coordinates**. 

- Then upload input file, csv and npz files and run ATLAS engine in different Tool executor

### ATLAS extension (`gui/dialogs/position_coordinates_dialogs.py`)
`PositionCoordinatesDialog` — the core ATLAS tool. Opened from **Tools → Position coordinates**. Connects to `viewer.preview_canvas` via matplotlib's `mpl_connect("button_press_event")` to pick XY positions. Exports:
- `<stem>_positions.csv` — columns: Point, Original_X, Original_Y, X (Angstrom), Y (Angstrom), Height, Z (Angstrom)
- `<stem>_positions.npz` — numpy arrays `x`, `y`, `z` (full STM grid in Angstrom)

Unit conversion from display units to Angstrom is done inline with a hardcoded dict (`pm`×0.01, `nm`×10, `m`×1e10).




