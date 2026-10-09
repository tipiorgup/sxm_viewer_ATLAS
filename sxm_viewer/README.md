# SXM Viewer package (internals)

This package is what `python -m sxm_viewer` loads. The old `sxm_grid_viewer.py`
remains only as a shim and delegates into this package.

## Package map
```
sxm_viewer/
  __init__.py
  config.py                 # user config, cache limits, defaults
  palettes.py               # color cycles and colormap helpers
  data/
    io.py                   # Omicron/Anfatec header + channel parsing
    spectroscopy.py         # .dat metadata, axis helpers, matrix detection
    matrix.py               # matrix dataset representations
  providers/
    nanonis/
      adapter.py            # Nanonis .sxm -> Omicron-style cache generator
      vendor/               # vendored nanonispy reader
  gui/
    main_window.py          # top-level Qt widget and app state
    main_window_layout.py   # layout helpers and shortcuts panel
    main_window_toolbar.py  # toolbar actions and dark-mode toggle
    main_window_spectro.py  # spectro dock and browser wiring
    viewer/                 # thumbnail load/render, preview, loader, measurement
    spectroscopy/           # overlays, controller, popups for spectroscopies
    dialogs/                # spectroscopy dialogs, profile dialog, exports
    canvases/               # canvas workspace window and tiles
  utils/                    # small helpers (units, logging, thumbnails)
```

## Running from source
- After installing dependencies (see top-level README), launch with:
  ```bash
  python -m sxm_viewer
  ```
- For legacy compatibility, `python sxm_grid_viewer.py` still forwards to the
  package entry point.

## Migration status
- Core image browsing, spectroscopy overlays, matrix explorer, and canvas live
  here.
- Remaining legacy utilities are isolated under `scripts/` or kept as thin
  shims; new features should target the modules above.

## Position monomer: sugar names and SMILES

In **Tools > Position monomer**, the sugar input accepts either manual SMILES or
a sugar name. Examples: `glucose`, `chair KDO`, `D chair 4C1 glucose`, and
`beta-D-glucose`. Optional leading descriptors include D/L, alpha/beta, chair,
boat, twist, envelope, half-chair, and the specific chairs 4C1/1C4.

Click **Build** to resolve names via PubChem in the background. The resolved
stereochemical SMILES and requested ring/anomer are filled into the table, then
the existing build, placement, rotation and export workflow continues. A blank
MISO **Name** is filled with the sugar name; an existing name is preserved.
Placed-unit labels use that name followed by the row and copy numbers, e.g.
`KDO.1.1`, `glucose.2.1`, and `glucose.2.2`, in the instance list, image
annotations and exports. Amino acids use their resolved residue name (e.g.
`Asn.3.1`); unnamed manual sugars keep the fallback name `Sugar1`, etc.
Conflicting typed descriptors and table selections are reported rather than
silently overridden. D/L is encoded in the retrieved SMILES, not a pucker filter.

After selecting a placed unit, click the unit list or the image and use the
arrow keys to move it in X/Y on screen (0.5 Å per press; hold Shift for 2.5 Å).
The COM X/Y fields update to match.
Specific chair selections only accept matching generated conformers; if none
are generated, the build reports a failure rather than substituting another chair.

Manual SMILES never requires a network connection. When a name lookup cannot
connect (10-second timeout per request), a small warning explains that SMILES
and sugar details must be supplied manually. Unknown names and PubChem service
errors have separate warnings. Failed lookups leave the table and existing
placed monomers unchanged. Only sugar-name queries are sent to PubChem; images,
positions and local files are not uploaded.

## Position monomer: automatic MISO YAML export

After building and positioning sugars, use **MISO YAML: sugar connections**:

1. Choose the **Root** unit explicitly (normally the reducing end).
2. Choose **Keep positioned geometry and rotations** or **Let MISO optimize
   orientations**. Both modes reuse the exact exported sugar conformers; only
   fixed mode applies the placed rotations.
3. Select donor and acceptor units by their readable labels, choose their
   hydroxyl-bearing carbon positions and the alpha/beta linkage, then click
   **Add connection**. Branches are supported. Connections are chemical choices,
   not guessed from proximity. Remove mistakes using **Remove selected connection**.
4. Click **Export CSV + MISO input YAML**. Alongside the existing CSV, positions, orientations,
   geometry pickle and image, this generates `<stem>.yml`; its full path is
   shown when export finishes. Give the YAML a final inspection (units, root,
   linkages, α/β, orientation mode) before running MISO. **Run MISO** opens
   with this configuration, companion paths and orientation mode prefilled.
   Browsing to an exported YAML also loads its paths and orientation mode.

Every sugar must belong to one connected structure. In optimization mode, every
donor must reach the root through donor-to-acceptor links. A carbon cannot be
reused in multiple connections. Invalid selections block sugar-only export
before files are written. Rebuilding clears connections and root selection.
Rows with the same Name must use identical SMILES and conformer geometry;
otherwise assign distinct Names and rebuild to avoid incorrect geometry reuse.

The YAML uses MISO's `<Name>_<position-index>` IDs (e.g. `KDO_0`) while controls
and CSV annotations retain readable labels (e.g. `KDO.1.1`). All positions are
exported in instance-list order. Companion paths are absolute so MISO can run
from a different working directory. Re-export after moving or rotating units.

This first version generates YAML for sugars only. Amino-acid or lipid exports
still save the existing CSV/image files, but explicitly warn that YAML was not
generated. Any older YAML is left unchanged and must not be reused blindly.

## Position coordinates: optional sugars per picked point

Picking points and **Export CSV** work exactly as before (CSV + NPZ + PNG).
Assigning a sugar is optional and can be done for any subset of points:

1. In the point table, type a sugar name (e.g. `D chair 4C1 glucose`,
   `chair KDO`) or a SMILES in **Sugar name / SMILES**, and optionally a
   **Name**, **Ring type** and **Anomer**. Typed entries are kept when more
   points are picked.
2. Click **Build monomers**. Names are looked up in PubChem in the background;
   without internet a pop-up asks you to enter the SMILES and details
   manually. Blank Names get automatic names (`Sugar1`, ...). Points with the
   same Name must be the same sugar.
3. Built sugars are drawn on the image at their points, labelled
   `<Name>.<point>`. Select a table row to rotate that sugar with RX/RY/RZ.
4. Use the same **MISO YAML: sugar connections** panel and **Guide me /
   optional LLM assistant** as in Position monomer, including the choice of
   keeping the rotations (fixed) or letting MISO optimize them.

On export with built sugars, the picked-points CSV is used directly as MISO's
`circle_input_path`, and `<stem>.yml`, `<stem>_monomer_data.pkl` and
`<stem>_orientations.csv` (used in fixed mode) are also written. YAML IDs use the picked point
index (e.g. `Glc_2`), so unassigned points stay in the CSV but are not used by
any sugar. If sugar entries were edited after the last build, export offers
coordinates only; rebuild to get the YAML.

## Guided YAML assistant: offline or optional hosted LLM

Click **Guide me / optional LLM assistant** after building sugars. The default
**Offline guide** asks about the root, orientation mode, and each chemical
connection in separate steps. Review the summary, click **Approve and apply**,
then **Export CSV + MISO input YAML** to generate the YAML. It needs no internet, account, model
download, or additional software. Closing without approval leaves settings unchanged.

Both tabs show a live **connection plan** next to the questions. Each sugar is
drawn top-down as positioned (heavy atoms, same X/Y and rotation as on the STM
image), with its linkable carbons labelled C1, C2, … (only carbons with a free
OH that MISO can link; e.g. glucose C5 is not offered). Arrows run from the
donor carbon to the acceptor carbon and show α/β; carbons in use are highlighted.
The root is gold, and unconnected units are grey. Online proposals are drawn there only after local
validation, and are marked as not applied until you approve them.

For optional conversational guidance, use the **Online LLM** tab. An administrator
can configure a provider once per computer:

- **OpenAI** or **Anthropic**: use the preset request URL, your provider's model
  identifier, and an API key with access to that model.
- **Institution / OpenAI-compatible**: obtain the complete HTTPS request URL
  (including `/chat/completions` and any required non-secret API-version query),
  model identifier, and API key from your institution. Choose Bearer authentication
  or the `api-key` header as specified by the administrator. Other protocols,
  browser SSO, and arbitrary OAuth flows are not supported by this first version.
- Click **Save configuration**. On Windows, **Remember API key** stores it in
  Windows Credential Manager using the existing pywin32 dependency, never in
  YAML, ordinary settings, or logs. Use **Load saved key** in later sessions.
  **Remove saved key** deletes the saved credential. Uncheck Remember to use a
  session-only key. On systems without Windows Credential Manager support, use
  session-only keys; the offline guide remains available.

Before the first **Send**, and whenever the service/model changes, a consent
prompt names the destination. Online requests contain sugar names/labels,
position indices, XYZ coordinates in Angstrom, available hydroxyl-bearing
carbons, current connection/root/mode choices, and chat history. Images, local
file paths, SMILES, and repository code are not included automatically. Do not
paste confidential material or credentials into chat. Provider retention rules
and charges apply. Only HTTPS endpoints are supported; redirects are refused
so credentials cannot be forwarded to another destination.

The model should ask about missing chemistry, not infer bonds from XYZ proximity.
Its proposals are parsed into a restricted structure and validated by the same
local MISO validator. Invalid or incomplete proposals cannot be applied. A valid
proposal is displayed with readable unit labels and still requires your explicit
approval; an LLM's plausible chemistry is not a substitute for scientific review.
All YAML writing remains local. Service failures and malformed replies are
reported explicitly, and the offline guide is always available. No real provider
request is required to use the wizard.

## Why did MISO fail? (explain-only troubleshooting)

When a MISO run ends with an error (not when you click **Stop**), the
**Why did MISO fail?** button in the MISO Runner becomes available. It opens a
window that never changes the YAML, CSV, or any setting:

- **Offline checks** (nothing is sent): missing companion files, fixed
  orientation without an orientation CSV, sugars without positions (or the
  reverse), duplicate positions, indices outside the positions CSV, an unknown
  `root_mol`, and malformed or duplicate glycosidic-bond sites.
- **Optional LLM explanation** using the same service settings as the YAML
  assistant. Click **Show exactly what will be sent** to review the data first;
  a consent prompt appears for each new service/model. Sent: the Python
  traceback and the last ~200 log lines, the YAML structure (including sugar
  SMILES), the run settings, and the offline findings. Local paths become
  `<path>/file name (found|MISSING)` and user/computer names become `<user>`.
  The model is asked to explain the step that failed, the likely cause with the
  log lines as evidence, and the related input. It does not propose fixes.

The data is offered to the model as read-only tools (`get_failure_log`,
`get_yaml_structure`, `get_run_settings`, `get_local_checks`) defined in
`utils/miso_troubleshoot.py` in MCP shape (name, description, inputSchema), so
they can later be published by an MCP server. Services without tool calling
receive the same data inline.
