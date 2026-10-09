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
4. Click **Export CSV**. Alongside the existing CSV, positions, orientations,
   geometry pickle and image, this generates `<stem>.yml`. **Run MISO** opens
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

## Guided YAML assistant: offline or optional hosted LLM

Click **Guide me / optional LLM assistant** after building sugars. The default
**Offline guide** asks about the root, orientation mode, and each chemical
connection in separate steps. Review the summary, click **Approve and apply**,
then **Export CSV** to generate the YAML. It needs no internet, account, model
download, or additional software. Closing without approval leaves settings unchanged.

Both tabs show a live skeletal **connection plan** next to the questions. Each
sugar label appears at its placed X/Y position, matching the STM image. Arrows
run from donor to acceptor and show the carbons and α/β. The root is gold, and
unconnected units are grey. Online proposals are drawn there only after local
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
