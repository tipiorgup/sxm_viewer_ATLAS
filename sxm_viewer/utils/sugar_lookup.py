"""Parse sugar descriptions and resolve stereochemical SMILES using PubChem."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class SugarLookupError(ValueError):
    """A sugar description or PubChem response could not be resolved."""


class SugarConnectionError(SugarLookupError):
    """The PubChem endpoint could not be reached."""


@dataclass(frozen=True)
class SugarRequest:
    name: str
    query: str
    ring: str
    anomer: str


@dataclass(frozen=True)
class SugarResult:
    smiles: str
    cid: int
    iupac_name: str


_PREFIX = re.compile(
    r"^(half[- _]chair|chair|boat|twist|envelope|half|4C1|1C4|alpha|beta|D|L)"
    r"(?:[\s-]+|$)", re.IGNORECASE)


def parse_sugar_description(text: str, ring: str = "Any",
                            anomer: str = "Any") -> SugarRequest:
    """Accept names with optional leading configuration, anomer and pucker."""
    base = text.strip()
    config = None
    typed_ring = None
    typed_anomer = None
    chair = None
    while match := _PREFIX.match(base):
        token = match.group(1).lower()
        base = base[match.end():].strip()
        if token in ("d", "l"):
            if config is not None and config != token.upper():
                raise SugarLookupError("Both D and L configurations were requested.")
            config = token.upper()
        elif token in ("alpha", "beta"):
            if typed_anomer is not None and typed_anomer != token:
                raise SugarLookupError("Both alpha and beta anomers were requested.")
            typed_anomer = token
        elif token in ("4c1", "1c4"):
            value = "4C1" if token == "4c1" else "1C4"
            if chair is not None and chair != value:
                raise SugarLookupError("Both 4C1 and 1C4 chairs were requested.")
            chair = value
        else:
            value = "half" if token.startswith("half") else token
            if typed_ring is not None and typed_ring != value:
                raise SugarLookupError("Conflicting ring types were requested.")
            typed_ring = value
    if not base:
        raise SugarLookupError("Enter a sugar name after the conformation details.")
    if chair:
        if typed_ring not in (None, "chair"):
            raise SugarLookupError("4C1 and 1C4 can only be used with a chair.")
        typed_ring = f"chair_{chair}"
    if typed_ring:
        if ring != "Any" and ring != typed_ring:
            if ring == "chair" and typed_ring.startswith("chair_"):
                ring = typed_ring
            elif typed_ring == "chair" and ring.startswith("chair_"):
                pass
            else:
                raise SugarLookupError("The description conflicts with the Ring type selection.")
        else:
            ring = typed_ring
    if typed_anomer:
        if anomer not in ("Any", typed_anomer):
            raise SugarLookupError("The description conflicts with the Anomer selection.")
        anomer = typed_anomer
    parts = [part for part in (None if anomer == "Any" else anomer, config, base) if part]
    return SugarRequest(base, "-".join(parts), ring, anomer)


def lookup_sugar(query: str) -> SugarResult:
    """A lookup doubles as the connectivity check; manual SMILES never needs it."""
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           f"{quote(query, safe='')}/property/SMILES,IUPACName/JSON")
    request = Request(url, headers={"User-Agent": "SXMViewer-SugarLookup/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise SugarLookupError(
                f"PubChem did not find '{query}'. Try a more specific sugar name "
                "or enter SMILES manually.") from exc
        raise SugarLookupError(
            f"PubChem returned HTTP {exc.code}. Try again or enter SMILES manually.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise SugarConnectionError(
            "No internet connection has been found, or PubChem cannot be reached. "
            "SMILES and sugar details need to be given manually.\n\n"
            f"Connection details: {exc}") from exc
    except (ValueError, UnicodeError) as exc:
        raise SugarLookupError("PubChem returned an invalid response. Enter SMILES manually.") from exc
    try:
        record = payload["PropertyTable"]["Properties"][0]
        smiles = record.get("SMILES") or record.get("IsomericSMILES")
        cid = record["CID"]
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError("Missing stereochemical SMILES")
        if not isinstance(cid, int) or isinstance(cid, bool):
            raise ValueError("Invalid CID")
        return SugarResult(smiles.strip(), cid, str(record.get("IUPACName", "")))
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        raise SugarLookupError(
            "PubChem did not return usable SMILES. Enter SMILES manually.") from exc
