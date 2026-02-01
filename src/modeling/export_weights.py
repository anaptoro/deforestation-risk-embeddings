from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import quote


BASE_GEE_EDITOR_URL = "https://code.earthengine.google.com/#"


def save_logit_weights(w: Iterable[float], b: Optional[float], out_path: str | Path) -> None:
    """Save logistic regression weights for Earth Engine usage.

    Output JSON format:
      {"w": [...], "b": ...}
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {"w": [float(x) for x in w]}
    if b is not None:
        payload["b"] = float(b)

    out_path.write_text(json.dumps(payload, indent=2) + "\n")


def load_logit_weights(path: str | Path) -> Tuple[list[float], Optional[float]]:
    """Load {"w": [...], "b": ...} from JSON."""
    path = Path(path)
    obj = json.loads(path.read_text())

    if "w" not in obj:
        raise ValueError(f"Missing key 'w' in weights file: {path}")

    w = [float(x) for x in obj["w"]]
    b = float(obj["b"]) if "b" in obj and obj["b"] is not None else None
    return w, b


def weights_csv(w: Iterable[float]) -> str:
    """Return comma-separated weights suitable for URL fragment param w=..."""
    return ",".join(f"{float(x):.10g}" for x in w)


def gee_params_string(w: Iterable[float], b: Optional[float] = None) -> str:
    """Return 'b=...;w=...' string ready to paste into GEE URL fragment params."""
    w_str = weights_csv(w)
    if b is None:
        return f"w={w_str}"
    return f"b={float(b):.10g};w={w_str}"


def _fmt(v: Any) -> str:
    """Format a value for our URL fragment (NOT query string)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return f"{v}"
    return str(v)


def _add(parts: list[str], key: str, val: Any) -> None:
    """Append key=val to parts if val is not None/empty."""
    if val is None:
        return
    s = _fmt(val)
    if s == "":
        return
    # We encode only the value portion (title/tag can include spaces).
    parts.append(f"{key}={quote(s, safe='-_.~,')}")


def gee_fragment(
    *,
    w: Iterable[float],
    b: Optional[float],
    title: str = "AEF + frontier menagerie (66D)",
    tag: str = "",
    year: int = 2022,
    lat: float = -9.5,
    lon: float = -62.5,
    zoom: int = 9,
    lo: float = -10,
    hi: float = 10,
    roadKm: float = 100,
    nfMaxKm: float = 30,
    s2m1: int = 7,
    s2m2: int = 9,
    s2cloud: float = 60,
    s2Years: Optional[str] = None,  # e.g. "2020,2021,2022,2023"
) -> str:
    """
    Build the full Earth Engine Code Editor fragment expected by your menagerie_loader.js.

    Returns something like:
      title=...;tag=...;year=2022;lat=-9.5;lon=-62.5;zoom=9;lo=-10;hi=10;roadKm=100;nfMaxKm=30;
      s2m1=7;s2m2=9;s2cloud=60;s2Years=2020,2021,2022,2023;b=...;w=...
    """
    parts: list[str] = []
    _add(parts, "title", title)
    if tag:
        _add(parts, "tag", tag)

    _add(parts, "year", year)
    _add(parts, "lat", lat)
    _add(parts, "lon", lon)
    _add(parts, "zoom", zoom)

    _add(parts, "lo", lo)
    _add(parts, "hi", hi)

    _add(parts, "roadKm", roadKm)
    _add(parts, "nfMaxKm", nfMaxKm)

    _add(parts, "s2m1", s2m1)
    _add(parts, "s2m2", s2m2)
    _add(parts, "s2cloud", s2cloud)

    # If not provided, your JS defaults to year-2..year+1, so we can omit it.
    if s2Years is not None and str(s2Years).strip():
        _add(parts, "s2Years", s2Years)

    # Add weights last (huge)
    if b is not None:
        parts.append(f"b={float(b):.10g}")
    parts.append(f"w={weights_csv(w)}")

    return ";".join(parts) + ";"


def gee_code_editor_url(**kwargs: Any) -> str:
    """Full URL = base + fragment."""
    return BASE_GEE_EDITOR_URL + gee_fragment(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate GEE URL fragment / full Code Editor URL for menagerie_loader.js")
    ap.add_argument("--weights", required=True, help="Path to JSON weights file (e.g. models/logit_weights_v5.json)")

    # Menagerie params
    ap.add_argument("--title", default="AEF + frontier menagerie (66D)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--lat", type=float, default=-9.5)
    ap.add_argument("--lon", type=float, default=-62.5)
    ap.add_argument("--zoom", type=int, default=9)
    ap.add_argument("--lo", type=float, default=-10)
    ap.add_argument("--hi", type=float, default=10)
    ap.add_argument("--roadKm", type=float, default=100)
    ap.add_argument("--nfMaxKm", type=float, default=30)
    ap.add_argument("--s2m1", type=int, default=7)
    ap.add_argument("--s2m2", type=int, default=9)
    ap.add_argument("--s2cloud", type=float, default=60)
    ap.add_argument("--s2Years", default=None, help='Optional, e.g. "2020,2021,2022,2023"')

    ap.add_argument("--print", choices=["fragment", "url", "both"], default="both")
    args = ap.parse_args()

    w, b = load_logit_weights(args.weights)

    frag = gee_fragment(
        w=w,
        b=b,
        title=args.title,
        tag=args.tag,
        year=args.year,
        lat=args.lat,
        lon=args.lon,
        zoom=args.zoom,
        lo=args.lo,
        hi=args.hi,
        roadKm=args.roadKm,
        nfMaxKm=args.nfMaxKm,
        s2m1=args.s2m1,
        s2m2=args.s2m2,
        s2cloud=args.s2cloud,
        s2Years=args.s2Years,
    )
    url = BASE_GEE_EDITOR_URL + frag

    if args.print in ("fragment", "both"):
        print("\n=== GEE FRAGMENT ===")
        print(frag)

    if args.print in ("url", "both"):
        print("\n=== FULL CODE EDITOR URL ===")
        print(url)


if __name__ == "__main__":
    main()
