"""Fetch real species-level data from the IUCN Red List API (v4) for the species used to train the model.

Token: just run the script and paste your token when asked (it is hidden and not saved). Or set it beforehand:
  Windows PowerShell:  $env:IUCN_TOKEN="your-token"      Mac / Linux:  export IUCN_TOKEN="your-token"
Never write the token in a file that goes to GitHub.
Trial run:  python fetch_iucn_species.py --limit 3
Full run:   python fetch_iucn_species.py     (safe to stop and re-run: finished species are skipped)
Output: data/iucn_species.jsonl  (one line per species)

Features come from the assessment published up to 2021 (the one behind each species' 2021 status), so the 2022
change is not leaked into the model. The latest assessment is also saved, for the threats and conservation actions shown on the site."""
import argparse, getpass, json, os, sys, time
from pathlib import Path

import requests

from core.data_loader import read_any, parse_status_changes

BASE = "https://api.iucnredlist.org/api/v4"
ROOT = Path(__file__).parent
OUT = ROOT / "data" / "iucn_species.jsonl"
CUTOFF_YEAR = 2021


class Api:
    def __init__(self, token, pause):
        self.token, self.pause, self.s, self.mode = token, pause, requests.Session(), "raw"

    def get(self, path, **params):
        for attempt in range(6):
            auth = self.token if self.mode == "raw" else f"Bearer {self.token}"
            r = self.s.get(BASE + path, params=params, headers={"Authorization": auth, "Accept": "application/json"}, timeout=30)
            time.sleep(self.pause)
            if r.status_code == 401 and self.mode == "raw":
                self.mode = "bearer"  # some setups need the "Bearer " prefix
                continue
            if r.status_code == 401:
                sys.exit("The IUCN API rejected the token (401). Check IUCN_TOKEN.")
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Too many retries for {path}")


def is_global(a):
    sc = a.get("scopes") or []
    return not sc or any(str(x.get("code")) == "1" or "global" in json.dumps(x.get("description", "")).lower() for x in sc)


def pick(assessments, year_max=None):
    c = [a for a in assessments if is_global(a) and a.get("year_published") is not None]
    if year_max is not None:
        c = [a for a in c if int(a["year_published"]) <= year_max]
    return max(c, key=lambda a: int(a["year_published"]), default=None)


def lab(x):
    if isinstance(x, dict):
        d = x.get("description", x.get("en"))
        return d.get("en") if isinstance(d, dict) else (d or x.get("code"))
    return x


def summarise(det, year):
    d = det or {}
    return {
        "assessment_id": d.get("assessment_id"), "year": year,
        "category": (d.get("red_list_category") or {}).get("code"),
        "population_trend": lab(d.get("population_trend")),
        "threats": [{"code": t.get("code"), "name": lab(t), "timing": lab(t.get("timing")), "scope": lab(t.get("scope")), "severity": lab(t.get("severity"))} for t in d.get("threats") or []],
        "habitats": [{"code": h.get("code"), "name": lab(h)} for h in d.get("habitats") or []],
        "conservation_actions": [{"code": a.get("code"), "name": lab(a)} for a in d.get("conservation_actions") or []],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only fetch this many species (for a trial run)")
    ap.add_argument("--pause", type=float, default=0.4, help="seconds between requests")
    a = ap.parse_args()
    token = os.environ.get("IUCN_TOKEN", "").strip() or getpass.getpass("Paste your IUCN API token (hidden, not saved): ").strip()
    if not token:
        sys.exit("No token given. Paste it when asked, or set the IUCN_TOKEN environment variable.")
    f = next((ROOT / "data").glob("Markov*.xlsx"), None)
    if f is None:
        sys.exit("Put the Excel file (Markov*.xlsx) in the data/ folder.")
    species, _ = parse_status_changes(read_any(f.read_bytes(), f.name))
    names = list(dict.fromkeys(species["Scientific name"].astype(str).str.strip()))
    done = set()
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["scientific_name"])
            except Exception:
                pass
    todo = [n for n in names if n not in done][: a.limit]
    api = Api(token, a.pause)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a", encoding="utf-8") as fh:
        for i, n in enumerate(todo, 1):
            parts = n.split()
            params = {"genus_name": parts[0], "species_name": parts[1] if len(parts) > 1 else ""}
            if len(parts) > 2:
                params["infra_name"] = parts[-1]
            res = api.get("/taxa/scientific_name", **params)
            rec = {"scientific_name": n, "found": bool(res)}
            if res:
                ass = res.get("assessments") or []
                pre, latest = pick(ass, CUTOFF_YEAR), pick(ass)
                rec["pre_year"] = pre and pre.get("year_published")
                rec["latest_year"] = latest and latest.get("year_published")
                if pre and pre.get("assessment_id"):
                    rec["pre"] = summarise(api.get(f"/assessment/{pre['assessment_id']}"), rec["pre_year"])
                if latest and latest.get("assessment_id"):
                    same = pre and latest["assessment_id"] == pre.get("assessment_id")
                    rec["latest"] = rec.get("pre") if same else summarise(api.get(f"/assessment/{latest['assessment_id']}"), rec["latest_year"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {n}: " + ("ok" if rec.get("pre") or rec.get("latest") else "no data"))


if __name__ == "__main__":
    main()
