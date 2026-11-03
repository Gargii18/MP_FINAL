"""Train the project's Random Forest and export it for the website.
Run from the biodiversity_app folder:  python export_model.py
Uses data/Markov*.xlsx if present, otherwise synthetic DEMO data (flagged on the site)."""
import json
import pandas as pd
from pathlib import Path
from core import classifier as clf
from core import trait_model as tm
from core.constants import STATES
from core.data_loader import find_file, read_any, parse_status_changes
from core.demo_data import demo_status_changes

ROOT = Path(__file__).parent
# Curated well-known species: (scientific, common, group, status). Verify against iucnredlist.org before relying on them.
CURATED = [
 ("Ailuropoda melanoleuca","Giant Panda","MAMMALIA","VU"),("Panthera tigris","Tiger","MAMMALIA","EN"),
 ("Panthera uncia","Snow Leopard","MAMMALIA","VU"),("Panthera leo","Lion","MAMMALIA","VU"),
 ("Acinonyx jubatus","Cheetah","MAMMALIA","VU"),("Elephas maximus","Asian Elephant","MAMMALIA","EN"),
 ("Loxodonta africana","African Savanna Elephant","MAMMALIA","EN"),("Loxodonta cyclotis","African Forest Elephant","MAMMALIA","CR"),
 ("Balaenoptera musculus","Blue Whale","MAMMALIA","EN"),("Diceros bicornis","Black Rhinoceros","MAMMALIA","CR"),
 ("Rhinoceros unicornis","Indian Rhinoceros","MAMMALIA","VU"),("Pongo abelii","Sumatran Orangutan","MAMMALIA","CR"),
 ("Ursus maritimus","Polar Bear","MAMMALIA","VU"),("Vulpes vulpes","Red Fox","MAMMALIA","LC"),
 ("Canis lupus","Grey Wolf","MAMMALIA","LC"),("Giraffa camelopardalis","Giraffe","MAMMALIA","VU"),
 ("Pan troglodytes","Chimpanzee","MAMMALIA","EN"),("Phocoena sinus","Vaquita","MAMMALIA","CR"),
 ("Haliaeetus leucocephalus","Bald Eagle","AVES","LC"),("Falco peregrinus","Peregrine Falcon","AVES","LC"),
 ("Pavo cristatus","Indian Peafowl","AVES","LC"),("Ardeotis nigriceps","Great Indian Bustard","AVES","CR"),
 ("Strigops habroptilus","Kakapo","AVES","CR"),("Raphus cucullatus","Dodo","AVES","EX"),
 ("Gavialis gangeticus","Gharial","REPTILIA","CR"),("Eretmochelys imbricata","Hawksbill Sea Turtle","REPTILIA","CR"),
 ("Chelonia mydas","Green Sea Turtle","REPTILIA","EN"),("Dermochelys coriacea","Leatherback Sea Turtle","REPTILIA","VU"),
 ("Varanus komodoensis","Komodo Dragon","REPTILIA","EN"),("Incilius periglenes","Golden Toad","AMPHIBIA","EX"),
]
THREATS = {
 "Panthera tigris":["Poaching for skins and bones","Habitat loss and fragmentation","Loss of prey species","Human-wildlife conflict"],
 "Diceros bicornis":["Poaching for horn","Illegal wildlife trade","Habitat loss"],
 "Loxodonta cyclotis":["Poaching for ivory","Forest loss and fragmentation","Bushmeat hunting"],
 "Loxodonta africana":["Poaching for ivory","Habitat loss","Human-elephant conflict"],
 "Elephas maximus":["Habitat loss and fragmentation","Human-elephant conflict","Poaching"],
 "Pongo abelii":["Deforestation (palm oil, logging)","Illegal pet trade","Hunting and conflict"],
 "Pan troglodytes":["Habitat loss","Hunting for bushmeat","Disease"],
 "Balaenoptera musculus":["Ship strikes","Entanglement in fishing gear","Ocean noise","Historic whaling"],
 "Phocoena sinus":["Bycatch in gillnets (illegal totoaba fishing)","Extremely small population"],
 "Ardeotis nigriceps":["Collisions with power lines","Grassland habitat loss","Hunting and disturbance"],
 "Strigops habroptilus":["Introduced predators (cats, stoats, rats)","Very small population","Low genetic diversity"],
 "Gavialis gangeticus":["River damming and sand mining","Fishing-net entanglement","Habitat loss"],
 "Eretmochelys imbricata":["Illegal shell trade","Egg harvesting","Loss of nesting beaches and coral reefs","Fisheries bycatch"],
 "Chelonia mydas":["Fisheries bycatch","Egg and turtle harvesting","Nesting-habitat loss","Climate change"],
 "Varanus komodoensis":["Climate change and sea-level rise","Habitat loss","Loss of prey"],
}
MEASURES = {
 "Panthera tigris":["Strengthen anti-poaching patrols and wildlife-crime enforcement","Protect and connect forest corridors","Restore prey populations","Support community programmes that reduce conflict"],
 "Diceros bicornis":["Ranger protection and intelligence-led anti-poaching","Move animals to secure reserves to grow populations","Reduce demand for horn and close illegal markets"],
 "Loxodonta cyclotis":["Enforce ivory-trade bans and stop poaching","Protect intact forest and corridors","Reduce bushmeat hunting through alternative livelihoods"],
 "Loxodonta africana":["Anti-poaching and ivory-trade enforcement","Secure migration corridors","Fund conflict mitigation such as beehive fences"],
 "Elephas maximus":["Protect and connect forest habitat","Early-warning systems and barriers to reduce crop-raiding conflict","Anti-poaching enforcement"],
 "Pongo abelii":["Stop forest clearing and promote deforestation-free palm oil","Enforce pet-trade laws; rescue and rehabilitate orangutans","Restore forest corridors"],
 "Pan troglodytes":["Protect forest habitat and corridors","Enforce hunting bans and promote alternative protein sources","Disease monitoring and veterinary care in protected areas"],
 "Balaenoptera musculus":["Shipping-lane changes and speed limits to reduce ship strikes","Whale-safe fishing gear","Reduce ocean noise","Enforce the whaling ban"],
 "Phocoena sinus":["Enforce the gillnet ban and remove ghost nets","Promote vaquita-safe fishing gear","Combat illegal totoaba trade"],
 "Ardeotis nigriceps":["Bury or mark power lines in key habitat","Protect grasslands and run captive breeding","Reduce hunting and disturbance"],
 "Strigops habroptilus":["Predator-free island sanctuaries","Intensive nest protection and supplementary feeding","Genetic management and assisted breeding"],
 "Gavialis gangeticus":["Protect nesting sandbanks and river flows","Regulate sand mining and fishing nets","Captive-rearing and release programmes"],
 "Eretmochelys imbricata":["Ban shell trade and enforce CITES","Protect nesting beaches and coral reefs","Turtle-excluder devices and bycatch reduction"],
 "Chelonia mydas":["Protect nesting beaches and stop egg harvesting","Reduce bycatch with turtle-excluder devices","Reduce plastic pollution"],
 "Varanus komodoensis":["Protect habitat and prey","Manage tourism and settlement","Monitor climate and sea-level impacts and plan refuges"],
}
CAT = {"least concern": "LC", "near threatened": "NT", "vulnerable": "VU", "endangered": "EN", "critically endangered": "CR",
       "extinct": "EX", "extinct in the wild": "EW", "data deficient": "DD",
       "lower risk/least concern": "LC", "lower risk/near threatened": "NT"}


def _col(df, *names):
    low = {str(c).strip().lower(): c for c in df.columns}
    return next((low[n.lower()] for n in names if n.lower() in low), None)


def load_full(folder):
    """Optional: full IUCN Red List species list from the IUCN search download (CSV files placed anywhere under data/).
    Needs simple_summary.csv + taxonomy.csv; common_names.csv is optional. Returns [[sci, common, group, status, ""], ...] or None."""
    sm, tx, cn = (next(folder.rglob(p), None) for p in ("*simple_summary*.csv", "*taxonomy*.csv", "*common*name*.csv"))
    if not (sm and tx):
        return None
    S = pd.read_csv(sm, low_memory=False, encoding_errors="replace")
    T = pd.read_csv(tx, low_memory=False, encoding_errors="replace")
    sid, ssci, scat = _col(S, "internalTaxonId"), _col(S, "scientificName"), _col(S, "redlistCategory")
    tid, tcls = _col(T, "internalTaxonId"), _col(T, "className")
    if not (sid and ssci and scat and tid and tcls):
        raise ValueError("IUCN download: need columns internalTaxonId, scientificName, redlistCategory (simple_summary.csv) "
                         f"and internalTaxonId, className (taxonomy.csv). Found {list(S.columns)[:8]} / {list(T.columns)[:8]}")
    df = S[[sid, ssci, scat]].merge(T[[tid, tcls]].drop_duplicates(tid), left_on=sid, right_on=tid, how="left")
    df["st"] = df[scat].astype(str).str.strip().str.lower().map(CAT)
    df = df[df["st"].notna() & df[ssci].notna()].drop_duplicates(ssci)
    names = {}
    if cn:
        C = pd.read_csv(cn, low_memory=False, encoding_errors="replace")
        cid, cnm, cmain, clang = _col(C, "internalTaxonId"), _col(C, "name"), _col(C, "main"), _col(C, "language")
        if cid and cnm:
            if clang is not None and C[clang].astype(str).str.lower().isin(["eng", "english"]).any():
                C = C[C[clang].astype(str).str.lower().isin(["eng", "english"])]
            if cmain is not None:
                C = C.sort_values(cmain, ascending=False)
            names = C.drop_duplicates(cid).set_index(cid)[cnm].astype(str).to_dict()
    return [[str(r[ssci]).strip(), names.get(r[sid], ""), str(r[tcls]).strip().upper() if r[tcls] == r[tcls] else "UNKNOWN", r["st"], ""]
            for _, r in df.iterrows()]


f = find_file(ROOT / "data", ["Markov*.xlsx", "Markov*.xls"])
if f:
    species, _ = parse_status_changes(read_any(f.read_bytes(), f.name)); demo = False
else:
    species = demo_status_changes(); demo = True
b = clf.train_random_forest(species)
model = {}
for g in list(b["taxonomies"]) + ["__unseen__"]:
    model[g] = {s: {r["IUCN status"]: round(float(r["Probability"]), 6) for _, r in clf.predict_status(b, g, s)["table"].iterrows()} for s in STATES}
db = {} if demo else {r["Scientific name"]: [r["Scientific name"], str(r["Common name"]) if r["Common name"] == r["Common name"] else "", r["Taxonomy"], r["Status_2021"], r["Status_2022"]] for _, r in species.iterrows()}
if demo:  # with real data the site lists exactly the species the model was trained on
    for sci, com, grp, st in CURATED:
        db.setdefault(sci, [sci, com, grp, st, ""])
full = load_full(ROOT / "data")
if full:  # official current status replaces the 2021 sheet and the hand-entered statuses
    db = {e[0]: e for e in full}
    print("Full IUCN list:", len(db), "species;", sum(e[2] in b["taxonomies"] for e in full), "in a group the model was trained on")
# Mammal trait model (optional): species_master_with_iucn_details.csv adds ~4,600 mammals with trait-based estimates and real IUCN details
TRAIT = {}   # scientific name -> probabilities (only for species that are not already covered by the status-change model)
INFO_CSV = {}
report = {}
mf = next((ROOT / "data").glob("species_master*.csv"), None)
if mf:
    raw_traits = tm.load(mf)
    table, report["trait_model"] = tm.train(raw_traits)
    det = raw_traits.set_index("sci_name")
    for _, r in table.iterrows():
        sci, st = r["sci"], r["status"]
        if not isinstance(st, str):
            continue  # no current IUCN category available
        x = det.loc[sci]
        th, ms = tm.threats_and_measures(x["iucn_threats"])
        yr = str(x["iucn_assessment_date"])[:4]
        INFO_CSV[sci] = {"t": x["iucn_population_trend"] if isinstance(x["iucn_population_trend"], str) else "", "y": yr if yr.isdigit() else "",
                         "h": tm.habitats(x["iucn_habitats"]), "th": th, "ca": ms}
        if sci not in db:
            db[sci] = [sci, "", "MAMMALS (Mammalia)", st, ""]
            if st in tm.CLASSES or st == "DD":
                TRAIT[sci] = {"p": r["p"]}
    print("Mammal trait model:", len(INFO_CSV), "species from the CSV,", len(TRAIT), "with trait-based estimates; CV accuracy", report["trait_model"]["cv_adjusted"]["accuracy"])

INFO = {}  # real species-level data from fetch_iucn_species.py (optional): population trend, habitats, threats, conservation actions
jl = ROOT / "data" / "iucn_species.jsonl"
if jl.exists():
    uniq = lambda xs, n: list(dict.fromkeys(x for x in xs if x))[:n]
    for line in jl.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        a = r.get("latest") or r.get("pre")
        if a:
            INFO[r["scientific_name"]] = {"t": a.get("population_trend") or "", "y": a.get("year") or "",
                "h": uniq([h.get("name") for h in a.get("habitats") or []], 6),
                "th": uniq([t.get("name") for t in a.get("threats") or []], 8),
                "ca": uniq([c.get("name") for c in a.get("conservation_actions") or []], 8)}
    print("IUCN species-level data found for", sum(k in INFO for k in db), "of", len(db), "species")
for k, v in INFO_CSV.items():
    INFO.setdefault(k, v)
for v in db.values():
    i = INFO.get(v[0], {})
    v.append(i.get("th") or THREATS.get(v[0], []))
    v.append(i.get("ca") or MEASURES.get(v[0], []))
    v.append({"t": i["t"], "h": i["h"], "y": i["y"]} if i else None)
    if v[0] in TRAIT:
        v.append(TRAIT[v[0]])
Path(ROOT / "model_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
out = {"demo": demo, "groups": b["taxonomies"], "model": model, "species": sorted(db.values(), key=lambda x: x[0])}
Path(ROOT / "docs" / "data.js").write_text(
    "var BIO = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\nif (typeof module !== 'undefined') module.exports = BIO;\n", encoding="utf-8")
print("Exported", "DEMO" if demo else "REAL", "model:", len(out["species"]), "species,", len(out["groups"]), "groups")
