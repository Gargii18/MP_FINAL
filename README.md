# 🌿 Biodiversity Loss Analysis & IUCN Extinction-Risk Prediction

One project with two parts that share the same code and data:

1. **Website (`docs/`)**: plain HTML, CSS and JavaScript, no login and no server. Type a species name to see its IUCN status, a model prediction, and (for EN/CR species) reasons for extinction risk and measures to conserve it. It lists **5,327 species**.
2. **Streamlit analysis app (`app.py`)**: trends and forecast, distribution fitting, Markov chain and status predictor pages.

## Project structure
```
README.md                 this file (the only README)
requirements.txt          Python packages
app.py                    Streamlit analysis app
export_model.py           retrains both models and rebuilds docs/data.js
fetch_iucn_species.py     optional: fetch more real IUCN details with your API token
model_report.json         accuracy and settings of the latest training run
core/                     model and data code (classifier.py, trait_model.py, markov.py, ...)
data/                     your datasets (see below)
docs/                     the website: index.html, style.css, script.js, data.js, .nojekyll
tests/test_smoke.py       renders every Streamlit page (pytest -q)
.streamlit/config.toml    Streamlit theme
.gitignore
```

## Data (in `data/`)
| File | Used for |
|---|---|
| `Markov_Random_Forest (1).xlsx` | Status-change model (738 usable species from all 3 sheets), Markov chain |
| `species_master_with_iucn_details.csv` | Mammal trait model (4,631 mammals) and real IUCN threats, habitats, population trend |
| `Distribution_Data.csv` | Distribution fitting (Streamlit app) |

Without the files the Streamlit app falls back to clearly labelled synthetic demo data. Check the IUCN Red List terms of use before making the data public.

## The two models
| Model | Trained on | Predicts | Inputs |
|---|---|---|---|
| Status-change model (Random Forest, `core/classifier.py`) | 738 species whose category changed 2021 to 2022 | The NEXT IUCN category | 2021 status, taxonomic group |
| Mammal trait model (gradient boosting + family/genus encoding, `core/trait_model.py`) | 4,104 mammals with a current category (LC to CR) | The category estimated from traits (also for Data Deficient mammals) | Body mass, geographic range, climate, human population pressure, diet, activity, life history, order |

Notes: the IUCN threats, habitats and population trend are shown on the site but are NOT model inputs (they are tied to the category). The mammal scores are adjusted (`PRIOR_ALPHA` in `core/trait_model.py`, default 0.5) so that the many LC species do not hide rarer, higher-risk categories. See `model_report.json` for accuracy; the mammal model separates threatened from non-threatened species well (AUC about 0.89) but is weak at the exact category.
Measures shown for mammals are standard actions matched to each species' IUCN threat types, not IUCN's own conservation-action lists.

## Model improvements (mammal trait model)
Measured with the same 5-fold cross-validation, the new model against the first version (Random Forest + median imputation), both stored in `model_report.json`:

| Metric | First version | Now |
|---|---|---|
| Threatened vs not, ROC-AUC | 0.868 | **0.887** (3 CV seeds, SD 0.001) |
| Threatened vs not, PR-AUC | 0.720 | **0.755** |
| Macro-F1, exact category | 0.389 | **0.420** |
| Within one category | 0.800 | **0.852** |
| Accuracy | 0.594 | **0.691** |

What changed: gradient boosting that handles missing traits natively; engineered range/mass features (log range area, range area per body mass, latitude/longitude span, island-like range); family and genus target encoding computed inside each CV fold so no label leaks. Balanced accuracy is about the same as before (0.42 vs 0.43): the new model predicts LC more often, which lifts accuracy, but recall for NT (0.16 vs 0.27) and CR (0.37 vs 0.46) is lower than the old model. Exact-category prediction is still weak.

The status-change model was not changed: with 738 rows, 99.5% of them species whose status changed, a simple lookup table (about 0.52 CV accuracy) still beats the Random Forest (about 0.46). More or different data would help, not tuning.

## Run
**Website** (no install): open `docs/index.html` in a browser, or run `python -m http.server` inside `docs/` and open http://localhost:8000.

**Rebuild the website data after changing data or settings**
```bash
pip install -r requirements.txt
python export_model.py        # retrains both models, rewrites docs/data.js and model_report.json
```

**Streamlit app**
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publish the website on GitHub Pages
1. Upload this whole project to a GitHub repository (the folder structure must stay as is).
2. Settings > Pages > Source: "Deploy from a branch", branch `main`, folder `/docs`.
3. Open `https://<username>.github.io/<repository>/` after about a minute (Ctrl+F5).

## Optional: more real species details
`python fetch_iucn_species.py --limit 3` asks for your IUCN API token (hidden, not saved) and saves details to `data/iucn_species.jsonl`; rerun `export_model.py` to add them to the site. Never put the token in a file you upload.

## Interpretation notes
* Both models are exploratory. The status-change model uses only two inputs and works much like a lookup table; its training data contains only species whose category changed.
* The site covers 5,327 species, not the full Red List (about 170,000). Outside mammals, only species that changed status in 2022 are included.
* Regression totals in the Streamlit app reflect assessment effort as well as true biodiversity loss; forecasts are illustrative.
