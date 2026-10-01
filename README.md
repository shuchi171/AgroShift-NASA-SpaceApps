# AgroShift 🌾
### NASA-Assisted Crop-Rotation Decision Support System

> **NASA Space Apps Challenge:** Field Shift: Adapting Farms with NASA Data

AgroShift combines farm location, current crop, soil texture, and farmer priority with NASA POWER environmental context to rank next-crop alternatives. The Streamlit dashboard explains a transparent, hand-designed scoring system. It does not predict yields, prices, drought risk, or an optimal multi-season sequence.

## Run locally

Use Python 3.10 or newer and an internet connection for NASA POWER requests.

```bash
git clone https://github.com/shuchi171/AgroShift-NASA-SpaceApps.git
cd AgroShift-NASA-SpaceApps
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## NASA POWER environmental context

The daily point API requests an inclusive **90-day** window ending four days before the request date to allow for publication latency. Availability varies by parameter; this buffer does not guarantee the newest data. The dashboard shows the returned date range and uses its latest complete 30-day period for scoring.

| Parameter | Use |
| --- | --- |
| T2M (°C) | Mean temperature for climate scoring |
| PRECTOTCORR (mm/day) | Accumulated rainfall for climate scoring |
| RH2M (%) | Relative humidity displayed as context |
| GWETTOP (0–1) | Surface soil wetness index for climate scoring |

NASA POWER supports historical daily time series and near-real-time updates with latency. Its meteorological reanalysis data have approximately **0.5° × 0.625°** spatial resolution, providing regional context rather than field sensor measurements. These data are not a weather forecast. See the [NASA POWER meteorology documentation](https://power.larc.nasa.gov/docs/methodology/meteorology/) and [API tutorial](https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/).

If NASA returns an error, no usable data, or an incomplete recent 30-day period, the dashboard displays a prominent notice and pauses recommendations. **No synthetic fallback data are generated.** The retry button clears cached requests; results otherwise remain cached for one hour.

## Transparent recommendation methodology

The selected current crop is excluded entirely, using crop IDs and normalized display names. Remaining candidates receive a **heuristic score out of 100**, not a probability, confidence estimate, or calibrated likelihood:

1. **Climate — 30 points:** Temperature fit (15) and wetness/rainfall fit (15).
2. **Soil — 25 points:** Matches the selected texture against preferred soils, with partial credit for other textures.
3. **Rotation — 25 points:** Same-family crops or explicitly incompatible predecessors receive 4 points. Poaceae cereals followed by legumes receive 25; legumes followed by non-legumes receive 24; other transitions receive 18. Family metadata comes from `crops.json`, so Rice → Wheat and Rice → Maize receive the same-family penalty. Fallow has no crop family and receives the neutral rotation score.
4. **Farmer priority — 20 points:** Water Conservation, Soil Health Restoration, Yield Potential, or Climate & Drought Resilience. These use qualitative database attributes; there is no economic or market-price model.

The top three crops are **alternative next crops**, each scored relative to the current crop. They are not consecutive seasons. The weights and thresholds are prototype rules, not validated agronomic prescriptions. Recent conditions do not represent the future growing season. Planting dates, cultivars, irrigation, local pests, and management still require local assessment.

## Data & References

`crops.json` is a small, manually curated prototype knowledge base. The current-crop selector includes every crop in database order plus Fallow (Bare Soil); soil choices include all preferred soils in the database. The original repository did not record field-by-field sources. Its temperature cutoffs, soil lists, tolerance labels, yield labels, and qualitative soil benefits remain **unverified heuristic inputs**. Unsupported numeric nitrogen-fixation, crop-duration, and heat-stress claims have been removed from descriptions. Benefits depend on cultivar, local conditions, and management. These references provide background and a basis for reviewing the data; they are not evidence that every stored value was extracted from these sources.

- [FAO ECOCROP documentation](https://www.fao.org/geospatial/data-and-tools/data-portals/ecocrop/) describes crop temperature, precipitation, and soil requirements. Requirements vary by crop and cultivar; this application is not a direct ECOCROP dataset export.
- [FAO: Crop Water Needs](https://www.fao.org/4/s2022e/s2022e02.htm) explains differences in water needs by crop, climate, and growth stage, informing interpretation of the simplified Low/Medium/High categories.
- [SARE: Legume Cover Crops](https://www.sare.org/publications/managing-cover-crops-profitably/legume-cover-crops/) describes nitrogen fixation and soil benefits. Nitrogen contribution depends on species and management; fixed kg N/ha values should not be assumed universally.
- [SARE: Cover Crops](https://www.sare.org/publications/building-soils-for-better-crops/cover-crops/) discusses residues, soil cover, and brassica biofumigation. Benefits are conditional, not guaranteed pest control.

## Project files and legacy material

- `app.py`: maintained Streamlit dashboard, NASA retrieval, and crop scoring.
- `crops.json`: prototype crop attributes and family metadata.
- `requirements.txt`: runtime dependencies.
- `tests/test_app.py`: offline scoring, API handling, and Streamlit regression checks. Run `python -m unittest discover -s tests -v` after installing dependencies.
- `archive/legacy_prototype.ipynb`: obsolete irrigation/Random Forest experiment, retained for history with outputs cleared and an explicit warning. Its former 100% accuracy result used a test set containing only class 0 and is not evidence of predictive skill. Do not run its file-writing cells over the maintained app.
- `project_screenshots/`: historical screenshots that may show earlier interface text.

The experimental ML section and its unsupported seven-day deficit forecast have been removed from the maintained application.
