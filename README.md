# AgroShift 🌾
### AI & NASA-Assisted Crop-Rotation Decision Support System

> **NASA Space Apps Challenge:** Field Shift: Adapting Farms with NASA Data  
> **Sub-Theme:** Climate Adaptation, Soil Health, and Resilient Agricultural Sequences

---

## 📌 Executive Summary
Climate change accelerates seasonal shifts, increases drought frequency, and disrupts traditional planting calendars. Monoculture and repetitive cropping deplete soil moisture and nitrogen, leaving smallholder farms vulnerable to abrupt climate shocks.

**AgroShift** pivots agricultural planning from reactive drought intervention to proactive **crop-rotation decision support**. By ingesting environmental parameters from the **NASA POWER API** and combining them with local soil characteristics, current crop status, and farmer objectives, AgroShift recommends optimal multi-season rotation sequences that regenerate soil, minimize water stress, and sustain yields.

---

## 🛰️ NASA Earth Data Architecture

AgroShift uses the **NASA POWER (Prediction of Worldwide Energy Resources) Agroclimatology API**. 

We query daily temporal point data using the following NASA satellite and assimilation parameters:
* **T2M (Temperature at 2 Meters, °C):** Assesses thermal suitability and risks of thermal crop shock.
* **PRECTOTCORR (Precipitation Corrected, mm/day):** Measures cumulative natural moisture replenishment.
* **RH2M (Relative Humidity at 2 Meters, %):** Gauges ambient vapor pressure deficit and atmospheric drying demand.
* **GWETTOP (Surface Soil Wetness, 0–1 index):** Monitors upper soil layer moisture saturation.

> **Operational Scope Note:** NASA POWER provides gridded satellite-reanalysis observations at a 0.5° × 0.5° spatial resolution. It offers regional environmental context and climate trends rather than high-frequency physical in-situ rootzone sensor measurements.

---

## ⚙️ Decision Support & Crop Rotation Methodology

Instead of treating crop choice as a black box, AgroShift implements a transparent, multi-factor agronomic scoring algorithm:

$$\text{Suitability Score} = S_{\text{Climate}} (30\%) + S_{\text{Soil}} (25\%) + S_{\text{Rotation}} (25\%) + S_{\text{Priority}} (20\%)$$

1. **Climate Suitability (30 pts):** Compares 30-day NASA temperature and moisture averages against crop biological thresholds.
2. **Soil Compatibility (25 pts):** Cross-references farm topsoil texture against root penetration and drainage profiles.
3. **Rotation & Biosecurity Fit (25 pts):** Enforces biological diversity rules:
   * Penalizes back-to-back planting of the same family (e.g., Cereal → Cereal) to prevent pest cycles.
   * Awards bonuses for alternating taproot legumes (e.g., *Rice → Lentil*) to fix atmospheric nitrogen ($30\text{–}50\text{ kg N/ha}$).
4. **Farmer Priority Alignment (20 pts):** Weights choices according to user objectives: *Water Conservation*, *Soil Health*, *Yield Optimization*, or *Climate Resilience*.

---

## 🔬 Experimental Machine Learning Model

The platform contains an **Experimental Drought-Risk Indicator** powered by a Random Forest Classifier:
* **Target:** Predicts the probability of topsoil moisture falling below critical deficit levels ($\text{GWETTOP} < 0.35$).
* **Leakage Prevention:** Evaluated strictly using temporal out-of-sample test splits. We report honest train vs. test metrics and make **no claims of false 100% accuracy**.
* **Role:** Designed purely as a secondary early-warning signal, keeping the primary focus on crop rotation and soil adaptation.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/agroshift.git](https://github.com/your-username/agroshift.git)
cd agroshift
