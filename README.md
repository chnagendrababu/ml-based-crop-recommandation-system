# 🌾 CropSense — AI-Powered Crop Recommendation System

> An XGBoost machine learning application that recommends the most suitable crop for your field based on soil nutrients and climate conditions — with a polished Streamlit interface, top-3 predictions, confidence scores, and downloadable PDF reports.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Input Parameters](#input-parameters)
- [Model Details](#model-details)
- [Output & Reports](#output--reports)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

CropSense is a machine learning-powered web application built with **Streamlit** that helps farmers and agronomists make data-driven crop decisions. By entering 7 soil and climate parameters, users receive an instant crop recommendation powered by a trained **XGBoost classifier**, along with confidence probabilities for the top 3 suggested crops.

The app is designed to be simple enough for everyday farming use while being technically robust — wrapping a production-grade ML pipeline in a clean, accessible interface.

---

## Features

- **🤖 XGBoost ML Model** — Fast, accurate gradient-boosted classifier trained on real crop dataset
- **📊 Confidence Scores** — Know how certain the model is about each recommendation
- **🔝 Top-3 Predictions** — See the 3 best-fit crops ranked by probability
- **📈 Visual Bar Chart** — Instant chart comparing top-3 crop confidence scores
- **📄 PDF Report Download** — Generate and download a full prediction report with inputs + results
- **🎨 Custom UI** — Themed interface with hero banner, grouped input cards, and result display
- **⚡ Fast Inference** — Real-time predictions with cached model loading

---

## Demo

```
Home Page  →  Enter Soil & Climate Parameters  →  Click "Recommend My Crop"
          →  View Top Crop + Confidence Score
          →  Explore Top 3 Crops + Bar Chart
          →  Download PDF Report
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit, Custom CSS |
| ML Model | XGBoost (`xgb_crop_model.pkl`) |
| Label Encoding | scikit-learn `LabelEncoder` (`label_encoder.pkl`) |
| Feature Config | Pickle (`feature_names.pkl`) |
| Visualization | Matplotlib, Seaborn |
| PDF Generation | ReportLab |
| Data Processing | NumPy |

---

## Project Structure

```
cropsense/
│
├── app.py                      # Main Streamlit application
│
├── xgb_crop_model.pkl          # Trained XGBoost model (required)
├── label_encoder.pkl           # Label encoder for crop classes (required)
├── feature_names.pkl           # Ordered list of input feature names (required)
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

> **Note:** The three `.pkl` files must be present in the root directory before running the app. Without them, the app will display an error and stop.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/chnagendrababu/ml-based-crop-recommandation-system.git
cd cropsense
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Model Files

Place the following files in the project root directory:

- `xgb_crop_model.pkl`
- `label_encoder.pkl`
- `feature_names.pkl`

### 5. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Usage

1. **Open the app** in your browser after running the command above
2. **Read the welcome section** to understand what the tool does
3. **Adjust the sliders** under each category:
   - Soil Nutrients — N, P, K values
   - Climate Conditions — Temperature, Humidity, Rainfall
   - Soil Property — pH level
4. **Click "🌱 Recommend My Crop"** to run the prediction
5. **View your results:**
   - Primary recommended crop with confidence %
   - Top 3 ranked crops with scores
   - Bar chart of confidence distribution
6. **Download the PDF report** for your records or sharing

---

## Input Parameters

The model accepts 7 parameters that describe your field's soil and climate:

| Parameter | Unit | Range | Description |
|---|---|---|---|
| **Nitrogen (N)** | kg/ha | 0 – 140 | Nitrogen content in soil |
| **Phosphorus (P)** | kg/ha | 0 – 140 | Phosphorus content in soil |
| **Potassium (K)** | kg/ha | 0 – 205 | Potassium content in soil |
| **Temperature** | °C | 5 – 45 | Average ambient temperature |
| **Humidity** | % | 10 – 100 | Relative humidity of the environment |
| **Soil pH** | — | 3.5 – 9.5 | Acidity/alkalinity of the soil |
| **Rainfall** | mm | 0 – 300 | Average annual/seasonal rainfall |

> **Tip:** Use soil testing reports from a certified lab for the most accurate N, P, K, and pH values.

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Input Features | 7 (N, P, K, temperature, humidity, pH, rainfall) |
| Output | Predicted crop class + probability distribution |
| Encoding | scikit-learn `LabelEncoder` for crop name mapping |
| Serialization | Python `pickle` format |

The model was trained on a labeled crop recommendation dataset. Feature names are loaded from `feature_names.pkl` to ensure the correct input order is maintained at inference time, regardless of how sliders are arranged in the UI.

---

## Output & Reports

### On-Screen Output
- **Recommended Crop** — The highest-probability crop for your inputs
- **Confidence Score** — Model certainty as a percentage
- **Top 3 Crops** — Ranked list with individual confidence scores
- **Bar Chart** — Visual comparison of the top 3 predictions

### PDF Report Contents
- All 7 input parameters entered by the user
- Primary recommended crop and confidence score
- Top 3 crop recommendations with probabilities
- Formatted for printing or sharing with agronomists

---

## Requirements

```
streamlit
numpy
xgboost
scikit-learn
matplotlib
seaborn
reportlab
```

Install all at once:

```bash
pip install streamlit numpy xgboost scikit-learn matplotlib seaborn reportlab
```

Full pinned versions can be found in `requirements.txt`.

---

## Troubleshooting

**❌ "Model files not found" error**
Make sure `xgb_crop_model.pkl`, `label_encoder.pkl`, and `feature_names.pkl` are all present in the same directory as `app.py`.

**❌ App doesn't open in browser**
Try visiting `http://localhost:8501` manually, or check if another process is using that port:
```bash
streamlit run app.py --server.port 8502
```

**❌ `ModuleNotFoundError` for any package**
Re-run the install command:
```bash
pip install -r requirements.txt
```

**❌ PDF download not working**
Ensure `reportlab` is installed. On some systems you may need:
```bash
pip install reportlab --upgrade
```

---

## Future Improvements

- 🗺️ **Location-based auto-fill** — Auto-detect climate values based on GPS/region
- 🌍 **Multi-language support** — Hindi, Marathi, and other regional languages
- 📱 **Mobile-optimized layout** — Improved slider UX on touch devices
- 🧾 **Season filter** — Recommend crops based on Kharif / Rabi / Zaid seasons
- 📊 **Historical comparison** — Track past recommendations across sessions
- ☁️ **Weather API integration** — Pull live temperature, humidity, and rainfall data
- 🎯 **Model explainability** — SHAP values to show which factors influenced the recommendation

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

<div align="center">

Made with ❤️ for smarter, data-driven farming.

**CropSense** · XGBoost ML · Streamlit · ReportLab

</div>
