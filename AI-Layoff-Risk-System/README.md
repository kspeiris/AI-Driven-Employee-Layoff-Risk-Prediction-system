AI-Driven Employee Layoff Risk Prediction
========================================

A workforce analytics system for estimating employee layoff risk from role,
industry, AI adoption, automation, and work-profile features. The project
contains a Flask prediction API, a Streamlit dashboard, trained model artifacts,
and a processed dataset used by the dashboard analytics pages.

Features
--------

- Single and batch layoff-risk prediction
- CatBoost, ANN, and DNN model support
- Streamlit dashboard for analytics and model exploration
- Industry and job-role intelligence pages
- SHAP and LIME explainability endpoints
- SQLite-backed prediction logging

Project Structure
-----------------

```text
AI-Layoff-Risk-System/
  backend/
    app.py                 Flask application entry point
    config.py              Paths, feature columns, and API settings
    database.py            SQLite setup and query helpers
    explainability.py      SHAP and LIME helper logic
    models/                Saved model artifacts
    routes/                API route modules
  dashboard/
    Home.py                Streamlit landing page
    pages/                 Dashboard pages
    utils/                 Data loading helpers
  data/
    processed_layoff_dataset.csv
  requirements.txt
```

Setup
-----

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Required model files should be available in `backend/models/`:

```text
ml_model.pkl
catboost_model.pkl
ann_layoff_model.keras
dnn_layoff_model.keras
scaler.pkl
```

Run The Backend
---------------

```powershell
cd backend
python app.py
```

The API runs at:

```text
http://127.0.0.1:5000
```

Useful endpoints:

```text
GET  /health
POST /predict
POST /predict/batch
GET  /analytics/summary
POST /explain/shap
POST /explain/lime
```

Run The Dashboard
-----------------

Open a second terminal from the project directory:

```powershell
streamlit run dashboard/Home.py
```

If the backend runs on a different host or port, set `API_URL` before launching
Streamlit:

```powershell
$env:API_URL="http://127.0.0.1:5000"
streamlit run dashboard/Home.py
```

Notes
-----

- `backend/ai_layoff.db` is created automatically at runtime.
- Python cache folders and training-output folders are intentionally ignored.
- The dashboard expects the processed dataset at `data/processed_layoff_dataset.csv`.
