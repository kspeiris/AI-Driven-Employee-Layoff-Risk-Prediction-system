# Deploying to Streamlit Cloud

Use these exact values in the Streamlit deploy form when your GitHub repo root is the `AI-Layoff-Risk-System` folder.

- Repository: paste your GitHub URL (example: `https://github.com/your-username/your-repo`)
- Branch: `main`
- Main file path: `streamlit_app.py`
- App URL (optional): pick a slug, e.g. `ai-layoff-risk` → final URL: `ai-layoff-risk.streamlit.app`

Notes
- Ensure you push the `AI-Layoff-Risk-System` folder contents to your GitHub repo `main` branch.
- Use forward slashes in the Streamlit form (already set since `streamlit_app.py` is at the repo root).
- Streamlit Cloud installs dependencies from `requirements.txt` at the repository root. This repo already references the inner requirements file via `-r AI-Layoff-Risk-System/requirements.txt`.

Local test
```bash
# from the workspace root (one level above AI-Layoff-Risk-System)
pip install -r AI-Layoff-Risk-System/requirements.txt
streamlit run AI-Layoff-Risk-System/streamlit_app.py
```

Troubleshooting
- "This file does not exist": double-check the repo root on GitHub — if the GitHub repo contains the `AI-Layoff-Risk-System` folder, use the path above; if you pushed only the folder contents (i.e., the folder's contents are at repo root), set `streamlit_app.py` as the Main file path instead.
- Ensure `streamlit_app.py` is committed and pushed to the `main` branch.

Want me to push these changes to your remote? Provide the repo remote name/URL and confirm.