# Scientific Discovery Copilot Frontend

Simple Streamlit UI for checking the FastAPI backend locally.

## Run

```powershell
cd C:\Users\admin\Desktop\Gemma4\scientific-discovery-copilot\frontend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The UI expects the backend at `http://127.0.0.1:8000/api/v1` by default.
