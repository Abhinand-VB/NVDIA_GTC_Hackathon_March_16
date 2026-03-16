# GPU Efficiency Advisor using NVIDIA Nemotron

A minimal Streamlit app that analyzes GPU training or inference configurations and job logs and returns optimization suggestions powered by **NVIDIA Nemotron**.

## What it does

- You paste your GPU config (model, GPU type, batch size, utilization, memory, etc.) or relevant log snippets.
- Click **Analyze** to send the text to NVIDIA Nemotron via the NIM API.
- You get structured suggestions for improving GPU utilization, memory use, speed, and cost (bottlenecks, 4–6 actionable tips, and estimated impact).

## Install dependencies

```bash
pip install -r requirements.txt
```

## API key

1. Copy `.env.example` to `.env`.
2. Get an API key from [NVIDIA API Catalog](https://build.nvidia.com) (e.g. for Nemotron).
3. Set it in `.env`:

   ```
   NVIDIA_API_KEY=your_actual_api_key_here
   ```

The app loads `.env` via `python-dotenv` and reads `NVIDIA_API_KEY`. If the key is missing, the app shows a warning and analysis is disabled.

## Run the app

**Option 1 – Run script (recommended)**  
From the project folder in PowerShell:
```powershell
.\run.ps1
```
Or double‑click `run.bat`. The script installs dependencies and starts Streamlit.

**Option 2 – Manual**  
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (e.g. `http://localhost:8501`). Paste a config, click **Analyze**, and view the suggestions.

## Project structure

- `app.py` – Single-file Streamlit app (UI + Nemotron API calls).
- `requirements.txt` – Python dependencies.
- `.env.example` – Example env file; copy to `.env` and add your key.
- `README.md` – This file.

Built for NVIDIA Hackathon at SJSU | Powered by NVIDIA Nemotron.
