# Credit Risk Decision Engine

An end-to-end ML system for loan default prediction and credit decisioning.

## Status: In Progress

cat prompt.txt | claude

.\venv\Scripts\Activate.ps1

mlflow ui --backend-store-uri ./mlruns

uvicorn src.api:app --port 8000 --reload
