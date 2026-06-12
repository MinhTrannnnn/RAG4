# Run Without `.venv`

This guide is only for backup cases where `.venv` is broken or cannot be
created. The recommended way is still the main README flow with `.venv`.

## When To Use This

Use plain Python only if:

- `.venv` creation fails
- activation is confusing or broken
- you are on a temporary exam machine and need a quick fallback

Plain Python installs packages into the selected Python environment, so it can
mix dependencies between projects. Use it only when needed.

## Check Python

If the machine only has VS Code, install Python 3.10 or newer first from
`https://www.python.org/downloads/`. During install, tick
`Add python.exe to PATH`.

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

Python 3.10 or newer is recommended.

If `python` opens Microsoft Store, disable Python app execution aliases in
Windows Settings or use `py`:

```powershell
py --version
```

## Install Dependencies

From the project folder:

```powershell
cd D:\InformationRetrieval\ONTHI\RAG4
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` does not work but `py` does:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

## Create `.env`

```powershell
Copy-Item .env.example .env
```

Edit `.env`, especially:

```env
STUDENT_ID=YOUR_STUDENT_ID
TEACHER_BASE_URL=http://192.168.50.218:8000/api/v1
EMBEDDING_MODEL_PATH=models/vietnamese-sbert
VECTOR_DB_PATH=data/vector_db.pkl
```

## Download Embedding Model

```powershell
python scripts/download_embedding_model.py
```

Expected final output:

```text
Embedding dimension: 768
```

## Run Server

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

Check:

```powershell
curl.exe http://127.0.0.1:5000/
```

## Register And Evaluate

Open another PowerShell window:

```powershell
python register.py
python evaluate.py
python result.py
```

`python evaluate.py` sends `document_received=false`, so the Teacher will call
`/upload` first. After upload has succeeded once and `data/vector_db.pkl` exists,
use this for later attempts:

```powershell
python evaluate.py --skip-upload
```

That sends `document_received=true`, so the Teacher skips `/upload` and only
sends questions.

If needed:

```powershell
python reset.py
```

## Common Fixes

If packages are missing:

```powershell
python -m pip install -r requirements.txt
```

If the wrong Python is being used:

```powershell
python -c "import sys; print(sys.executable)"
```

If model is missing:

```powershell
python scripts/download_embedding_model.py
```

If port `5000` is busy:

```powershell
netstat -ano | findstr :5000
Stop-Process -Id PROCESS_ID
```
