# RAG Competition Student Server

FastAPI server for the RAG exam. The server exposes the two required endpoints:

- `POST /upload`
- `POST /ask`

This repository is intentionally lightweight. It does not include the embedding
model or the virtual environment. After cloning, create `.venv`, install
dependencies, download the embedding model, then run the server.

The code is split by responsibility:

- `app/main.py`: FastAPI endpoints
- `app/config.py`: environment/config values
- `app/schemas.py`: request/response models
- `app/rag.py`: chunking, embedding, retrieval
- `app/llm.py`: Teacher Proxy LLM call and answer parsing
- `app/logging_utils.py`: JSONL debug logs

Extra guides:

- `READMEPYTHON.md`: backup flow for running with plain Python if `.venv` fails
- `TUNING.md`: what to adjust when the score is low

## What Is Included

Included in Git:

- source code
- helper scripts
- `requirements.txt`
- `.env.example`
- `models/.gitkeep`

Not included in Git:

- `.venv/`
- `.env`
- `models/vietnamese-sbert/`
- `logs/`
- `data/vector_db.pkl`

## Fresh Windows Machine With Only VS Code

If the machine only has VS Code, install/check these first:

1. Install Python 3.10 or newer from `https://www.python.org/downloads/`.
   During install, tick `Add python.exe to PATH`.
2. Install Git for Windows from `https://git-scm.com/download/win` if `git`
   is not available.
3. Open VS Code, then open a PowerShell terminal:
   `Terminal -> New Terminal`.
4. Check:

```powershell
python --version
git --version
```

If `python` opens Microsoft Store, disable Python app execution aliases in
Windows Settings, or reinstall Python and tick `Add python.exe to PATH`.

Clone the repository:

```powershell
cd D:\InformationRetrieval\ONTHI
git clone https://github.com/MinhTrannnnn/RAG4.git
cd RAG4
```

Then continue with the quick start below.

If VS Code asks for a Python interpreter after `.venv` is created, choose:

```text
.\.venv\Scripts\python.exe
```

VS Code may auto-activate `.venv` in new terminals. That is fine, but the
explicit commands below work even without activation.

## Quick Start After Clone

Go to the project folder:

```powershell
cd D:\InformationRetrieval\ONTHI\RAG4
```

Check Python:

```powershell
python --version
```

Python 3.10 or newer is recommended.

Create `.venv` and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `.env`:

```powershell
Copy-Item .env.example .env
```

Open `.env` and update at least:

```env
STUDENT_ID=YOUR_STUDENT_ID
STUDENT_PORT=5000
TEACHER_BASE_URL=http://192.168.50.218:8000/api/v1
EMBEDDING_MODEL_PATH=models/vietnamese-sbert
VECTOR_DB_PATH=data/vector_db.pkl
```

Download the embedding model:

```powershell
.\.venv\Scripts\python.exe scripts/download_embedding_model.py
```

Expected final output includes:

```text
Embedding dimension: 768
```

Run the server:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 5000
```

Check locally:

```powershell
curl.exe http://127.0.0.1:5000/
```

Expected response includes:

```json
{
  "status": "running",
  "embedding_model_path_exists": true
}
```

## If VS Code Auto Activates `.venv`

VS Code may automatically run a command like this when opening a new terminal:

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
```

This is normal if the VS Code Python extension is installed and auto activation
is enabled. It only changes the current terminal so `python` points to the
selected virtual environment.

Check which Python is active:

```powershell
python -c "import sys; print(sys.executable)"
```

For this project, the output should point to:

```text
D:\InformationRetrieval\ONTHI\RAG4\.venv\Scripts\python.exe
```

If VS Code activates the wrong project environment, either select the correct
interpreter with `Python: Select Interpreter`, or run commands with the explicit
`.venv` path shown in this README. The explicit commands work even when the
terminal is not activated.

## Evaluation Flow

The slide says that after you call `POST /competition/evaluate`, the Teacher
Server actively calls your Student Server:

1. `POST /upload` once with the source document, timeout up to 120 seconds.
2. `POST /ask` 100 times, timeout up to 60 seconds for each question.

The Teacher may omit `doc_id` in `/upload`; this is valid because `doc_id` is
optional.

The exam announcement says each student should submit at most 5 evaluation
attempts. The `document_received` body field controls whether the Teacher sends
the document again:

- first attempt or when `data/vector_db.pkl` does not exist: `false`
- later attempts after upload/vector DB already succeeded: `true`

This project saves the vector DB at `data/vector_db.pkl` after `/upload`, then
loads it automatically when the server starts again.

`/ask` retrieves local context, calls the Teacher Proxy LLM, and returns exactly
one letter: `A`, `B`, `C`, or `D`.

If the Teacher Proxy times out or fails, `/ask` now returns an HTTP error instead
of guessing locally. A local guess usually hides the real issue and is unlikely
to help scoring, because the evaluation expects the RAG answer to come from the
retrieved context plus Teacher Proxy LLM.

Updated Teacher Server schemas from the slide:

```text
POST /competition/register
Header: X-Student-ID
Body: { "server_url": "http://YOUR_LAN_IP:5000" }
Response: message, student_id, server_url

POST /competition/evaluate
Header: X-Student-ID
Body: { "document_received": false } on first upload, or true after vector DB exists
Response: message, final_score

POST /competition/reset
Header: X-Student-ID
Body: none
Response: status, message, score

GET /competition/result
Header: X-Student-ID
Response: student_id, score, status, current_question
```

## Register With Teacher Server

Keep the FastAPI server running, then open another PowerShell window.

Register:

```powershell
.\.venv\Scripts\python.exe register.py
```

If the script guesses the wrong LAN IP, pass it manually:

```powershell
.\.venv\Scripts\python.exe register.py --server-url "http://YOUR_LAN_IP:5000"
```

Start evaluation:

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

This is the first-attempt/full flow and sends:

```json
{
  "document_received": false
}
```

After `/upload` succeeded once and `data/vector_db.pkl` exists, re-submit without
uploading the document again:

```powershell
.\.venv\Scripts\python.exe evaluate.py --skip-upload
```

That sends:

```json
{
  "document_received": true
}
```

The helper allows up to 900 seconds for this request, because the Teacher Server
may wait for `/upload` plus 100 `/ask` calls before returning `final_score`.

Check result:

```powershell
.\.venv\Scripts\python.exe result.py
```

Reset after a crash:

```powershell
.\.venv\Scripts\python.exe reset.py
```

## Local Postman Test

First call `POST http://127.0.0.1:5000/upload`:

```json
{
  "text": "RAG is Retrieval-Augmented Generation. FastAPI is used to create APIs."
}
```

Then call `POST http://127.0.0.1:5000/ask`:

```json
{
  "question": "RAG la gi? A. Database B. Retrieval-Augmented Generation C. Network D. OS"
}
```

`/ask` calls the Teacher Proxy LLM, so it needs access to the competition LAN.

## Troubleshooting

### `python` Is Not Recognized

Install Python 3.10 or newer, then reopen PowerShell:

```powershell
python --version
```

If Windows opens Microsoft Store, disable Python app execution aliases in Windows
Settings.

### `.venv` Does Not Exist

Create it again:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Package Is Missing

Example:

```text
ModuleNotFoundError: No module named 'fastapi'
ModuleNotFoundError: No module named 'sentence_transformers'
```

Fix:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `pip install` Is Slow Or Fails

Make sure the machine still has Internet. The heavy packages are mainly
`torch`, `transformers`, and `sentence-transformers`.

Try again with:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Embedding Model Is Missing

Example:

```text
Embedding model path not found: models/vietnamese-sbert
```

Fix:

```powershell
.\.venv\Scripts\python.exe scripts/download_embedding_model.py
```

### Model Download Fails

Check Internet, then retry:

```powershell
.\.venv\Scripts\python.exe scripts/download_embedding_model.py
```

After download, the folder should exist:

```powershell
Get-ChildItem models\vietnamese-sbert
```

### Offline Model Test

Use this if you want to confirm the model can load without Internet:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('models/vietnamese-sbert', local_files_only=True); print(m.get_sentence_embedding_dimension())"
```

Expected output:

```text
768
```

### Port 5000 Is Already In Use

Find the process:

```powershell
netstat -ano | findstr :5000
```

Stop it:

```powershell
Stop-Process -Id PROCESS_ID
```

Or run on another port and update `.env`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 5001
```

```env
STUDENT_PORT=5001
```

### `curl http://127.0.0.1:5000/` Cannot Connect

The server is not running or crashed during startup. Start it again and read the
error in that PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 5000
```

### Register Script Uses The Wrong LAN IP

Check your LAN IP:

```powershell
ipconfig
```

Then pass the URL manually:

```powershell
.\.venv\Scripts\python.exe register.py --server-url "http://YOUR_LAN_IP:5000"
```

### Teacher Server Cannot Call Your Server

Make sure:

- the server is running with `--host 0.0.0.0`
- you registered `http://YOUR_LAN_IP:5000`, not `127.0.0.1`
- your machine and Teacher Server are on the same LAN
- Windows Firewall allows Python on port `5000`

### `/ask` Fails But `/upload` Works

`/upload` only uses local embedding.

`/ask` also calls the Teacher Proxy:

```text
http://192.168.50.218:8000/api/v1/proxy
```

So `/ask` needs the competition LAN and correct `TEACHER_BASE_URL` in `.env`.

## Debug Score

After evaluation, inspect:

```text
logs/ask_logs.jsonl
```

Each `/ask` call logs the question, retrieved chunks, raw LLM answer, final
answer, and any LLM error.
