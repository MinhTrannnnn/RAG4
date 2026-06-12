# Tuning Guide

Use this guide when the server runs correctly but the score is low.

The main files are:

- `app/rag.py`: chunking and retrieval
- `app/llm.py`: prompt and answer parsing
- `.env`: tuning values
- `logs/ask_logs.jsonl`: debug output after evaluation

## First Check The Logs

After evaluation, open:

```text
logs/ask_logs.jsonl
```

For each question, check:

- `question`: what Teacher Server asked
- `sources`: retrieved chunks
- `raw_answer`: what Teacher Proxy returned
- `answer`: final parsed answer
- `llm_error`: timeout/proxy/parse error if any

If `sources` do not contain the relevant legal/text evidence, tune retrieval.
If `sources` look right but `answer` is wrong, tune the prompt.

## Tune Retrieval

These values are in `.env`.

### `TOP_K`

Default:

```env
TOP_K=8
```

Increase when the right evidence is nearby but not included:

```env
TOP_K=10
TOP_K=12
```

Tradeoff: higher `TOP_K` gives the LLM more context, but can add noise.

### `CHUNK_SIZE`

Default:

```env
CHUNK_SIZE=900
```

If chunks are too short and lose context, try:

```env
CHUNK_SIZE=1100
CHUNK_SIZE=1300
```

If chunks are too long and retrieval is vague, try:

```env
CHUNK_SIZE=700
CHUNK_SIZE=800
```

### `CHUNK_OVERLAP`

Default:

```env
CHUNK_OVERLAP=180
```

If answers are split across chunk boundaries, increase:

```env
CHUNK_OVERLAP=220
CHUNK_OVERLAP=260
```

Keep overlap smaller than chunk size.

### `MAX_CONTEXT_CHARS`

Default:

```env
MAX_CONTEXT_CHARS=9000
```

If the LLM misses evidence because context is cut off, increase:

```env
MAX_CONTEXT_CHARS=11000
MAX_CONTEXT_CHARS=13000
```

Tradeoff: too much context can make the LLM less focused and slower.

## Tune Scoring Weights

In `app/rag.py`, retrieval combines semantic similarity and lexical overlap:

```python
score = semantic_score + (0.08 * lexical_score)
```

If questions contain exact terms that should strongly match the document, try:

```python
score = semantic_score + (0.12 * lexical_score)
score = semantic_score + (0.18 * lexical_score)
```

If lexical overlap adds noise, reduce it:

```python
score = semantic_score + (0.04 * lexical_score)
```

## Tune Chunk Splitting

In `app/rag.py`, `SPLIT_PATTERN` decides where text is split before chunking.

If the source document has headings like `Điều`, `Khoản`, `Chương`, or `Mục`,
add normalized heading forms that appear in the uploaded text.

Example:

```python
SPLIT_PATTERN = re.compile(
    r"\n\s*\n+|\n(?=\s*(?:Chuong|Bai|Muc|Cau|Slide|Phan|Dieu|Khoan)\b)",
    re.IGNORECASE,
)
```

Only change this after checking `logs/ask_logs.jsonl`.

## Tune Prompt

Prompt code is in:

```text
app/llm.py
```

If retrieved chunks are correct but answers are wrong, edit `build_prompt()`.

Useful prompt additions:

```text
Choose the option that is explicitly supported by the context.
If multiple options look plausible, prefer the one with the strongest direct evidence.
Return only A, B, C, or D.
```

Avoid asking for explanation because the required answer is one letter.

## Tune Timeout

In `.env`:

```env
LLM_TIMEOUT_SECONDS=45
```

Slide says each `/ask` has up to 60 seconds. If Teacher Proxy is slow, try:

```env
LLM_TIMEOUT_SECONDS=55
```

Do not set much higher than 60 because Teacher Server may time out first.

## Suggested Tuning Order

1. Run evaluation.
2. Open `logs/ask_logs.jsonl`.
3. If sources are wrong, tune `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, then scoring weight.
4. If sources are right but answer is wrong, tune prompt in `app/llm.py`.
5. If errors mention timeout/proxy, check LAN/Teacher Proxy and maybe set `LLM_TIMEOUT_SECONDS=55`.
6. Run `reset.py` or register again, restart server if needed, then run
   `evaluate.py --skip-upload` if `data/vector_db.pkl` already exists.

## Reasonable Presets

Balanced:

```env
CHUNK_SIZE=900
CHUNK_OVERLAP=180
TOP_K=8
MAX_CONTEXT_CHARS=9000
```

More context:

```env
CHUNK_SIZE=1100
CHUNK_OVERLAP=240
TOP_K=10
MAX_CONTEXT_CHARS=12000
```

More focused:

```env
CHUNK_SIZE=700
CHUNK_OVERLAP=140
TOP_K=6
MAX_CONTEXT_CHARS=7000
```
