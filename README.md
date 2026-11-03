# Evidence — EPITA RAG V2

A local web application for asking questions about selected `.txt` and `.md` documents, with inspectable evidence. V2 retains V1’s fixed character windows and dense cosine retrieval, and adds the C7 two-corpus evaluation and Words analytics. Assignment deadline: **November 3, 2026, 11:59 a.m.**

## Features and architecture

The browser manages sources, chunk size, overlap, top-k, questions, answers, references and baseline results. FastAPI serves both the API and static frontend; there is no separate frontend build or Node runtime requirement. Pydantic validates requests and generated JSON. Local `all-MiniLM-L6-v2` embeddings feed a persistent Chroma cosine index. Gemini generates answers through the existing OpenAI-compatible integration.

Indexing: UTF-8 source → overlapping character windows → local embeddings → Chroma text/vector/metadata.
Answering: selected source IDs + question → same embedding model → filtered top-k search → Gemini → schema/citation validation → answer and source passages.

The original `ask_rag_google.py` and `ask_norag_google.py` course demos remain available and use their original `chroma_db/` store. The web application uses a separate ignored `data/` directory.

## Installation

Use Python 3.12 (the tested version), a writable local filesystem and internet access for the first MiniLM download. A Google API key with access to the selected Gemini model is needed for generation; retrieval evaluation does not require it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env`. Optional `GOOGLE_MODEL` defaults to the course demo's `gemini-3.5-flash-lite`; model availability depends on your account. Optional `RAG_DATA_DIR` changes the application data directory. Environment variables take precedence over `.env`. Never commit keys. The first embeddings run downloads the public MiniLM model; later runs can reuse its local cache.

`requirements-lock.txt` records the exact tested macOS/Python 3.12 environment, including transitive dependencies. Use `pip install -r requirements-lock.txt` for matching environments; use the smaller requirement files on other platforms.

## Start backend and frontend

From the repository root:

```bash
source .venv/bin/activate
python -m uvicorn rag_app.api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. The same process serves the frontend. API documentation is at `/docs`. Use one worker: document/index operations are protected by a process-local lock. This is a local single-user homework application.

1. Upload `docs/vellora-duo-espresso-machine.txt` with **Upload source**. Uploaded sources persist across restarts. Each file is limited to 5 MiB and must contain nonempty UTF-8 text.
2. Check one or more documents. New uploads are selected automatically. Identical content is deduplicated even with a different filename; the first filename is kept.
3. Set chunk size (100–1,000 characters), overlap (0 to size minus one), and top-k (1–20).
4. Click **Build selected index**, or ask a question to build missing indexes automatically. New size/overlap settings use a new index. Top-k reuses existing embeddings.
5. Try **How long is the Vellora Duo warranty?** The measured live answer was two years from purchase.
6. Open a supporting reference to see its exact retrieved text, filename, start line and character offsets. Offsets are zero-based, end exclusive, in decoded source characters. The UI also exposes every retrieved passage, including on abstentions.
7. Delete removes the uploaded source and its chunks from every web-app index. It does not delete the original repository file in `docs/`.

## Corpus and evaluation

The baseline uses the unchanged 31,036-character Vellora Duo manual supplied in the professor's repository. Its header identifies it as a fictional course document: the application did not invent it. It spans 11 sections and many similar error codes, making both answers and retrieval failures verifiable. The two existing Crime and Punishment files remain available for uploads, but the shortened/full copies are excluded from the baseline to avoid duplicate evidence. Corpus origin: `https://github.com/bcs-s5-2026/norag-rag`, `docs/vellora-duo-espresso-machine.txt`.

```bash
python -m evaluation.run
# For comparisons, preserve baseline.json and write another file:
python -m evaluation.run --top-k 3 --output evaluation/comparison.json
```

The evaluator creates an isolated temporary corpus/index and uses real embeddings and Chroma, never an LLM. `evaluation/questions.json` freezes 15 questions: 12 answerable (4 easy, 4 precise, 4 hard), 3 unanswerable. Each answerable question has an expected answer, source, section and verbatim evidence span. **Hit@k** is the fraction of answerable questions where a retrieved chunk from the expected document contains the full expected span. Matching spans rather than chunk IDs allows different chunk sizes to be compared, though split spans can still count as misses. V2 retains this dataset/corpus and records settings and hashes.

Measured V1: **Hit@5 = 7/12 = 58.3%** at size 800, overlap 80. `evaluation/baseline.json` contains settings, hashes and individual results. Three unanswerable queries are retrieved and recorded but excluded from Hit@k: nearest-neighbor search always returns candidates, so that metric cannot establish abstention accuracy. The frontend displays the saved baseline, independently of current library selection/settings.

## Tests

```bash
python -m pytest -q
node --check rag_app/static/app.js  # optional JavaScript syntax check
```

The original 30 V1 tests use real MiniLM/Chroma; hosted generation is stubbed to test valid and invalid structured output predictably. Tests cover upload, deduplication, deletion across configurations, selection, offsets, 384-dimensional embeddings, retrieval, settings, schema/citation integrity, abstention normalization, missing credentials and simulated embedding/database/API failures. A separate actual Gemini HTTP smoke test is recorded in `reports/live_smoke.json`; it includes a supported answer and two abstentions. Automated tests do not call the paid API.

## V2: two corpora, evaluation and Words

C7’s V2 specification requires measurement and word exploration, without prescribing hybrid retrieval or a new embedding model. The original service, embedding model, upload/delete/selection, settings and citation validation are retained. No new dependencies.

The five unchanged course Kestrel documents are distributed under `docs/kestrel/`; the unchanged 10 starter questions are in `evaluation/kestrel.jsonl`. Vellora remains the own corpus: `evaluation/vellora.jsonl` converts the 15 frozen V1 questions to Kestrel JSONL format without changing their questions, expected answers or passages. Both corpora are fictional course material. `/Tasks/` contains local source material and is ignored, including its PDFs; the runtime needs none of it.

From **Evaluation**, choose a corpus and click **Load corpus into library** to upload and select just that corpus. Choose chunk settings, then **Run evaluation**. Evaluation uses an isolated temporary index, independently of the current library selection, so extra uploads cannot contaminate the benchmark. Metrics always use k=1/3/5/10. Top-k controls evidence for optional answer generation. The default run retrieves only; the answer checkbox generates validated answers, and the no-context checkbox asks the exact questions with no documents, expected answers or evidence supplied to Gemini. Hosted calls are spaced six seconds apart; a run may take several minutes. API failures are saved per question rather than scored as wrong answers.

Terminal equivalents (use fresh output filenames; existing files are refused):

```bash
python -m evaluation.v2 --corpus kestrel --output evaluation/results/kestrel-new-800-80.json
python -m evaluation.v2 --corpus vellora --chunk-size 500 --chunk-overlap 100 --output evaluation/results/vellora-new-500-100.json
python -m evaluation.v2 --corpus kestrel --top-k 10 --answers --no-context --output evaluation/results/kestrel-new-answers.json
python -m evaluation.v2 --corpus vellora --no-context --output evaluation/results/vellora-new-no-context.json
```

Measured 800/80 → 500/100 Hit@5: **Vellora 7/12 (58.3%) → 8/12 (66.7%)**; **Kestrel 7/8 (87.5%) → 6/8 (75.0%)**. Smaller chunks help some facts and hurt others; the app defaults remain 800/80. Kestrel’s best measured depth is k=10 with 800/80, used for the answer-review run. See REPORT for all depths, types and question differences. A hit means at least one expected literal quote occurs in a retrieved chunk from the expected source. Quote recall also measures the fraction of all expected quotes found; this exposes partial evidence in split/two-hop questions. Unanswerable questions are excluded from both retrieval metrics. A hit does not prove the generated answer is correct.

Use **Open measured run for review** to open a saved CLI/project result in a local grading copy. Expand questions, read expected answers, generated answers and references, then enter your reviewer name, grade, notes and no-context correctness flag. Grading rules are saved before grading and shown in the page. **Save human grade** writes to the same local results JSON. **Download saved results** exports that reviewed file for the project; copy it to `evaluation/results/` with a new filename. A no-context flag is true only when the answer supplies the expected facts; an abstention is not prior knowledge. Preserve questions even if flagged and discuss them in the report. No manual grades are fabricated by the implementation agent.

Browser runs are saved under ignored `data/evaluations/`. Copy the displayed run ID to reopen after a restart, or use the measured-run selector for committed evidence. Local evaluations are single-user synchronous requests; keep the tab open until completion. The saved V1 baseline panel remains available.

The **Words** tab operates on checked library documents and current chunk size/overlap, without an LLM or embeddings. Corpus mode lists words, their occurrence counts and number of containing chunks; enter or click a word to inspect exact matching passages/counts. Question mode accepts typed text or a question-set selection and shows present and absent words and matching chunks. Tokenization case-folds Unicode words, keeps hyphenated identifiers and internal apostrophes together, and uses no stemming or stopword removal. Counts refer to chunk windows, so overlap duplicates count again. Question words are shown once each. Exact source offsets are preserved. Large uploaded corpora can produce large tables; intended for the course-sized corpora.

V2 API additions: `POST /api/search`, `/api/words`, `/api/corpora/{corpus}`, `/api/evaluations`; `GET /api/question-sets`, `/api/measured-runs`, `/api/evaluations/{run_id}`; and result-review routes documented in `/docs`. The CLI and API call the same application retrieval service and evaluation function. The C7-required standalone `/about` page is deliberately unlinked from the frontend.

## Controlled failures and limitations

Bad uploads/settings/questions produce 400/413/422 errors; missing documents produce 404; embedding/storage/API failures produce a safe 503; invalid generated JSON or citations produce 502. The UI displays the message and restores controls. Invalid model output is rejected rather than repaired or shown. Failed indexes have no completion marker and are cleared/rebuilt on retry.

V1 dense retrieval misses some exact codes and boundary-spanning facts. There is no calibrated relevance threshold; evidence sufficiency is decided by the prompted LLM. Schema validation checks structure and citation membership, not whether every factual statement follows from a cited passage. Prompt instructions reduce but cannot eliminate hallucination or document prompt injection. Long/token-dense chunks may be truncated by MiniLM's 256-wordpiece limit even within the character cap. Similarities are not probabilities.

This release has no authentication, multi-user isolation, background indexing, hybrid search, reranking, cost dashboard or V3–V4 features. Source passages and the question are sent to Google for generation. It is intended for loopback use with non-sensitive course material. Single-process indexing serializes retrieval; first indexing can take time. Old parameter-specific indexes remain until their documents are deleted. Browser smoke testing reached the V2 tabs and corpus loading; full result rendering, Words interaction and responsive layout remain unverified after browser access timed out. API workflows and JS syntax were verified.

See `REPORT.md` for decisions and observed failures, `reports/RAG_V2_IMPLEMENTATION_REPORT.md` for current delivery evidence and Git status, and the preserved V1 report for historical evidence.


Kestrel's completed owner review is saved in `evaluation/results/kestrel-reviewed.json`: 6 correct, 3 partial, 1 wrong; all 10 no-context flags false. All 15 Vellora no-context responses are AI-assessed with explicit provenance; q09 and q10 supply the expected facts. No human Vellora review is claimed. See the latest review-status section in REPORT for release readiness.
