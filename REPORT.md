# RAG V1 — team and engineering report

## Team details

Feras (GitHub: Ferasmich) is the sole developer and project owner. Implementation assistance was provided by Codex. No teammates are invented. The course presentation specifies teams of two or three; individual participation still requires confirmation with the professor. The student should review and personalize these decisions and be ready to explain the entire implementation.

## Decision log

| Decision | Options | Choice and reason | Trade-off |
|---|---|---|---|
| Web stack | FastAPI + static JS; React/Next; Streamlit | FastAPI + static HTML/CSS/JS keeps the existing Python stack and uses one startup command | Manual browser state management; less component tooling |
| Pipeline reuse | Import demo directly; adapt its steps | Keep demos unchanged and implement the same MiniLM/Chroma/Gemini pipeline in a reusable service; demo import performs interactive/global initialization | Small duplication, but isolated application lifecycle and tests |
| Embeddings | Local MiniLM; hosted embeddings | Same local all-MiniLM-L6-v2 for chunks/questions, as in the course | 256-wordpiece limit; poor exact-code matching |
| Chunking | Fixed windows; paragraph/token-aware chunks | Naive 800-character windows, 80-character overlap by default; retain all source characters | Split words/facts and repeated passages; provides a measurable V1 baseline |
| Vector store | Chroma; in-memory vectors; hosted service | Local persistent Chroma cosine collections | Local disk dependency, one-process scope |
| Index identity | Filename; content hash | SHA-256 decoded content IDs; collection fingerprint includes model, algorithm, size and overlap; top-k excluded | Identical documents keep first filename; unused settings consume disk |
| Index recovery | Assume add succeeds; completion marker | Only complete per-document builds get a marker; interrupted builds are cleared and retried | Filesystem and Chroma are not a single transaction; managed application storage must stay together |
| Generation | Preserve streaming free text; validated JSON | Existing Gemini-compatible endpoint with non-streaming JSON, Pydantic validation and exact retrieved-ID membership checks | Wait for whole answer; malformed output is rejected; semantic grounding still needs human review |
| No answer | Always answer; score cutoff; model abstention | Explicit model abstention, empty citations, canonical no-evidence text | No calibrated score threshold; LLM may still misjudge relevance |
| References | Filename only; chunk IDs and offsets | Exact retrieved chunk snapshot with filename, start line and character range | Snapshot remains in prior responses even if a source later changes/deletes; UI clears displayed answer on deletion |
| Corpus | Whole novel; novel subsets; Vellora manual | Existing course manual: 31,036 characters, verifiable facts, many confusable error codes | Narrow fictional domain; not evidence of broad-domain quality |
| Evaluation | LLM grading; exact chunk IDs; source-span Hit@k | Fixed expected verbatim spans, source identity and Hit@5, isolated real retrieval | Small dataset; a split evidence span counts as miss; does not measure answer quality |
| Frontend safety | Render Markdown/HTML; plain text | Render all model/source/filename data through textContent | No rich formatting, but uploaded text cannot inject HTML into the page |
| Deployment | Hosted/multi-user; local | Loopback single-user process for V1 | No authentication or multi-worker synchronization |

## Retrieval baseline

`evaluation/questions.json` version 1 contains 15 questions: 12 answerable (four easy, four precise, four hard) and three unanswerable. Each contains expected answer and source passage/location, or explicit unanswerability. Run `python -m evaluation.run`. The evaluator uploads the unchanged course manual to an isolated temporary Chroma store, indexes once and queries all questions with the same model.

For each answerable question, a hit means at least one of the top-k chunks from the expected document contains the full expected literal evidence span. Hit@k = hit questions / answerable questions. Default settings: size 800 characters, overlap 80, k=5, MiniLM, cosine search. Actual measured **7/12 = 58.3% Hit@5**. Raw individual results and corpus/dataset hashes are in `evaluation/baseline.json`. The five misses are q01, q05, q07, q09 and q11. Unanswerable queries are excluded from this retrieval metric; their returned chunks are recorded, not labeled as correct evidence.

Keep the dataset and corpus unchanged for V2 and write new results to a separate file. Report all changed parameters. This small, manually selected dataset is a regression baseline, not a general accuracy estimate. No answer-accuracy percentage is claimed.

## Observed failures

| Test question | Expected | Observed | Possible cause | Potential V2 experiment |
|---|---|---|---|---|
| How large is the Vellora Duo water tank? | Retrieve section 2, 1.8 litres | q01 misses expected span in top 5. Live Gemini abstains, despite fact existing in the corpus | Generic water-tank passages outrank the parts list; chunk boundaries dilute the fact | Sentence/paragraph-aware chunks, compare same dataset |
| What should I do for error E47? | Retrieve the upper-burr/foreign-object procedure | q05 misses the expected span | Dense embeddings do not distinguish short exact codes reliably | Hybrid lexical/dense retrieval in V2 |
| How do I clear error E86? | Hold menu for five seconds | q07 misses expected span | Similar error-code passages and weak representation of exact digits | Measure lexical retrieval on precise-code subset |
| The grounds container was emptied but the counter has not reset. Why? | Remove for five seconds with machine on | q09 misses expected span | Long indirect query and boundary/context effects | Evaluate chunk strategy and retrieval depth |
| What preparation is needed after the machine sits unused for three weeks? | Fill water circuit after more than two weeks unused | q11 misses expected span | Temporal paraphrase and fixed-window retrieval | Query wording and chunking experiments |

Additional observed checks: the live Wi-Fi-password question correctly abstained, while the warranty question returned two years with a valid retrieved citation. Invalid generated JSON and fabricated citation IDs produced handled 502 responses in tests. These injected failures verify error handling; they are not claims that the real Gemini model exhibited those failures.

Browser visual QA was attempted: the browser inventory was empty and Safari inspection timed out. Consequently no visual or responsive-browser pass is claimed. Static routes, JS syntax and live HTTP workflow passed. No general hallucination-resistance claim follows from the two abstention samples. The exact returned evidence is retained in `reports/live_smoke.json`.

## Understanding the implementation

`rag_app/service.py` owns upload/index/retrieve/generate/validate. `rag_app/models.py` defines input and output rules. `rag_app/api.py` maps HTTP requests to service calls and safe errors. The static frontend sends selected IDs and numeric settings; it never receives API credentials. The same embedding function is attached to the Chroma collection for both add and query. At answer time, metadata filters exclude every unselected document before top-k selection. Returned references are assembled from the actual retrieval lookup rather than model-supplied source text.

A schema-valid citation is an integrity check: the cited passage was retrieved. It is not proof that the passage supports every generated claim. Opening and reading references is part of using V1 responsibly. The goal of V2 is to improve these measured retrieval misses one component at a time.
