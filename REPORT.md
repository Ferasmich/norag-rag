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


## V2 — C7 implementation and evaluation (review draft)

Authoritative scope: C7, Session 3, Part III, “RAG Project - V2”, 14 pages. Deadline **November 3, 2026 at 11:59 a.m.**, superseding the C6 schedule. This section is an implementation-assistant draft based on actual code and measured runs; Feras must review, personalize the decision log and enter the required hand grades. It does not claim student-authored observations or decisions. C7 page 3 asks for a student-written report; AI assistance is disclosed rather than presented as that personal work.

### Requirements and implementation evidence

| C7 page | Requirement | Implementation / verification |
|---|---|---|
| 2 | Deadline; repo submission and team details | Deadline corrected here and in README; sole developer recorded above. Submission email remains the owner’s responsibility. |
| 3 | Preserve V1 web app, references, validation, management and settings | Core service unchanged; original 30 tests retained and pass in the expanded suite; `v1` unchanged. |
| 4–5 | Two corpora, Kestrel starter 10 and own >=10 questions | Five Kestrel documents and starter JSONL copied unchanged; 15 fixed Vellora questions converted losslessly to JSONL. |
| 6 | No-context tests for both sets; flag correct answers | Exact questions sent without evidence/expected answers; real responses/errors saved. Kestrel human flags verified; all 15 Vellora flags independently AI-assessed with explicit provenance. Retain flagged questions to preserve baseline comparability. |
| 7–8 | Verification distinct from evaluation; file output, CLI + frontend; k=1/3/5/10, per type; two configurations | Shared evaluator, real MiniLM/Chroma runs, structured saved evidence and API/frontend controls; tables below. Unanswerable excluded. |
| 9 | Optional answer generation, grading rules before hand grades, best config on one set, diagnose failure stage | Kestrel 800/80 k=10 selected; actual responses saved with references and prior grading rubric. Manual grade/notes/reviewer controls and JSON export; all 10 Kestrel human reviews verified. |
| 10–11 | Words tab, corpus and question modes; choices justified | Case-folded Unicode token counts, chunk frequency, word-to-chunk counts, typed/picked questions, absence detection; exact offsets and selection maintained. |
| 12 | Diagnose two retrieval misses, one per corpus | Same Words analysis function as UI; actual counts and target chunks in `reports/words_diagnoses.json`, discussed below. |
| 13 | Unlinked `/about` page with prescribed large text | Dedicated HTML route; HTTP content and absence of frontend link checked in tests. |
| 14 | JSONL sets + no-context, results + grades, README/REPORT, v2 tag | Sets and real evaluation artifacts provided; Kestrel hand grades verified; Vellora no-context AI assessments complete. Git release status is recorded in the implementation report. |

### V2 decisions for owner review

| Choice | Options considered | Implemented choice and rationale | Limitation |
|---|---|---|---|
| Retrieval experiment | Chunk parameters; hybrid/reranker/new model | Compare 800/80 with 500/100 using existing dense retrieval. C7 requires configuration comparison, not a specific advanced retriever. | Improvement depends on corpus and k. |
| Metric | Any expected quote; require every quote | HitRate uses any source/quote, plus macro quote recall for all quotes; Vellora’s single-quote metric matches V1 exactly. | Partial multi-hop evidence can count as a hit; recall and hand grading expose this. |
| Corpora | Replace Vellora; retain it with Kestrel | Preserve V1 corpus and all 15 questions; add unchanged Kestrel files/questions outside ignored Tasks. | Vellora type labels are mapped from V1 difficulty; not all six challenge types are present, so Kestrel is used for answer grading. |
| Evaluation entry points | Separate implementations; shared app service | CLI and frontend share the evaluator and RagService; no HTTP hop inside the evaluator. Public `/api/search` exposes the identical retrieval path for direct inspection. | Runs synchronously; browser generation may take minutes. |
| Words | Stemmed/stopword-filtered NLP; literal words | Python regex, case folding, internal hyphens/apostrophes; no new dependency or linguistic normalization. | Inflections differ; chunk boundaries can split a word; overlap repeats occurrences. |
| Grading | Automatic LLM judge; human grades | Save expected answers, real responses and references; owner records grades and no-context flags. | No answer-accuracy result until owner completes review. |
| Hosted failure handling | Abort and lose results; save per-question errors | Retain successes and handled errors. A paced rerun spaces calls six seconds apart. | No general retry/rate-limit framework; provider availability remains external. |

### Measurement method and actual configuration comparison

All four retrieval runs use the unchanged `all-MiniLM-L6-v2` CPU embeddings, cosine Chroma search and exact source substrings. Separate temporary indexes avoid cross-corpus contamination. Kestrel includes both billing policies, not just the files with expected quotes. Only chunk size and overlap change: A=800/80 (720-character step), B=500/100 (400-character step). Query once at depth 10, score prefixes 1/3/5/10. The app default answer top-k stays 5; optional Kestrel grading uses 10 because the best measured retrieval completeness is at that depth. Metric and dataset/corpus SHA-256 hashes, settings, timestamps and exact evidence snapshots are stored in each JSON result.

V1 is reproduced: `evaluation/baseline.json` and original dataset/corpus remain byte-for-byte intact, **7/12 Hit@5** at 800/80. V2 Vellora at 800/80 is also **7/12**. At 500/100 it is **8/12**, an increase of one question / 8.3 percentage points. This is an in-dataset comparison, not a general accuracy claim. At k=1, 500/100 is worse on both corpora; Kestrel also becomes worse at k=5. No universal improvement is claimed and no global default is silently changed.

### Kestrel: measured retrieval

| Chunk size / overlap | Type | Answerable | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Quote recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| 800 / 80 | overall | 8 | 62.5% | 87.5% | 87.5% | 100.0% | 93.8% |
| 800 / 80 | exact_identifier | 2 | 50.0% | 50.0% | 50.0% | 100.0% | 100.0% |
| 800 / 80 | not_in_corpus | 0 | excluded | excluded | excluded | excluded | excluded |
| 800 / 80 | split_answer | 2 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 800 / 80 | two_hop | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 50.0% |
| 800 / 80 | versions | 2 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 800 / 80 | vocabulary | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 500 / 100 | overall | 8 | 25.0% | 75.0% | 75.0% | 100.0% | 90.6% |
| 500 / 100 | exact_identifier | 2 | 0.0% | 50.0% | 50.0% | 100.0% | 100.0% |
| 500 / 100 | not_in_corpus | 0 | excluded | excluded | excluded | excluded | excluded |
| 500 / 100 | split_answer | 2 | 50.0% | 100.0% | 100.0% | 100.0% | 87.5% |
| 500 / 100 | two_hop | 1 | 0.0% | 0.0% | 0.0% | 100.0% | 50.0% |
| 500 / 100 | versions | 2 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 500 / 100 | vocabulary | 1 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

Per-question Hit@5 (fixed questions):

| Question | Type | 800/80 | 500/100 | Change |
|---|---|---|---|---|
| q01 | exact_identifier | False | False | unchanged |
| q02 | exact_identifier | True | True | unchanged |
| q03 | split_answer | True | True | unchanged |
| q04 | split_answer | True | True | unchanged |
| q05 | vocabulary | True | True | unchanged |
| q06 | versions | True | True | unchanged |
| q07 | versions | True | True | unchanged |
| q08 | two_hop | True | False | worse |
| q09 | not_in_corpus | None | None | excluded |
| q10 | not_in_corpus | None | None | excluded |

### Vellora: measured retrieval

| Chunk size / overlap | Type | Answerable | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Quote recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| 800 / 80 | overall | 12 | 50.0% | 50.0% | 58.3% | 83.3% | 83.3% |
| 800 / 80 | exact_identifier | 4 | 50.0% | 50.0% | 50.0% | 75.0% | 75.0% |
| 800 / 80 | not_in_corpus | 0 | excluded | excluded | excluded | excluded | excluded |
| 800 / 80 | other | 4 | 75.0% | 75.0% | 75.0% | 75.0% | 75.0% |
| 800 / 80 | vocabulary | 4 | 25.0% | 25.0% | 50.0% | 100.0% | 100.0% |
| 500 / 100 | overall | 12 | 16.7% | 50.0% | 66.7% | 83.3% | 83.3% |
| 500 / 100 | exact_identifier | 4 | 0.0% | 25.0% | 50.0% | 50.0% | 50.0% |
| 500 / 100 | not_in_corpus | 0 | excluded | excluded | excluded | excluded | excluded |
| 500 / 100 | other | 4 | 50.0% | 75.0% | 100.0% | 100.0% | 100.0% |
| 500 / 100 | vocabulary | 4 | 0.0% | 50.0% | 50.0% | 100.0% | 100.0% |

Per-question Hit@5 (fixed questions):

| Question | Type | 800/80 | 500/100 | Change |
|---|---|---|---|---|
| q01 | other | False | True | improved |
| q02 | other | True | True | unchanged |
| q03 | other | True | True | unchanged |
| q04 | other | True | True | unchanged |
| q05 | exact_identifier | False | False | unchanged |
| q06 | exact_identifier | True | True | unchanged |
| q07 | exact_identifier | False | False | unchanged |
| q08 | exact_identifier | True | True | unchanged |
| q09 | vocabulary | False | True | improved |
| q10 | vocabulary | True | True | unchanged |
| q11 | vocabulary | False | False | unchanged |
| q12 | vocabulary | True | False | worse |
| q13 | not_in_corpus | None | None | excluded |
| q14 | not_in_corpus | None | None | excluded |
| q15 | not_in_corpus | None | None | excluded |


### Two misses diagnosed through Words

These counts were computed using `rag_app.words.analyze`, the exact function used by the Words UI, on 800/80 windows. Both corpora yield 43 nonempty chunks. Full source excerpts, word counts and exact offsets are in `reports/words_diagnoses.json`.

**Kestrel q01, KST-2044:** no expected quote in top 5; a quote appears by top 10. The identifier `kst-2044` occurs exactly once in one chunk, including the expected answer window (`support-runbook-errors.md`, characters 2880–3680). Generic `error` occurs in nine chunks, `it` in 21 and `and` in all 43. `mean`, `i` and `fix` are absent from the whole corpus. The answer-bearing chunk has the unique identifier but not the generic query terms `what`, `does`, `error`, `how`, `do`. Interpretation for review: the dense model may underweight exact digits and prefer semantically similar error-code neighbors. These counts establish an inspectable lexical distinction; they do not prove the embedding’s internal cause. At 500/100 q01 still misses at k=5.

**Vellora q11, preparation after three weeks unused:** no expected quote in top 5 in either configuration. `sits` and `unused` are absent from the corpus; `machine` occurs in 38/43 chunks, `the` in all 43, and `for` in 26. The expected answer window (characters 2880–3680) contains `three` and `weeks`, but not `preparation`, `needed` or `after`. `weeks` occurs in just one chunk. Interpretation for review: common wording is nondiscriminative while the question’s temporal/paraphrase relation is hard for a single dense query. Lexical presence alone cannot resolve “three weeks” versus the source’s “longer than two weeks” condition. No query rewriting or lexical retrieval is introduced in this assignment.

### Known failures and verification limits

The exact-code q05/q07 and temporal q11 Vellora misses persist at k=5. The smaller windows gain q01/q09 but lose q12; Kestrel loses the q08 two-hop Hit@5. At k=10 all Kestrel answerable questions have at least one quote, but quote recall remains below 100%, so complete evidence is still missing for some questions. Hand grades must distinguish partial answers, wrong answers and unsupported claims. Unanswerable retrieval candidates are not judged relevant and do not enter retrieval denominators.

Hosted generation/no-context errors in initial real runs are preserved, not graded as model mistakes. Real no-context replies can abstain, give generic advice or match parts of an expected answer; only owner review can establish which to flag. Questions will be retained rather than replaced, with flags disclosed. The implementation agent has not supplied fictional manual grades or claimed student review. The completed owner answer-quality evaluation is recorded below.

V1’s limitations still apply: embeddings may truncate inputs, schema-valid references do not prove semantic support, no calibrated relevance threshold, no authentication/multi-user isolation. Words returns all matches for course-sized corpora and can become large. Browser automation reached the V2 page, tab controls and Kestrel load; an evaluation POST returned 200, but final result rendering and Words click-through were not fully observed after browser access timed out. No complete interactive browser pass is claimed. API workflow and deterministic generation validation are covered by the expanded tests; precise test commands/results are recorded in `reports/RAG_V2_IMPLEMENTATION_REPORT.md`.

Future versions may investigate lexical retrieval for exact codes and safer/ measured generation, but hybrid search, reranking, logging/cost dashboards and new guardrails were not implemented here.


### Hosted-run completion and owner review

Paced Kestrel 800/80/k=10 completed with 10/10 validated generated responses and 10/10 no-context responses, no recorded hosted errors (`evaluation/results/kestrel-answers-no-context-paced.json`). Paced Vellora no-context completed 15/15, also without recorded hosted errors (`evaluation/results/vellora-no-context-paced.json`). Earlier incomplete attempts remain in their original result files. The two `*-no-context.jsonl` result sets include the original questions, expected evidence and actual no-context responses. Kestrel correctness flags reflect the verified owner review; Vellora flags now reflect separately saved AI assessments, not fabricated human reviews.

The owner completed Kestrel grades. Reviewable local runs are `1b9aa8d25e274a92ba0079d58cae8ec9` (Kestrel, grade plus prior-knowledge flag) and `b9fecacce5e04d46a1b8388a942dd191` (Vellora, prior-knowledge flags). The verified Kestrel grade table appears below. Final commit/push/v2 tag status is recorded in the submission report.


## Verified Kestrel human review

The owner completed all 10 Kestrel reviews as **Feras M**. The downloaded `kestrel-1b9aa8d25e274a92ba0079d58cae8ec9.json` exactly matches the runtime saved run. Each row has a valid grade, nonblank notes/reviewer, review timestamp and explicit boolean no-context correctness flag. Every non-review field compares unchanged against the original paced run. The reviewed artifact is `evaluation/results/kestrel-reviewed.json`; the Kestrel no-context JSONL also includes these human flags and review metadata.

| Question | Human grade | Human diagnosis / rationale |
|---|---|---|
| q01 | correct | Exact identifier, both allowlist IPs and intermittent-failure explanation supported. |
| q02 | partial | Omits the invitation's seven-day validity condition despite retrieved evidence. |
| q03 | correct | Current-policy retries, grace period, read-only state and suspension supported. |
| q04 | partial | Omits testing with one Admin and enforcing SSO for all members. |
| q05 | partial | Omits identity-verification duration of up to two business days. |
| q06 | correct | Owner accepts the supported comparison of old/current refund policies. |
| q07 | correct | Owner accepts the supported current/old annual discounts. |
| q08 | wrong | Abstains on an answerable question; necessary Messaging team evidence missing from retrieval. |
| q09 | correct | Correctly abstains on unavailable mobile-app information. |
| q10 | correct | Correctly abstains on unavailable seat pricing. |

Totals: **6 correct, 3 partial, 1 wrong, 0 unsupported** across 10 questions. Both unanswerable questions abstain correctly. These are the owner's judgments on this small fixed set, not agent-assigned grades or a general accuracy estimate. All 10 Kestrel no-context flags are false according to the owner. No questions are replaced. The full original notes remain in the reviewed JSON rather than being rewritten by the agent.

Vellora no-context assessment is now complete with transparent AI provenance; see the final assessment section below. No human Vellora review is claimed.


## Final no-context assessment and comparison

At the owner's explicit request, Codex independently assessed all 15 saved Vellora no-context responses against the expected answers and corpus evidence. These are **AI assessments, not human reviews**. They are saved separately in `evaluation/results/vellora-ai-assessed.json` with per-question reasoning, assessor, timestamp and `human_reviewed: false`; the original model responses and runtime review file are preserved. The submission-format `vellora-no-context.jsonl` includes the same transparent assessment provenance. No human reviewer identity was invented.

Vellora flags: **q09 and q10 true; the other 13 false**. q09 supplies both removal for at least five seconds and machine-on conditions. q10 gives no dishwasher and explains loss of necessary lubricant. q04 offers a generic two-month interval for another filter but not the required 50-litre condition, so it is false. q13–q15 do not provide factual answers; abstentions never count as prior knowledge. All questions are retained, with these flags disclosed rather than silently replaced. Owner review of the AI judgments remains recommended, but C7 explicitly requires hand grading of generated RAG answers on one set; the existing Kestrel human review fulfills that requirement.

RAG versus no-context: on Kestrel, the owner rated 6/10 RAG answers correct, 3 partial and 1 wrong (0 unsupported), while none of the 10 no-context responses supplied the expected facts. The wrong q08 is primarily a retrieval failure: team/channel evidence is missing; q02/q04/q05 omit details despite available source information, indicating incomplete generation. The owner's acceptance of q06/q07 is preserved exactly, despite the previously noted version-selection caveat. Vellora has 2/15 no-context expected-fact matches according to AI assessment; it was evaluated for retrieval at all four k values, but no comprehensive Vellora RAG-answer quality percentage is claimed. Retrieval hits are not answer grades. The small question sets and mixed human/AI assessment provenance limit broader conclusions.

Verification limitations: complete responsive/browser click-through remains unverified; browser smoke and HTTP workflows are recorded above. Student report personalization and the actual submission email remain owner responsibilities. No V3/V4 functionality was added.
