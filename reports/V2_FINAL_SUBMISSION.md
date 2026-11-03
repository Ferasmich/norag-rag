# RAG V2 final submission

Deadline: November 3, 2026, 11:59 a.m., Europe/Paris.

## Commits, branch and remote

- Final verified implementation SHA: `c7b4c1deee2629b67e6ffd31edf4fa92e64a23ca`.
- Branch: `main`.
- Origin: `https://github.com/Ferasmich/norag-rag.git`.
- Implementation push: **PASS**, remote main matched that exact SHA after push.
- Final release commit includes this documentation-only follow-up and is identified by annotated tag `v2`. Its exact SHA is `git rev-parse 'v2^{commit}'`; tag-object SHA is `git rev-parse v2`. A committed report cannot embed its own commit hash or the hash of a tag created afterward. The final delivery message supplies the literal verified release SHA, and the commands below reproduce both hashes from GitHub.

## Tests and security

**PASS:** 40 tests passed in 8.32 seconds; real MiniLM/Chroma with mocked hosted generation. JavaScript syntax, dependency consistency, staged diff whitespace, all result JSON/JSONL parsing and staged secret-value/path scans passed. Frozen V1 service, corpus, dataset, baseline and report compare unchanged. No Tasks files, environments, credentials or local database files staged. Kestrel documents and starter questions are byte-identical copies of supplied material.

## Evaluation completeness

**PASS:** both corpora, two configurations, k=1/3/5/10, overall/per-type HitRate and quote recall; real evidence and settings/hashes saved. Vellora Hit@5: 800/80 = 7/12 (58.3%), 500/100 = 8/12 (66.7%). Kestrel: 7/8 (87.5%) versus 6/8 (75.0%). Original V1 baseline preserved and reproduced. No universal improvement claimed.

**PASS:** Kestrel's 10 human reviews verified against downloaded and runtime results: 6 correct, 3 partial, 1 wrong, 0 unsupported; every row has notes, reviewer, timestamp and no-context flag. The original evidence and answers are unchanged. Human no-context matches: 0/10.

**PASS with explicit AI provenance:** all 15 Vellora no-context responses assessed independently at the user's request. q09/q10 provide expected facts; other flags false. Reasoning and `human_reviewed: false` provenance saved separately in `vellora-ai-assessed.json` and submission JSONL. The blank runtime human-review copy is untouched. No fabricated human identity. C7 requires hand grading of RAG answers on one question set, satisfied by Kestrel; no Vellora human review is claimed. Questions are retained and flagged.

**PASS:** shared CLI/frontend evaluator, optional validated generation/no-context, saved reviews/export, Words corpus/question modes and exact chunk counts/offsets. API tests and actual HTTP workflow passed. Two measured lexical miss diagnoses and failure-stage analysis are in REPORT and the complete implementation report.

## Limitations and submission status

**PARTIAL:** complete browser click-through and responsive visual QA unavailable after browser access timed out; native browser smoke reached tabs/corpus loading/evaluation launch, and HTTP routes passed. Owner should review/personalize the AI-assisted report and AI Vellora assessments. Exact identifiers and two-hop evidence still fail in measured cases; citation schema validity is not semantic proof. Earlier hosted errors are preserved; paced final runs completed with no recorded errors. No full Vellora RAG-answer accuracy measurement claimed.

GitHub publication is the authorized submission artifact. Submission email to the professor was **not sent**; the owner remains responsible for that action and any team-policy confirmation. No force push, upstream push or v1 modification is permitted. The release procedure publishes the documentation commit and annotated v2 tag, then verifies remote main and dereferenced tag equal local HEAD and v1 is unchanged.

```bash
git rev-parse HEAD v2 'v2^{commit}'
git ls-remote origin refs/heads/main refs/tags/v2 'refs/tags/v2^{}' refs/tags/v1 'refs/tags/v1^{}'
git status --short
```

## Run and reproduce

```bash
source .venv/bin/activate
python -m uvicorn rag_app.api:app --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000 — Evaluation and Words tabs
python -m pytest -q
python -m evaluation.v2 --corpus kestrel --output /tmp/kestrel-fresh.json
python -m evaluation.v2 --corpus vellora --chunk-size 500 --chunk-overlap 100 --output /tmp/vellora-fresh.json
```

Use fresh output paths. Optional `--answers --no-context` requires configured Gemini credentials and sends only the relevant questions/evidence, not expected answers. See README and `RAG_V2_IMPLEMENTATION_REPORT.md` for exact methodology and grading rules.
