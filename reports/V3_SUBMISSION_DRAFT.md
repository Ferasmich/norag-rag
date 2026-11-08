# Local submission draft — do not send yet

> **Superseded (8 Nov 2026, afternoon).** This file describes the first, partial v1 graph (gemini-3.5-flash-lite, 10 relations). The submitted V3 results, rebuilt graph (`concepts/kestrel_graph_v2/`) and the extraction-quality review (`evaluation/graph_review/kestrel-v2-10-triples.json`) are in REPORT.md, section "V3 final". The v1 grades below were AI-drafted reasons confirmed in chat, not an independent review.


To: [professor's submission address]

Subject: EPITA AI Systems Integration & Operations — C8 RAG V3 — Feras M

Hello,

Please find my RAG V3 work for the C8 assignment:

- Repository: https://github.com/Ferasmich/norag-rag
- Release tag: `v3` [pending owner-approved commit, tag and publication]
- Main report: `REPORT.md`; detailed V3 implementation and evaluation: `reports/V3_FINAL_INTEGRATION.md`.
- Graph and extraction audit: `concepts/kestrel_graph/`.
- Ten manually graded triples: `evaluation/graph_review/kestrel-10-triples.md` (Feras M: 10 correct, 0 wrong, 0 unsupported).
- Final retrieval runs: `evaluation/results/v3/submission_2026-11-08/`.
- Augmented Kestrel questions: `evaluation/kestrel-v3.jsonl`; original questions are preserved and q11/q12 appended.

The implementation includes persistent BM25 for both corpora, independent dense/BM25/graph retrievers, configurable RRF, and citation/path provenance. The graph is partial after the approved 43-request extraction cap; the report distinguishes original model output from nine offline context-expanded relations and documents failures and incomplete usage. The measured defaults remain dense for Kestrel and BM25 for Vellora. All 144 regression tests pass, including the four V2 dense baseline comparisons.

Best regards,
Feras M

---

This is an unsent local draft for a sole-developer submission. Confirm the recipient, final author identity and final repository/tag links before sending. The remote repository does not yet contain these uncommitted V3 changes. No commit, push, tag or email send has been authorized by approval of the extraction alone. The owner requested final-verification approval first.
