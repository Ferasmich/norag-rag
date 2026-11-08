const $ = (id) => document.getElementById(id);
let busy = false;
function node(tag, text, cls) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function status(text, error = false) {
  $("status").textContent = text;
  $("status").className = error ? "error" : "";
}
async function api(path, options) {
  const response = await fetch("/api/" + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed.");
  return data;
}
function payload() {
  return {
    document_ids: Array.from(
      document.querySelectorAll(".doc input:checked"),
    ).map((e) => e.value),
    settings: {
      chunk_size: Number($("size").value),
      chunk_overlap: Number($("overlap").value),
      top_k: Number($("topk").value),
    },
  };
}
function retrievalConfig() {
  const retrievers = ["dense", "bm25", "graph"].filter(
    (name) => $("retriever-" + name).checked,
  );
  if (!retrievers.length) throw new Error("Select at least one retriever.");
  return {
    retrievers,
    candidate_depth: Number($("candidate-depth").value),
    rrf_k: Number($("rrf-k").value),
  };
}
function post(body) {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}
async function run(action) {
  if (busy) return;
  busy = true;
  document.querySelectorAll("button").forEach((e) => (e.disabled = true));
  try {
    await action();
  } catch (error) {
    status(error.message, true);
  } finally {
    busy = false;
    document.querySelectorAll("button").forEach((e) => (e.disabled = false));
  }
}
async function refresh() {
  const docs = await api("documents");
  const old = new Map(
    Array.from(document.querySelectorAll(".doc input")).map((e) => [
      e.value,
      e.checked,
    ]),
  );
  $("documents").replaceChildren();
  $("count").textContent = `${docs.length} sources`;
  for (const doc of docs) {
    const row = node("div", undefined, "doc");
    const label = node("label");
    const check = node("input");
    check.type = "checkbox";
    check.value = doc.id;
    check.checked = old.has(doc.id) ? old.get(doc.id) : true;
    label.append(check, node("span", doc.name));
    const del = node("button", "Delete");
    del.setAttribute("aria-label", `Delete ${doc.name}`);
    del.onclick = () =>
      run(async () => {
        await api("documents/" + doc.id, { method: "DELETE" });
        $("result").replaceChildren(
          node("p", "Source deleted. Ask again to refresh evidence."),
        );
        await refresh();
        status("Document and its indexed passages deleted.");
      });
    row.append(label, del);
    $("documents").append(row);
  }
}
$("upload").onsubmit = (event) => {
  event.preventDefault();
  run(async () => {
    status("Uploading source…");
    const form = new FormData();
    form.append("file", $("file").files[0]);
    const data = await api("documents", { method: "POST", body: form });
    await refresh();
    $("file").value = "";
    status(
      data.duplicate
        ? "This content is already in your library."
        : "Source added. It will be indexed on your next question.",
    );
  });
};
$("index").onclick = () =>
  run(async () => {
    status("Building index… The first run may download the embedding model.");
    const data = await api("index", post(payload()));
    status(
      `Index ready: ${data.chunks} passages across ${data.documents} documents.`,
    );
  });
$("ask").onsubmit = (event) => {
  event.preventDefault();
  run(async () => {
    $("result").replaceChildren(
      node("p", "Retrieving evidence and preparing an answer…", "hint"),
    );
    status(
      "Working… Indexing new sources, retrieving passages, then asking the model.",
    );
    const data = await api(
      "ask",
      post({ ...payload(), question: $("question").value, retrieval: retrievalConfig() }),
    );
    const result = $("result");
    result.replaceChildren(
      node(
        "p",
        data.answerable ? "GENERATED ANSWER" : "INSUFFICIENT EVIDENCE",
        "eyebrow",
      ),
      node("p", data.answer),
    );
    if (data.references.length)
      result.append(node("h2", "Supporting references"));
    for (const ref of data.references) result.append(reference(ref));
    const all = node("details", undefined, "reference");
    all.append(
      node(
        "summary",
        `Inspect all ${data.retrieved.length} retrieved passages`,
      ),
    );
    for (const ref of data.retrieved) all.append(reference(ref));
    result.append(all);
    if (data.retrieval_status)
      result.append(node("pre", JSON.stringify(data.retrieval_status, null, 2)));
    if (data.graph_trace) renderGraph(result, data.graph_trace, data.references);
    status("Answer validated. Open a passage to check the evidence.");
  });
};
function reference(ref) {
  const detail = node("details", undefined, "reference");
  detail.append(
    node(
      "summary",
      `${ref.name} · line ${ref.line} · characters ${ref.start}–${ref.end}`,
    ),
    node("p", ref.text),
  );
  if (ref.origins) {
    detail.append(node("small", Object.entries(ref.origins)
      .map(([name, info]) => `${name}: rank ${info.rank}, raw score ${Number(info.raw_score).toFixed(3)}`)
      .join(" · ")));
    detail.append(node("p", `Fused rank ${ref.fused_rank} · RRF ${Number(ref.rrf_score).toFixed(6)} (not a confidence score)`));
  } else if (typeof ref.similarity === "number") {
    detail.append(node("small", `Cosine similarity ${ref.similarity.toFixed(3)} · not a confidence score`));
  }
  return detail;
}
function renderGraph(host, trace, references = []) {
  const section = node("section");
  section.append(node("h3", "Graph evidence paths"));
  if (!trace.complete)
    section.append(node("p", `Partial extraction: ${trace.coverage.valid_chunks}/${trace.coverage.total_chunks} chunks validated. Missing graph facts can affect retrieval.`, "hint"));
  section.append(node("p", "Question entities: " + (trace.matched_entities.map((e) => `${e.label} (${e.matched_by})`).join(", ") || "none")));
  if (trace.reason) section.append(node("p", trace.reason));
  const cited = new Set(references.flatMap((r) => r.origins?.graph?.path_ids || []));
  const paths = cited.size ? trace.paths.filter((p) => cited.has(p.id)) : trace.paths;
  section.append(node("p", cited.size ? "Paths attached to cited references:" : "Retrieved paths (not a claim that an answer used every path):"));
  const chunks = new Map((trace.chunks || []).map((c) => [c.id, c]));
  for (const path of paths.slice(0, 20)) {
    const view = node("details", undefined, "graph-path");
    view.append(node("summary", path.nodes.map((n) => n.label).join(" → ")));
    const labels = new Map(path.nodes.map((n) => [n.id, n.label]));
    for (const edge of path.edges) {
      view.append(node("p", `${labels.get(edge.subject_id)} — ${edge.relation} → ${labels.get(edge.object_id)} (walk ${edge.direction})`));
      if (edge.condition_quote) view.append(node("p", "Condition: " + edge.condition_quote));
      for (const proof of edge.evidence) {
        const passage = node("details");
        const rank = trace.fused_chunk_ranks?.[proof.chunk_id];
        passage.append(node("summary", `${proof.source_document} · characters ${proof.quote_start}–${proof.quote_end} · ${rank ? "fused rank " + rank : "outside final fused selection"}`));
        passage.append(node("blockquote", proof.quote));
        const chunk = chunks.get(proof.chunk_id);
        if (chunk) passage.append(node("pre", chunk.text));
        view.append(passage);
      }
    }
    if (!path.edges.length) {
      view.append(node("p", "Direct entity match; source chunks:"));
      for (const id of path.chunk_ids) {
        const chunk = chunks.get(id);
        if (chunk) { const passage = node("details"); passage.append(node("summary", `${chunk.name} · ${id}`), node("pre", chunk.text)); view.append(passage); }
      }
    }
    section.append(view);
  }
  if (paths.length > 20) section.append(node("p", `Showing 20 of ${paths.length} paths.`));
  if (trace.traversal_limit_reached) section.append(node("p", "The bounded traversal reached its exploration limit."));
  host.append(section);
}
refresh().catch((e) => status(e.message, true));
api("evaluation")
  .then((data) => {
    const host = $("evaluation");
    if (data.hit_at_k !== undefined) {
      host.append(
        node(
          "h3",
          `Hit@${data.settings.top_k}: ${(data.hit_at_k * 100).toFixed(1)}%`,
        ),
        node(
          "p",
          `${data.hits} of ${data.answerable_count} answerable questions retrieved the expected passage. ${data.unanswerable_count} unanswerable questions excluded from this metric.`,
          "hint",
        ),
      );
    }
    const details = node("details");
    details.append(
      node("summary", "Dataset, settings and per-question results"),
      node("pre", JSON.stringify(data, null, 2)),
    );
    host.append(details);
  })
  .catch((e) => ($("evaluation").textContent = e.message));

// C7 evaluation and Words share the existing selected sources and settings.
document.querySelectorAll("[data-tab]").forEach((button) => {
  button.onclick = () => {
    document.querySelectorAll(".tab-view").forEach((view) => {
      view.hidden = view.id !== button.dataset.tab;
    });
    document
      .querySelectorAll("[data-tab]")
      .forEach((tab) =>
        tab.setAttribute("aria-pressed", String(tab === button)),
      );
  };
});
api("question-sets")
  .then((sets) => {
    for (const [corpus, questions] of Object.entries(sets)) {
      for (const q of questions) {
        const option = node("option", `${corpus} / ${q.id}: ${q.question}`);
        option.value = q.question;
        $("word-question").append(option);
      }
    }
  })
  .catch((e) => status(e.message, true));
$("word-question").onchange = () => {
  $("words-question").value = $("word-question").value;
  $("words-mode").value = "question";
};
$("load-corpus").onclick = () =>
  run(async () => {
    const docs = await api("corpora/" + $("eval-corpus").value, post({}));
    await refresh();
    const ids = new Set(docs.map((d) => d.id));
    document.querySelectorAll(".doc input").forEach((c) => {
      c.checked = ids.has(c.value);
    });
    const kestrel = $("eval-corpus").value === "kestrel";
    $("retriever-dense").checked = kestrel;
    $("retriever-bm25").checked = true;
    $("retriever-graph").checked = kestrel;
    status(`Corpus loaded. Measured default: ${kestrel ? "dense + BM25 + graph (Recall@5 0.76, two-hop 0.78)" : "BM25 (Recall@5 0.92)"}. You can change the retrievers above.`);
  });
function downloadJSON(data, filename) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const link = node("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function showRun(runId, result) {
  $("saved-run").value = runId;
  localStorage.setItem("rag-last-evaluation", runId);
  const host = $("eval-results");
  host.replaceChildren();
  host.append(node("p", `Saved run: ${runId}`));
  const download = node("button", "Download saved results");
  download.onclick = () =>
    run(async () =>
      downloadJSON(
        await api("evaluations/" + runId),
        `${result.corpus}-${runId}.json`,
      ),
    );
  host.append(
    download,
    node("h3", "Overall and per-type metrics"),
    node(
      "pre",
      JSON.stringify(
        {
          settings: result.settings,
          retrieval: result.retrieval,
          question_set: result.question_set,
          overall: result.overall,
          by_type: result.by_type,
        },
        null,
        2,
      ),
    ),
  );
  const rules = node("details");
  rules.append(
    node("summary", "Grading rules (read before grading)"),
    node("pre", JSON.stringify(result.grading_rules, null, 2)),
  );
  host.append(rules);
  for (const row of result.results) {
    const detail = node("details");
    detail.append(node("summary", `${row.id} / ${row.type}: ${row.question}`));
    detail.append(node("p", `Expected: ${row.expected_answer}`));
    if (row.retrieval_seconds !== undefined)
      detail.append(node("p", `Retrieval ${(row.retrieval_seconds * 1000).toFixed(2)} ms · LLM calls ${row.llm_calls}`));
    if (row.graph_trace) renderGraph(detail, row.graph_trace, row.response?.references || []);
    detail.append(
      node(
        "pre",
        JSON.stringify(
          {
            hits: row.hits,
            quote_recall: row.quote_recall,
            evidence: row.evidence,
          },
          null,
          2,
        ),
      ),
    );
    if (row.response || row.no_context_answer !== undefined) {
      if (row.response) {
        detail.append(node("p", `Generated: ${row.response.answer}`));
        for (const ref of row.response.references)
          detail.append(reference(ref));
      }
      const grading = node("form");
      const gradeLabel = node("label", "Grade");
      const grade = node("select");
      for (const value of ["", "correct", "partial", "wrong", "unsupported"]) {
        const op = node("option", value || "Choose after review");
        op.value = value;
        grade.append(op);
      }
      grade.value = row.grade || "";
      grade.required = !!row.response;
      grade.disabled = !row.response;
      gradeLabel.append(grade);
      const reviewerLabel = node("label", "Reviewer");
      const reviewer = node("input");
      reviewer.required = true;
      reviewer.maxLength = 100;
      reviewer.value = row.reviewer || "";
      reviewerLabel.append(reviewer);
      const noteLabel = node(
        "label",
        "Notes: supporting facts; retrieval or generation failure",
      );
      const notes = node("textarea");
      notes.required = true;
      notes.maxLength = 3000;
      notes.value = row.grade_notes || "";
      noteLabel.append(notes);
      const ncLabel = node(
        "label",
        "No-context answer supplies expected facts?",
      );
      const nc = node("select");
      for (const [value, label] of [
        ["", "Not reviewed"],
        ["true", "Yes — flag prior knowledge"],
        ["false", "No"],
      ]) {
        const op = node("option", label);
        op.value = value;
        nc.append(op);
      }
      nc.value =
        row.no_context_correct === null ? "" : String(row.no_context_correct);
      nc.disabled = row.no_context_answer === undefined;
      nc.required = row.no_context_answer !== undefined;
      ncLabel.append(nc);
      const save = node("button", "Save human grade");
      save.type = "submit";
      grading.append(gradeLabel, reviewerLabel, noteLabel, ncLabel, save);
      grading.onsubmit = (e) => {
        e.preventDefault();
        run(async () => {
          await api(
            `evaluations/${runId}/grades`,
            post({
              question_id: row.id,
              grade: grade.value || null,
              notes: notes.value,
              reviewer: reviewer.value,
              no_context_correct: nc.value === "" ? null : nc.value === "true",
            }),
          );
          status("Grade saved to results file.");
        });
      };
      detail.append(grading);
    }
    if (row.generation_error)
      detail.append(node("p", row.generation_error, "error"));
    if (row.no_context_answer !== undefined)
      detail.append(node("p", `Without retrieval: ${row.no_context_answer}`));
    if (row.no_context_error)
      detail.append(node("p", row.no_context_error, "error"));
    const passages = node("details");
    passages.append(node("summary", "All retrieved passages"));
    for (const ref of row.retrieved) passages.append(reference(ref));
    detail.append(passages);
    host.append(detail);
  }
}
$("evaluate").onclick = () =>
  run(async () => {
    status(
      "Running evaluation… Hosted generation, if selected, may take several minutes.",
    );
    const retrieval = retrievalConfig();
    const hosted = $("eval-answers").checked || $("eval-no-context").checked;
    if (hosted && (retrieval.retrievers.join() !== "dense" || $("eval-question-set").value !== "v2"))
      throw new Error("Hosted review supports original dense V2 only. Clear hosted options for a V3 retrieval evaluation.");
    const body = {corpus: $("eval-corpus").value, settings: payload().settings};
    const data = hosted ? await api("evaluations", post({...body, answers: $("eval-answers").checked, no_context: $("eval-no-context").checked}))
      : await api("evaluations/v3", post({...body, retrieval, question_set: $("eval-question-set").value}));
    showRun(data.run_id, data.result);
    status("Evaluation saved. Review per-question results and any errors.");
  });
$("saved-run").value = localStorage.getItem("rag-last-evaluation") || "";
$("restore-run").onclick = () =>
  run(async () => {
    const id = $("saved-run").value;
    showRun(id, await api("evaluations/" + encodeURIComponent(id)));
    status("Saved evaluation opened.");
  });
$("analyze").onclick = () =>
  run(async () => {
    const data = await api(
      "words",
      post({
        ...payload(),
        question:
          $("words-mode").value === "question" ? $("words-question").value : "",
        word: $("words-mode").value === "corpus" ? $("word-query").value : "",
      }),
    );
    const host = $("word-results");
    host.replaceChildren(
      node("p", `${data.chunk_count} chunks. ${data.counting}`),
    );
    for (const word of data.words) {
      const detail = node("details");
      detail.append(
        node(
          "summary",
          `${word.word}: ${word.present ? `${word.occurrences} occurrences in ${word.chunks} chunks` : "absent from corpus"}`,
        ),
      );
      for (const passage of word.passages) {
        const item = node("details");
        item.append(
          node(
            "summary",
            `${passage.name}: ${passage.start}–${passage.end} / ${passage.count} occurrences`,
          ),
          node("p", passage.text),
        );
        detail.append(item);
      }
      host.append(detail);
    }
    if ($("words-mode").value === "corpus") {
      const table = node("table");
      const head = node("tr");
      for (const title of ["Word", "Occurrences", "Chunks"])
        head.append(node("th", title));
      table.append(head);
      for (const row of data.vocabulary) {
        const tr = node("tr");
        const cell = node("td");
        const link = node("button", row.word, "secondary");
        link.onclick = () => {
          $("word-query").value = row.word;
          $("analyze").click();
        };
        cell.append(link);
        tr.append(
          cell,
          node("td", String(row.occurrences)),
          node("td", String(row.chunks)),
        );
        table.append(tr);
      }
      host.append(table);
    }
    status("Word analysis ready.");
  });

api("measured-runs")
  .then((names) => {
    for (const name of names) {
      const option = node("option", name);
      option.value = name;
      $("measured-run").append(option);
    }
  })
  .catch((e) => status(e.message, true));
$("open-measured-run").onclick = () =>
  run(async () => {
    const data = await api(
      "measured-runs/" + encodeURIComponent($("measured-run").value),
      post({}),
    );
    showRun(data.run_id, data.result);
    status(
      "Measured run copied to local review. Save grades and download the reviewed JSON.",
    );
  });
