# My Understanding and Learning Experience — RAG Project V1 to V4

**Student:** Feras Michael  
**Course:** AI Systems Integration and Operations  
**Professor:** Denis Amselem  
**Project:** norag-rag  
**Team:** Feras Michael and Makuochukwu Okoene

## 1. Introduction — From Basic Search to RAG

My understanding of RAG started with a simple question: how does a computer find the correct information inside a document?

Imagine we have a large PDF file containing hundreds of pages. We want to ask a question and receive the exact information from that document.

The simplest approach is keyword search. We can use Python or another programming language to search for a particular word or sentence in a PDF or text file.

For example, if we search for "payment", the program can find passages containing that word.

But I understood that this has a limitation. A user may ask a question using different words from those written in the document. The document might say "payment deadline", while the user asks "When is the money due?"

A basic exact-word search may miss the correct information because the wording is different.

The next question for me was how to make the computer understand similar meanings rather than only matching exact words.

We could try using dictionaries, synonyms or other language-processing techniques. But human language is complicated. The meaning of a word also depends on the sentence where it appears.

For example, the word "work" can mean different things in "I work at a company" and "This solution does not work."

This made me interested in semantic search.

Instead of finding only matching words, semantic search attempts to find text that has a similar meaning to the user's question.

This is one of the important ideas behind Retrieval-Augmented Generation (RAG).

RAG combines information retrieval with a language model. The retrieval system searches the documents, and the language model uses the retrieved passages to generate an answer.

I also learned that sending an entire document to an LLM every time is not necessarily efficient. More text means more tokens, more computation, potentially higher cost, and often higher latency.

Even if the complete document fits inside the context window, the model may overlook relevant information.

RAG attempts to reduce this problem by selecting useful passages before generation. However, this introduces another problem: the retrieval system must actually find the correct passages.

## 2. Embeddings, Vectors and Cosine Similarity

### From text to numbers

Another question I studied was how a computer can compare the meaning of sentences mathematically.

Initially, I thought about assigning an ID to every word, sentence or document. IDs are useful for identifying information, but they do not explain whether two sentences have similar meanings.

For example, sentence number 100 is not necessarily more similar to sentence number 101 than sentence number 500.

We need a different representation.

This is where I learned about embeddings.

An embedding model converts text into a numerical vector. The vector contains many numbers, and the model is trained to represent useful semantic relationships in that numerical space.

For example, a sentence can become a vector with hundreds of dimensions.

I first tried to understand this using two-dimensional vectors because they are easier to imagine.

In a two-dimensional space, we can draw two vectors and measure the angle between them.

Cosine similarity compares the direction of the vectors.

The formula is:

\[
\operatorname{cosine}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}
\]

The dot product compares their components, while the denominator accounts for their lengths.

If two nonzero vectors point in exactly the same direction, their cosine similarity is 1.

If they are perpendicular, the similarity is 0.

If they point in opposite directions, the similarity is -1.

During my practice, I used the example of vectors [1, 2] and [10, 20]. Their magnitudes are different, but they point in the same direction, so the cosine similarity is 1.

The same mathematical principle works in 384 or more dimensions, even though we cannot easily draw those vectors.

I understood that cosine similarity does not read sentences like a human. It compares their numerical representations, while the quality of those representations depends on the embedding model.

### Chunking and overlap

Before generating embeddings, we need to decide how to divide a document.

A PDF might contain thousands of sentences. We could create one embedding for the entire document, but that would make it difficult to locate individual facts, and embedding models have input-length limits.

Instead, we divide the document into smaller pieces called chunks.

In our V1 implementation, the default chunk size was 800 characters, with an overlap of 80 characters.

This introduced another important problem: a sentence or fact might be split between two chunks.

For example, one chunk could contain the beginning of a procedure, and the next chunk could contain the actual instruction.

If we retrieve only the first chunk, we might lose the important answer.

Overlap helps by repeating some content between neighbouring chunks. It reduces the risk of losing information at a boundary.

However, overlap does not completely solve the problem. It also creates repeated information and can increase the number of stored chunks.

I learned that chunk size and overlap are engineering parameters that should be measured rather than selected randomly.

### What V1 taught me

Our first retrieval baseline used a local MiniLM embedding model, Chroma for vector storage, cosine similarity and top-k retrieval.

The V1 evaluation produced 7 successful evidence hits out of 12 answerable questions at top-5.

That is approximately 58.3% HitRate@5.

This result helped me understand something important: an application can run without errors while still failing to retrieve the information the user needs.

A working user interface does not prove that the underlying RAG system is accurate.

## 3. BM25, Hybrid Retrieval and RRF

### Why semantic search is not enough

After learning cosine similarity, I started to understand its weaknesses.

Dense semantic search is useful when the question and the document have different wording but similar meanings.

However, it can struggle with exact identifiers.

For example, a document may contain an error code such as KST-2044. The user asks about that exact error, but semantic search could rank another error passage more highly because both passages discuss similar technical problems.

This is dangerous because one character in an error code can completely change the meaning of the answer.

That is why I studied BM25.

BM25 is a lexical ranking algorithm. It considers whether query terms appear in a document, how frequently they occur, how common they are across the collection, and the length of the document or passage.

It is particularly useful for technical keywords, exact identifiers and specialised terminology.

Dense retrieval and BM25 solve different problems.

Dense retrieval is strong at semantic similarity. BM25 can be stronger when the exact words are important.

This led me to hybrid retrieval.

### Combining retrieval methods

Instead of depending on only one retrieval algorithm, a system can run multiple retrieval methods and combine their results.

In our V3 project, we used dense retrieval, BM25 and Concept Graph retrieval.

Each method returns a ranking of relevant chunks.

But cosine similarity scores and BM25 scores are calculated differently. We cannot simply assume that a BM25 score of 10 is directly comparable to a cosine similarity of 0.8.

One approach is to combine the rankings rather than their raw scores.

That is the purpose of Reciprocal Rank Fusion (RRF).

Its formula is:

\[
RRF(d)=\sum_{r\in R}\frac{1}{k+\operatorname{rank}_r(d)}
\]

Here, each retriever contributes according to the position of a document or chunk in its ranking.

A chunk that ranks highly across different retrievers can receive a strong combined score.

I understood that RRF does not make the retrieval systems smarter. It combines the evidence provided by their rankings.

A weak or irrelevant ranking can still affect the result.

### What our evaluations showed

Our V3 results demonstrated that different corpora can benefit from different retrieval configurations.

For the Kestrel corpus, dense retrieval reached Recall@5 of 0.75, while the combination of dense, BM25 and Concept Graph reached 0.79 on the V3 question set.

However, this did not mean the hybrid configuration was better on every question.

Some questions improved, especially multi-hop questions, while other questions became worse.

For Vellora, BM25 alone reached approximately 0.92 Recall@5, compared with approximately 0.58 for dense retrieval.

This was an important result because a simpler lexical retriever performed better on that particular technical manual.

I learned that the best algorithm depends on the documents, the questions and the required evidence.

### What about reranking?

I also studied reranking as another possible step after retrieval.

A retriever can quickly return candidate chunks. A reranker then examines the question and each candidate more closely to reorder them.

This can improve relevance, but it also increases processing time and computational cost.

Our confirmed school implementation uses RRF rather than a separate reranking model.

I consider reranking an interesting future experiment, particularly for comparing retrieval quality before and after an additional ranking stage.

## 4. Concept Graphs, Evaluation and Answer Generation

### Concept Graphs

One of the more advanced parts of our project is the Concept Graph.

Instead of representing information only as independent text chunks, a graph represents entities and the relationships between them.

For example:

An error code may be associated with a service.

The service may be owned by a team.

The team may have a particular communication channel.

A question might require following all these relationships to produce the correct answer.

This is different from searching only for a similar paragraph.

In our project, the graph is built from relationships extracted from document chunks, with evidence from the original sources.

One thing I found important is source provenance.

A graph relationship should not be accepted merely because the LLM generated it. We need to verify that the relationship is supported by the original text.

The Concept Map Explorer allows us to inspect these relationships and their supporting passages.

I also learned that graph retrieval can fail when the correct entity is not recognised or when an important relationship was never extracted.

Adding a graph does not guarantee a better answer.

### Evaluating retrieval

The next question is how we know whether the retrieved results are correct.

For this, we need a set of questions with expected supporting passages.

One metric is HitRate@K.

This tells us whether the first K retrieved chunks contain at least one expected supporting passage.

Another metric is Recall@K.

Recall becomes especially useful when an answer requires several pieces of evidence.

For example, a question may require information from three separate passages.

If we retrieve only one of them, we may have a hit but still lack the complete evidence required to answer.

I also practised precision and recall using examples of correct and incorrect classifications.

One important lesson was that high precision does not automatically mean high recall.

A system may be correct about the evidence it identifies while still missing many relevant passages.

### Evaluating generated answers

Retrieval evaluation and answer evaluation are different.

A correct passage can be retrieved, but the LLM may still produce an incomplete or incorrect answer.

Likewise, an LLM might generate an answer that sounds reasonable even though the documents do not support it.

During my practice, I studied hallucinations, faithfulness and Natural Language Inference (NLI).

NLI considers whether one statement is supported by another, contradicted by it, or not determined by it.

This helped me understand why we should compare an answer against the retrieved source evidence rather than judge it only by how convincing it sounds.

Human review is also important.

In our V4 evaluation, eight generated answers were reviewed. Seven were marked correct and one partial.

The partial answer contained useful information but missed an important after-hours escalation rule.

This illustrates why generating an answer and providing a citation do not automatically prove that the answer is complete.

## 5. V4 — Observability, Privacy and What I Learned

### Why observability matters

In the earlier versions, I focused mainly on whether a question returned the correct passage or answer.

V4 introduced another engineering question: how can we understand what happened inside the system?

For example, if a result is incorrect, the problem could be in document indexing, retrieval, graph extraction, rank fusion or answer generation.

Tracing helps us inspect the steps of a request.

Our V4 implementation records individual spans for operations such as retrieval, fusion, redaction, generation and validation.

A trace can show which retrievers ran, which chunks were selected, how long each stage took and how many tokens were consumed by the LLM.

This is useful because performance and cost are part of the quality of an AI application.

One finding from the V4 measurements was that a question involving the Enterprise concept used significantly more input tokens than most other questions.

The reason was that additional graph-path identifiers were included in the information sent to the LLM.

Those identifiers increased token usage without providing useful natural-language evidence.

This showed an important practical lesson: sending unnecessary metadata to an LLM can increase cost even when the retrieval result is correct.

### Personal-data redaction

Another important part of V4 is responsible AI.

When a user asks a question, the application may send the question and relevant document passages to an external model provider.

These passages might contain personal information.

We therefore introduced redaction for email addresses and phone numbers.

The application replaces detected contact details with placeholders before sending the text to the LLM.

It can then restore the original values locally when appropriate.

I understood that this provides partial protection, not complete anonymisation.

For example, our current implementation does not reliably detect personal names or all possible formats of personal information.

I also learned that redaction can affect answer quality. In our tests, the model sometimes invented an address instead of preserving the intended placeholder.

This required additional validation.

### What I learned about AI systems

At the beginning of my RAG learning, I mainly thought about how an AI system could find the answer to a question.

After studying and working through the different versions, I understood that a RAG system contains many separate engineering problems.

We need to prepare documents correctly, choose embeddings, retrieve suitable passages, combine results, evaluate evidence, validate generated answers and protect information.

Each improvement introduces its own trade-offs.

Larger chunks may provide more context but reduce retrieval precision.

Smaller chunks may make individual facts easier to locate but separate related information.

BM25 may improve exact matching while losing semantic relationships.

Hybrid retrieval may improve some questions and reduce the ranking of important passages for others.

Graph retrieval may help multi-hop questions, but its quality depends on extracting and connecting the correct entities.

Tracing adds observability but also introduces responsibility for storing data safely.

The most important lesson I take from this project is that we cannot determine the quality of a RAG system simply by asking it one question and receiving an answer.

We need controlled evaluations, real test cases, source verification, and an understanding of the cost and limitations of every component.

I would like to continue studying reranking, embedding-model comparisons, retrieval evaluation and more reliable answer verification.

This project has helped me understand the difference between simply using an LLM and engineering an application around one.
