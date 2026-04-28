# RAG System Design — Reddit Political Discussion Corpus
## Part 2: Conversational QA System

---

## 0. Project Context

- **Corpus**: Cleaned Reddit posts and comments from a political subreddit, Jun–Nov 2024
- **Part 1 status**: Topic/stance analysis files exist but will be reworked later. RAG system is built independently of Part 1 outputs — it reads directly from `data/cleaned/`.
- **Goal**: Retrieval-Augmented Generation QA system with two LLM backends and a rigorous evaluation framework.

---

## 1. Environment Setup

### 1.1 Dependencies

```bash
pip install chromadb sentence-transformers groq google-generativeai \
            rouge-score bert-score pandas numpy tqdm
```

### 1.2 API Keys

**Groq**
1. Sign up at https://console.groq.com
2. Navigate to API Keys → Create API Key
3. Add to environment: `GROQ_API_KEY=your_key_here`
4. Models available:
   - `llama-3.1-8b-instant` — 128k context, fast, free tier: 6000 tokens/min
   - `llama-3.3-70b-versatile` — 128k context, higher quality, free tier: 6000 tokens/min
   - `mixtral-8x7b-32768` — 32k context, free tier: 6000 tokens/min
   - **Recommended**: `llama-3.3-70b-versatile` for quality; `llama-3.1-8b-instant` as fast comparison model

**Google AI Studio**
1. Sign in at https://aistudio.google.com
2. Click "Get API key" → Create API key in new project
3. Add to environment: `GOOGLE_API_KEY=your_key_here`
4. Models available:
   - `gemini-1.5-flash` — 1M token context, free tier: 15 RPM, 1500 RPD
   - `gemini-2.0-flash` — 1M token context, faster, same limits
   - **Recommended**: `gemini-2.0-flash`

### 1.3 Token Budget Analysis

Context windows are not a constraint here. Worst-case estimate for a retrieval context:
- 20 posts × ~300 tokens avg = 6,000 tokens
- 5 comments per post × 20 posts × ~80 tokens avg = 8,000 tokens
- System prompt + query = ~500 tokens
- **Total: ~14,500 tokens** — well within all free-tier models

Chunking decisions are therefore driven entirely by **retrieval quality**, not token limits.

---

## 2. Embedding Model

**Model**: `sentence-transformers/all-mpnet-base-v2`

**Rationale over all-MiniLM-L6-v2**:
- 768-dim vs 384-dim vectors — richer semantic representation
- Better on opinion and stance-heavy language (important for Reddit political content)
- MiniLM is already cached from Part 1 (BERTopic); mpnet is the retrieval-specific upgrade
- Marginally slower at indexing but retrieval latency is identical at this corpus size

**Note on Part 1**: The fuzzy BERTopic topics are more likely caused by `--min-topic-size` being too low and 39% outlier rate than by the embedding model. Revisit BERTopic separately; use mpnet for RAG regardless.

---

## 3. Corpus Preparation & Chunking

### 3.1 Source Files

| File | Content | Use |
|------|---------|-----|
| `data/cleaned/posts_clean.jsonl` | Full post corpus | Primary retrieval units |
| `data/cleaned/comments_clean.jsonl` | Full comment corpus | Supporting context per post |

Do **not** depend on Part 1 outputs (topic labels, stance labels) at index time — the RAG system must work independently. Topic/stance metadata can be added as an enhancement later.

### 3.2 Post Chunks

Each post becomes **one chunk**. Structure:

```
[{link_flair_text}] {title}

{selftext}   ← omit if empty
```

**Embed**: `title + " " + selftext` (full text field)

**Metadata stored per chunk**:
```python
{
    "doc_id": "post_{post_id}",
    "doc_type": "post",
    "post_id": post_id,
    "title": title,
    "created_month": "YYYY-MM",
    "created_iso": "...",
    "score": int,
    "num_comments": int,
    "link_flair_text": "...",
    "permalink": "https://reddit.com{permalink}",
}
```

### 3.3 Comment Chunks

Store only the **top 5 comments by score** per post. Each comment chunk prepends the parent post title for standalone interpretability:

```
[Post: {title}]
{comment_body}
```

**Embed**: the full prefixed string above.

**Metadata stored per chunk**:
```python
{
    "doc_id": "comment_{comment_id}",
    "doc_type": "comment",
    "comment_id": comment_id,
    "post_id": post_id,
    "parent_post_title": title,
    "created_iso": "...",
    "score": int,
    "permalink": "https://reddit.com{permalink}",
}
```

### 3.4 Two ChromaDB Collections

| Collection | Contents | Size estimate |
|------------|---------|---------------|
| `reddit_posts` | One chunk per post | ~20k documents |
| `reddit_comments` | Top-5 comments per post | ~100k documents |

---

## 4. Retrieval Pipeline

### 4.1 Overview

```
query
  → embed (mpnet)
  → posts collection: top-30 by cosine similarity
  → comments collection: top-50 by cosine similarity
  → score-weighted re-ranking (separate for posts and comments)
  → adversarial threshold check
  → diversity filter
  → final context: top-5 posts + associated comments
  → prompt construction
  → LLM A (Groq) ‖ LLM B (Gemini)  [parallel async]
  → return answers + retrieved sources
```

### 4.2 Step-by-Step

**Step 1 — Dense Retrieval**

Query both collections independently using ChromaDB's cosine similarity:
- Posts: retrieve top-30 candidates
- Comments: retrieve top-50 candidates

```python
post_results = post_collection.query(
    query_embeddings=[query_embedding],
    n_results=30,
    include=["documents", "metadatas", "distances"]
)
```

ChromaDB returns `distances` as cosine distance (0 = identical, 2 = opposite).
Convert to similarity: `similarity = 1 - (distance / 2)`

**Step 2 — Score-Weighted Re-ranking**

For posts:
```python
final_score = cosine_sim * log(1 + max(post_score, 0))
```

For comments:
```python
final_score = cosine_sim * log(1 + max(comment_score, 0))
```

The log dampens extreme outliers (a post with 10k upvotes shouldn't completely bury a relevant post with 50). Floor score at 0 to handle downvoted content.

**Step 3 — Adversarial / No-Answer Threshold**

Compute the **maximum cosine similarity** across all retrieved post candidates (before re-ranking).

```python
MAX_COSINE_SIM = max(post_similarities)
THRESHOLD = 0.35

if MAX_COSINE_SIM < THRESHOLD:
    no_answer_flag = True  # Pass to prompt — instruct LLM to decline
```

This is the primary defence for adversarial questions. The threshold of 0.35 is a starting point — calibrate against your known adversarial eval questions.

**Step 4 — Post Selection with Diversity Filter**

Sort re-ranked posts descending. Walk the list and select up to **5 posts**, with a cap of **2 posts per unique flair category**. This prevents a single heavily-discussed topic (e.g. all posts flaired "Elections") from dominating the context when a cross-topic question is asked.

```python
selected_posts = []
flair_counts = {}
for post in sorted_posts:
    flair = post["link_flair_text"]
    if flair_counts.get(flair, 0) >= 2:
        continue
    selected_posts.append(post)
    flair_counts[flair] = flair_counts.get(flair, 0) + 1
    if len(selected_posts) == 5:
        break
```

**Step 5 — Comment Association**

For each of the 5 selected posts, retrieve its top comments from the comment collection filtered by `post_id`. Use the pre-stored top-5-by-score comments (already limited at index time). If the post has fewer than 5 comments in the collection, use all available.

```python
for post in selected_posts:
    post_comments = comment_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"post_id": post["post_id"]},
        include=["documents", "metadatas", "distances"]
    )
```

This keeps the comment retrieval semantically relevant to the query (not just top-scored), while the at-index-time top-5 cap ensures comment quality.

**Final context shape**: 5 posts + up to 25 comments = up to 30 chunks.

---

## 5. Prompt Design

### 5.1 System Prompt

```
You are an assistant that answers questions about political discussions on Reddit.

Rules:
1. Answer ONLY using the provided Reddit excerpts below.
2. If the excerpts do not contain enough information, respond exactly with:
   "The corpus does not contain sufficient information to answer this question."
3. Do not use any external knowledge or make inferences beyond what is stated.
4. When citing a specific claim, reference the post title or permalink in parentheses.
5. For opinion questions, attribute views to "Reddit users" or "commenters", not as facts.
```

### 5.2 Context Block Format

Each retrieved post is formatted as:

```
---
[POST] {title}
Flair: {link_flair_text} | Score: {score} | Date: {created_month}
{selftext if non-empty}

  [COMMENT] Score: {score}
  {comment_body}

  [COMMENT] Score: {score}
  {comment_body}
---
```

### 5.3 User Turn

```
Reddit excerpts:
{formatted context block}

Question: {user_query}
```

### 5.4 No-Answer Variant

When `no_answer_flag = True`, append to the system prompt:

```
Note: The search returned no closely matching content for this query.
It is very likely the answer is not present in this corpus. You should
say so clearly rather than speculate.
```

---

## 6. LLM Abstraction Layer

Both providers should be called through a common interface:

```python
def query_llm(provider: str, system_prompt: str, context: str, question: str) -> str:
    """
    provider: "groq" | "gemini"
    Returns the model's answer as a plain string.
    """
```

Call both providers in **parallel using asyncio** for the evaluation run — 15 questions × 2 models synchronously would be slow, especially under Groq's rate limits.

---

## 7. File & Directory Structure

```
project_root/
├── data/
│   ├── cleaned/
│   │   ├── posts_clean.jsonl         ← input
│   │   └── comments_clean.jsonl      ← input
│   └── chroma_db/                    ← ChromaDB persistence directory
│       ├── reddit_posts/
│       └── reddit_comments/
├── scripts/
│   ├── build_index.py                ← Step 1: build ChromaDB collections
│   ├── rag_query.py                  ← Step 2: retrieval + LLM query
│   ├── evaluate_rag.py               ← Step 3: run eval set, compute metrics
│   └── eval_set.json                 ← Ground-truth QA pairs (hand-authored)
└── outputs/
    └── evaluation_report.md          ← Final comparative report
```

---

## 8. Evaluation Framework

### 8.1 Ground-Truth Eval Set (`eval_set.json`)

**Write these BEFORE running the system**, based on direct reading of the data.
Minimum 15 QA pairs structured as:

```json
[
  {
    "id": "q01",
    "type": "factual",
    "question": "...",
    "reference_answer": "...",
    "answerable": true
  },
  ...
]
```

**Question type breakdown (minimum)**:

| Type | Count | Description |
|------|-------|-------------|
| Factual | 5 | Verifiable from post content or aggregate stats |
| Opinion-summary | 6 | "What do users think about X?" — require synthesis |
| Adversarial | 2–3 | Answers genuinely absent from corpus |

**Adversarial question design guidance**: choose topics that are plausibly related to the subreddit but fall outside the date range or data scope — e.g. post-election Senate results, moderation history, all-time statistics. Avoid questions that are merely hard; they must be definitively unanswerable from this corpus.

### 8.2 Metrics

**ROUGE-L** (`rouge-score` library)
- Measures longest common subsequence overlap between generated and reference answer
- Penalises hallucination indirectly — fabricated content won't overlap with reference
- Compute per question, report mean across set

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
score = scorer.score(reference, generated)["rougeL"].fmeasure
```

**BERTScore** (`bert-score` library)
- Semantic similarity via contextual embeddings — handles paraphrases ROUGE-L misses
- Backbone: `microsoft/deberta-xlarge-mnli` if compute allows, else default `roberta-large`
- Report F1 score; compute per question, report mean

```python
from bert_score import score as bert_score
P, R, F1 = bert_score([generated], [reference], lang="en", model_type="roberta-large")
```

**Faithfulness (manual binary flag)**
- For each answer, read it and mark `1` if every factual claim is traceable to a retrieved chunk, `0` if any claim is hallucinated or uses external knowledge
- Report as percentage across the test set per model
- This is the most diagnostic metric for a RAG system

**Answer Relevance (manual binary flag)**
- Mark `1` if the answer actually addresses the question asked, `0` if it's off-topic or a refusal on an answerable question
- Report as percentage per model
- Distinguishes "faithful but unhelpful" from "hallucinated but relevant"

**Adversarial Refusal Rate**
- For adversarial questions only: mark `1` if the model correctly declines to answer
- Report separately — a model that hallucinates a confident answer on adversarial questions is unsafe regardless of its ROUGE-L on other questions

### 8.3 Results Table Format

| Question ID | Type | Groq ROUGE-L | Gemini ROUGE-L | Groq BERTScore | Gemini BERTScore | Groq Faithful | Gemini Faithful | Groq Relevant | Gemini Relevant |
|-------------|------|-------------|---------------|---------------|----------------|--------------|----------------|--------------|----------------|
| q01 | factual | ... | ... | ... | ... | 1 | 1 | 1 | 1 |
| ... | | | | | | | | | |
| **Mean** | | | | | | **%** | **%** | **%** | **%** |

**Adversarial sub-table**:

| Question ID | Groq Refused Correctly | Gemini Refused Correctly |
|-------------|----------------------|------------------------|
| q14 | Yes / No | Yes / No |
| q15 | Yes / No | Yes / No |

### 8.4 Qualitative Analysis Prompts

After computing metrics, write a short analysis covering:
1. Where does each model succeed? (e.g. Gemini better at synthesis, Groq faster/more concise)
2. Where does each model fail? (hallucination patterns, over-refusal, citation errors)
3. How does retrieval quality affect answer quality? (note cases where wrong chunks were retrieved)
4. Adversarial behaviour: did the threshold correctly trigger? Did models respect the no-answer instruction?

---

## 9. Implementation Order (Step-by-Step for Agent)

```
Step 1  build_index.py
        - Load posts_clean.jsonl
        - For each post: build chunk text, extract metadata
        - Load comments_clean.jsonl
        - Group by post_id, keep top-5 by score per post
        - For each comment: prepend parent post title, extract metadata
        - Initialise ChromaDB PersistentClient at data/chroma_db/
        - Create collections: reddit_posts, reddit_comments
        - Embed all chunks with all-mpnet-base-v2
        - Upsert to respective collections
        - Print collection sizes to confirm

Step 2  rag_query.py
        - Implement embed_query(text) → vector
        - Implement retrieve(query, no_answer_flag) → (posts, comments)
          following the 5-step pipeline in Section 4
        - Implement format_context(posts, comments) → str
        - Implement build_prompt(context, question, no_answer_flag) → (system, user)
        - Implement query_groq(system, user) → str
        - Implement query_gemini(system, user) → str
        - Implement query_both_async(question) → {groq: str, gemini: str}
        - Add CLI: python rag_query.py --question "..."

Step 3  eval_set.json
        - Hand-author 15+ QA pairs (5 factual, 6 opinion, 2-3 adversarial)
        - Must be written before running evaluate_rag.py

Step 4  evaluate_rag.py
        - Load eval_set.json
        - For each question: call query_both_async
        - Compute ROUGE-L and BERTScore for each answer vs reference
        - Output CSV with all scores for manual annotation
        - Manual pass: fill in Faithful and Relevant binary flags
        - Generate evaluation_report.md with results table + qualitative analysis
```

---

## 10. Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | all-mpnet-base-v2 | Richer 768-dim vectors; better for opinion-heavy retrieval |
| Vector store | ChromaDB | Native persistence, metadata filtering, simple API |
| Collections | Two (posts + comments) | Clean separation; allows per-type retrieval strategies |
| Comments per post | Top-5 by score at index time | Quality cap; ~100k total comments is manageable |
| Posts retrieved | Top-5 after diversity filter | Balances coverage with context window efficiency |
| Comments retrieved | Up to 5 per selected post, filtered by query | Semantically relevant, not just highest-scored |
| Re-ranking | cosine × log(1 + score) | Rewards community-validated content without over-weighting viral posts |
| Diversity filter | Max 2 posts per flair | Prevents topic monopolisation in cross-topic queries |
| Adversarial defence | Cosine threshold 0.35 + system prompt instruction | Two-layer defence; threshold triggers prompt flag |
| LLMs | Groq llama-3.3-70b + Gemini 2.0 Flash | Both free-tier; different architectures for meaningful comparison |
| Parallelism | asyncio for eval run | Avoids rate limit bottlenecks across 15 × 2 calls |