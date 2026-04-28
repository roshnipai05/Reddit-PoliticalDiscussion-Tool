# Implementation Update Playbook

This document defines the order in which the pending codebase updates should be addressed and the procedure an agent should follow for each item. It is based on the current repository state, not only on the requested feature list.

## Goal

Bring the system from a partial prototype to a consistent end-to-end analyst tool:

1. fix backend correctness and dataset coverage first
2. improve analysis quality next
3. connect backend outputs to the frontend after interfaces stabilize
4. refine UI and add new exploratory features after the core outputs are trustworthy

## Implementation Notes

### 2026-04-29

- The top-bar question input now exposes a frontend question-type helper.
- Clicking or focusing the question bar opens a dropdown with examples for:
  - `focused:`
  - `aggregate:`
  - `comparison:`
  - `multi-hop:`
- The current UI contract now requires analysts to prefix QA prompts explicitly instead of relying on implicit query-type detection.
- The frontend validates the prefix locally and surfaces an inline hint when the format is invalid.
- Backend routing in [`scripts/rag_query.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\rag_query.py) now accepts the same prefixed contract and resolves queries into:
  - `focused`
  - `aggregate`
  - `comparison`
  - `multi-hop`
- `run_rag_query()` now returns `query_type`, `question_body`, and `route_metadata` in addition to the existing answer fields.
- Aggregate mode now reads deterministic corpus aggregates from `data/topic_analysis/aggregate_stats.json` and top modeled topics from `data/topic_analysis/topic_summary.json`.
- Aggregate mode is intentionally conservative:
  - it supports corpus totals, flair frequencies, and topic-level synthesis
  - it does not yet support canonicalized person/entity rankings
  - unsupported aggregate requests should refuse rather than hallucinate
- Comparison mode now performs side-by-side retrieval for two parsed comparison targets when the question format is parseable.
- Multi-hop mode now decomposes the question into a small set of hop subqueries and fuses the resulting evidence before answer generation.
- Gemini calls are now paced through a centralized delay helper before request execution.
- A local app server now exists at [`scripts/local_app_server.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\local_app_server.py).
- The localhost server is responsible for:
  - serving the `app/` directory
  - exposing `/api/query` for routed QA
  - exposing `/api/status` for bundle / topic / stance / RAG readiness
  - exposing pipeline actions for topic analysis, stance preview, and bundle rebuild
- [`scripts/build_app_bundle.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\build_app_bundle.py) now tolerates missing stance-preview artifacts so the app can still boot while that pipeline is unfinished.
- The frontend now talks to the local API instead of relying only on static file fetches.

## Current State Snapshot

### RAG / QA

- The current RAG is an English-only dense retrieval pipeline over ChromaDB.
- Query flow today:
  1. non-English question is translated to English in [`scripts/rag_query.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\rag_query.py)
  2. English query is embedded with `sentence-transformers/all-mpnet-base-v2`
  3. posts are retrieved from `reddit_posts`
  4. posts are re-ranked by semantic similarity times `log1p(score)`
  5. a diversity filter caps posts per flair
  6. comments are retrieved per selected post from `reddit_comments`
  7. answer is generated in English
  8. answer is translated back to the input language
- Focused-mode retrieval is still dense vector retrieval plus score-weighted re-ranking plus per-post comment expansion.
- Aggregate, comparison, and multi-hop questions are now routed explicitly from the user prefix instead of being inferred implicitly.
- Aggregate mode now uses a hybrid of structured analytics plus supporting retrieval, but entity-ranking questions still require a richer aggregate layer to become answerable.
- The answer prompt no longer requires inline citation by post title.
- Gemini calls now use centralized pacing logic before request execution.

### Topic Analysis

- Topic analysis is currently run from `data/cleaned/posts_subset.jsonl`, not the full dataset.
- [`data/topic_analysis/run_metadata.json`](C:\Users\91887\Documents\Reddit-Topic-Analysis\data\topic_analysis\run_metadata.json) shows:
  - `post_count = 5000`
  - input source is `posts_subset.jsonl`
  - `month_axis = ["2024-07", "2024-08"]`
- Topic descriptions are currently generated from keywords, top flair, and one representative title. That explains the weak summaries.
- Trend labels are backend-computed in [`scripts/topic_modeling_analysis.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\topic_modeling_analysis.py) using:
  - active months
  - monthly share of posts
  - recent-window vs early-window share
  - slope over time
  - coefficient of variation

### Stance Analysis

- Stance analysis is currently broad clustering, not argument summarization.
- Comments are embedded, split into two clusters with MiniBatchKMeans, then the larger/weighted cluster becomes `support` and the other becomes `opposing`.
- Summaries are built from top keywords plus representative comment excerpts. This is why they read like keyword summaries rather than coherent arguments.
- The checked-in preview is also subset-based. [`data/topic_stance_preview/run_metadata.json`](C:\Users\91887\Documents\Reddit-Topic-Analysis\data\topic_stance_preview\run_metadata.json) points to `comments_subset.jsonl`.

### Frontend

- Sidebar navigation buttons in [`app/index.html`](C:\Users\91887\Documents\Reddit-Topic-Analysis\app\index.html) are placeholders only. They do not route anywhere yet.
- The topic frequency chart in [`app/app.js`](C:\Users\91887\Documents\Reddit-Topic-Analysis\app\app.js) is currently an SVG line chart, not daily bars across six months.
- The current frontend bundle is assembled in [`scripts/build_app_bundle.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\build_app_bundle.py).
- The top-bar QA input now shows a dropdown of query-type examples and requires a prefix format such as `aggregate: "..."`.
- `conversationInput` is now wired to `/api/query`.
- The top bar also exposes model selection and query submission controls.
- The sidebar now exposes pipeline controls for:
  - refresh status
  - rebuild bundle
  - run topic analysis
  - run stance preview
- Topic-analysis and stance-preview runs now trigger a bundle rebuild automatically from the frontend flow so refreshed outputs appear in the app without a second manual action.
- The main content area now includes a QA results panel that renders:
  - answer text
  - routed query metadata
  - top retrieved source posts

## Recommended Implementation Order

### Phase 1: Topic discovery refinement and taxonomy quality

Do this first. The current weakness is not only the prose summary. The discovered topics themselves need post-discovery cleanup before they are usable in the UI or in stance analysis.

1. keep unsupervised topic discovery as the draft stage
2. add a robust refinement pass after discovery
3. infer major topics as broad analyst-facing domains from the refined sub-topics
4. keep sub-topics as sharper issue areas within those domains
5. merge, relabel, or collapse weak draft clusters before writing summaries

Methodology:
- unsupervised discovery
- merge / relabel / refine
- optional semi-supervised analyst naming rules
- no hard-coded end-state taxonomy, but the induced major topics should converge toward broad domains such as elections, institutions/law, economy/domestic policy, identity/rights, foreign policy, and parties/media narratives when the evidence supports them

### Phase 2: Improve topic and stance summary quality

1. rewrite topic summaries to be corpus-synthesis summaries
2. rewrite stance summaries to describe arguments, not keywords
3. preserve top posts as evidence cards, but decouple them from the summary text

Reason:
- even strong prose will fail if the topic boundaries are poor
- stance summaries become clearer automatically once sub-topic boundaries are sharper

### Phase 3: Restore full-data completeness and regenerate artifacts

1. run the refined topic pipeline on the full cleaned posts
2. run the refined stance pipeline on the full cleaned comments
3. regenerate topic, stance, and bundle artifacts
4. verify six-month coverage in output metadata

Reason:
- there is no point paying full-data runtime cost before the refined pipeline is stable

### Phase 4: Fix RAG retrieval and answer behavior

Do this before frontend integration so the UI connects to the final QA contract once.

1. verify multilingual query flow end to end
2. add Gemini throttling / sleep spacing
3. remove inline source citations from generated answers
4. make answers synthesis-heavy across similar posts
5. keep top 5 retrieved posts as a separate display payload
6. decide whether aggregate questions need a second retrieval mode

Reason:
- the desired QA output format changes both backend prompt design and frontend rendering
- aggregate-style analyst questions are a retrieval-design issue, not a frontend issue

### Phase 5: Connect backend outputs to product flows

1. connect top-bar QA input to the RAG backend
2. display answer plus top retrieved posts
3. pass the prefixed question type from the frontend to the QA backend contract
4. add query-language toggle behavior
5. expose trend legend in frontend
6. map left-nav buttons to real pages or views

Reason:
- this is the first point where the system becomes a coherent tool

### Phase 6: Visual and exploratory feature expansion

1. replace topic trend line with six-month daily bar chart plus event annotations
2. design stance-overlap exploration between any two topics/sub-topics
3. add analyst-oriented features only after core flows work

Reason:
- these features depend on stable backend outputs and clean frontend information architecture

## Detailed Procedure Per Phase

## Phase 1: Topic discovery refinement

### Target design

- major topics = broad analyst-facing domains
- sub-topics = sharper issue areas within those domains
- major topics should still be discovered organically
- the system is allowed to use a post-discovery refinement layer and analyst naming rules
- fewer, sharper sub-topics are preferred over many weak ones

### Procedure

1. Treat BERTopic or any equivalent clustering step as a draft generator only.
2. Build topic profiles from multiple signals:
   - top weighted phrases from titles
   - top weighted phrases from post bodies
   - flair distribution
   - representative high-engagement posts
   - temporal pattern
3. Infer issue-area labels for draft clusters.
4. Merge or collapse draft clusters when:
   - they resolve to the same issue area
   - they are semantically near-duplicate
   - one is broad noise and the other is the clearer issue-area version
5. Infer broader major-topic domains from the refined issue-area clusters.
6. Only after refinement, write the final `topic_summary` outputs and `post_topics` assignments.

### Acceptance criteria

- noisy labels like accidental phrase fragments should disappear
- sub-topic titles should read like issue areas, not post headlines
- the major-topic hierarchy should be compact and interpretable

## Phase 2: Topic and stance summary rewrite

### Procedure

1. Replace the current keyword-driven topic descriptions with synthesis across multiple posts.
2. Summaries should explain:
   - what the sub-topic is about
   - what recurring policy conflict or issue area ties the posts together
   - what recurring thread types or events are driving attention within that issue area
3. Use the Groq API as the preferred prose layer for:
   - sub-topic descriptions
   - dominant stance summaries
   - supporting argument summaries
   - opposing argument summaries
4. Keep a deterministic fallback path so the scripts still run if the API is unavailable.
5. Replace stance summaries so they describe the main supporting and opposing arguments, not just keywords and excerpts.
6. Keep evidence items separate from the prose summaries.

### Acceptance criteria

- topic descriptions read like analyst summaries
- stance summaries read like argument maps
- evidence cards remain available without dominating the narrative text

## Phase 3: Full-data rerun

### Files to inspect

- [`scripts/topic_modeling_analysis.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\topic_modeling_analysis.py)
- [`scripts/topic_stance_analysis.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\topic_stance_analysis.py)
- [`scripts/build_app_bundle.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\build_app_bundle.py)
- [`data/topic_analysis/run_metadata.json`](C:\Users\91887\Documents\Reddit-Topic-Analysis\data\topic_analysis\run_metadata.json)
- [`data/topic_stance_preview/run_metadata.json`](C:\Users\91887\Documents\Reddit-Topic-Analysis\data\topic_stance_preview\run_metadata.json)

### Procedure

1. Run the refined topic pipeline on `data/cleaned/posts_clean.jsonl`.
2. Run the refined stance pipeline on `data/cleaned/comments_clean.jsonl` against the refined topic outputs.
3. Rebuild `app/data.bundle.json`.
4. Verify:
   - six-month month axis is present
   - major-topic and sub-topic counts remain compact
   - stance metadata no longer points to subset files
   - the refined labels and descriptions persist in the bundle

### Acceptance criteria

- output metadata references full cleaned files, not subset files
- month axis spans the intended six-month window
- frontend bundle is rebuilt from full-data outputs

## Phase 2: RAG and multilingual QA

### What kind of RAG is currently implemented

Current system = dense bilingual-through-translation retrieval over an English index.

More precisely:

1. translate non-English question to English
2. embed English query with `all-mpnet-base-v2`
3. retrieve nearest English post vectors from Chroma
4. re-rank by semantic similarity and Reddit score
5. select up to 5 posts with flair diversity
6. retrieve semantically similar comments for those posts
7. build one answer from those retrieved excerpts
8. translate the answer back to the source language

This should answer focused questions reasonably well if the relevant evidence exists in semantically similar posts.

This now has explicit routed modes, but the aggregate route is only partially complete:

- supported today:
  - corpus totals
  - flair distribution
  - topic-level prevalence from modeled topic summaries
- not supported yet:
  - canonicalized politician/entity rankings
  - full graph-style relation queries

### Files to inspect

- [`scripts/rag_query.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\rag_query.py)
- [`scripts/build_index.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\build_index.py)
- [`data/chroma_db/index_metadata.json`](C:\Users\91887\Documents\Reddit-Topic-Analysis\data\chroma_db\index_metadata.json)

### Procedure

1. Parse explicit question-type prefixes in `rag_query.py` and support an optional `--query-type` override.
2. Keep focused mode as the default when no prefix is present.
3. Replace the old single-prompt flow with routed prompt builders for:
   - focused
   - aggregate
   - comparison
   - multi-hop
4. Keep `source_posts` as a separate response field for frontend display.
5. Add rate-limit protection around Gemini calls:
   - include a conservative sleep before each Gemini request
   - centralize the wait logic in one helper
6. For aggregate mode:
   - answer from structured analytics first
   - use retrieval as supporting evidence
   - refuse unsupported entity-ranking requests rather than improvising
7. For comparison mode:
   - parse a left and right comparison target when possible
   - retrieve each side separately
   - synthesize from the paired evidence
8. For multi-hop mode:
   - derive a small set of hop subqueries
   - retrieve each hop separately
   - synthesize only after hop evidence is assembled

### Recommended design decision

For analyst questions like "Which politicians are most discussed?" do not rely on retrieval-only QA.

Preferred approach:

1. accept explicit analyst routing from the question prefix
2. answer aggregate requests from structured analytics when possible
3. use RAG only as supporting evidence and examples
4. refuse unsupported aggregate scopes until richer entity aggregates exist

### Acceptance criteria

- multilingual QA still works end to end
- Gemini calls no longer fail from burst rate issues under normal usage
- answer text contains no inline source citations
- top 5 retrieved posts are still returned separately
- answers read as cross-post synthesis, not isolated-thread summaries
- routed responses expose enough metadata for the frontend to show which QA mode was used
- the localhost app can boot even if stance preview outputs are currently missing

## Trend logic reference

### Current backend trend logic

Trend type is computed from:

- `active_months`
- `total_months`
- `recent_share`
- `early_share`
- `overall_share`
- linear `slope`
- share `cv`

Rules today:

- `Persistent`: active in most months, low variance, recent share close to overall share
- `Trending`: recent share materially above overall and early share, positive slope
- `Declining`: early share materially above overall and recent share, negative slope
- `Episodic`: everything else

## Phase 5: Frontend wiring and legend

### Files to inspect

- [`app/index.html`](C:\Users\91887\Documents\Reddit-Topic-Analysis\app\index.html)
- [`app/app.js`](C:\Users\91887\Documents\Reddit-Topic-Analysis\app\app.js)
- [`app/styles.css`](C:\Users\91887\Documents\Reddit-Topic-Analysis\app\styles.css)
- [`scripts/build_app_bundle.py`](C:\Users\91887\Documents\Reddit-Topic-Analysis\scripts\build_app_bundle.py)

### Procedure

1. Define the page map for the left navigation before coding routes:
   - Topic Map
   - Trend Monitor
   - Conversation QA
   - Historic Events
   - Reports
2. For each button, decide whether it becomes:
   - separate page
   - routed view
   - drawer/modal
3. Preserve the explicit query-type prefix contract in the frontend:
   - clicking the question bar opens example query types
   - the analyst prefixes the prompt with `focused:`, `aggregate:`, `comparison:`, or `multi-hop:`
   - invalid prefixes are blocked or warned before QA submission
4. Connect `conversationInput` and the language toggle to the QA endpoint or script runner.
   - pass the prefixed question string as entered
   - map the UI language state to the backend `lang` code
   - surface backend `query_type` and `route_metadata` in the result payload
   - expose model choice in the UI and pass it through to the backend
5. Display:
   - generated answer
   - top 5 retrieved posts for validation
   - language state / active query language
6. Add a trend legend panel sourced from backend definitions.

### Acceptance criteria

- top bar QA teaches the analyst the supported question modes before submission
- top bar QA actually executes a query
- source post cards are visible separately from the answer
- sidebar navigation no longer contains dead buttons
- pipeline controls can run the linked backend scripts from the UI

## Phase 6: Trend chart redesign

### Current gap

The detail chart is rendered as a line chart from `timeline.post_counts`. The requirement is daily bars across six months, with major events marked on the x-axis.

### Procedure

1. Extend backend output to include daily counts, not only monthly counts.
2. Add event alignment data at the same date granularity.
3. Replace the SVG line chart with a daily bar chart.
4. Ensure event labels do not overcrowd the axis:
   - selective labeling
   - tooltip or marker strategy

### Acceptance criteria

- chart is daily-bar based
- event markers align with dates in the visible window
- six-month topic activity can be correlated with events

## Phase 7: Topic overlap / stance overlap feature

This is feasible, but it should be treated as a new feature after the current pipeline is stabilized.

### Proposed backend idea

For any two selected topics or sub-topics:

1. identify overlapping `author_hash` users
2. compute each user’s dominant stance in each selected topic
3. produce overlap stats:
   - total overlapping users
   - share of topic A users also active in topic B
   - share of topic B users also active in topic A
   - stance-to-stance transition matrix
4. produce graph-ready nodes and edges

### Proposed frontend idea

- two topic/sub-topic selectors
- overlap stats summary cards
- bipartite or Sankey-like view of stance overlap
- drill-down table of overlapping users and comment counts

### Feasibility notes

- the existing `author_hash` fields and topic-level stance outputs are enough for a first version
- a careful definition is needed for "holds an opinion" so low-activity users do not distort results

## Suggested Additional Features For A Political Analyst

Implement only after the core backlog above.

1. event-to-topic impact view
2. politician/entity co-mention view
3. query presets for analyst workflows
4. compare two time windows for the same topic
5. exportable topic brief per topic
6. uncertainty / confidence indicators on topic and stance summaries

## Execution Rules For Any Agent Working This Plan

1. Do not start frontend polish before full-data backend outputs are regenerated.
2. Do not wire the QA UI before the QA response schema is finalized.
3. Keep evidence cards separate from generated summaries.
4. Preserve old output directories until the new outputs are verified.
5. After each phase, rebuild the app bundle and test the affected flow.

## Minimum Validation Sequence

After each major phase:

1. run the relevant backend script
2. inspect output metadata json
3. rebuild `app/data.bundle.json` if topic or stance outputs changed
4. open the frontend and verify the affected view
5. record any schema changes before moving to the next phase

## Immediate Next Step

Start with Phase 1. The current subset-based topic and stance outputs will otherwise contaminate the evaluation of every later change.
