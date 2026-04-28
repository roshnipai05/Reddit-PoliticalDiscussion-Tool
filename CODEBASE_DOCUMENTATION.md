# Reddit Topic Analysis Codebase Documentation

## 1. Purpose and Scope

This repository implements a local analyst-facing system for exploring political Reddit discussions through two connected layers:

1. a backend analysis pipeline that cleans data, builds topic and stance artifacts, and indexes the corpus for retrieval
2. a local web application that exposes hierarchical topic exploration and routed retrieval-augmented question answering

The codebase is not a generic RAG demo. It is structured around a specific analyst workflow:

- model the main themes in the subreddit corpus
- summarize how those themes evolve over time
- preview disagreement patterns inside each topic
- let an analyst ask targeted, aggregate, comparison, and multi-hop questions against the same corpus

Two planning documents define the intended architecture and the recent update sequence:

- [rag_design_plan.md](/C:/Users/91887/Documents/Reddit-Topic-Analysis/rag_design_plan.md)
- [implementation_update_playbook.md](/C:/Users/91887/Documents/Reddit-Topic-Analysis/implementation_update_playbook.md)

The implementation follows both, but it also diverges in some important ways:

- the original RAG design described a single focused retrieval flow with inline source references; the current implementation has explicit routed query modes and intentionally removes inline citations from answer text
- the original plan described topic analysis as needing refinement; the current topic pipeline already contains a significant post-discovery refinement layer
- the playbook expected the checked-in stance preview to still be subset-based, but the current repo also contains a newer `topic_stance_preview_check_groq` run on full cleaned comments for one topic sample, while the app’s default `data/topic_stance_preview/` directory is still missing

This document explains the repository as it exists now.

## 2. Repository Structure

Top-level layout:

- [app/](/C:/Users/91887/Documents/Reddit-Topic-Analysis/app) contains the static frontend bundle
- [scripts/](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts) contains all pipeline and server scripts
- [data/](/C:/Users/91887/Documents/Reddit-Topic-Analysis/data) contains cleaned inputs, analysis outputs, and the Chroma index
- [outputs/](/C:/Users/91887/Documents/Reddit-Topic-Analysis/outputs) is reserved for generated reports and evaluations
- [original_data/](/C:/Users/91887/Documents/Reddit-Topic-Analysis/original_data) appears to hold source/raw data snapshots

Important files:

- [scripts/topic_modeling_analysis.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/topic_modeling_analysis.py): topic discovery, topic refinement, trend classification, report generation
- [scripts/topic_stance_analysis.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/topic_stance_analysis.py): per-topic stance clustering and stance-summary generation
- [scripts/build_index.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/build_index.py): ChromaDB index build for posts and comments
- [scripts/rag_query.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/rag_query.py): routed QA engine
- [scripts/build_app_bundle.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/build_app_bundle.py): transforms analysis outputs into a compact frontend bundle
- [scripts/local_app_server.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/local_app_server.py): local HTTP server and JSON API
- [scripts/llm_summaries.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/llm_summaries.py): shared Groq JSON helper with pacing
- [app/index.html](/C:/Users/91887/Documents/Reddit-Topic-Analysis/app/index.html): shell layout
- [app/app.js](/C:/Users/91887/Documents/Reddit-Topic-Analysis/app/app.js): frontend state management, rendering, and API calls
- [app/styles.css](/C:/Users/91887/Documents/Reddit-Topic-Analysis/app/styles.css): visual system and responsive layout

## 3. Data and Artifact Flow

The system is built around a staged data flow:

1. cleaned Reddit posts and comments live in `data/cleaned/`
2. topic modeling reads cleaned posts and writes topic summaries, per-post topic assignments, trend outputs, and aggregate statistics into `data/topic_analysis/`
3. stance analysis reads cleaned comments plus topic assignments and writes topic-level stance outputs into a stance-preview directory
4. the index builder reads cleaned posts/comments and writes a two-collection ChromaDB store into `data/chroma_db/`
5. the app-bundle builder reads topic analysis and stance outputs and compiles `app/data.bundle.json`
6. the local server serves the app and forwards QA requests to the routed RAG engine

This separation is a deliberate design choice:

- topic and stance analytics are deterministic-ish offline artifacts
- QA is an online query path backed by vector retrieval and LLM synthesis
- the frontend reads one prebuilt bundle for exploration instead of performing large client-side joins

## 4. Current Repository State

Based on the checked-in metadata:

- [data/topic_analysis/run_metadata.json](/C:/Users/91887/Documents/Reddit-Topic-Analysis/data/topic_analysis/run_metadata.json) shows the topic pipeline has been run on `data/cleaned/posts_clean.jsonl`
- the current topic run covers `13,766` posts with `13` refined topics
- the month axis is `2024-07` through `2024-12`
- [data/chroma_db/index_metadata.json](/C:/Users/91887/Documents/Reddit-Topic-Analysis/data/chroma_db/index_metadata.json) shows the RAG index is built from full cleaned posts and comments
- the Chroma index contains `13,766` post chunks and `7,285` comment chunks
- the comment count is lower than the raw corpus comment total because the index intentionally stores only the top `5` comments per post
- [data/topic_stance_preview_check_groq/run_metadata.json](/C:/Users/91887/Documents/Reddit-Topic-Analysis/data/topic_stance_preview_check_groq/run_metadata.json) shows a newer stance-preview run exists on full cleaned comments, but only for one topic sample with `400` comments
- `data/topic_stance_preview/` does not currently exist, so the app’s default bundle path has to degrade gracefully when stance outputs are missing

That last point matters architecturally: the frontend and bundle builder were explicitly hardened so topic exploration can still work while stance analysis is incomplete.

## 5. Backend Architecture Overview

The backend is split into four main subsystems:

1. topic discovery and refinement
2. stance clustering and stance summarization
3. vector retrieval and routed RAG
4. local serving and orchestration

This is a strong design choice for maintainability. Topic modeling, stance analysis, and QA evolve at different speeds and have different failure modes. Keeping them separate makes it possible to:

- rerun topic analysis without rebuilding the index
- improve RAG prompts without changing the topic taxonomy
- tolerate missing stance artifacts while still exposing the rest of the tool

## 6. Topic Analysis Pipeline

### 6.1 Main responsibility

[scripts/topic_modeling_analysis.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/topic_modeling_analysis.py) is responsible for:

- loading cleaned posts
- computing corpus-level aggregate stats
- discovering draft topics with BERTopic
- refining those draft topics into sharper analyst-facing issue areas
- assigning broader major-topic domains
- generating topic descriptions
- classifying topic trends over time
- writing structured outputs for downstream consumption

### 6.2 Inputs and outputs

Primary input:

- `data/cleaned/posts_clean.jsonl`

Primary outputs:

- `data/topic_analysis/aggregate_stats.json`
- `data/topic_analysis/post_topics.csv`
- `data/topic_analysis/topic_summary.json`
- `data/topic_analysis/topic_summary.csv`
- `data/topic_analysis/topic_monthly_trends.csv`
- `data/topic_analysis/topic_flair_breakdown.csv`
- `data/topic_analysis/topic_analysis_report.md`
- `data/topic_analysis/topic_share_chart.html`
- `data/topic_analysis/run_metadata.json`

### 6.3 Topic modeling strategy

The code does not simply fit BERTopic and expose whatever labels fall out. The actual strategy is two-stage:

1. unsupervised draft discovery
2. post-discovery refinement and relabeling

The draft discovery stack is:

- `SentenceTransformer` embeddings
- `UMAP` for dimensionality reduction
- `HDBSCAN` for clustering
- `BERTopic` for topic extraction and keyword production

Important configuration choices:

- default embedding model: `all-MiniLM-L6-v2`
- vectorizer: `CountVectorizer` with political/domain stopwords and `ngram_range=(1, 3)`
- `nr_topics` target default: `15`
- `min_topic_size` default: `90`
- optional outlier reduction via `model.reduce_outliers(..., strategy="embeddings")`

The choice to keep BERTopic is pragmatic: it gives good enough draft clustering and keyword extraction while leaving room for a rule-based refinement pass.

### 6.4 Why topic quality improved

The implementation playbook correctly identified that weak topic labels were not just a summary-writing problem. The current code improves quality through several concrete strategies:

1. domain-specific stopword control
2. title-phrase scoring
3. salient phrase extraction from title plus selftext
4. issue-area inference rules
5. major-domain mapping
6. topic merging by normalized merge keys
7. LLM-assisted topic descriptions with deterministic fallback

These are the main mechanisms.

#### 6.4.1 Domain-specific stopwording

The script extends standard English stopwords with political stopwords such as:

- `trump`
- `biden`
- `harris`
- `election`
- `politics`
- `vote`

This prevents the topic model from wasting label capacity on generic political vocabulary that appears everywhere.

#### 6.4.2 Phrase extraction biased toward meaningful titles

Functions such as `score_title_phrases()` and `extract_salient_phrases()` intentionally give weight to phrases that recur in titles and short post text, rather than relying only on BERTopic keyword lists.

This matters because Reddit titles often encode the issue frame more clearly than body text. The code explicitly:

- builds 2-gram and 3-gram title phrases
- filters noisy and generic terms
- rewards phrases that recur across documents
- boosts phrases that also occur in titles

That is a targeted design choice for social-discussion data.

#### 6.4.3 Rule-based issue-area relabeling

The script defines `ISSUE_AREA_RULES` and `DOMAIN_ARCHETYPES`. These are not a hard-coded end-state taxonomy in the strictest sense, but they are a controlled post-discovery guidance layer. The refinement logic uses:

- representative titles
- flair distributions
- raw topic keywords
- salient phrases

to infer labels like:

- `Polling, debate performance and electoral momentum`
- `Supreme Court power and constitutional constraints`
- `Inflation, housing costs and consumer pressure`

This is the main reason labels are analyst-readable instead of accidental keyword bundles.

#### 6.4.4 Major-topic grouping

Each refined issue area is also mapped into a broader analyst-facing domain such as:

- `Elections & Campaigns`
- `Institutions, Courts & Law`
- `Economy, Labor & Domestic Policy`
- `Identity, Rights & Social Conflict`
- `Foreign Policy & Geopolitics`
- `Parties, Media & Political Narratives`

This hierarchy is then surfaced directly in the frontend topic tree.

#### 6.4.5 Merge-key based refinement

Draft topics are grouped by a merge key of the form:

- `major_topic_slug::normalized_label`

All raw topic IDs that collapse into the same refined label/domain pair are merged before final summaries are written. This is a high-value design decision because it:

- reduces near-duplicate topics
- keeps the top-level taxonomy compact
- separates the clustering stage from the final analyst-facing taxonomy

### 6.5 Trend analysis strategy

Trend classification is not derived from a single statistic. The script computes:

- active months
- total months
- recent share
- early share
- overall topic share
- linear slope over time
- coefficient of variation

It then classifies each topic as:

- `Persistent`
- `Trending`
- `Declining`
- `Episodic`

The playbook documents the rule logic, and the code refreshes trend labels from those metrics rather than treating trend names as static text.

This is a good design choice because it keeps trend labels reproducible and tied to observable temporal behavior.

### 6.6 Topic summary generation

Topic descriptions are generated by `build_topic_description_generative()`.

Strategy:

1. build a deterministic fallback summary from major topic, keywords, top flairs, and representative titles
2. if Groq is available, send a compact structured prompt asking for a 2-3 sentence analytical summary in JSON
3. fall back to deterministic text if the API is unavailable or errors

This hybrid approach is repeated across the codebase. The design pattern is:

- use LLMs for prose quality
- keep deterministic logic as the structural backbone
- never make the pipeline wholly dependent on remote generation

## 7. Stance Analysis Pipeline

### 7.1 Main responsibility

[scripts/topic_stance_analysis.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/topic_stance_analysis.py) turns topic-assigned comments into a preview of disagreement structure within each topic.

It writes:

- per-comment stance assignments
- topic-level support/opposition counts
- disagreement metrics
- representative comments
- user-group previews
- support/opposition summaries

### 7.2 Conceptual model

This is not a supervised stance classifier. The script says that explicitly. The current implementation is a discourse-camp proxy:

1. embed comments within a topic
2. split them into two clusters
3. choose the dominant cluster as `support`
4. choose the other cluster as `opposing`
5. summarize the main argument patterns on each side

This is a coarse but operationally useful strategy when gold stance labels do not exist.

### 7.3 Why stance analysis improved over the earlier keyword-only approach

The playbook notes that earlier stance outputs read like keyword summaries. The current code improves that in several ways.

#### 7.3.1 Embedding-based two-camp clustering

Per-topic comments are embedded with a sentence-transformer model and clustered via `MiniBatchKMeans(n_clusters=2)`.

This is intentionally simple:

- it assumes the major conflict inside a topic is roughly bipolar
- it is computationally cheap
- it is easy to explain to an analyst

The tradeoff is that it will flatten more complex multi-sided debates into two camps.

#### 7.3.2 Weighted dominance instead of raw size only

The dominant cluster is not chosen only by comment count. The code calculates:

- weighted size using `1 + log1p(score)` per comment
- support cue counts
- oppose cue counts

The chosen dominant cluster is the one with the best combination of weighted size and support-minus-oppose cues.

That improves robustness because a cluster with many low-value comments does not automatically outrank a smaller but higher-engagement cluster.

#### 7.3.3 Confidence estimation

For each comment, the script computes cosine similarity to both cluster centroids and stores the absolute difference as `stance_confidence`.

This is a useful UI/analysis hook because it creates a continuous signal rather than only a hard label.

#### 7.3.4 Better theme extraction

`cluster_keywords()` uses TF-IDF n-grams with custom stance stopwords and filters generic political vocabulary. This is more targeted than taking top raw terms from the whole topic.

#### 7.3.5 Representative comment selection

Representative comments are not chosen only by score. The ranking uses:

- proximity to the cluster centroid
- Reddit score

This is the right strategy for explanation cards because it favors comments that are both central to the cluster and visible in the discussion.

#### 7.3.6 Generative argument-map summaries

`generate_stance_summaries_generative()` uses:

- support keywords
- opposing keywords
- representative support comments
- representative opposing comments

to ask Groq for structured JSON with:

- `dominant_position_summary`
- `support_argument_summary`
- `opposing_argument_summary`

Fallback logic exists through `infer_dominant_position()` and `summarize_cluster_arguments()`.

This is the key change that shifts the output from keyword dump to argument summary.

### 7.4 User-group preview strategy

The stance pipeline also aggregates by `author_hash` and records for each user:

- dominant stance
- support comment count
- opposing comment count
- total score
- average confidence

Only a preview subset is kept, but this creates the base needed for a future overlap or stance-transition feature, which the playbook explicitly proposes.

### 7.5 Known limitations

The code still has clear limitations:

- exactly two camps are forced per topic
- stance labels are derived from cluster structure, not claim-target semantics
- cluster polarity can be topic-relative rather than globally comparable
- the app currently expects `data/topic_stance_preview/`, but the checked-in full-data sample lives in `data/topic_stance_preview_check_groq/`

## 8. Vector Index and Retrieval Data Design

### 8.1 Index builder responsibilities

[scripts/build_index.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/build_index.py) reads cleaned posts and comments and builds a persistent Chroma index.

The design follows the original RAG plan closely.

### 8.2 Two-collection design

The index uses two Chroma collections:

- `reddit_posts`
- `reddit_comments`

This is an important architectural decision. It allows the system to:

- rank posts as primary evidence units
- rank comments as supporting detail
- apply different retrieval and post-processing logic to posts and comments

If everything were indexed as one flat collection, the system would lose control over post/comment balance.

### 8.3 Chunking strategy

Posts:

- one chunk per post
- chunk text format: flair + title + selftext

Comments:

- only top 5 comments per post by score are kept
- comment chunk text prepends the parent post title

This is a very deliberate design:

- post-level retrieval gives stable topical anchors
- top-comment capping keeps the index manageable and biases toward higher-signal discussion
- parent-title prefixing makes isolated comment chunks interpretable during retrieval

### 8.4 Metadata strategy

The builder stores structured metadata per chunk, including:

- post ID / comment ID
- type
- title
- score
- month
- flair
- permalink

That metadata is later used for:

- reranking
- flair diversity filtering
- UI display
- source-post cards

### 8.5 Embedding choice

The RAG stack uses `sentence-transformers/all-mpnet-base-v2`, which is intentionally stronger than the topic pipeline’s default MiniLM setup.

This separation is sensible:

- topic modeling benefits from fast draft clustering
- retrieval benefits from richer semantic vectors

The design plan argued for exactly this upgrade, and the implementation follows it.

## 9. Routed QA Architecture

### 9.1 Why routing exists

The most important recent backend change is in [scripts/rag_query.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/rag_query.py).

The original plan mostly described one focused QA flow. The current implementation explicitly routes questions into four modes:

- `focused`
- `aggregate`
- `comparison`
- `multi-hop`

This is one of the strongest architectural choices in the repo because not all analyst questions should be handled by the same retrieval pattern.

### 9.2 Frontend-backend query contract

The contract is prefix-based:

- `focused:`
- `aggregate:`
- `comparison:`
- `multi-hop:`

`parse_question_type()` strips the prefix and returns:

- `query_type`
- `question_body`

The frontend now validates this locally before submission, which avoids ambiguous backend heuristics and makes analyst intent explicit.

### 9.3 Shared query flow

Across modes, the common steps are:

1. optionally translate non-English input into English
2. parse query type
3. route to the mode-specific builder
4. retrieve post and comment evidence
5. build a prompt tailored to the route
6. call Groq, Gemini, or both
7. optionally translate the answer back to the input language
8. return answer text plus source-post metadata and route metadata

### 9.4 Focused mode

Focused mode is the direct descendant of the original design.

Pipeline:

1. embed the English question with `all-mpnet-base-v2`
2. retrieve top candidate posts from `reddit_posts`
3. convert Chroma distance into a similarity score
4. rerank posts by `cosine_sim * log1p(reddit_score)`
5. apply a flair-diversity cap
6. retrieve top semantically relevant comments within each selected post
7. build a post-plus-comments context block
8. prompt the model to synthesize only from the supplied excerpts

Important constants:

- `CANDIDATE_POSTS = 30`
- `FINAL_POSTS = 5`
- `MAX_POSTS_PER_FLAIR = 2`
- `CANDIDATE_COMMENTS = 5`
- `NO_ANSWER_THRESHOLD = 0.35`

This is how the query is being routed in concrete terms: not just “dense retrieval,” but dense retrieval with score-weighted reranking and a diversity cap before comment expansion.

### 9.5 Aggregate mode

Aggregate questions are intentionally not treated as pure retrieval questions.

This is a direct response to the playbook’s concern that analyst questions like “Which politicians are most discussed?” should not be answered through naive nearest-neighbor retrieval.

Current aggregate mode strategy:

1. run a broader retrieval window for supporting examples
2. load deterministic corpus aggregates from `data/topic_analysis/aggregate_stats.json`
3. load modeled topic summaries from `data/topic_analysis/topic_summary.json`
4. build a structured aggregate brief
5. allow retrieved excerpts to act only as supporting examples
6. refuse unsupported scopes, especially entity/person ranking

Important constants:

- `AGGREGATE_CANDIDATE_POSTS = 50`
- `AGGREGATE_FINAL_POSTS = 8`
- `AGGREGATE_MAX_POSTS_PER_FLAIR = 3`
- `AGGREGATE_COMMENTS_PER_POST = 3`

The key design choice is conservative scope control. `aggregate_support_reason()` blocks unsupported entity-centric requests by design rather than hallucinating rankings that the system cannot compute canonically.

### 9.6 Comparison mode

Comparison mode handles side-by-side evidence gathering when the query can be parsed into two targets.

Flow:

1. parse comparison sides using regex patterns such as `vs`, `versus`, `difference between`, or `compare X and Y`
2. transform each side into a focused subquestion like `What do Reddit users say about <side>?`
3. retrieve each side independently
4. build separate left and right contexts
5. merge unique source posts for display
6. prompt the model to compare the two evidence sets explicitly

This is a good example of route specialization. Instead of retrieving one blended neighborhood for a comparison query, the code forces evidence separation first and synthesis second.

### 9.7 Multi-hop mode

Multi-hop mode decomposes a question into a small set of subqueries.

Current heuristic:

- keep the full question
- look for connectors such as `after`, `before`, `during`, `because`, `following`
- split around the first connector
- deduplicate and cap the hop list to three subqueries

Then:

1. retrieve each hop separately
2. build one context block per hop
3. mark hop-level low-confidence conditions
4. merge unique source posts
5. prompt the model to answer only after connecting the hops

This is intentionally lightweight. It is not a full planner, but it is enough to distinguish simple temporal or causal bridging questions from ordinary single-hop queries.

### 9.8 No-answer strategy

The design plan called for an adversarial threshold, and the implementation keeps that defense.

Mechanism:

- compute max retrieval similarity before reranking
- set `no_answer_flag` when it is below threshold
- append a stricter no-answer addendum to the system prompt

In aggregate mode, unsupported question types also force a no-answer path.

This two-layer design is solid:

- retrieval confidence acts as a structural guard
- prompt instructions act as a behavioral guard

### 9.9 Multilingual query routing

The RAG system is still English-indexed, but it supports multilingual usage through translation:

1. translate non-English questions to English using Groq
2. run English retrieval and English answer generation
3. translate the final answer back into the source language

Supported language codes are:

- `en`
- `hi`
- `es`
- `fr`

The UI currently exposes a Hindi toggle, but the backend contract is broader.

### 9.10 Model abstraction and pacing

The RAG layer can call:

- Groq
- Gemini
- both in parallel

Gemini calls are paced through a centralized lock and minimum interval helper. That mirrors the same pattern in [scripts/llm_summaries.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/llm_summaries.py), where Groq JSON calls also use centralized rate spacing.

This is a practical operational decision, not just a cleanliness detail. It reduces burst-related failures during repeated analyst usage and evaluation runs.

### 9.11 Response payload design

`run_rag_query()` returns more than answer text. It returns:

- `question`
- `question_body`
- `query_type`
- `english_question`
- `lang`
- `no_answer_flag`
- `max_cosine_sim`
- `retrieved_post_ids`
- `source_posts`
- `context`
- `route_metadata`
- provider-specific answers

This is the correct shape for a productized analyst tool because the frontend can expose:

- which route was used
- whether the route was low confidence
- what evidence posts were surfaced

without needing to reverse-engineer backend behavior.

## 10. Prompting Strategy

The prompting layer in `rag_query.py` uses:

- one shared base system prompt
- one route-specific guidance block per query type
- an optional low-confidence addendum
- route-specific user-context construction

Important design choices:

- answers must use only supplied evidence
- answers must return a fixed insufficient-information sentence when unsupported
- answer text should synthesize across evidence rather than summarize one thread
- answer text should not include inline citations
- source-post validation is handled separately through `source_posts`

That last point is a major product decision. The app chooses evidence transparency through separate cards rather than cluttering the answer body with citations.

## 11. Local Server and Product Integration

### 11.1 Server responsibilities

[scripts/local_app_server.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/local_app_server.py) is the integration layer.

It:

- serves the static app directory
- exposes `/api/bundle`
- exposes `/api/status`
- exposes `/api/query`
- exposes actions for topic analysis, stance analysis, and bundle rebuild

### 11.2 Runtime design

The server stores an `AppRuntime` object that:

- caches Chroma collections
- forwards QA requests to `rag_query.run_rag_query()`
- shells out to analysis/build scripts
- computes readiness state from the filesystem

This avoids pushing orchestration logic into the frontend.

### 11.3 Status endpoint design

`/api/status` reports readiness for:

- the app bundle
- the RAG index
- topic analysis outputs
- stance analysis outputs

This is important because the repository intentionally tolerates partially completed pipeline states. The frontend can render operational status instead of failing invisibly.

## 12. App Bundle Design

### 12.1 Purpose

[scripts/build_app_bundle.py](/C:/Users/91887/Documents/Reddit-Topic-Analysis/scripts/build_app_bundle.py) converts backend artifacts into a compact JSON file for the frontend.

This bundle is a deliberate anti-coupling layer. Rather than making the browser read multiple CSVs and JSON files directly, the builder pre-joins:

- aggregate stats
- topic run metadata
- topic summaries
- stance previews
- user-group previews
- comment previews
- timeline data
- known event markers

### 12.2 Enrichment strategy

Each topic is enriched with:

- `stance_preview`
- `user_groups_preview`
- `comment_preview`
- a `timeline` object containing months, post counts, shares, and aligned events

The bundle also builds `topic_tree`, grouping refined topics under major-topic nodes. This lets the frontend render a domain-first exploration view with almost no client-side transformation.

### 12.3 Graceful degradation

The builder uses `load_json_if_exists()` for stance artifacts. If stance outputs are absent:

- the bundle still builds
- stance mode is marked unavailable
- the rest of the topic explorer still works

This matches the implementation playbook’s requirement that the app must boot while stance work is unfinished.

## 13. Frontend Architecture

### 13.1 Overall structure

The frontend is intentionally lightweight:

- plain HTML
- one JS module
- one CSS file
- JSON fetched from the local API

There is no framework runtime. That is a reasonable choice for a local analyst tool because the application state is moderate and the deployment target is a single local host.

### 13.2 Layout strategy

[app/index.html](/C:/Users/91887/Documents/Reddit-Topic-Analysis/app/index.html) divides the UI into:

- a top bar with brand, query input, filters, language toggle, and model controls
- a left sidebar with navigation, dataset stats, model quality notes, and pipeline controls
- a main area with QA results and the topic explorer

The information architecture reflects two workflows running in parallel:

- “ask a question”
- “browse the topic map”

### 13.3 Frontend state model

[app/app.js](/C:/Users/91887/Documents/Reddit-Topic-Analysis/app/app.js) keeps a single `state` object with:

- `bundle`
- `appStatus`
- `activeTopicId`
- `expandedMajorTopics`
- `zoom`
- `language`
- query-menu state
- QA result state
- pipeline execution state

This is minimal but sufficient because the app is mostly server-driven and bundle-driven.

### 13.4 UI design choices

The current visual system in [app/styles.css](/C:/Users/91887/Documents/Reddit-Topic-Analysis/app/styles.css) is deliberate and not generic dashboard boilerplate.

Key choices:

- warm paper-toned background instead of stark white
- navy top bar for contrast and editorial feel
- muted red, gold, green, and blue used as semantic accents
- rounded cards and heavy shadows to separate analytic regions
- large grid-based shell with clear separation between operational controls and exploratory content

Specific design rationale visible in the CSS:

- the `:root` palette uses beige/sand surfaces to make the interface read more like a research workspace than a SaaS admin dashboard
- major-topic cards use a navy-blue gradient tint to visually establish them as higher-level taxonomy objects
- trend badges encode state semantically: green for persistent, red for trending, gold for declining, blue for neutral/episodic
- QA results and pipeline logs are rendered as separate cards so operational state and analytic output do not blur together

### 13.5 Query input design

The question input is one of the most intentional product surfaces.

Design choices:

- clicking or focusing the input opens a query-type menu
- the menu shows example prompts for each supported route
- invalid prefixes are marked inline before network submission
- the backend is called only after the contract is locally validated

This is better than implicit mode detection for an analyst tool because it teaches the supported question taxonomy directly in the UI.

### 13.6 Topic explorer design

The topic explorer is built around the refined taxonomy:

- major-topic accordion sections
- child issue-area cards
- zoom controls
- search, flair, date, and trend filters
- a detail panel for the selected topic

This matches the backend data model closely and exposes the refinement work instead of flattening it into a single list.

### 13.7 Topic detail design

The topic detail panel emphasizes:

- label and major-topic context
- summary description
- key metrics
- top keywords
- top flairs
- timeline visualization
- representative threads
- stance diagnostics

That is a good analyst-oriented composition because it mixes abstract summary, temporal trend, and evidence access in one place.

### 13.8 Timeline design

The current chart is a custom SVG line chart built from monthly counts. Event markers are projected vertically into the same coordinate system.

This is functional, but it is also one of the clearest places where the implementation still lags the playbook. The playbook explicitly calls for:

- daily bars across six months
- event-aligned markers at daily granularity

The current code still renders monthly line data.

### 13.9 QA result design

The QA panel displays:

- synthesized answer text
- route metadata
- top retrieved posts

This is a strong product decision. It separates:

- what the model concluded
- how the route behaved
- what evidence anchor posts were used

That structure fits the system’s emphasis on analyst validation rather than black-box chat output.

### 13.10 Pipeline controls

The sidebar can trigger:

- status refresh
- bundle rebuild
- topic analysis rerun
- stance preview rerun

This turns the app from a passive report into a thin operational console for the local pipeline.

## 14. How the Plans Map to the Current Implementation

### 14.1 `rag_design_plan.md`

This document supplied the baseline RAG architecture:

- two Chroma collections
- one post chunk per post
- top comments per post
- mpnet embeddings
- similarity plus log-score reranking
- flair diversity cap
- adversarial threshold
- common LLM abstraction

The implementation follows nearly all of that directly.

The main evolution beyond the plan is routed QA:

- aggregate mode
- comparison mode
- multi-hop mode
- multilingual input handling
- separate source-post display payload

### 14.2 `implementation_update_playbook.md`

This document is closer to the current product behavior. Major implemented items include:

- explicit prefixed query-type contract
- local frontend validation
- backend route metadata
- aggregate route backed by structured analytics
- comparison side-by-side retrieval
- multi-hop subquery decomposition
- centralized Gemini pacing
- local app server
- bundle build tolerance for missing stance outputs
- frontend API wiring to `/api/query`

Some backlog items are still visibly incomplete:

- sidebar nav buttons are still placeholders
- trend chart is still monthly SVG line, not daily bars
- app’s default stance-preview path is not populated
- aggregate entity-ranking support is still intentionally unsupported

## 15. Major Design Patterns Across the Codebase

Several design patterns repeat throughout the repository.

### 15.1 Deterministic structure plus generative prose

Used in:

- topic descriptions
- stance summaries
- QA answering

Pattern:

- deterministic extraction, clustering, or aggregation provides the structure
- LLMs improve narrative quality
- fallbacks preserve pipeline robustness

### 15.2 Graceful partial readiness

Used in:

- app bundle build
- status endpoint
- frontend quality panel

The code assumes some artifacts may be missing and tries to keep the rest of the tool working.

### 15.3 Explicit analyst contracts

Used in:

- prefixed question routing
- route metadata returned to the UI
- source-post evidence cards

The system avoids hidden inference where explicit mode selection is safer.

### 15.4 Separation of offline analytics and online QA

The code intentionally does not push all questions through one retrieval-only path. Aggregate questions can use precomputed corpus statistics while focused questions use evidence retrieval.

That is exactly the right separation for this domain.

## 16. Gaps, Risks, and Next Logical Improvements

The playbook’s remaining priorities are still technically sound.

Most important open items:

1. populate `data/topic_stance_preview/` with the current refined stance pipeline outputs so the app’s default bundle includes stance cards
2. replace the monthly line chart with the planned daily bar chart and daily event alignment
3. turn sidebar navigation buttons into actual routed views
4. extend aggregate analytics if entity-ranking questions are a real product need
5. consider whether two-cluster stance splitting is sufficient for topics with more than two persistent camps

Important modeling risks:

- topic labels are improved substantially, but the refinement layer is still rule-guided and may need periodic rule maintenance as the corpus domain shifts
- stance polarity is topic-local and may not map neatly across topics
- translation-mediated multilingual QA depends on translation quality, especially for politically charged nuance
- aggregate mode is correctly conservative today, but that means some analyst questions will still refuse until richer structured aggregates exist

## 17. Summary

This codebase has evolved from a set of analysis scripts into a coherent local analyst tool.

Its strongest technical characteristics are:

- a refined topic-modeling pipeline that goes beyond raw BERTopic output
- a stance-preview pipeline that converts clustered discourse into argument summaries
- a routed QA engine that chooses different strategies for focused, aggregate, comparison, and multi-hop questions
- a bundle-driven frontend that exposes both exploratory analytics and evidence-backed QA
- a pragmatic fallback philosophy that keeps the tool usable while some artifacts remain incomplete

The most important architectural idea in the repository is that different analyst questions require different evidence paths. The code reflects that clearly:

- focused questions use dense retrieval plus reranking
- aggregate questions use structured corpus summaries first
- comparison questions retrieve each side separately
- multi-hop questions decompose and fuse evidence across hops

That explicit routing strategy, combined with post-discovery topic refinement and stance summarization improvements, is what turns the repository from a prototype into an analyst-oriented system rather than a generic NLP experiment.
