# Reddit Political Discussion Analysis

This project is a local NLP-based analysis tool built on posts and comments from [`r/PoliticalDiscussion`](https://www.reddit.com/r/PoliticalDiscussion/). It is designed for analyst-style exploration rather than casual browsing: the application helps a user study what political topics dominated the subreddit, how those topics changed over time, where disagreement was concentrated, and how different issue areas connect through shared participation.

The tool combines an offline analysis pipeline with an interactive local web application. Topic modelling organises the corpus into refined issue areas, trend analysis separates persistent themes from event-driven spikes, stance analysis gives a support-versus-opposition preview within each topic, and the routed RAG system allows direct question answering over the same corpus.

## What the Application Does

The application is built around a few connected analysis features:

- **Corpus overview:** shows dataset-wide statistics such as post volume, user count, date range, and other aggregate properties of the collected Reddit data.
<img src="assets\Agg-Stats.jpg">

- **Topic map:** groups posts into analyst-readable political issue areas and then further into broader major domains such as elections, institutions, economy, rights, foreign policy, and media narratives.
<img src="assets\Topic-Map.jpg">

- **Trend monitoring:** identifies whether a topic is persistent, trending, declining, or episodic across the July--December 2024 time window.
  
- **Stance preview:** approximates disagreement inside each topic by splitting comments into two discourse camps and summarising the dominant and opposing arguments.
<img src="assets\Stance-Analysis.jpg">

- **User demographics / participation overlap:** lets the user compare selected topics and inspect how much their participant bases overlap. This is useful for understanding whether the same repeat users are driving multiple discussions.
<img src="assets\User-Demographics.jpg">

- **Conversation QA:** supports question answering over the corpus through routed query modes, including focused, aggregate, comparison, and multi-hop questions.
<img src="assets\Query-System.jpg">

- **Multilingual access:** supports translation-based querying so the same English corpus can be explored through Hindi and other supported languages.

## How It Can Be Used as a Data Analysis Tool

The application is most useful when its features are used together.

An analyst can begin with the corpus statistics to understand the scale and coverage of the dataset, move into the topic map to identify major issue areas, inspect monthly topic trajectories to see what rose or faded, and then open stance summaries to understand how arguments were structured inside those topics. From there, the user-demographics view can be used to see whether the same users are active across multiple issues, and the QA system can be used to ask targeted follow-up questions grounded in the corpus.

This makes the tool useful for research questions such as:

- Which political issues dominated subreddit attention during the 2024 US election cycle?
- Which discussions were persistent structural concerns, and which were short-lived event reactions?
- Where was disagreement strongest, and what were the main lines of argument on each side?
- Do some topics share the same core participant base, suggesting recurring political coalitions or discourse blocs?
- How does discussion of one political figure, policy issue, or institution compare to another?

In short, the project functions as a compact political-discourse analysis workspace rather than just a topic-model visualiser or a generic chatbot.

## Running the Local App

Run the local server from the repository root:

```powershell
python scripts/local_app_server.py
```

Then open:

```text
http://127.0.0.1:8000
```

The server:

- serves the frontend from `app/`
- exposes `/api/query` for routed RAG QA
- exposes `/api/status` for pipeline readiness
- exposes UI-triggerable actions for topic analysis, stance preview, and app bundle rebuild


