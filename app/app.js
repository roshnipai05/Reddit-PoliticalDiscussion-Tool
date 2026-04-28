const state = {
  bundle: null,
  appStatus: null,
  activeTopicId: null,
  expandedMajorTopics: new Set(),
  zoom: 1,
  language: "en",
  queryTypeMenuOpen: false,
  qaResult: null,
  qaBusy: false,
  pipelineBusy: false,
  pipelineLog: "No pipeline action has been run yet.",
};

const QUESTION_TYPES = [
  {
    prefix: "focused:",
    label: "Focused",
    example: 'focused: "What are users saying about Biden dropping out?"',
  },
  {
    prefix: "aggregate:",
    label: "Aggregate",
    example: 'aggregate: "What topics dominate the corpus?"',
  },
  {
    prefix: "comparison:",
    label: "Comparison",
    example: 'comparison: "How does discussion of Harris differ from Trump?"',
  },
  {
    prefix: "multi-hop:",
    label: "Multi-hop",
    example: 'multi-hop: "How did views on Harris shift after Biden dropped out?"',
  },
];

const DEFAULT_QUERY_HINT =
  "Prefix each question with `focused:`, `aggregate:`, `comparison:`, or `multi-hop:`. Click the bar to see examples.";
const INVALID_QUERY_HINT =
  "Question type prefix required. Start with focused:, aggregate:, comparison:, or multi-hop:.";

const els = {
  subredditLabel: document.getElementById("subredditLabel"),
  overviewStats: document.getElementById("overviewStats"),
  aggregateGrid: document.getElementById("aggregateGrid"),
  qualityPanel: document.getElementById("qualityPanel"),
  topicTree: document.getElementById("topicTree"),
  topicDetail: document.getElementById("topicDetail"),
  topicCountLabel: document.getElementById("topicCountLabel"),
  searchInput: document.getElementById("searchInput"),
  trendFilter: document.getElementById("trendFilter"),
  flairFilter: document.getElementById("flairFilter"),
  dateFilter: document.getElementById("dateFilter"),
  zoomIn: document.getElementById("zoomIn"),
  zoomOut: document.getElementById("zoomOut"),
  zoomLabel: document.getElementById("zoomLabel"),
  languageToggle: document.getElementById("languageToggle"),
  modelSelect: document.getElementById("modelSelect"),
  askButton: document.getElementById("askButton"),
  conversationInput: document.getElementById("conversationInput"),
  queryTypeMenu: document.getElementById("queryTypeMenu"),
  queryTypeHint: document.getElementById("queryTypeHint"),
  qaStatusText: document.getElementById("qaStatusText"),
  qaResultPanel: document.getElementById("qaResultPanel"),
  refreshStatusButton: document.getElementById("refreshStatusButton"),
  rebuildBundleButton: document.getElementById("rebuildBundleButton"),
  runTopicAnalysisButton: document.getElementById("runTopicAnalysisButton"),
  runStanceAnalysisButton: document.getElementById("runStanceAnalysisButton"),
  pipelineStatus: document.getElementById("pipelineStatus"),
};

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value ?? 0);
}

function formatPercent(value) {
  return `${((value || 0) * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function trendClass(value) {
  return String(value || "").toLowerCase();
}

function normalizeQuestionPrefix(value) {
  return String(value || "").trimStart().toLowerCase();
}

function hasValidQuestionPrefix(value) {
  const normalized = normalizeQuestionPrefix(value);
  return QUESTION_TYPES.some((item) => normalized.startsWith(item.prefix));
}

function setQueryHint(message, isError = false) {
  els.queryTypeHint.textContent = message;
  els.queryTypeHint.classList.toggle("is-error", isError);
}

function renderQueryTypeMenu() {
  els.queryTypeMenu.innerHTML = QUESTION_TYPES.map(
    (item) => `
      <button class="query-type-option" type="button" data-prefix="${item.prefix}">
        <span class="query-type-label">${item.label}</span>
        <span class="query-type-example">${escapeHtml(item.example)}</span>
      </button>
    `
  ).join("");

  [...els.queryTypeMenu.querySelectorAll(".query-type-option")].forEach((button) => {
    button.addEventListener("click", () => {
      els.conversationInput.value = `${button.dataset.prefix} ""`;
      els.conversationInput.focus();
      els.conversationInput.setSelectionRange(
        button.dataset.prefix.length + 2,
        button.dataset.prefix.length + 2
      );
      validateConversationInput();
      closeQueryTypeMenu();
    });
  });
}

function openQueryTypeMenu() {
  state.queryTypeMenuOpen = true;
  els.queryTypeMenu.hidden = false;
}

function closeQueryTypeMenu() {
  state.queryTypeMenuOpen = false;
  els.queryTypeMenu.hidden = true;
}

function validateConversationInput() {
  const hasPrefix = hasValidQuestionPrefix(els.conversationInput.value);
  const isEmpty = !els.conversationInput.value.trim();
  const isInvalid = !isEmpty && !hasPrefix;
  els.conversationInput.classList.toggle("invalid-prefix", isInvalid);
  setQueryHint(isInvalid ? INVALID_QUERY_HINT : DEFAULT_QUERY_HINT, isInvalid);
  return !isInvalid;
}

function getTopicById(topicId) {
  if (!state.bundle) return null;
  return state.bundle.topics.find((topic) => Number(topic.topic_id) === Number(topicId)) || null;
}

function currentMonthFilter() {
  return els.dateFilter.value;
}

function topicMatchesFilters(topic) {
  const search = els.searchInput.value.trim().toLowerCase();
  const trend = els.trendFilter.value;
  const flair = els.flairFilter.value;
  const date = currentMonthFilter();
  const haystack = [
    topic.label,
    topic.topic_description,
    ...(topic.keywords || []),
    ...((topic.top_flairs || []).map((item) => item.flair)),
    topic.major_topic,
  ]
    .join(" ")
    .toLowerCase();

  const matchesSearch = !search || haystack.includes(search);
  const matchesTrend = trend === "all" || topic.trend_type === trend;
  const matchesFlair =
    flair === "all" || (topic.top_flairs || []).some((item) => item.flair === flair);
  const matchesDate =
    date === "all" ||
    (topic.timeline?.months || []).includes(date) ||
    topic.representative_posts?.some((post) => post.created_month === date);
  return matchesSearch && matchesTrend && matchesFlair && matchesDate;
}

function filteredTree() {
  if (!state.bundle) return [];
  return state.bundle.topic_tree
    .map((majorTopic) => ({
      ...majorTopic,
      children: majorTopic.children.filter((child) => {
        const topic = getTopicById(child.topic_id);
        return topic ? topicMatchesFilters(topic) : false;
      }),
    }))
    .filter((majorTopic) => majorTopic.children.length > 0);
}

function renderOverview() {
  if (!state.bundle) {
    els.subredditLabel.textContent = "Bundle not loaded";
    els.overviewStats.innerHTML = '<article class="metric-card"><div class="metric-label">Status</div><div class="metric-value">No bundle</div></article>';
    els.aggregateGrid.innerHTML = '<div class="sidebar-stat"><div class="metric-label">Bundle</div><div class="sidebar-value">Missing</div></div>';
    els.qualityPanel.innerHTML = '<div class="quality-item"><strong>Waiting for bundle</strong><p>Build or rebuild the bundle from the pipeline panel.</p></div>';
    return;
  }

  const bundle = state.bundle;
  els.subredditLabel.textContent = `${bundle.app_meta.subreddit} • ${bundle.app_meta.analysis_scope}`;

  const cards = [
    ["Major Topic Groups", bundle.topic_tree.length],
    ["Model Topics", bundle.overview.topic_count],
    ["Trending", bundle.overview.trending_topics],
    ["Persistent", bundle.overview.persistent_topics],
    ["Comments Analyzed", bundle.stance_preview_metadata.comment_count_analyzed || 0],
  ];
  els.overviewStats.innerHTML = cards
    .map(
      ([label, value]) => `
        <article class="metric-card">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${formatNumber(value)}</div>
        </article>
      `
    )
    .join("");

  const aggregateCards = [
    ["Posts", formatNumber(bundle.aggregate_stats.total_posts)],
    ["Users", formatNumber(bundle.aggregate_stats.total_unique_users)],
    ["Comments", formatNumber(bundle.aggregate_stats.total_comments)],
    ["Upvotes", formatNumber(bundle.aggregate_stats.total_upvotes)],
    [
      "Date Span",
      `${bundle.aggregate_stats.date_range_start} to ${bundle.aggregate_stats.date_range_end}`,
    ],
  ];
  els.aggregateGrid.innerHTML = aggregateCards
    .map(
      ([label, value]) => `
        <div class="sidebar-stat">
          <div class="metric-label">${label}</div>
          <div class="sidebar-value">${value}</div>
        </div>
      `
    )
    .join("");

  const stanceReady = bundle.app_meta.stance_mode !== "unavailable";
  els.qualityPanel.innerHTML = `
    <div class="quality-item">
      <strong>Topic coverage</strong>
      <p>${formatPercent(
        bundle.topic_run_metadata.assigned_non_outlier_posts / bundle.topic_run_metadata.post_count
      )} of retained posts were assigned to non-outlier topics.</p>
    </div>
    <div class="quality-item">
      <strong>Labeling</strong>
      <p>Labels and descriptions are derived from political title phrases, topic keywords, and representative threads.</p>
    </div>
    <div class="quality-item">
      <strong>Stance preview</strong>
      <p>${
        stanceReady
          ? `${formatNumber(bundle.stance_preview_metadata.comment_count_analyzed || 0)} comments were grouped into support/opposition previews across ${bundle.stance_preview_metadata.topic_count_analyzed || 0} major topics.`
          : "Stance preview outputs are not available yet. Topic exploration still works."
      }</p>
    </div>
  `;
}

function initFilters() {
  els.flairFilter.innerHTML = '<option value="all">All flairs</option>';
  els.dateFilter.innerHTML = '<option value="all">All months</option>';
  if (!state.bundle) {
    return;
  }

  const flairs = new Set();
  state.bundle.topics.forEach((topic) => {
    (topic.top_flairs || []).forEach((item) => flairs.add(item.flair));
  });

  [...flairs].sort().forEach((flair) => {
    const option = document.createElement("option");
    option.value = flair;
    option.textContent = flair;
    els.flairFilter.append(option);
  });

  state.bundle.app_meta.month_axis.forEach((month) => {
    const option = document.createElement("option");
    option.value = month;
    option.textContent = month;
    els.dateFilter.append(option);
  });
}

function renderTopicTree() {
  if (!state.bundle) {
    els.topicCountLabel.textContent = "Topic bundle unavailable";
    els.topicTree.innerHTML = '<div class="empty-state">Build the app bundle to browse topics.</div>';
    return;
  }

  const majorTopics = filteredTree();
  const filteredTopicCount = majorTopics.reduce((sum, node) => sum + node.children.length, 0);
  els.topicCountLabel.textContent = `${filteredTopicCount} topics visible across ${majorTopics.length} major groups`;
  if (!majorTopics.length) {
    els.topicTree.innerHTML = '<div class="empty-state">No topics match the active filters.</div>';
    return;
  }

  els.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  els.topicTree.style.transform = `scale(${state.zoom.toFixed(2)})`;
  els.topicTree.style.transformOrigin = "top left";

  els.topicTree.innerHTML = majorTopics
    .map((majorTopic) => {
      const expanded = state.expandedMajorTopics.has(majorTopic.id);
      return `
        <section class="major-topic ${expanded ? "expanded" : ""}" data-major-id="${majorTopic.id}">
          <button class="major-topic-button" type="button" data-major-id="${majorTopic.id}">
            <div>
              <div class="major-topic-title">${escapeHtml(majorTopic.label)}</div>
              <div class="major-topic-meta">${formatPercent(majorTopic.topic_share)} of posts • ${formatNumber(
                majorTopic.post_count
              )} posts</div>
            </div>
            <span class="expand-indicator">${expanded ? "−" : "+"}</span>
          </button>
          <div class="topic-branch ${expanded ? "expanded" : ""}">
            ${majorTopic.children
              .map((child) => {
                const topic = getTopicById(child.topic_id);
                return `
                  <article class="topic-node ${
                    Number(state.activeTopicId) === Number(child.topic_id) ? "active" : ""
                  }" data-topic-id="${child.topic_id}">
                    <div class="topic-node-header">
                      <div class="topic-node-title">${escapeHtml(child.label)}</div>
                      <span class="badge ${trendClass(child.trend_type)}">${child.trend_type}</span>
                    </div>
                    <div class="topic-node-copy">${escapeHtml(topic?.topic_description || "")}</div>
                    <div class="topic-node-meta">
                      <span>${formatPercent(child.topic_share)}</span>
                      <span>${formatNumber(child.post_count)} posts</span>
                    </div>
                  </article>
                `;
              })
              .join("")}
          </div>
        </section>
      `;
    })
    .join("");

  [...els.topicTree.querySelectorAll(".major-topic-button")].forEach((button) => {
    button.addEventListener("click", () => {
      const majorId = button.dataset.majorId;
      if (state.expandedMajorTopics.has(majorId)) {
        state.expandedMajorTopics.delete(majorId);
      } else {
        state.expandedMajorTopics.add(majorId);
      }
      renderTopicTree();
    });
  });

  [...els.topicTree.querySelectorAll(".topic-node")].forEach((node) => {
    node.addEventListener("click", () => {
      state.activeTopicId = Number(node.dataset.topicId);
      renderTopicTree();
      renderTopicDetail();
    });
  });
}

function polylinePoints(values, width, height, padding) {
  const maxValue = Math.max(...values, 1);
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;
  return values
    .map((value, index) => {
      const x = padding + (usableWidth * index) / Math.max(values.length - 1, 1);
      const y = padding + usableHeight - (usableHeight * value) / maxValue;
      return `${x},${y}`;
    })
    .join(" ");
}

function renderTimeline(topic) {
  const timeline = topic.timeline;
  if (!timeline || !timeline.post_counts?.length) {
    return '<div class="timeline-empty">No monthly trend data available.</div>';
  }

  const width = 520;
  const height = 220;
  const padding = 24;
  const points = polylinePoints(timeline.post_counts, width, height, padding);
  const eventMarkers = (timeline.events || [])
    .map((event) => {
      const monthIndex = timeline.months.indexOf(event.month);
      if (monthIndex < 0) return "";
      const x = padding + ((width - padding * 2) * monthIndex) / Math.max(timeline.months.length - 1, 1);
      return `
        <line x1="${x}" y1="${padding}" x2="${x}" y2="${height - padding}" class="event-line"></line>
        <text x="${x + 6}" y="${padding + 10}" class="event-label">${escapeHtml(event.label)}</text>
      `;
    })
    .join("");

  const axisLabels = timeline.months
    .map((month, index) => {
      const x = padding + ((width - padding * 2) * index) / Math.max(timeline.months.length - 1, 1);
      return `<text x="${x}" y="${height - 6}" text-anchor="middle" class="axis-label">${escapeHtml(
        month.slice(5)
      )}</text>`;
    })
    .join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" class="timeline-chart" role="img" aria-label="Topic frequency chart">
      <rect x="0" y="0" width="${width}" height="${height}" rx="18" class="chart-bg"></rect>
      ${[0.25, 0.5, 0.75].map((fraction) => {
        const y = padding + (height - padding * 2) * fraction;
        return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" class="grid-line"></line>`;
      }).join("")}
      ${eventMarkers}
      <polyline points="${points}" class="trend-line"></polyline>
      ${timeline.post_counts
        .map((value, index) => {
          const x = padding + ((width - padding * 2) * index) / Math.max(timeline.months.length - 1, 1);
          const maxValue = Math.max(...timeline.post_counts, 1);
          const y = padding + (height - padding * 2) - ((height - padding * 2) * value) / maxValue;
          return `<circle cx="${x}" cy="${y}" r="4" class="trend-point"></circle>`;
        })
        .join("")}
      ${axisLabels}
    </svg>
  `;
}

function renderStancePanel(topic) {
  const preview = topic.stance_preview;
  if (!preview) {
    return '<div class="detail-copy muted">No stance preview is available for this topic yet.</div>';
  }
  return `
    <div class="stance-grid">
      <article class="stance-card">
        <div class="detail-subtitle">Dominant position</div>
        <p class="detail-copy">${escapeHtml(preview.dominant_position_summary)}</p>
      </article>
      <article class="stance-card">
        <div class="detail-subtitle">Agreement / disagreement</div>
        <div class="stance-meter">
          <div class="stance-support" style="width:${preview.support_share * 100}%"></div>
          <div class="stance-opposing" style="width:${preview.opposing_share * 100}%"></div>
        </div>
        <div class="detail-meta">
          <span>${formatPercent(preview.support_share)} support</span>
          <span>${formatPercent(preview.opposing_share)} opposing</span>
          <span>Index ${preview.disagreement_index.toFixed(2)}</span>
        </div>
      </article>
      <article class="stance-card">
        <div class="detail-subtitle">Support-side summary</div>
        <p class="detail-copy">${escapeHtml(preview.support_argument_summary)}</p>
      </article>
      <article class="stance-card">
        <div class="detail-subtitle">Opposing-side summary</div>
        <p class="detail-copy">${escapeHtml(preview.opposing_argument_summary)}</p>
      </article>
    </div>
  `;
}

function renderTopicDetail() {
  const topic = getTopicById(state.activeTopicId);
  if (!topic) {
    els.topicDetail.innerHTML = '<div class="empty-state">Select a topic to inspect it.</div>';
    return;
  }

  const topFlairs = (topic.top_flairs || [])
    .map((item) => `<span class="chip">${escapeHtml(item.flair)} ${formatPercent(item.share_within_topic)}</span>`)
    .join("");
  const keywords = (topic.keywords || [])
    .slice(0, 10)
    .map((keyword) => `<span class="chip keyword-chip">${escapeHtml(keyword)}</span>`)
    .join("");
  const representativePosts = (topic.representative_posts || [])
    .slice(0, 4)
    .map(
      (post) => `
        <article class="post-card">
          <div class="post-title">${escapeHtml(post.title)}</div>
          <div class="detail-meta">
            <span>${escapeHtml(post.link_flair_text || "Unspecified")}</span>
            <span>${formatNumber(post.score)} score</span>
            <span>${formatNumber(post.num_comments)} comments</span>
            <span>${escapeHtml(post.created_month)}</span>
          </div>
          <a href="https://reddit.com${post.permalink}" target="_blank" rel="noreferrer">Open Reddit thread</a>
        </article>
      `
    )
    .join("");

  els.topicDetail.innerHTML = `
    <div class="detail-header">
      <div>
        <div class="eyebrow">${escapeHtml(topic.major_topic)}</div>
        <h3>${escapeHtml(topic.label)}</h3>
      </div>
      <div class="detail-badges">
        <span class="badge ${trendClass(topic.trend_type)}">${topic.trend_type}</span>
        <span class="badge neutral">${formatPercent(topic.topic_share)} of posts</span>
      </div>
    </div>

    <p class="detail-copy">${escapeHtml(topic.topic_description)}</p>

    <div class="detail-metrics">
      <div class="metric-pill">
        <span class="metric-label">Posts</span>
        <strong>${formatNumber(topic.post_count)}</strong>
      </div>
      <div class="metric-pill">
        <span class="metric-label">Active months</span>
        <strong>${formatNumber(topic.active_months)}</strong>
      </div>
      <div class="metric-pill">
        <span class="metric-label">Trend lift</span>
        <strong>${Number(topic.trend_lift || 0).toFixed(2)}x</strong>
      </div>
    </div>

    <section class="detail-section">
      <div class="detail-subtitle">Top keywords</div>
      <div class="chip-row">${keywords}</div>
    </section>

    <section class="detail-section">
      <div class="detail-subtitle">Top flairs</div>
      <div class="chip-row">${topFlairs}</div>
    </section>

    <section class="detail-section">
      <div class="detail-subtitle">Frequency over time</div>
      ${renderTimeline(topic)}
    </section>

    <section class="detail-section">
      <div class="detail-subtitle">Representative threads</div>
      <div class="post-list">${representativePosts}</div>
    </section>

    <section class="detail-section">
      <div class="detail-subtitle">Stance diagnostics</div>
      ${renderStancePanel(topic)}
    </section>
  `;
}

function ensureValidActiveTopic() {
  const visibleTree = filteredTree();
  const visibleTopicIds = visibleTree.flatMap((majorTopic) =>
    majorTopic.children.map((child) => Number(child.topic_id))
  );
  if (!visibleTopicIds.length) {
    state.activeTopicId = null;
    return;
  }
  if (!visibleTopicIds.includes(Number(state.activeTopicId))) {
    state.activeTopicId = visibleTopicIds[0];
  }
  visibleTree.forEach((majorTopic) => {
    if (majorTopic.children.some((child) => Number(child.topic_id) === Number(state.activeTopicId))) {
      state.expandedMajorTopics.add(majorTopic.id);
    }
  });
}

function renderPipelineStatus() {
  const status = state.appStatus;
  if (!status) {
    els.pipelineStatus.innerHTML = '<div class="quality-item">Status not loaded yet.</div>';
    return;
  }

  const topicReady = status.topic_analysis?.ready ? "Ready" : "Missing";
  const stanceReady = status.stance_analysis?.ready ? "Ready" : "Missing";
  const ragReady = status.rag?.index_ready ? "Ready" : "Missing";
  const bundleReady = status.app?.bundle_exists ? "Ready" : "Missing";
  els.pipelineStatus.innerHTML = `
    <div class="quality-item">
      <strong>Backend status</strong>
      <p>Bundle: ${bundleReady} • RAG index: ${ragReady} • Topic analysis: ${topicReady} • Stance preview: ${stanceReady}</p>
    </div>
    <pre class="pipeline-log">${escapeHtml(state.pipelineLog)}</pre>
  `;
}

function renderQaResult() {
  if (!state.qaResult) {
    els.qaResultPanel.innerHTML = '<div class="empty-state">No QA result yet.</div>';
    return;
  }

  const result = state.qaResult;
  const answerCandidates = [result.groq_answer, result.gemini_answer].filter(Boolean);
  const answer =
    answerCandidates.find((item) => !String(item).startsWith("[")) ||
    answerCandidates[0] ||
    "No answer returned.";
  const sources = (result.source_posts || [])
    .map(
      (post) => `
        <article class="post-card">
          <div class="post-title">${escapeHtml(post.title)}</div>
          <div class="detail-meta">
            <span>${escapeHtml(post.flair)}</span>
            <span>sim ${Number(post.cosine_sim).toFixed(4)}</span>
            <span>${formatNumber(post.score)} score</span>
            <span>${escapeHtml(post.created_month)}</span>
          </div>
          ${post.permalink ? `<a href="${escapeHtml(post.permalink)}" target="_blank" rel="noreferrer">Open Reddit thread</a>` : ""}
        </article>
      `
    )
    .join("");

  els.qaStatusText.textContent = `Last query ran in ${result.query_type} mode using ${result.lang.toUpperCase()} input.`;
  els.qaResultPanel.innerHTML = `
    <div class="qa-grid">
      <article class="qa-card">
        <div class="detail-subtitle">Answer</div>
        <div class="qa-meta">
          <span>Mode: ${escapeHtml(result.query_type)}</span>
          <span>Model: ${escapeHtml(els.modelSelect.value)}</span>
          <span>Max similarity: ${Number(result.max_cosine_sim || 0).toFixed(4)}</span>
          <span>${result.no_answer_flag ? "Low-confidence route" : "Evidence-backed route"}</span>
        </div>
        <div class="qa-answer">${escapeHtml(answer)}</div>
      </article>
      <article class="qa-card">
        <div class="detail-subtitle">Route metadata</div>
        <pre class="pipeline-log">${escapeHtml(JSON.stringify(result.route_metadata || {}, null, 2))}</pre>
      </article>
    </div>
    <div class="qa-card">
      <div class="detail-subtitle">Top retrieved posts</div>
      <div class="qa-source-list">${sources || '<div class="muted">No source posts returned.</div>'}</div>
    </div>
  `;
}

function render() {
  ensureValidActiveTopic();
  renderOverview();
  renderTopicTree();
  renderTopicDetail();
  renderPipelineStatus();
  renderQaResult();
}

async function loadBundle() {
  try {
    state.bundle = await apiFetch(`/api/bundle?ts=${Date.now()}`, { method: "GET" });
    state.expandedMajorTopics = new Set((state.bundle.topic_tree || []).slice(0, 2).map((node) => node.id));
  } catch (error) {
    state.bundle = null;
    state.pipelineLog = `Bundle load failed: ${error.message}`;
  }
}

async function loadStatus() {
  try {
    state.appStatus = await apiFetch("/api/status", { method: "GET" });
  } catch (error) {
    state.appStatus = null;
    state.pipelineLog = `Status load failed: ${error.message}`;
  }
}

async function refreshAppData() {
  await Promise.all([loadBundle(), loadStatus()]);
  initFilters();
  render();
}

function setQaBusy(isBusy) {
  state.qaBusy = isBusy;
  els.askButton.disabled = isBusy;
  els.askButton.textContent = isBusy ? "Running..." : "Run QA";
}

async function submitQuery() {
  if (!validateConversationInput()) {
    return;
  }

  const question = els.conversationInput.value.trim();
  if (!question) {
    return;
  }

  setQaBusy(true);
  els.qaStatusText.textContent = "Query in progress...";
  try {
    state.qaResult = await apiFetch("/api/query", {
      method: "POST",
      body: JSON.stringify({
        question,
        lang: state.language,
        model: els.modelSelect.value,
        query_type: "auto",
      }),
    });
    renderQaResult();
  } catch (error) {
    state.qaResult = {
      query_type: "error",
      lang: state.language,
      max_cosine_sim: 0,
      no_answer_flag: true,
      route_metadata: {},
      source_posts: [],
      groq_answer: `Query failed: ${error.message}`,
      gemini_answer: null,
    };
    renderQaResult();
  } finally {
    setQaBusy(false);
  }
}

function setPipelineBusy(isBusy, button = null) {
  state.pipelineBusy = isBusy;
  [
    els.refreshStatusButton,
    els.rebuildBundleButton,
    els.runTopicAnalysisButton,
    els.runStanceAnalysisButton,
  ].forEach((element) => {
    element.disabled = isBusy;
  });
  if (button) {
    button.textContent = isBusy ? "Running..." : button.dataset.label;
  }
}

async function runPipelineAction(button, endpoint, successLabel, rebuildBundleAfter = false) {
  setPipelineBusy(true, button);
  state.pipelineLog = `${successLabel} started...`;
  renderPipelineStatus();
  try {
    const result = await apiFetch(endpoint, { method: "POST", body: "{}" });
    let rebuildResult = null;
    if (result.ok && rebuildBundleAfter) {
      rebuildResult = await apiFetch("/api/actions/rebuild-bundle", { method: "POST", body: "{}" });
    }
    state.pipelineLog = [
      `${successLabel}: ${result.ok ? "completed" : "failed"}`,
      "",
      "STDOUT:",
      result.stdout || "(no stdout)",
      "",
      "STDERR:",
      result.stderr || "(no stderr)",
      ...(rebuildResult
        ? [
            "",
            "Bundle rebuild:",
            rebuildResult.ok ? "completed" : "failed",
            "",
            "BUNDLE STDOUT:",
            rebuildResult.stdout || "(no stdout)",
            "",
            "BUNDLE STDERR:",
            rebuildResult.stderr || "(no stderr)",
          ]
        : []),
    ].join("\n");
    await refreshAppData();
  } catch (error) {
    state.pipelineLog = `${successLabel} failed: ${error.message}`;
    renderPipelineStatus();
  } finally {
    setPipelineBusy(false, button);
  }
}

async function boot() {
  renderQueryTypeMenu();
  await refreshAppData();

  [els.searchInput, els.trendFilter, els.flairFilter, els.dateFilter].forEach((element) => {
    element.addEventListener("input", render);
    element.addEventListener("change", render);
  });

  els.zoomIn.addEventListener("click", () => {
    state.zoom = Math.min(1.6, state.zoom + 0.1);
    renderTopicTree();
  });
  els.zoomOut.addEventListener("click", () => {
    state.zoom = Math.max(0.8, state.zoom - 0.1);
    renderTopicTree();
  });

  els.conversationInput.addEventListener("focus", openQueryTypeMenu);
  els.conversationInput.addEventListener("click", openQueryTypeMenu);
  els.conversationInput.addEventListener("input", validateConversationInput);
  els.conversationInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      submitQuery();
    }
  });
  els.conversationInput.addEventListener("blur", () => {
    window.setTimeout(closeQueryTypeMenu, 120);
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".prompt-box")) {
      closeQueryTypeMenu();
    }
  });

  els.languageToggle.addEventListener("click", () => {
    state.language = state.language === "en" ? "hi" : "en";
    const hindiMode = state.language === "hi";
    els.languageToggle.setAttribute("aria-pressed", String(hindiMode));
    els.languageToggle.textContent = `Hindi Mode: ${hindiMode ? "On" : "Off"}`;
  });

  els.askButton.addEventListener("click", submitQuery);
  els.refreshStatusButton.dataset.label = "Refresh status";
  els.rebuildBundleButton.dataset.label = "Rebuild bundle";
  els.runTopicAnalysisButton.dataset.label = "Run topic analysis";
  els.runStanceAnalysisButton.dataset.label = "Run stance preview";
  els.refreshStatusButton.addEventListener("click", refreshAppData);
  els.rebuildBundleButton.addEventListener("click", () =>
    runPipelineAction(els.rebuildBundleButton, "/api/actions/rebuild-bundle", "Bundle rebuild")
  );
  els.runTopicAnalysisButton.addEventListener("click", () =>
    runPipelineAction(els.runTopicAnalysisButton, "/api/actions/run-topic-analysis", "Topic analysis", true)
  );
  els.runStanceAnalysisButton.addEventListener("click", () =>
    runPipelineAction(els.runStanceAnalysisButton, "/api/actions/run-stance-analysis", "Stance preview", true)
  );

  validateConversationInput();
  render();
}

boot();
