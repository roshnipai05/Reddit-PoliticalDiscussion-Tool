const state = {
  bundle: null,
  appStatus: null,
  activeTopicId: null,
  expandedMajorTopics: new Set(),
  zoom: 1,
  language: "en",
  model: "both",
  activeView: "topic-map",
  queryTypeMenuOpen: false,
  qaResult: null,
  qaBusy: false,
  pipelineBusy: false,
  pipelineLog: "No pipeline action has been run yet.",
  selectedDemographicTopics: [],
  demographicStanceFilter: "both",
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
  demographicsTopicPicker: document.getElementById("demographicsTopicPicker"),
  demographicsSummary: document.getElementById("demographicsSummary"),
  overlapChart: document.getElementById("overlapChart"),
  overlapLegend: document.getElementById("overlapLegend"),
  overlapTable: document.getElementById("overlapTable"),
  demographicsNote: document.getElementById("demographicsNote"),
  navItems: [...document.querySelectorAll(".nav-item[data-view]")],
  pageSections: [...document.querySelectorAll(".page-section[data-page]")],
  languageToggles: [...document.querySelectorAll(".flag-toggle[data-language]")],
  modelToggles: [...document.querySelectorAll(".model-toggle[data-model]")],
  stanceToggles: [...document.querySelectorAll(".stance-toggle[data-stance-filter]")],
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

function formatMonthLabel(value) {
  const [year, month] = String(value || "").split("-");
  if (!year || !month) return String(value || "");
  const date = new Date(`${year}-${month}-01T00:00:00Z`);
  return new Intl.DateTimeFormat("en-US", { month: "short" }).format(date);
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

function maybeOpenQueryTypeMenu() {
  if (!els.conversationInput.value.trim()) {
    openQueryTypeMenu();
  }
}

function setActiveView(view) {
  state.activeView = view;
  els.navItems.forEach((item) => {
    item.classList.toggle("active", item.dataset.view === view);
  });
  els.pageSections.forEach((section) => {
    section.hidden = section.dataset.page !== view;
  });
  if (view !== "conversation-qa") {
    closeQueryTypeMenu();
  }
}

function syncLanguageToggles() {
  els.languageToggles.forEach((button) => {
    const active = button.dataset.language === state.language;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function syncModelToggles() {
  els.modelToggles.forEach((button) => {
    const active = button.dataset.model === state.model;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function getTopicById(topicId) {
  if (!state.bundle) return null;
  return state.bundle.topics.find((topic) => Number(topic.topic_id) === Number(topicId)) || null;
}

function getDemographicTopics() {
  if (!state.bundle) return [];
  return state.bundle.topics
    .filter((topic) => topic.user_groups_preview?.users?.length)
    .sort((left, right) => {
      const leftUsers = left.user_groups_preview?.users?.length || 0;
      const rightUsers = right.user_groups_preview?.users?.length || 0;
      return rightUsers - leftUsers;
    });
}

function getFilteredDemographicUsers(topic, stanceFilter = state.demographicStanceFilter) {
  const users = topic?.user_groups_preview?.users || [];
  if (stanceFilter === "both") {
    return users;
  }
  return users.filter((user) => user.dominant_stance === stanceFilter);
}

function getTopicSupportOpposingCounts(topic) {
  const preview = topic?.user_groups_preview;
  return {
    support: Number(preview?.support_users || 0),
    opposing: Number(preview?.opposing_users || 0),
  };
}

function getSelectedDemographicTopics() {
  return state.selectedDemographicTopics
    .map((topicId) => getTopicById(topicId))
    .filter((topic) => topic?.user_groups_preview?.users?.length);
}

function syncDemographicStanceToggles() {
  els.stanceToggles.forEach((button) => {
    const active = button.dataset.stanceFilter === state.demographicStanceFilter;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
}

function ensureValidDemographicSelection() {
  const availableTopicIds = new Set(getDemographicTopics().map((topic) => Number(topic.topic_id)));
  state.selectedDemographicTopics = state.selectedDemographicTopics
    .filter((topicId) => availableTopicIds.has(Number(topicId)))
    .slice(0, 4);

  if (state.selectedDemographicTopics.length >= 2) {
    return;
  }

  const defaults = getDemographicTopics()
    .slice(0, 2)
    .map((topic) => Number(topic.topic_id));
  state.selectedDemographicTopics = defaults;
}

function toggleDemographicTopic(topicId) {
  const normalizedId = Number(topicId);
  const next = [...state.selectedDemographicTopics];
  const currentIndex = next.indexOf(normalizedId);
  if (currentIndex >= 0) {
    next.splice(currentIndex, 1);
  } else if (next.length < 4) {
    next.push(normalizedId);
  }
  state.selectedDemographicTopics = next;
  renderUserDemographics();
}

function formatPercentFromCount(count, total) {
  if (!total) return "0.0%";
  return `${((count / total) * 100).toFixed(1)}%`;
}

function computeDemographicOverlap(topics, stanceFilter = state.demographicStanceFilter) {
  const topicSets = topics.map((topic) => {
    const filteredUsers = getFilteredDemographicUsers(topic, stanceFilter);
    return {
      topic,
      users: filteredUsers,
      userSet: new Set(filteredUsers.map((user) => user.author_hash)),
    };
  });

  const participants = new Set();
  topicSets.forEach((entry) => {
    entry.userSet.forEach((authorHash) => participants.add(authorHash));
  });

  const masks = new Map();
  participants.forEach((authorHash) => {
    const membership = topicSets
      .map((entry, index) => (entry.userSet.has(authorHash) ? String(index) : null))
      .filter(Boolean);
    const key = membership.join("|");
    if (!key) return;
    masks.set(key, (masks.get(key) || 0) + 1);
  });

  const pairwise = [];
  for (let leftIndex = 0; leftIndex < topicSets.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < topicSets.length; rightIndex += 1) {
      const sharedUsers = [...topicSets[leftIndex].userSet].filter((authorHash) =>
        topicSets[rightIndex].userSet.has(authorHash)
      ).length;
      pairwise.push({
        leftTopic: topicSets[leftIndex].topic,
        rightTopic: topicSets[rightIndex].topic,
        sharedUsers,
      });
    }
  }

  return {
    topicSets,
    participantCount: participants.size,
    exactMasks: [...masks.entries()]
      .map(([key, count]) => ({
        key,
        count,
        indexes: key.split("|").map((value) => Number(value)),
      }))
      .sort((left, right) => right.count - left.count),
    pairwise,
  };
}

function getOverlapLayout(topicCount) {
  if (topicCount === 2) {
    return [
      { x: 260, y: 208, rx: 120, ry: 98, rotation: -8 },
      { x: 500, y: 208, rx: 120, ry: 98, rotation: 8 },
    ];
  }
  if (topicCount === 3) {
    return [
      { x: 270, y: 222, rx: 116, ry: 95, rotation: -14 },
      { x: 490, y: 222, rx: 116, ry: 95, rotation: 14 },
      { x: 380, y: 118, rx: 120, ry: 98, rotation: 0 },
    ];
  }
  return [
    { x: 270, y: 130, rx: 108, ry: 88, rotation: -18 },
    { x: 494, y: 130, rx: 108, ry: 88, rotation: 18 },
    { x: 270, y: 286, rx: 108, ry: 88, rotation: 14 },
    { x: 494, y: 286, rx: 108, ry: 88, rotation: -14 },
  ];
}

function renderOverlapChart(analysis) {
  const { topicSets, exactMasks, participantCount } = analysis;
  const layout = getOverlapLayout(topicSets.length);
  const colors = ["#305987", "#b33a31", "#396c55", "#9f7f3a"];
  const maxTopicUsers = Math.max(...topicSets.map((entry) => entry.userSet.size), 1);

  const circles = topicSets
    .map((entry, index) => {
      const base = layout[index];
      const ratio = Math.sqrt(entry.userSet.size / maxTopicUsers || 0);
      const rx = Math.max(72, base.rx * (0.72 + ratio * 0.28));
      const ry = Math.max(58, base.ry * (0.72 + ratio * 0.28));
      return { ...base, rx, ry, color: colors[index], topic: entry.topic, count: entry.userSet.size };
    });

  const circleMarkup = circles
    .map(
      (circle, index) => `
        <g>
          <ellipse
            cx="${circle.x}"
            cy="${circle.y}"
            rx="${circle.rx}"
            ry="${circle.ry}"
            transform="rotate(${circle.rotation} ${circle.x} ${circle.y})"
            fill="${circle.color}22"
            stroke="${circle.color}"
            stroke-width="2.5"
          ></ellipse>
          <text x="${circle.x}" y="${circle.y - circle.ry - 16}" text-anchor="middle" class="venn-topic-label">
            ${escapeHtml(circle.topic.label)}
          </text>
          <text x="${circle.x}" y="${circle.y - circle.ry - 1}" text-anchor="middle" class="venn-topic-count">
            ${formatNumber(circle.count)} users
          </text>
        </g>
      `
    )
    .join("");

  const labelMarkup = exactMasks
    .filter((mask) => mask.count > 0)
    .map((mask, index) => {
      const centers = mask.indexes.map((topicIndex) => layout[topicIndex]);
      const averageX = centers.reduce((sum, item) => sum + item.x, 0) / centers.length;
      const averageY = centers.reduce((sum, item) => sum + item.y, 0) / centers.length;
      const yOffset = mask.indexes.length === 1 ? 24 : mask.indexes.length === topicSets.length ? -8 : 0;
      const xOffset = ((index % 3) - 1) * 12;
      return `
        <g>
          <rect x="${averageX - 44 + xOffset}" y="${averageY - 20 + yOffset}" width="88" height="38" rx="12" class="venn-label-bg"></rect>
          <text x="${averageX + xOffset}" y="${averageY - 4 + yOffset}" text-anchor="middle" class="venn-label-value">
            ${formatNumber(mask.count)}
          </text>
          <text x="${averageX + xOffset}" y="${averageY + 11 + yOffset}" text-anchor="middle" class="venn-label-share">
            ${formatPercentFromCount(mask.count, participantCount)}
          </text>
        </g>
      `;
    })
    .join("");

  return `
    <svg viewBox="0 0 760 420" class="overlap-chart" role="img" aria-label="Cross-topic user overlap chart">
      <rect x="0" y="0" width="760" height="420" rx="24" class="chart-bg"></rect>
      ${circleMarkup}
      ${labelMarkup}
    </svg>
  `;
}

function renderUserDemographics() {
  if (!els.demographicsTopicPicker) {
    return;
  }

  const demographicTopics = getDemographicTopics();
  if (!state.bundle || !demographicTopics.length) {
    els.demographicsTopicPicker.innerHTML = "";
    els.demographicsSummary.innerHTML = "";
    els.overlapChart.innerHTML =
      '<div class="empty-state compact-empty">Stance data is unavailable. Run stance preview to populate this view.</div>';
    els.overlapLegend.innerHTML = "";
    els.overlapTable.innerHTML = "";
    els.demographicsNote.textContent =
      "This view depends on topic-level stance user groups. When those outputs are missing, the page remains read-only.";
    return;
  }

  ensureValidDemographicSelection();
  syncDemographicStanceToggles();

  const selectedIds = new Set(state.selectedDemographicTopics.map((topicId) => Number(topicId)));
  els.demographicsTopicPicker.innerHTML = demographicTopics
    .map((topic) => {
      const selected = selectedIds.has(Number(topic.topic_id));
      const disabled = !selected && state.selectedDemographicTopics.length >= 4;
      return `
        <button
          class="topic-picker-pill ${selected ? "selected" : ""}"
          type="button"
          data-topic-toggle="${topic.topic_id}"
          ${disabled ? "disabled" : ""}
        >
          <span class="topic-picker-label">${escapeHtml(topic.label)}</span>
          <span class="topic-picker-meta">${formatNumber(topic.user_groups_preview.users.length)} users</span>
        </button>
      `;
    })
    .join("");

  [...els.demographicsTopicPicker.querySelectorAll("[data-topic-toggle]")].forEach((button) => {
    button.addEventListener("click", () => toggleDemographicTopic(button.dataset.topicToggle));
  });

  const selectedTopics = getSelectedDemographicTopics();
  if (selectedTopics.length < 2) {
    els.demographicsSummary.innerHTML = `
      <article class="metric-card compact-card">
        <div class="metric-label">Selection</div>
        <div class="metric-value">${formatNumber(selectedTopics.length)}</div>
      </article>
    `;
    els.overlapChart.innerHTML =
      '<div class="empty-state compact-empty">Select at least two topics to compute overlap.</div>';
    els.overlapLegend.innerHTML = "";
    els.overlapTable.innerHTML = "";
    els.demographicsNote.textContent =
      "Percentages will use the union of users who appear in at least one currently selected topic under the active stance filter.";
    return;
  }

  const analysis = computeDemographicOverlap(selectedTopics);
  els.demographicsSummary.innerHTML = [
    ["Selected topics", selectedTopics.length],
    ["Participating users", analysis.participantCount],
    [
      "Largest pair overlap",
      analysis.pairwise.length ? Math.max(...analysis.pairwise.map((pair) => pair.sharedUsers)) : 0,
    ],
    ["Active stance mode", state.demographicStanceFilter === "both" ? "Both" : state.demographicStanceFilter],
  ]
    .map(
      ([label, value]) => `
        <article class="metric-card compact-card">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${typeof value === "number" ? formatNumber(value) : escapeHtml(value)}</div>
        </article>
      `
    )
    .join("");

  els.overlapChart.innerHTML = renderOverlapChart(analysis);

  els.overlapLegend.innerHTML = analysis.topicSets
    .map((entry) => {
      const counts = getTopicSupportOpposingCounts(entry.topic);
      const totalUsers = counts.support + counts.opposing;
      const activeUsers = entry.userSet.size;
      return `
        <article class="legend-card">
          <div class="legend-title-row">
            <strong>${escapeHtml(entry.topic.label)}</strong>
            <span class="badge neutral">${formatNumber(activeUsers)} active users</span>
          </div>
          <div class="stance-meter">
            <div class="stance-support" style="width:${totalUsers ? (counts.support / totalUsers) * 100 : 0}%"></div>
            <div class="stance-opposing" style="width:${totalUsers ? (counts.opposing / totalUsers) * 100 : 0}%"></div>
          </div>
          <div class="detail-meta">
            <span>${formatNumber(counts.support)} support</span>
            <span>${formatNumber(counts.opposing)} opposing</span>
            <span>${formatPercentFromCount(activeUsers, analysis.participantCount)} of participants</span>
          </div>
        </article>
      `;
    })
    .join("");

  els.overlapTable.innerHTML = analysis.pairwise.length
    ? `
      <div class="overlap-table">
        <div class="overlap-table-header">Topic pair</div>
        <div class="overlap-table-header">Shared users</div>
        <div class="overlap-table-header">% of participants</div>
        ${analysis.pairwise
      .map(
        (pair) => `
              <div class="overlap-cell">${escapeHtml(pair.leftTopic.label)} <span class="muted">vs</span> ${escapeHtml(
          pair.rightTopic.label
        )}</div>
              <div class="overlap-cell">${formatNumber(pair.sharedUsers)}</div>
              <div class="overlap-cell">${formatPercentFromCount(pair.sharedUsers, analysis.participantCount)}</div>
            `
      )
      .join("")}
      </div>
    `
    : '<div class="muted">No pairwise overlap is available for the current selection.</div>';

  const stanceModeLabel =
    state.demographicStanceFilter === "both"
      ? "support and opposing users together"
      : `${state.demographicStanceFilter} users only`;
  els.demographicsNote.textContent = `Denominator: unique users who appear in at least one selected topic under ${stanceModeLabel}. A user keeps their existing per-topic dominant stance, so the same author can overlap across topics even when their stance changes from one topic to another.`;
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
      <p>${stanceReady
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
                  <article class="topic-node ${Number(state.activeTopicId) === Number(child.topic_id) ? "active" : ""
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

function renderTimeline(topic) {
  const timeline = topic.timeline;
  if (!timeline || !timeline.daily_post_counts?.length) {
    return '<div class="timeline-empty">No daily trend data available.</div>';
  }

  const width = 620;
  const height = 290;
  const chartLeft = 28;
  const chartRight = width - 24;
  const chartTop = 18;
  const axisY = 202;
  const monthLabelY = 224;
  const eventBaseY = 252;
  const plotHeight = axisY - chartTop;
  const values = timeline.daily_post_counts;
  const maxValue = Math.max(...values, 1);
  const step = (chartRight - chartLeft) / Math.max(values.length - 1, 1);
  const barWidth = Math.max(1.5, Math.min(7, step * 0.82));
  const bars = values
    .map((value, index) => {
      const x = chartLeft + step * index - barWidth / 2;
      const barHeight = (plotHeight * value) / maxValue;
      const y = axisY - barHeight;
      return `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${Math.max(
        barHeight,
        value > 0 ? 1.5 : 0
      ).toFixed(2)}" rx="1.5" class="trend-bar"></rect>`;
    })
    .join("");

  const monthTicks = (timeline.day_axis || [])
    .map((day, index, days) => {
      const previousMonth = index > 0 ? days[index - 1].slice(0, 7) : null;
      const currentMonth = day.slice(0, 7);
      if (index !== 0 && currentMonth === previousMonth) {
        return "";
      }
      const x = chartLeft + step * index;
      return `
        <line x1="${x.toFixed(2)}" y1="${chartTop}" x2="${x.toFixed(2)}" y2="${axisY}" class="month-tick"></line>
        <text x="${x.toFixed(2)}" y="${monthLabelY}" text-anchor="middle" class="axis-label">${escapeHtml(
        formatMonthLabel(currentMonth)
      )}</text>
      `;
    })
    .join("");

  const eventMarkers = (timeline.events || [])
    .map((event, index) => {
      const dayIndex = (timeline.day_axis || []).indexOf(event.date);
      if (dayIndex < 0) return "";
      const x = chartLeft + step * dayIndex;
      const eventLabelY = eventBaseY + (index % 2) * 14;
      return `
        <line x1="${x.toFixed(2)}" y1="${chartTop}" x2="${x.toFixed(2)}" y2="${axisY}" class="event-line"></line>
        <circle cx="${x.toFixed(2)}" cy="${axisY}" r="3.5" class="event-dot"></circle>
        <text x="${x.toFixed(2)}" y="${eventLabelY}" text-anchor="start" transform="rotate(35 ${x.toFixed(
        2
      )} ${eventLabelY})" class="event-label">${escapeHtml(event.label)}</text>
      `;
    })
    .join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" class="timeline-chart" role="img" aria-label="Topic frequency chart">
      <rect x="0" y="0" width="${width}" height="${height}" rx="18" class="chart-bg"></rect>
      ${[0.25, 0.5, 0.75].map((fraction) => {
    const y = chartTop + plotHeight * fraction;
    return `<line x1="${chartLeft}" y1="${y}" x2="${chartRight}" y2="${y}" class="grid-line"></line>`;
  }).join("")}
      <line x1="${chartLeft}" y1="${axisY}" x2="${chartRight}" y2="${axisY}" class="axis-line"></line>
      ${bars}
      ${monthTicks}
      ${eventMarkers}
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
          <span>Model: ${escapeHtml(state.model)}</span>
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
  ensureValidDemographicSelection();
  renderOverview();
  renderTopicTree();
  renderTopicDetail();
  renderUserDemographics();
  renderPipelineStatus();
  renderQaResult();
}

async function loadBundle() {
  try {
    state.bundle = await apiFetch(`/api/bundle?ts=${Date.now()}`, { method: "GET" });
    state.expandedMajorTopics = new Set((state.bundle.topic_tree || []).slice(0, 2).map((node) => node.id));
    ensureValidDemographicSelection();
  } catch (error) {
    state.bundle = null;
    state.pipelineLog = `Bundle load failed: ${error.message}`;
    state.selectedDemographicTopics = [];
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

  closeQueryTypeMenu();

  setQaBusy(true);
  els.qaStatusText.textContent = "Query in progress...";
  try {
    state.qaResult = await apiFetch("/api/query", {
      method: "POST",
      body: JSON.stringify({
        question,
        lang: state.language,
        model: state.model,
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

  els.navItems.forEach((item) => {
    item.addEventListener("click", () => {
      setActiveView(item.dataset.view);
    });
  });

  els.stanceToggles.forEach((button) => {
    button.addEventListener("click", () => {
      state.demographicStanceFilter = button.dataset.stanceFilter;
      renderUserDemographics();
    });
  });

  els.conversationInput.addEventListener("focus", maybeOpenQueryTypeMenu);
  els.conversationInput.addEventListener("click", maybeOpenQueryTypeMenu);
  els.conversationInput.addEventListener("input", () => {
    validateConversationInput();
    if (els.conversationInput.value.trim()) {
      closeQueryTypeMenu();
    }
  });
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

  els.languageToggles.forEach((button) => {
    button.addEventListener("click", () => {
      closeQueryTypeMenu();
      state.language = button.dataset.language;
      syncLanguageToggles();
    });
  });
  els.modelToggles.forEach((button) => {
    button.addEventListener("click", () => {
      closeQueryTypeMenu();
      state.model = button.dataset.model;
      syncModelToggles();
    });
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
  syncLanguageToggles();
  syncModelToggles();
  setActiveView(state.activeView);
  render();
}

boot();
