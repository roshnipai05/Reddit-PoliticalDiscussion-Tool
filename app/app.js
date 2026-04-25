const state = {
  bundle: null,
  activeTopicId: null,
  expandedMajorTopics: new Set(),
  zoom: 1,
  hindiMode: false,
};

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
};

function formatNumber(value) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function trendClass(value) {
  return String(value || "").toLowerCase();
}

function getTopicById(topicId) {
  return state.bundle.topics.find((topic) => Number(topic.topic_id) === Number(topicId));
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
  return state.bundle.topic_tree
    .map((majorTopic) => ({
      ...majorTopic,
      children: majorTopic.children.filter((child) => topicMatchesFilters(getTopicById(child.topic_id))),
    }))
    .filter((majorTopic) => majorTopic.children.length > 0);
}

function renderOverview() {
  const bundle = state.bundle;
  els.subredditLabel.textContent = `${bundle.app_meta.subreddit} • ${bundle.app_meta.analysis_scope}`;

  const cards = [
    ["Major Topic Groups", bundle.topic_tree.length],
    ["Model Topics", bundle.overview.topic_count],
    ["Trending", bundle.overview.trending_topics],
    ["Persistent", bundle.overview.persistent_topics],
    ["Comments Analyzed", bundle.stance_preview_metadata.comment_count_analyzed],
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
      <p>${formatNumber(
        bundle.stance_preview_metadata.comment_count_analyzed
      )} comments were grouped into support/opposition previews across ${bundle.stance_preview_metadata.topic_count_analyzed} major topics.</p>
    </div>
  `;
}

function initFilters() {
  const flairs = new Set();
  state.bundle.topics.forEach((topic) => {
    (topic.top_flairs || []).forEach((item) => flairs.add(item.flair));
  });
  els.flairFilter.innerHTML = '<option value="all">All flairs</option>';
  [...flairs]
    .sort()
    .forEach((flair) => {
      const option = document.createElement("option");
      option.value = flair;
      option.textContent = flair;
      els.flairFilter.append(option);
    });

  els.dateFilter.innerHTML = '<option value="all">All months</option>';
  state.bundle.app_meta.month_axis.forEach((month) => {
    const option = document.createElement("option");
    option.value = month;
    option.textContent = month;
    els.dateFilter.append(option);
  });
}

function renderTopicTree() {
  const majorTopics = filteredTree();
  const filteredTopicCount = majorTopics.reduce((sum, node) => sum + node.children.length, 0);
  els.topicCountLabel.textContent = `${filteredTopicCount} topics visible across ${majorTopics.length} major groups`;
  if (!majorTopics.length) {
    els.topicTree.innerHTML = '<div class="empty-state">No topics match the active filters.</div>';
    return;
  }

  const scale = state.zoom.toFixed(2);
  els.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  els.topicTree.style.transform = `scale(${scale})`;
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
                    <div class="topic-node-copy">${escapeHtml(topic.topic_description)}</div>
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

function render() {
  ensureValidActiveTopic();
  renderTopicTree();
  renderTopicDetail();
}

async function boot() {
  const response = await fetch("./data.bundle.json");
  state.bundle = await response.json();
  state.bundle.topic_tree.slice(0, 2).forEach((node) => state.expandedMajorTopics.add(node.id));

  initFilters();
  renderOverview();
  render();

  [els.searchInput, els.trendFilter, els.flairFilter, els.dateFilter].forEach((el) => {
    el.addEventListener("input", render);
    el.addEventListener("change", render);
  });

  els.zoomIn.addEventListener("click", () => {
    state.zoom = Math.min(1.6, state.zoom + 0.1);
    renderTopicTree();
  });
  els.zoomOut.addEventListener("click", () => {
    state.zoom = Math.max(0.8, state.zoom - 0.1);
    renderTopicTree();
  });
  els.languageToggle.addEventListener("click", () => {
    state.hindiMode = !state.hindiMode;
    els.languageToggle.setAttribute("aria-pressed", String(state.hindiMode));
    els.languageToggle.textContent = `Hindi Mode: ${state.hindiMode ? "On" : "Off"}`;
  });
}

boot();
