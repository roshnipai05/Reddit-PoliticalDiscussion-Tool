const state = {
  bundle: null,
  filteredTopics: [],
  activeTopicId: null,
};

const els = {
  subredditLabel: document.getElementById("subredditLabel"),
  overviewStats: document.getElementById("overviewStats"),
  aggregateGrid: document.getElementById("aggregateGrid"),
  topicBars: document.getElementById("topicBars"),
  qualityPanel: document.getElementById("qualityPanel"),
  topicGrid: document.getElementById("topicGrid"),
  topicInspector: document.getElementById("topicInspector"),
  topicCountLabel: document.getElementById("topicCountLabel"),
  searchInput: document.getElementById("searchInput"),
  trendFilter: document.getElementById("trendFilter"),
  stanceFilter: document.getElementById("stanceFilter"),
  flairFilter: document.getElementById("flairFilter"),
};

function formatNumber(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function titleCaseTrend(value) {
  return String(value || "").toLowerCase();
}

function trendBadgeClass(value) {
  return titleCaseTrend(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function initFilters(bundle) {
  const flairs = new Set();
  bundle.topics.forEach((topic) => {
    (topic.top_flairs || []).forEach((item) => flairs.add(item.flair));
  });
  [...flairs].sort().forEach((flair) => {
    const option = document.createElement("option");
    option.value = flair;
    option.textContent = flair;
    els.flairFilter.append(option);
  });
}

function renderOverview(bundle) {
  els.subredditLabel.textContent = `${bundle.app_meta.subreddit} • ${bundle.app_meta.analysis_scope}`;
  const cards = [
    ["Topics", bundle.overview.topic_count],
    ["Persistent", bundle.overview.persistent_topics],
    ["Trending", bundle.overview.trending_topics],
    ["Stance Preview Topics", bundle.overview.stance_preview_topics],
    ["Posts", bundle.aggregate_stats.total_posts],
    ["Comments", bundle.aggregate_stats.total_comments],
  ];
  els.overviewStats.innerHTML = cards
    .map(
      ([label, value]) => `
        <div class="stat-box">
          <div class="metric-label">${label}</div>
          <div class="stat-value">${formatNumber(value)}</div>
        </div>
      `
    )
    .join("");

  const aggregateCards = [
    ["Total Posts", formatNumber(bundle.aggregate_stats.total_posts)],
    ["Unique Users", formatNumber(bundle.aggregate_stats.total_unique_users)],
    ["Total Comments", formatNumber(bundle.aggregate_stats.total_comments)],
    ["Total Upvotes", formatNumber(bundle.aggregate_stats.total_upvotes)],
    ["Date Range", `${bundle.aggregate_stats.date_range_start} to ${bundle.aggregate_stats.date_range_end}`],
  ];
  els.aggregateGrid.innerHTML = aggregateCards
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${value}</div>
        </div>
      `
    )
    .join("");

  els.qualityPanel.innerHTML = `
    <div class="quality-item"><strong>Topic model</strong><div class="summary-copy">BERTopic with dense semantic embeddings (${bundle.topic_run_metadata.embedding_model}). ${bundle.topic_run_metadata.topic_count} final topics across ${formatNumber(bundle.topic_run_metadata.post_count)} cleaned posts.</div></div>
    <div class="quality-item"><strong>Topic assignment coverage</strong><div class="summary-copy">${formatPercent(bundle.topic_run_metadata.assigned_non_outlier_posts / bundle.topic_run_metadata.post_count)} of cleaned posts were assigned to non-outlier topics. ${formatNumber(bundle.topic_run_metadata.outlier_posts)} posts remain outliers and should be reviewed during refinement.</div></div>
    <div class="quality-item"><strong>Stance analysis status</strong><div class="summary-copy">Current Part 1.4 view is based on a preview sample of ${formatNumber(bundle.stance_preview_metadata.comment_count_analyzed)} comments across ${bundle.stance_preview_metadata.topic_count_analyzed} topics. This app surfaces the raw evidence so you can judge whether the stance split is coherent before scaling to the full corpus.</div></div>
  `;
}

function getFilteredTopics() {
  const search = els.searchInput.value.trim().toLowerCase();
  const trend = els.trendFilter.value;
  const stanceFilter = els.stanceFilter.value;
  const flair = els.flairFilter.value;

  return state.bundle.topics.filter((topic) => {
    const haystack = [
      topic.label,
      ...(topic.keywords || []),
      ...((topic.top_flairs || []).map((item) => item.flair)),
    ]
      .join(" ")
      .toLowerCase();

    const matchesSearch = !search || haystack.includes(search);
    const matchesTrend = trend === "all" || topic.trend_type === trend;
    const hasPreview = Boolean(topic.stance_preview);
    const matchesPreview =
      stanceFilter === "all" ||
      (stanceFilter === "with_preview" && hasPreview) ||
      (stanceFilter === "without_preview" && !hasPreview);
    const matchesFlair =
      flair === "all" || (topic.top_flairs || []).some((item) => item.flair === flair);

    return matchesSearch && matchesTrend && matchesPreview && matchesFlair;
  });
}

function renderTopicBars(topics) {
  const maxShare = Math.max(...topics.map((topic) => topic.topic_share), 0.0001);
  els.topicBars.innerHTML = topics
    .slice()
    .sort((a, b) => b.topic_share - a.topic_share)
    .map(
      (topic) => `
        <div class="topic-bar">
          <div class="topic-bar-row">
            <strong>${escapeHtml(topic.label)}</strong>
            <span>${formatPercent(topic.topic_share)}</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill ${trendBadgeClass(topic.trend_type)}" style="width:${(topic.topic_share / maxShare) * 100}%"></div>
          </div>
        </div>
      `
    )
    .join("");
}

function renderTopicGrid(topics) {
  els.topicCountLabel.textContent = `${topics.length} topics shown`;
  els.topicGrid.innerHTML = topics
    .map((topic) => {
      const preview = topic.stance_preview;
      const supportShare = preview ? preview.support_share : 0;
      const opposingShare = preview ? preview.opposing_share : 0;
      return `
        <article class="topic-card ${state.activeTopicId === topic.topic_id ? "active" : ""}" data-topic-id="${topic.topic_id}">
          <div class="topic-card-header">
            <span class="badge ${trendBadgeClass(topic.trend_type)}">${topic.trend_type}</span>
            ${preview ? '<span class="badge preview">Stance Preview</span>' : ""}
          </div>
          <div class="topic-title">${escapeHtml(topic.label)}</div>
          <div class="topic-keywords">${escapeHtml((topic.keywords || []).slice(0, 6).join(", "))}</div>
          <div class="mini-metrics">
            <span class="chip">Share ${formatPercent(topic.topic_share)}</span>
            <span class="chip">${formatNumber(topic.post_count)} posts</span>
            ${preview ? `<span class="chip">${formatNumber(preview.comment_count)} preview comments</span>` : ""}
          </div>
          ${
            preview
              ? `
                <div>
                  <div class="split-row">
                    <strong>Stance split</strong>
                    <span>${formatPercent(supportShare)} support / ${formatPercent(opposingShare)} opposing</span>
                  </div>
                  <div class="split-meter">
                    <div class="split-support" style="width:${supportShare * 100}%"></div>
                    <div class="split-opposing" style="width:${opposingShare * 100}%"></div>
                  </div>
                </div>
              `
              : `<div class="muted">No stance preview yet for this topic.</div>`
          }
          <div class="chip-row">
            ${(topic.top_flairs || [])
              .slice(0, 3)
              .map((item) => `<span class="chip">${escapeHtml(item.flair)}</span>`)
              .join("")}
          </div>
        </article>
      `;
    })
    .join("");

  [...els.topicGrid.querySelectorAll(".topic-card")].forEach((card) => {
    card.addEventListener("click", () => {
      state.activeTopicId = Number(card.dataset.topicId);
      render();
    });
  });
}

function renderTopicInspector(topic) {
  if (!topic) {
    els.topicInspector.classList.add("empty");
    els.topicInspector.innerHTML =
      '<div class="empty-state">Select a topic to inspect its summaries, posts, and stance evidence.</div>';
    return;
  }

  els.topicInspector.classList.remove("empty");
  const preview = topic.stance_preview;
  const posts = topic.representative_posts || [];
  const supportComments = preview?.support_representative_comments || [];
  const opposingComments = preview?.opposing_representative_comments || [];
  const commentPreview = topic.comment_preview || [];
  const userGroups = topic.user_groups_preview || preview?.user_groups || null;

  els.topicInspector.innerHTML = `
    <div class="inspector-header">
      <h2>${escapeHtml(topic.label)}</h2>
      <span class="badge ${trendBadgeClass(topic.trend_type)}">${topic.trend_type}</span>
      ${preview ? '<span class="badge preview">Stance Preview</span>' : ""}
    </div>

    <div class="mini-metrics">
      <span class="chip">Topic share ${formatPercent(topic.topic_share)}</span>
      <span class="chip">${formatNumber(topic.post_count)} posts</span>
      <span class="chip">${formatNumber(topic.active_months)} active months</span>
      ${
        preview
          ? `<span class="chip">Disagreement index ${preview.disagreement_index.toFixed(3)}</span>`
          : ""
      }
    </div>

    <div class="inspector-grid">
      <div class="inspector-panel">
        <h3>Topic Summary</h3>
        <div class="summary-copy">Keywords: ${escapeHtml((topic.keywords || []).join(", "))}</div>
        <div class="chip-row">
          ${(topic.top_flairs || [])
            .map((item) => `<span class="chip">${escapeHtml(item.flair)} ${formatPercent(item.share_within_topic)}</span>`)
            .join("")}
        </div>
      </div>

      <div class="inspector-panel">
        <h3>Trend Diagnostics</h3>
        <div class="summary-copy">Recent share: ${formatPercent(topic.recent_share)}<br />Early share: ${formatPercent(topic.early_share)}<br />Slope: ${topic.trend_slope.toFixed(4)}<br />Variance ratio: ${topic.share_cv.toFixed(3)}</div>
      </div>
    </div>

    <div class="inspector-grid">
      <div class="inspector-panel">
        <h3>Representative Posts</h3>
        <div class="post-list">
          ${posts
            .map(
              (post) => `
                <div class="post-item">
                  <strong>${escapeHtml(post.title)}</strong>
                  <div class="meta-row">
                    <span>Score ${formatNumber(post.score)}</span>
                    <span>${formatNumber(post.num_comments)} comments</span>
                    <span>${escapeHtml(post.created_month)}</span>
                  </div>
                  <a href="https://reddit.com${post.permalink}" target="_blank" rel="noreferrer">Open Reddit thread</a>
                </div>
              `
            )
            .join("")}
        </div>
      </div>

      <div class="inspector-panel">
        <h3>Stance Summary</h3>
        ${
          preview
            ? `
              <div class="summary-copy"><strong>Dominant position</strong><br />${escapeHtml(preview.dominant_position_summary)}</div>
              <div class="summary-copy"><strong>Support-side arguments</strong><br />${escapeHtml(preview.support_argument_summary)}</div>
              <div class="summary-copy"><strong>Opposing-side arguments</strong><br />${escapeHtml(preview.opposing_argument_summary)}</div>
              <div class="split-row">
                <strong>Comment split</strong>
                <span>${formatNumber(preview.support_comment_count)} support / ${formatNumber(preview.opposing_comment_count)} opposing</span>
              </div>
              <div class="split-meter">
                <div class="split-support" style="width:${preview.support_share * 100}%"></div>
                <div class="split-opposing" style="width:${preview.opposing_share * 100}%"></div>
              </div>
            `
            : `<div class="summary-copy">No stance preview is available for this topic yet.</div>`
        }
      </div>
    </div>

    ${
      preview
        ? `
          <div class="inspector-grid">
            <div class="inspector-panel">
              <h3>Support Evidence</h3>
              <div class="chip-row">
                ${(preview.support_keywords || []).map((word) => `<span class="chip">${escapeHtml(word)}</span>`).join("")}
              </div>
              <div class="evidence-list">
                ${supportComments
                  .map(
                    (item) => `
                      <div class="evidence-item">
                        <div class="comment-body">${escapeHtml(item.excerpt)}</div>
                        <div class="meta-row">
                          <span>Score ${formatNumber(item.score)}</span>
                          <span>Confidence ${item.stance_confidence.toFixed(3)}</span>
                        </div>
                        <a href="https://reddit.com${item.permalink}" target="_blank" rel="noreferrer">Open comment</a>
                      </div>
                    `
                  )
                  .join("")}
              </div>
            </div>

            <div class="inspector-panel">
              <h3>Opposing Evidence</h3>
              <div class="chip-row">
                ${(preview.opposing_keywords || []).map((word) => `<span class="chip">${escapeHtml(word)}</span>`).join("")}
              </div>
              <div class="evidence-list">
                ${opposingComments
                  .map(
                    (item) => `
                      <div class="evidence-item">
                        <div class="comment-body">${escapeHtml(item.excerpt)}</div>
                        <div class="meta-row">
                          <span>Score ${formatNumber(item.score)}</span>
                          <span>Confidence ${item.stance_confidence.toFixed(3)}</span>
                        </div>
                        <a href="https://reddit.com${item.permalink}" target="_blank" rel="noreferrer">Open comment</a>
                      </div>
                    `
                  )
                  .join("")}
              </div>
            </div>
          </div>
        `
        : ""
    }

    ${
      userGroups
        ? `
          <div class="inspector-grid">
            <div class="inspector-panel">
              <h3>User Grouping</h3>
              <div class="summary-copy">Support users: ${formatNumber(userGroups.support_users)}<br />Opposing users: ${formatNumber(userGroups.opposing_users)}</div>
              <div class="user-list">
                ${(userGroups.users || [])
                  .slice(0, 12)
                  .map(
                    (user) => `
                      <div class="user-item">
                        <div><strong>${escapeHtml(user.author_hash)}</strong></div>
                        <div class="meta-row">
                          <span>${escapeHtml(user.dominant_stance)}</span>
                          <span>${formatNumber(user.support_comments)} support</span>
                          <span>${formatNumber(user.opposing_comments)} opposing</span>
                          <span>Score ${formatNumber(user.total_score)}</span>
                        </div>
                      </div>
                    `
                  )
                  .join("")}
              </div>
            </div>

            <div class="inspector-panel">
              <h3>Raw Preview Comments</h3>
              <div class="comment-list">
                ${commentPreview
                  .map(
                    (item) => `
                      <div class="comment-item">
                        <div class="meta-row">
                          <span class="badge ${item.stance_label === "support" ? "persistent" : "trending"}">${escapeHtml(item.stance_label)}</span>
                          <span>Confidence ${item.stance_confidence.toFixed(3)}</span>
                          <span>Score ${formatNumber(item.score)}</span>
                          <span>${escapeHtml(item.link_flair_text)}</span>
                        </div>
                        <div><strong>${escapeHtml(item.topic_post_title)}</strong></div>
                        <div class="comment-body">${escapeHtml(item.body)}</div>
                        <div class="meta-row">
                          <span>${escapeHtml(item.author_hash)}</span>
                          <span>${escapeHtml(item.created_iso)}</span>
                        </div>
                        <a href="https://reddit.com${item.permalink}" target="_blank" rel="noreferrer">Open comment</a>
                      </div>
                    `
                  )
                  .join("")}
              </div>
            </div>
          </div>
        `
        : ""
    }
  `;
}

function render() {
  const topics = getFilteredTopics();
  state.filteredTopics = topics;
  if (!topics.some((topic) => topic.topic_id === state.activeTopicId)) {
    state.activeTopicId = topics[0]?.topic_id ?? null;
  }
  renderTopicBars(topics);
  renderTopicGrid(topics);
  renderTopicInspector(topics.find((topic) => topic.topic_id === state.activeTopicId));
}

async function boot() {
  const response = await fetch("./data.bundle.json");
  state.bundle = await response.json();
  initFilters(state.bundle);
  renderOverview(state.bundle);

  [els.searchInput, els.trendFilter, els.stanceFilter, els.flairFilter].forEach((el) =>
    el.addEventListener("input", render)
  );
  [els.trendFilter, els.stanceFilter, els.flairFilter].forEach((el) =>
    el.addEventListener("change", render)
  );

  render();
}

boot();
