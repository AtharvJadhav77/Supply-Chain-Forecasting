const $ = (id) => document.getElementById(id);

const state = {
  meta: null,
  history: null,
  chart: null,
  loading: false,
  toastTimer: null,
};

const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const inr2 = new Intl.NumberFormat("en-IN", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

function money(value) {
  return `₹${inr.format(Math.round(Number(value) || 0))}`;
}

function parseISO(value) {
  if (!value) return null;
  const [year, month, day] = value.split("-").map(Number);
  if (!year || !month || !day) return null;
  return new Date(year, month - 1, day);
}

function toISO(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function addDays(iso, days) {
  const date = parseISO(iso);
  if (!date) return iso;
  date.setDate(date.getDate() + days);
  return toISO(date);
}

function daysBetween(fromISO, toISOValue) {
  const from = parseISO(fromISO);
  const to = parseISO(toISOValue);
  if (!from || !to) return 0;
  return Math.round((to - from) / 86400000);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let payload = {};
  try {
    payload = await response.json();
  } catch {
    payload = {};
  }
  if (!response.ok) {
    const detail = payload.detail || `Request failed with status ${response.status}`;
    throw new Error(detail);
  }
  return payload;
}

function showToast(message, type = "error") {
  const toast = $("toast");
  toast.textContent = message;
  toast.className = `toast show ${type === "success" ? "success" : ""}`;
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => {
    toast.className = "toast";
  }, 3600);
}

function setApiStatus(status, label) {
  const container = $("apiStatus");
  container.className = `header-status ${status}`;
  container.querySelector(".status-text").textContent = label;
}

function fillSelect(select, values, preferred) {
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
  if (preferred && values.includes(preferred)) {
    select.value = preferred;
  }
}

function updateHeroStats(dataset) {
  if (!dataset) return;
  $("statRows").textContent = inr.format(dataset.rows);
  $("statDistricts").textContent = inr.format(dataset.districts);
  $("statCommodities").textContent = inr.format(dataset.commodities);
  $("statPairs").textContent = inr.format(dataset.pairs);
  $("datasetRange").textContent = `${dataset.first_date} → ${dataset.last_date}`;
}

function updateCommodities(preferred) {
  if (!state.meta) return;
  const district = $("districtSelect").value;
  const list = state.meta.availability[district] || [];
  fillSelect($("commoditySelect"), list, preferred);
}

function updateHorizonUI() {
  if (!state.meta) return;
  const days = daysBetween(state.meta.dataset.last_date, $("dateInput").value);
  const horizonValue = $("horizonValue");
  const horizonBar = $("horizonBar");

  if (days <= 0) {
    horizonValue.textContent = "—";
    horizonBar.style.width = "0%";
  } else {
    horizonValue.textContent = `${days} day${days === 1 ? "" : "s"}`;
    horizonBar.style.width = `${Math.min(100, (days / state.meta.limits.max_horizon_days) * 100)}%`;
  }

  document.querySelectorAll(".preset-row button").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.days) === days);
  });
}

function setPreset(days) {
  if (!state.meta) return;
  const target = addDays(state.meta.dataset.last_date, days);
  const maxDate = addDays(state.meta.dataset.last_date, state.meta.limits.max_horizon_days);
  $("dateInput").value = target > maxDate ? maxDate : target;
  updateHorizonUI();
  resetForecastView("Horizon updated — run a forecast to refresh the projection.");
  if (state.history) {
    renderChart(state.history.series, [], state.history.district, state.history.commodity);
  }
}

function resetForecastView(message = "Run a forecast to load market intelligence.") {
  $("kpiPrice").textContent = "—";
  $("kpiConfidence").textContent = "—";
  $("kpiRisk").textContent = "—";
  $("kpiStock").textContent = "—";
  $("kpiTrend").textContent = "—";
  $("kpiTrend").className = "chip";
  $("confidenceBar").style.width = "0%";
  $("riskBar").style.width = "0%";
  $("confidenceHint").textContent = "—";
  $("riskHint").textContent = "News-driven volatility";
  $("rangeValue").textContent = "—";
  $("latestValue").textContent = "—";
  $("moveValue").textContent = "—";
  $("moveValue").style.color = "";
  $("trendChip").textContent = "—";
  $("trendChip").className = "chip";
  $("advisoryText").textContent = message;
  $("explanationText").textContent = "The model explanation will appear here.";
  $("newsMeta").textContent = "—";
  $("newsList").innerHTML = '<li class="empty-state">News signals will appear after a forecast.</li>';
  $("chartSubtitle").textContent = "Historical modal price";
}

function setLoading(loading) {
  state.loading = loading;
  const button = $("forecastBtn");
  button.disabled = loading;
  button.classList.toggle("loading", loading);
  button.querySelector(".btn-label").textContent = loading ? "Forecasting…" : "Run forecast";
  $("resultsRoot").classList.toggle("is-loading", loading);
}

function renderChart(history, projection, district, commodity) {
  const canvas = $("priceChart");
  if (!canvas || typeof Chart === "undefined") return;

  const historyPoints = Array.isArray(history) ? history : [];
  const projectionPoints = Array.isArray(projection) ? projection : [];
  const labels = [
    ...historyPoints.map((point) => point.date),
    ...projectionPoints.map((point) => point.date),
  ];

  const historyData = [
    ...historyPoints.map((point) => Number(point.modal_price)),
    ...projectionPoints.map(() => null),
  ];

  const forecastData = new Array(Math.max(historyPoints.length - 1, 0)).fill(null);
  if (historyPoints.length) {
    forecastData.push(Number(historyPoints[historyPoints.length - 1].modal_price));
  }
  projectionPoints.forEach((point) => forecastData.push(Number(point.price)));

  const subtitle = projectionPoints.length
    ? `${commodity} · ${district} · ${projectionPoints.length}-day projection`
    : `${commodity} · ${district} · recent market history`;
  $("chartSubtitle").textContent = subtitle;

  const datasets = [
    {
      label: "Historical",
      data: historyData,
      borderColor: "#38bdf8",
      backgroundColor: "rgba(56, 189, 248, 0.08)",
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: "#38bdf8",
      tension: 0.32,
      fill: false,
    },
    {
      label: "Forecast",
      data: forecastData,
      borderColor: "#34d399",
      borderWidth: 2.4,
      borderDash: [6, 5],
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: "#34d399",
      tension: 0.32,
      fill: true,
      backgroundColor: (context) => {
        const chart = context.chart;
        const { ctx, chartArea } = chart;
        if (!chartArea) return "rgba(52, 211, 153, 0.08)";
        const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
        gradient.addColorStop(0, "rgba(52, 211, 153, 0.28)");
        gradient.addColorStop(1, "rgba(52, 211, 153, 0)");
        return gradient;
      },
    },
  ];

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 650 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(8, 14, 26, 0.95)",
        borderColor: "rgba(148, 163, 184, 0.22)",
        borderWidth: 1,
        titleColor: "#e9eff8",
        bodyColor: "#b8c5d8",
        padding: 12,
        displayColors: true,
        callbacks: {
          label: (context) => {
            if (context.parsed.y === null || Number.isNaN(context.parsed.y)) return null;
            return `${context.dataset.label}: ${money(context.parsed.y)}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        border: { display: false },
        ticks: {
          color: "#7f8fa8",
          maxTicksLimit: 8,
          maxRotation: 0,
          font: { family: "Inter", size: 10 },
          callback: (value) => {
            const label = labels[value];
            if (!label) return "";
            return label.slice(5);
          },
        },
      },
      y: {
        grid: { color: "rgba(148, 163, 184, 0.08)" },
        border: { display: false },
        ticks: {
          color: "#7f8fa8",
          maxTicksLimit: 6,
          font: { family: "Inter", size: 10 },
          callback: (value) => money(value),
        },
      },
    },
  };

  if (!state.chart) {
    state.chart = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { labels, datasets },
      options,
    });
  } else {
    state.chart.data.labels = labels;
    state.chart.data.datasets = datasets;
    state.chart.options = options;
    state.chart.update();
  }
}

function renderForecast(data) {
  const history = data.history || [];
  const latest = history.length ? Number(history[history.length - 1].modal_price) : null;

  $("kpiPrice").textContent = money(data.fusion_forecast);

  const trendChip = $("kpiTrend");
  const trendLabel = data.trend.charAt(0).toUpperCase() + data.trend.slice(1);
  trendChip.textContent = trendLabel;
  trendChip.className = `chip chip-${data.trend}`;

  $("kpiConfidence").textContent = `${data.confidence_percent.toFixed(1)}%`;
  $("confidenceBar").style.width = `${Math.min(100, data.confidence_percent)}%`;
  $("confidenceHint").textContent = `${data.forecast_horizon_days}-day horizon · decays with time`;

  const risk = Number(data.risk_index);
  $("kpiRisk").textContent = risk.toFixed(2);
  $("riskBar").style.width = `${Math.min(100, (risk / 10) * 100)}%`;
  $("riskHint").textContent =
    risk > 8 ? "High news-driven risk" : risk > 4 ? "Moderate news-driven risk" : "Low news-driven risk";

  $("kpiStock").textContent = inr.format(Math.round(data.recommended_safety_stock));

  $("rangeValue").textContent = `${money(data.forecast_low)} – ${money(data.forecast_high)}`;
  $("latestValue").textContent = latest === null ? "—" : money(latest);

  const move = Number(data.expected_change_percent);
  const moveText = `${move >= 0 ? "+" : ""}${move.toFixed(2)}%`;
  $("moveValue").textContent = moveText;
  $("moveValue").style.color = move > 0 ? "#86efac" : move < 0 ? "#fda4af" : "";

  const trendBadge = $("trendChip");
  trendBadge.textContent = trendLabel;
  trendBadge.className = `chip chip-${data.trend}`;

  $("advisoryText").textContent = data.advisory || "No advisory available.";
  $("explanationText").textContent = data.explanation || "No explanation available.";

  renderNews(data.news || [], data);
  renderChart(history, data.projection, data.district, data.commodity);
}

function renderNews(items, data) {
  const list = $("newsList");
  const meta = $("newsMeta");
  meta.textContent = `${data.news_count || items.length} scored · sentiment ${Number(
    data.sentiment_score || 0
  ).toFixed(2)}`;

  if (!items.length) {
    list.innerHTML = '<li class="empty-state">No live headlines available right now.</li>';
    return;
  }

  list.innerHTML = items
    .map((item) => {
      const sentiment = ["positive", "negative"].includes(item.sentiment)
        ? item.sentiment
        : "neutral";
      const source = item.source ? `<span>${escapeHtml(item.source)}</span>` : "";
      const published = item.published ? `<span>${escapeHtml(item.published)}</span>` : "";
      const title = escapeHtml(item.title || "Untitled headline");
      const content = item.url
        ? `<a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">${title}</a>`
        : `<span class="news-title">${title}</span>`;

      return `
        <li class="news-item">
          ${content}
          <div class="news-meta">
            <span class="sentiment ${sentiment}">${sentiment}</span>
            ${source}
            ${published}
          </div>
        </li>
      `;
    })
    .join("");
}

async function loadHistory() {
  if (!state.meta) return;
  const district = $("districtSelect").value;
  const commodity = $("commoditySelect").value;
  if (!district || !commodity) return;

  try {
    const data = await api(
      `/api/history?district=${encodeURIComponent(district)}&commodity=${encodeURIComponent(
        commodity
      )}&limit=120`
    );
    state.history = data;
    renderChart(data.series, [], district, commodity);
  } catch (error) {
    showToast(error.message);
  }
}

async function runForecast(notify = true) {
  if (!state.meta || state.loading) return;

  const payload = {
    district: $("districtSelect").value,
    commodity: $("commoditySelect").value,
    forecast_date: $("dateInput").value,
  };

  if (!payload.district || !payload.commodity || !payload.forecast_date) {
    showToast("Select a district, commodity, and forecast date first.");
    return;
  }

  const days = daysBetween(state.meta.dataset.last_date, payload.forecast_date);
  if (days <= 0) {
    showToast("Forecast date must be after the latest available data.");
    return;
  }
  if (days > state.meta.limits.max_horizon_days) {
    showToast(`Maximum horizon is ${state.meta.limits.max_horizon_days} days.`);
    return;
  }

  setLoading(true);
  try {
    const data = await api("/api/forecast", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.history = {
      district: data.district,
      commodity: data.commodity,
      series: data.history,
      stats: null,
    };
    renderForecast(data);
    if (notify) showToast("Forecast updated with live market intelligence.", "success");
  } catch (error) {
    showToast(error.message);
  } finally {
    setLoading(false);
  }
}

function bindEvents() {
  $("districtSelect").addEventListener("change", () => {
    updateCommodities();
    resetForecastView();
    loadHistory();
  });

  $("commoditySelect").addEventListener("change", () => {
    resetForecastView();
    loadHistory();
  });

  $("dateInput").addEventListener("change", () => {
    updateHorizonUI();
    resetForecastView("Horizon updated — run a forecast to refresh the projection.");
    if (state.history) {
      renderChart(
        state.history.series,
        [],
        state.history.district,
        state.history.commodity
      );
    }
  });

  document.querySelectorAll(".preset-row button").forEach((button) => {
    button.addEventListener("click", () => setPreset(Number(button.dataset.days)));
  });

  $("forecastBtn").addEventListener("click", () => runForecast(true));
}

async function init() {
  bindEvents();

  try {
    await api("/api/health");
    setApiStatus("online", "API live");
  } catch {
    setApiStatus("offline", "API offline");
    showToast("Backend is not reachable. Start FastAPI and refresh.");
    return;
  }

  try {
    const meta = await api("/api/meta");
    state.meta = meta;
    updateHeroStats(meta.dataset);

    const { defaults, dataset, limits } = meta;
    fillSelect($("districtSelect"), meta.districts, defaults.district);
    updateCommodities(defaults.commodity);

    const dateInput = $("dateInput");
    dateInput.min = addDays(dataset.last_date, 1);
    dateInput.max = addDays(dataset.last_date, limits.max_horizon_days);
    dateInput.value = defaults.forecast_date;
    updateHorizonUI();

    resetForecastView();
    await loadHistory();
    await runForecast(false);
  } catch (error) {
    showToast(error.message);
  }
}

document.addEventListener("DOMContentLoaded", init);
