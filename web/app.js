const state = {
  summary: null,
  exploration: null,
  board: { version: 1, items: [] },
};

const numberFormatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
const compactFormatter = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

const $ = (id) => document.getElementById(id);

function number(value) {
  return value === null || value === undefined ? "—" : numberFormatter.format(value);
}

function compact(value) {
  return value === null || value === undefined ? "—" : compactFormatter.format(value);
}

function setText(id, value) {
  const element = $(id);
  if (element) element.textContent = value;
}

function titleize(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatDate(value) {
  if (!value) return "unknown time";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

async function fetchJson(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function sendJson(path, method, payload) {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

function showError(error) {
  $("error-banner").hidden = false;
  setText("error-message", error instanceof Error ? error.message : String(error));
  $("service-dot").className = "status-dot offline";
  setText("service-status", "Unavailable");
}

function showOnline() {
  $("error-banner").hidden = true;
  $("service-dot").className = "status-dot online";
  setText("service-status", "Live local data");
}

function populateSelect(explorations) {
  const select = $("run-select");
  const current = select.value;
  select.replaceChildren();
  explorations.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.run_id;
    option.textContent = `${item.run_id} · ${number(item.probe_user_count)} users · ${number(item.probe_eligible_item_count)} items`;
    select.append(option);
  });
  const preferred = state.summary?.default_run_id || current || explorations[0]?.run_id;
  if (preferred && explorations.some((item) => item.run_id === preferred)) select.value = preferred;
}

function renderKpis(data) {
  const dimensions = data.dimensions || {};
  const split = data.split || {};
  const probe = data.pilot_support_probe || {};
  setText("kpi-ratings", compact(dimensions.rating_count));
  setText("kpi-users", compact(dimensions.user_count));
  setText("kpi-items", compact(dimensions.item_count));
  setText("kpi-eval-users", compact(split.evaluation_eligible_user_count));
  setText("kpi-support-items", number(probe.eligible_item_count));
  setText("kpi-support-foot", `${number(probe.support_threshold)} positives / item`);
}

function renderSupport(data) {
  const probe = data.pilot_support_probe || {};
  const observed = Number(probe.observed_item_count || 0);
  const eligible = Number(probe.eligible_item_count || 0);
  const ratio = observed ? (eligible / observed) * 100 : 0;
  setText("probe-size", `${number(probe.user_count)} users`);
  setText("support-number", number(eligible));
  setText("support-ratio", `${ratio.toFixed(1)}% of observed items`);
  setText("observed-items", number(observed));
  setText("probe-interactions", number(probe.positive_train_interaction_count));
  setText("minimum-support", number(probe.support_threshold));
  $("support-bar-fill").style.width = `${Math.min(100, ratio)}%`;

  const quantiles = probe.item_support_quantiles || {};
  const chart = $("quantile-chart");
  chart.replaceChildren();
  const entries = Object.entries(quantiles);
  const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  entries.forEach(([quantile, value]) => {
    const numeric = Number(value) || 0;
    const column = document.createElement("div");
    column.className = "quantile-column";
    const label = quantile === "0.1" || quantile === "0.25" || quantile === "0.5" || quantile === "0.75" || quantile === "0.9" || quantile === "0.99" ? `${Number(quantile) * 100}%` : quantile === "0" ? "min" : "max";
    column.innerHTML = `<span class="quantile-value">${compact(numeric)}</span><span class="quantile-bar" style="height:${Math.max(3, Math.round((Math.log1p(numeric) / Math.log1p(max)) * 100))}%"></span><span class="quantile-label">${label}</span>`;
    chart.append(column);
  });
}

function renderProtocol(data) {
  const split = data.split || {};
  setText("split-partition", titleize(split.partition_object));
  setText("split-ordering", titleize(split.ordering));
  setText("split-fractions", `${Math.round((split.train_fraction || 0) * 100)} / ${Math.round((split.validation_fraction || 0) * 100)} / ${Math.round((split.test_fraction || 0) * 100)}`);
  setText("split-tie-seed", number(split.tie_break_seed));
  setText("split-positive-filter", titleize(split.positive_filter_timing));
  setText("tie-groups", number(split.timestamp_tie_groups));
}

function renderRatings(data) {
  const counts = (data.ratings || {}).count_by_value || {};
  const chart = $("rating-chart");
  chart.replaceChildren();
  const entries = Object.entries(counts);
  const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  entries.forEach(([rating, value]) => {
    const numeric = Number(value) || 0;
    const column = document.createElement("div");
    column.className = "rating-column";
    const positive = Number(rating) >= 4;
    column.innerHTML = `<span class="rating-value">${compact(numeric)}</span><span class="rating-bar${positive ? " positive" : ""}" style="height:${Math.max(2, Math.round((numeric / max) * 100))}%"></span><span class="rating-label">${rating}</span>`;
    chart.append(column);
  });
}

function renderRuns(runs) {
  const body = $("run-table-body");
  body.replaceChildren();
  if (!runs.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="5">No planned or completed run records found.</td>';
    body.append(row);
    return;
  }
  runs.forEach((run) => {
    const row = document.createElement("tr");
    const status = run.status || "pending";
    const support = run.support_check_passed ? "Passed" : "Not run";
    const worktree = run.worktree_clean ? "Clean" : "Modified";
    row.innerHTML = `<td>${titleize(run.run_id)}</td><td>${titleize(run.variant_id)}</td><td><span class="status-pill status-${status}">${status}</span></td><td class="${run.support_check_passed ? "status-yes" : "status-no"}">${support}</td><td class="${run.worktree_clean ? "status-yes" : "status-no"}">${worktree}</td>`;
    body.append(row);
  });
}

function boardInput(field, value, placeholder = "") {
  const input = document.createElement("input");
  input.dataset.field = field;
  input.value = value || "";
  input.placeholder = placeholder;
  return input;
}

function boardSelect(field, value, options) {
  const select = document.createElement("select");
  select.dataset.field = field;
  options.forEach(([optionValue, label]) => {
    const option = document.createElement("option");
    option.value = optionValue;
    option.textContent = label;
    option.selected = optionValue === value;
    select.append(option);
  });
  return select;
}

function boardCell(child) {
  const cell = document.createElement("td");
  cell.append(child);
  return cell;
}

const BOARD_COMMAND_OPTIONS = [
  ["make explore", "make explore"],
  ["make validate-experiment", "make validate-experiment"],
  ["create_run_record.py", "create_run_record.py"],
];

function parameterText(parameters) {
  return Object.entries(parameters || {}).map(([key, value]) => `${key}=${value}`).join(" ");
}

function parseParameters(value) {
  const parameters = {};
  const text = value.trim();
  if (!text) return parameters;
  text.split(/\s+/).forEach((token) => {
    const separator = token.indexOf("=");
    if (separator <= 0) throw new Error("Parameters must use KEY=value pairs.");
    parameters[token.slice(0, separator)] = token.slice(separator + 1);
  });
  return parameters;
}

function applyBoardItemToRow(row, item) {
  row.querySelectorAll("[data-field]").forEach((field) => {
    const value = field.dataset.field === "parameters" ? parameterText(item.parameters) : item[field.dataset.field];
    field.value = value || "";
  });
  const lastRun = row.querySelector("[data-last-run]");
  if (!item.last_run) {
    lastRun.textContent = "Not run";
    return;
  }
  const status = item.last_run.status || "started";
  const outcome = status === "running" ? "Running" : status === "completed" ? "Completed" : status === "failed" ? "Failed" : "Started";
  lastRun.textContent = `${outcome} · pid ${item.last_run.pid}`;
}

function setBoardMessage(message, error = false) {
  const element = $("board-message");
  element.textContent = message;
  element.className = `board-message${error ? " board-message-error" : ""}`;
}

async function refreshBoard() {
  const board = await fetchJson("/api/board");
  state.board = board;
  renderBoard(board);
}

async function updateBoardItem(itemId, row) {
  const payload = {};
  try {
    row.querySelectorAll("[data-field]").forEach((field) => {
      payload[field.dataset.field] = field.dataset.field === "parameters" ? parseParameters(field.value) : field.value;
    });
    const item = await sendJson(`/api/board/items/${encodeURIComponent(itemId)}`, "PATCH", payload);
    const index = state.board.items.findIndex((entry) => entry.id === itemId);
    if (index >= 0) state.board.items[index] = item;
    applyBoardItemToRow(row, item);
    setBoardMessage("Saved.");
  } catch (error) {
    setBoardMessage(error.message, true);
  }
}

async function runBoardItem(itemId, row) {
  try {
    const result = await sendJson(`/api/board/items/${encodeURIComponent(itemId)}/run`, "POST", {});
    const index = state.board.items.findIndex((entry) => entry.id === itemId);
    if (index >= 0) state.board.items[index] = result.item;
    applyBoardItemToRow(row, result.item);
    setBoardMessage(`Started ${result.item.command} (job ${result.job_id}).`);
    pollBoardJob(itemId, row, result.job_id);
  } catch (error) {
    setBoardMessage(error.message, true);
  }
}

async function pollBoardJob(itemId, row, jobId) {
  try {
    const job = await fetchJson(`/api/board/jobs/${encodeURIComponent(jobId)}`);
    const index = state.board.items.findIndex((entry) => entry.id === itemId);
    if (index >= 0) {
      const item = state.board.items[index];
      item.last_run = { ...(item.last_run || {}), ...job };
      applyBoardItemToRow(row, item);
    }
    if (job.status === "running") {
      window.setTimeout(() => pollBoardJob(itemId, row, jobId), 1000);
      return;
    }
    const outcome = job.status === "completed" ? "completed" : "failed";
    setBoardMessage(`${job.command} ${outcome}.`, job.status !== "completed");
  } catch (error) {
    setBoardMessage(`Job status unavailable: ${error.message}`, true);
  }
}

async function deleteBoardItem(itemId, row) {
  if (!window.confirm("Delete this research task?")) return;
  try {
    await sendJson(`/api/board/items/${encodeURIComponent(itemId)}`, "DELETE", {});
    state.board.items = state.board.items.filter((item) => item.id !== itemId);
    row.remove();
    if (!state.board.items.length) renderBoard(state.board);
    setBoardMessage("Deleted.");
  } catch (error) {
    setBoardMessage(error.message, true);
  }
}

function renderBoard(board) {
  state.board = board;
  const body = $("board-table-body");
  body.replaceChildren();
  const items = board.items || [];
  if (!items.length) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="8">No research tasks yet. Add one above.</td>';
    body.append(row);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("tr");
    row.dataset.itemId = item.id;
    row.append(
      boardCell(boardInput("title", item.title, "Task")),
      boardCell(boardSelect("status", item.status, [["todo", "To do"], ["doing", "In progress"], ["blocked", "Blocked"], ["done", "Done"]])),
      boardCell(boardSelect("priority", item.priority, [["low", "Low"], ["normal", "Normal"], ["high", "High"]])),
      boardCell(boardSelect("command", item.command, BOARD_COMMAND_OPTIONS)),
      boardCell(boardInput("parameters", parameterText(item.parameters), "KEY=value ...")),
      boardCell(boardInput("owner", item.owner, "Owner")),
      boardCell(boardInput("notes", item.notes, "Notes")),
    );
    const actions = document.createElement("td");
    actions.className = "board-actions";
    const run = document.createElement("button");
    run.className = "button board-action board-run";
    run.type = "button";
    run.textContent = "Run";
    run.addEventListener("click", () => runBoardItem(item.id, row));
    const save = document.createElement("button");
    save.className = "button button-outline board-action";
    save.type = "button";
    save.textContent = "Save";
    save.addEventListener("click", () => updateBoardItem(item.id, row));
    const remove = document.createElement("button");
    remove.className = "button button-danger board-action";
    remove.type = "button";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => deleteBoardItem(item.id, row));
    const lastRun = document.createElement("small");
    const initialStatus = item.last_run?.status || "started";
    const initialOutcome = initialStatus === "running" ? "Running" : initialStatus === "completed" ? "Completed" : initialStatus === "failed" ? "Failed" : "Started";
    lastRun.textContent = item.last_run ? `${initialOutcome} · pid ${item.last_run.pid}` : "Not run";
    actions.append(run, save, remove, lastRun);
    row.append(actions);
    body.append(row);
  });
}

function markdownText(value) {
  return value.replace(/\*\*/g, "").replace(/`/g, "").trim();
}

function proposalTable(lines) {
  const table = document.createElement("table");
  table.className = "proposal-table";
  const header = lines[0].split("|").map((cell) => cell.trim()).filter(Boolean);
  const headRow = document.createElement("tr");
  header.forEach((cell) => {
    const th = document.createElement("th");
    th.textContent = markdownText(cell);
    headRow.append(th);
  });
  const thead = document.createElement("thead");
  thead.append(headRow);
  table.append(thead);
  const body = document.createElement("tbody");
  lines.slice(2).forEach((line) => {
    const row = document.createElement("tr");
    line.split("|").map((cell) => cell.trim()).filter(Boolean).forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = markdownText(cell);
      row.append(td);
    });
    body.append(row);
  });
  table.append(body);
  return table;
}

function renderProposal(markdown) {
  const content = $("proposal-content");
  content.replaceChildren();
  const lines = markdown.split(/\r?\n/);
  let index = 0;
  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line) {
      index += 1;
      continue;
    }
    if (line.startsWith("|")) {
      const rows = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        rows.push(lines[index].trim());
        index += 1;
      }
      if (rows.length >= 2) content.append(proposalTable(rows));
      continue;
    }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const element = document.createElement(heading[1].length === 3 ? "h4" : "h3");
      element.textContent = markdownText(heading[2]);
      content.append(element);
      index += 1;
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      const list = document.createElement("ul");
      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        const item = document.createElement("li");
        item.textContent = markdownText(lines[index].replace(/^\s*[-*]\s+/, ""));
        list.append(item);
        index += 1;
      }
      content.append(list);
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      const list = document.createElement("ol");
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        const item = document.createElement("li");
        item.textContent = markdownText(lines[index].replace(/^\s*\d+\.\s+/, ""));
        list.append(item);
        index += 1;
      }
      content.append(list);
      continue;
    }
    const paragraph = document.createElement("p");
    paragraph.textContent = markdownText(line);
    content.append(paragraph);
    index += 1;
  }
}

async function loadProposal() {
  try {
    const proposal = await fetchJson("/api/proposal");
    renderProposal(proposal.markdown);
  } catch (error) {
    setText("proposal-content", `Proposal unavailable: ${error.message}`);
  }
}



function renderFigures(data) {
  const figures = data.figures || [];
  setText("figure-count", `${figures.length} figures`);
  const gallery = $("figure-gallery");
  gallery.replaceChildren();
  figures.forEach((figure) => {
    const card = document.createElement("figure");
    card.className = "figure-card";
    const label = figure.path.split("/").pop().replace(/^\d+_/, "").replace(/\.png$/, "").replace(/_/g, " ");
    const image = document.createElement("img");
    image.src = `/${figure.path}`;
    image.alt = `${label} exploration figure`;
    image.loading = "lazy";
    const caption = document.createElement("figcaption");
    caption.className = "figure-caption";
    caption.textContent = titleize(label);
    card.append(image, caption);
    image.onerror = () => {
      image.remove();
      card.classList.add("figure-missing");
      caption.textContent = `${titleize(label)} · not generated locally — run make explore`;
    };
    gallery.append(card);
  });
}

function render(data) {
  state.exploration = data;
  renderKpis(data);
  renderSupport(data);
  renderProtocol(data);
  renderRatings(data);
  renderFigures(data);
  setText("last-updated", `Snapshot created ${formatDate(data.created_at_utc)}`);
  const inputs = data.inputs || {};
  setText("footer-dataset", `${inputs.dataset_id || "Tracked metadata"} · ${inputs.dataset_version || "no version"} · no model run executed`);
}

async function loadExploration(runId) {
  if (!runId) return;
  try {
    const data = await fetchJson(`/api/explorations/${encodeURIComponent(runId)}`);
    render(data);
    showOnline();
  } catch (error) {
    showError(error);
  }
}

async function loadDashboard() {
  try {
    const summary = await fetchJson("/api/summary");
    state.summary = summary;
    populateSelect(summary.explorations || []);
    renderRuns(summary.runs || []);
    renderBoard(summary.board || { version: 1, items: [] });
    await loadProposal();
    await loadExploration($("run-select").value || summary.default_run_id);
    showOnline();
  } catch (error) {
    showError(error);
  }
}

$("run-select").addEventListener("change", (event) => loadExploration(event.target.value));
$("refresh-button").addEventListener("click", loadDashboard);
$("board-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  try {
    await sendJson("/api/board/items", "POST", {
      title: $("board-title").value,
      status: $("board-status").value,
      priority: $("board-priority").value,
      command: $("board-command").value,
      parameters: parseParameters($("board-parameters").value),
      owner: $("board-owner").value,
      notes: "",
    });
    form.reset();
    $("board-parameters").value = "RUN_ID=exploration";
    await refreshBoard();
    setBoardMessage("Added.");
  } catch (error) {
    setBoardMessage(error.message, true);
  }
});
loadDashboard();
