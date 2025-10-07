const DEFAULT_SYMBOLS = ["AAPL", "MSFT"];

const form = document.getElementById("control-form");
const statusEl = document.getElementById("status");
const paperMetricsEl = document.getElementById("paper-metrics");
const trainingMetricsEl = document.getElementById("training-metrics");
const parameterMetricsEl = document.getElementById("parameter-metrics");
const rankingTableBody = document.querySelector("#ranking-table tbody");
const paperWindowEl = document.getElementById("paper-window");
const trainingWindowEl = document.getElementById("training-window");
const chartEl = document.getElementById("chart");
const equityEl = document.getElementById("equity-chart");
const symbolSelect = document.getElementById("symbol-select");
const runButton = document.getElementById("run-button");
const chartTitle = document.getElementById("chart-title");
const availableSymbolsSelect = document.getElementById("available-symbols");
const extraSymbolsInput = document.getElementById("extra-symbols");
const stopRangeToggle = document.getElementById("use_stop_range");
const stopRangeInputs = document.querySelectorAll("[data-stop-range]");
const stopSingleValueToggle = document.getElementById("stop_single_value");
const stopMinInput = document.getElementById("stop_min");
const stopMaxInput = document.getElementById("stop_max");
const stopStepInput = document.getElementById("stop_step");
const holdInfiniteToggle = document.getElementById("hold_infinite");
const holdOnlyInfiniteToggle = document.getElementById("hold_only_infinite");
const holdRangeInputs = document.querySelectorAll("[data-hold-range]");

let latestData = null;

function handleStopRangeToggle() {
    if (!stopRangeInputs) {
        return;
    }
    const rangeEnabled = !!(stopRangeToggle && stopRangeToggle.checked);
    stopRangeInputs.forEach((input) => {
        input.disabled = !rangeEnabled;
    });
    applyStopSingleState();
}

if (stopRangeToggle) {
    stopRangeToggle.addEventListener("change", handleStopRangeToggle);
}

function applyStopSingleState() {
    if (!stopMaxInput && !stopStepInput) {
        return;
    }
    const rangeEnabled = !!(stopRangeToggle && stopRangeToggle.checked);
    const singleMode = rangeEnabled && !!(stopSingleValueToggle && stopSingleValueToggle.checked);
    if (singleMode && stopMinInput && stopMaxInput) {
        stopMaxInput.value = stopMinInput.value;
    }
    if (singleMode && stopStepInput) {
        stopStepInput.value = "1";
    }
    if (stopMaxInput) {
        stopMaxInput.disabled = !rangeEnabled || singleMode;
    }
    if (stopStepInput) {
        stopStepInput.disabled = !rangeEnabled || singleMode;
    }
}

if (stopSingleValueToggle) {
    stopSingleValueToggle.addEventListener("change", applyStopSingleState);
}

function handleHoldOnlyInfiniteToggle() {
    if (!holdRangeInputs) {
        return;
    }
    const onlyInfinite = !!(holdOnlyInfiniteToggle && holdOnlyInfiniteToggle.checked);
    holdRangeInputs.forEach((input) => {
        input.disabled = onlyInfinite;
    });
    if (onlyInfinite && holdInfiniteToggle) {
        holdInfiniteToggle.checked = true;
    }
}

if (holdOnlyInfiniteToggle) {
    holdOnlyInfiniteToggle.addEventListener("change", handleHoldOnlyInfiniteToggle);
}

async function loadAvailableSymbols() {
    if (!availableSymbolsSelect) {
        return;
    }
    availableSymbolsSelect.innerHTML = "";
    let symbols = DEFAULT_SYMBOLS;
    try {
        const response = await fetch("/api/symbols");
        if (response.ok) {
            const payload = await response.json();
            if (Array.isArray(payload.symbols) && payload.symbols.length) {
                symbols = payload.symbols;
            }
        }
    } catch (error) {
        console.warn("Failed to load cached symbols", error);
    }

    symbols.forEach((symbol) => {
        const option = document.createElement("option");
        option.value = symbol;
        option.textContent = symbol;
        if (DEFAULT_SYMBOLS.includes(symbol)) {
            option.selected = true;
        }
        availableSymbolsSelect.append(option);
    });
}

function gatherSymbols(formData) {
    const selected = availableSymbolsSelect
        ? Array.from(availableSymbolsSelect.selectedOptions).map((option) => option.value.toUpperCase())
        : [];
    const manual = (formData.get("extra_symbols") || "")
        .split(",")
        .map((value) => value.trim().toUpperCase())
        .filter(Boolean);
    const combined = [...selected, ...manual];
    const unique = Array.from(new Set(combined));
    return unique.length ? unique : DEFAULT_SYMBOLS;
}

function parseIntOr(value, fallback) {
    const parsed = parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function parseFloatOr(value, fallback) {
    const parsed = parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function buildParameterSpec(formData) {
    const toInt = (name, fallback) => parseIntOr(formData.get(name), fallback);
    const holdOnlyInfinite = formData.has("hold_only_infinite");

    const spec = {
        entry_threshold: {
            minimum: toInt("entry_min", 5),
            maximum: toInt("entry_max", 15),
            step: toInt("entry_step", 5),
        },
        exit_threshold: {
            minimum: toInt("exit_min", 60),
            maximum: toInt("exit_max", 80),
            step: toInt("exit_step", 10),
        },
        max_hold_days: holdOnlyInfinite
            ? {
                minimum: 0,
                maximum: 0,
                step: 1,
                include_infinite: true,
                only_infinite: true,
            }
            : {
                minimum: toInt("hold_min", 3),
                maximum: toInt("hold_max", 10),
                step: toInt("hold_step", 2),
                include_infinite: formData.has("hold_infinite"),
            },
        target_position_pct: {
            minimum: toInt("target_min", 10),
            maximum: toInt("target_max", 20),
            step: toInt("target_step", 5),
        },
        stop_loss_pct: null,
        include_no_stop_loss: formData.has("include_no_stop_loss"),
        lot_size: toInt("lot_size", 10),
    };

    if (formData.has("use_stop_range")) {
        const singleStop = formData.has("stop_single_value");
        const stopMin = toInt("stop_min", 5);
        const stopMax = singleStop ? stopMin : toInt("stop_max", 10);
        const stopStep = singleStop ? 1 : toInt("stop_step", 5);
        spec.stop_loss_pct = {
            minimum: stopMin,
            maximum: stopMax,
            step: stopStep,
        };
    }

    return spec;
}

async function runBacktest(event) {
    if (event) {
        event.preventDefault();
    }

    const formData = new FormData(form);
    const symbols = gatherSymbols(formData);

    if (!symbols.length) {
        setStatus("Please select at least one symbol to optimize.", "error");
        return;
    }

    const payload = {
        symbols,
        initial_cash: parseFloatOr(formData.get("initial_cash"), 100000),
        limit: parseIntOr(formData.get("limit"), 250),
        auto_fetch: formData.has("auto_fetch"),
        paper_days: parseIntOr(formData.get("paper_days"), 365),
        training_years: parseFloatOr(formData.get("training_years"), 2),
        parameter_spec: buildParameterSpec(formData),
    };

    setStatus("Searching for optimal parameters...", "info");
    runButton.disabled = true;
    try {
        const response = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            let errorMessage = `Request failed with status ${response.status}`;
            try {
                const detail = await response.json();
                if (typeof detail?.detail === "string") {
                    errorMessage = detail.detail;
                } else if (detail?.detail?.message) {
                    errorMessage = detail.detail.message;
                }
            } catch (parseError) {
                const fallback = await response.text();
                if (fallback) {
                    errorMessage = fallback;
                }
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();
        latestData = data;
        updateChartSymbolOptions(data.symbols || []);
        renderSymbolData();
    } catch (error) {
        console.error(error);
        setStatus(error.message || "Failed to run optimization.", "error");
    } finally {
        runButton.disabled = false;
    }
}

function updateChartSymbolOptions(symbols) {
    symbolSelect.innerHTML = "";
    symbols.forEach((symbol) => {
        const option = document.createElement("option");
        option.value = symbol;
        option.textContent = symbol;
        symbolSelect.append(option);
    });
    if (symbols.length) {
        const active = symbols.includes(symbolSelect.value) ? symbolSelect.value : symbols[0];
        symbolSelect.value = active;
        chartTitle.textContent = `${active} Candlestick`;
    } else {
        chartTitle.textContent = "Candlestick Chart";
    }
}

function getActiveSymbol() {
    if (!latestData || !Array.isArray(latestData.symbols) || !latestData.symbols.length) {
        return null;
    }
    const candidate = symbolSelect.value;
    if (candidate && latestData.symbols.includes(candidate)) {
        return candidate;
    }
    return latestData.symbols[0];
}

function getSymbolResult(symbol) {
    if (!latestData || !symbol) {
        return null;
    }
    return latestData.results?.[symbol] || null;
}

function renderSymbolData() {
    if (!latestData) {
        Plotly.purge(chartEl);
        Plotly.purge(equityEl);
        paperMetricsEl.innerHTML = "";
        trainingMetricsEl.innerHTML = "";
        parameterMetricsEl.innerHTML = "";
        rankingTableBody.innerHTML = "";
        paperWindowEl.textContent = "";
        trainingWindowEl.textContent = "";
        setStatus("Run an optimization to see results.", "info");
        return;
    }

    const symbol = getActiveSymbol();
    if (!symbol) {
        Plotly.purge(chartEl);
        Plotly.purge(equityEl);
        setStatus("No symbols returned from optimization.", "warning");
        return;
    }

    const result = getSymbolResult(symbol);
    renderCandles(symbol, result);
    renderEquity(result);
    renderOptimization(result);
    updateStatusForSymbol(symbol, result);
}

function updateStatusForSymbol(symbol, result) {
    if (!symbol || !result) {
        setStatus(`No optimization output for ${symbol || "selected symbol"}.`, "warning");
        return;
    }
    const optimization = result.optimization || {};
    const trainingRange = formatRange(optimization.training);
    const paperRange = formatRange(optimization.paper);
    let message = `Optimized ${latestData.symbols?.length || 0} symbol(s). Showing ${symbol} — Train ${trainingRange}, Paper ${paperRange}.`;
    let level = "success";
    if (latestData.warnings) {
        const missing = Array.isArray(latestData.warnings.missing)
            ? latestData.warnings.missing
                .map((item) => item.symbol || item.path || item.reason)
                .filter(Boolean)
                .join(", ")
            : "";
        if (missing) {
            message += ` Missing: ${missing}`;
        } else if (latestData.warnings.message) {
            message += ` ${latestData.warnings.message}`;
        }
        level = "warning";
    }
    setStatus(message, level);
}

function renderCandles(symbol, result) {
    if (!result || !Array.isArray(result.candles) || !result.candles.length) {
        Plotly.purge(chartEl);
        chartTitle.textContent = symbol ? `${symbol} Candlestick (no data)` : "Candlestick Chart";
        return;
    }

    const candles = result.candles;
    const buys = result.buy_signals || [];

    const traceCandles = {
        type: "candlestick",
        name: `${symbol} price`,
        x: candles.map((candle) => candle.timestamp),
        open: candles.map((candle) => candle.open),
        high: candles.map((candle) => candle.high),
        low: candles.map((candle) => candle.low),
        close: candles.map((candle) => candle.close),
        increasing: { line: { color: "#2ecc71" } },
        decreasing: { line: { color: "#e74c3c" } },
    };

    const traceBuys = {
        type: "scatter",
        mode: "markers",
        name: "Buy Signal",
        x: buys.map((point) => point.timestamp),
        y: buys.map((point) => point.price),
        marker: {
            color: "#f39c12",
            size: 10,
            symbol: "triangle-up",
            line: { width: 1, color: "#8e44ad" },
        },
    };

    const layout = {
        margin: { t: 40, r: 10, b: 50, l: 60 },
        xaxis: { title: "Date", rangeslider: { visible: false } },
        yaxis: { title: "Price" },
        legend: { orientation: "h", x: 0, y: 1.1 },
        dragmode: "pan",
        hovermode: "x unified",
    };

    const config = { responsive: true, displaylogo: false };
    const traces = buys.length ? [traceCandles, traceBuys] : [traceCandles];
    Plotly.newPlot(chartEl, traces, layout, config);
    chartTitle.textContent = `${symbol} Candlestick`;
}

function renderEquity(result) {
    if (!result || !Array.isArray(result.equity_curve) || !result.equity_curve.length) {
        Plotly.purge(equityEl);
        return;
    }

    const trace = {
        type: "scatter",
        mode: "lines",
        name: "Equity",
        x: result.equity_curve.map((point) => point.timestamp),
        y: result.equity_curve.map((point) => point.equity),
        line: { color: "#3498db", width: 3 },
    };

    const layout = {
        margin: { t: 40, r: 10, b: 50, l: 60 },
        xaxis: { title: "Date" },
        yaxis: { title: "Equity" },
        hovermode: "x unified",
    };

    Plotly.newPlot(equityEl, [trace], layout, { responsive: true, displaylogo: false });
}

function renderOptimization(result) {
    if (!result || !result.optimization) {
        paperMetricsEl.innerHTML = "";
        trainingMetricsEl.innerHTML = "";
        parameterMetricsEl.innerHTML = "";
        rankingTableBody.innerHTML = "";
        paperWindowEl.textContent = "";
        trainingWindowEl.textContent = "";
        if (result?.metrics) {
            renderMetricCards(paperMetricsEl, buildMetricRows(result.metrics));
        }
        return;
    }

    const { paper, training, best_parameters: bestParams, rankings } = result.optimization;

    renderMetricCards(paperMetricsEl, buildMetricRows(paper?.metrics || result.metrics));
    renderMetricCards(trainingMetricsEl, buildMetricRows(training?.metrics));
    renderParameterCards(bestParams);
    renderRankingTable(rankings || []);

    paperWindowEl.textContent = `Window: ${formatRange(paper)}`;
    trainingWindowEl.textContent = `Window: ${formatRange(training)}`;
}

function buildMetricRows(metrics) {
    if (!metrics) {
        return [];
    }
    return [
        { label: "Final Equity", value: formatCurrency(metrics.final_equity) },
        { label: "Net Profit", value: formatCurrency(metrics.net_profit) },
        { label: "Total Return", value: formatPercent(metrics.total_return) },
        { label: "CAGR", value: formatPercent(metrics.cagr) },
        { label: "Sharpe Ratio", value: formatNumber(metrics.sharpe_ratio) },
        { label: "Max Drawdown", value: formatPercent(-Math.abs(metrics.max_drawdown || 0)) },
        { label: "Trades", value: formatNumber(metrics.trade_count, 0) },
        { label: "Final Cash", value: formatCurrency(metrics.final_cash) },
    ];
}

function renderMetricCards(container, rows) {
    if (!container) {
        return;
    }
    if (!rows || !rows.length) {
        container.innerHTML = "<p class=\"metric-empty\">No metrics available</p>";
        return;
    }
    container.innerHTML = rows
        .map(
            (row) => `
        <div class="metric-card">
          <span class="metric-label">${row.label}</span>
          <span class="metric-value">${row.value ?? "–"}</span>
        </div>
      `,
        )
        .join("");
}

function renderParameterCards(params) {
    if (!parameterMetricsEl) {
        return;
    }
    if (!params) {
        parameterMetricsEl.innerHTML = "<p class=\"metric-empty\">No parameters selected</p>";
        return;
    }

    const rows = [
        { label: "Entry RSI", value: formatNumber(params.entry_threshold, 0) },
        { label: "Exit RSI", value: formatNumber(params.exit_threshold, 0) },
        { label: "Max Hold Days", value: formatNumber(params.max_hold_days, 0) },
        { label: "Target Position", value: formatPercent(params.target_position_pct) },
        {
            label: "Stop Loss",
            value:
                params.stop_loss_pct === null || params.stop_loss_pct === undefined
                    ? "None"
                    : formatPercent(params.stop_loss_pct),
        },
        { label: "Lot Size", value: formatNumber(params.lot_size, 0) },
    ];

    parameterMetricsEl.innerHTML = rows
        .map(
            (row) => `
        <div class="metric-card">
          <span class="metric-label">${row.label}</span>
          <span class="metric-value">${row.value ?? "–"}</span>
        </div>
      `,
        )
        .join("");
}

function renderRankingTable(rows) {
    if (!rankingTableBody) {
        return;
    }
    if (!rows.length) {
        rankingTableBody.innerHTML = "<tr><td colspan=\"4\">No candidate parameters evaluated</td></tr>";
        return;
    }

    rankingTableBody.innerHTML = rows
        .map((row) => {
            const cagr = formatPercent(row.cagr);
            const totalReturn = formatPercent(row.total_return);
            const equity = formatCurrency(row.final_equity);
            return `
        <tr>
          <td>${row.params}</td>
          <td>${cagr}</td>
          <td>${totalReturn}</td>
          <td>${equity}</td>
        </tr>
      `;
        })
        .join("");
}

function setStatus(message, level = "info") {
    statusEl.textContent = message;
    statusEl.className = `status ${level}`;
}

function formatCurrency(value) {
    if (value === undefined || value === null || Number.isNaN(value)) {
        return "–";
    }
    const formatter = new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
    });
    return formatter.format(value);
}

function formatPercent(value) {
    if (value === undefined || value === null || Number.isNaN(value)) {
        return "–";
    }
    const formatter = new Intl.NumberFormat("en-US", {
        style: "percent",
        maximumFractionDigits: 2,
    });
    return formatter.format(value);
}

function formatNumber(value, digits = 2) {
    if (value === undefined || value === null || Number.isNaN(value)) {
        return "–";
    }
    const formatter = new Intl.NumberFormat("en-US", {
        maximumFractionDigits: digits,
    });
    return formatter.format(value);
}

function formatRange(range) {
    if (!range) {
        return "–";
    }
    const start = range.start || range[0];
    const end = range.end || range[1];
    if (!start || !end) {
        return "–";
    }
    return `${formatDate(start)} → ${formatDate(end)}`;
}

function formatDate(value) {
    if (!value) {
        return "–";
    }
    const dateValue = typeof value === "string" ? value : value.toString();
    const parsed = new Date(dateValue);
    if (Number.isNaN(parsed.getTime())) {
        return dateValue;
    }
    return parsed.toISOString().split("T")[0];
}

form.addEventListener("submit", runBacktest);
symbolSelect.addEventListener("change", () => {
    renderSymbolData();
});

document.addEventListener("DOMContentLoaded", async () => {
    handleStopRangeToggle();
    applyStopSingleState();
    handleHoldOnlyInfiniteToggle();
    await loadAvailableSymbols();
    await runBacktest();
});
