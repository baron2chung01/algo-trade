const DEFAULT_SYMBOLS = ["AAPL", "MSFT"];
const STRATEGY_MEAN_REVERSION = "mean_reversion";
const STRATEGY_BREAKOUT = "breakout";

const pageId = document.body?.dataset?.page || "";

const elements = {
    form: document.getElementById("control-form"),
    status: document.getElementById("status"),
    paperMetrics: document.getElementById("paper-metrics"),
    trainingMetrics: document.getElementById("training-metrics"),
    parameterMetrics: document.getElementById("parameter-metrics"),
    rankingTableBody: document.querySelector("#ranking-table tbody"),
    paperWindow: document.getElementById("paper-window"),
    trainingWindow: document.getElementById("training-window"),
    chart: document.getElementById("chart"),
    equityChart: document.getElementById("equity-chart"),
    symbolSelect: document.getElementById("symbol-select"),
    runButton: document.getElementById("run-button"),
    chartTitle: document.getElementById("chart-title"),
    availableSymbolsSelect: document.getElementById("available-symbols"),
    extraSymbolsInput: document.getElementById("extra-symbols"),
    strategySelect: document.getElementById("strategy"),
    meanReversionSection: document.getElementById("mean-reversion-params"),
    breakoutSection: document.getElementById("breakout-params"),
};

const meanReversionControls = {
    stopRangeToggle: document.getElementById("use_stop_range"),
    stopRangeInputs: document.querySelectorAll("[data-stop-range]"),
    stopSingleValueToggle: document.getElementById("stop_single_value"),
    stopMinInput: document.getElementById("stop_min"),
    stopMaxInput: document.getElementById("stop_max"),
    stopStepInput: document.getElementById("stop_step"),
    holdInfiniteToggle: document.getElementById("hold_infinite"),
    holdOnlyInfiniteToggle: document.getElementById("hold_only_infinite"),
    holdRangeInputs: document.querySelectorAll("[data-hold-range]"),
};

const breakoutControls = {
    stopToggle: document.getElementById("breakout_use_stop_range"),
    stopInputs: document.querySelectorAll("[data-breakout-stop-range]"),
    trailingToggle: document.getElementById("breakout_use_trailing_range"),
    trailingInputs: document.querySelectorAll("[data-breakout-trailing-range]"),
    profitToggle: document.getElementById("breakout_use_profit_range"),
    profitInputs: document.querySelectorAll("[data-breakout-profit-range]"),
    holdInfiniteToggle: document.getElementById("breakout_hold_infinite"),
    holdOnlyInfiniteToggle: document.getElementById("breakout_hold_only_infinite"),
    holdRangeInputs: document.querySelectorAll("[data-breakout-hold-range]"),
};

let latestData = null;

function initialize() {
    if (!elements.form) {
        return;
    }
    setupMeanReversionControls();
    setupBreakoutControls();
    toggleParameterSections();

    elements.form.addEventListener("submit", runBacktest);
    if (elements.symbolSelect) {
        elements.symbolSelect.addEventListener("change", renderSymbolData);
    }
    if (elements.strategySelect && elements.strategySelect.tagName === "SELECT") {
        elements.strategySelect.addEventListener("change", () => {
            toggleParameterSections();
        });
    }

    loadAvailableSymbols().then(runBacktest);
}

function setupMeanReversionControls() {
    const controls = meanReversionControls;
    if (controls.stopRangeToggle) {
        controls.stopRangeToggle.addEventListener("change", () => {
            applyStopRangeState(controls);
        });
    }
    if (controls.stopSingleValueToggle) {
        controls.stopSingleValueToggle.addEventListener("change", () => {
            applyStopRangeState(controls);
        });
    }
    if (controls.holdOnlyInfiniteToggle) {
        controls.holdOnlyInfiniteToggle.addEventListener("change", () => {
            applyHoldState(controls.holdOnlyInfiniteToggle, controls.holdInfiniteToggle, controls.holdRangeInputs);
        });
    }

    applyStopRangeState(controls);
    applyHoldState(controls.holdOnlyInfiniteToggle, controls.holdInfiniteToggle, controls.holdRangeInputs);
}

function setupBreakoutControls() {
    const controls = breakoutControls;
    createRangeToggle(controls.stopToggle, controls.stopInputs);
    createRangeToggle(controls.trailingToggle, controls.trailingInputs);
    createRangeToggle(controls.profitToggle, controls.profitInputs);
    applyHoldState(controls.holdOnlyInfiniteToggle, controls.holdInfiniteToggle, controls.holdRangeInputs);
    if (controls.holdOnlyInfiniteToggle) {
        controls.holdOnlyInfiniteToggle.addEventListener("change", () => {
            applyHoldState(controls.holdOnlyInfiniteToggle, controls.holdInfiniteToggle, controls.holdRangeInputs);
        });
    }
}

function applyStopRangeState(controls) {
    if (!controls.stopRangeToggle || !controls.stopRangeInputs) {
        return;
    }
    const rangeEnabled = controls.stopRangeToggle.checked;
    controls.stopRangeInputs.forEach((input) => {
        input.disabled = !rangeEnabled;
    });
    if (controls.stopSingleValueToggle) {
        const singleMode = rangeEnabled && controls.stopSingleValueToggle.checked;
        if (singleMode && controls.stopMinInput && controls.stopMaxInput) {
            controls.stopMaxInput.value = controls.stopMinInput.value;
        }
        if (controls.stopStepInput) {
            controls.stopStepInput.disabled = !rangeEnabled || singleMode;
            if (singleMode) {
                controls.stopStepInput.value = "1";
            }
        }
        if (controls.stopMaxInput) {
            controls.stopMaxInput.disabled = !rangeEnabled || singleMode;
        }
    }
}

function createRangeToggle(toggle, inputs) {
    if (!toggle || !inputs) {
        return;
    }
    const applyState = () => {
        const enabled = toggle.checked;
        inputs.forEach((input) => {
            input.disabled = !enabled;
        });
    };
    toggle.addEventListener("change", applyState);
    applyState();
}

function applyHoldState(onlyToggle, includeToggle, inputs) {
    if (!onlyToggle || !inputs) {
        return;
    }
    const onlyInfinite = onlyToggle.checked;
    inputs.forEach((input) => {
        input.disabled = onlyInfinite;
    });
    if (onlyInfinite && includeToggle) {
        includeToggle.checked = true;
    }
}

function toggleParameterSections() {
    const strategy = getSelectedStrategy();
    if (elements.meanReversionSection) {
        elements.meanReversionSection.style.display = strategy === STRATEGY_MEAN_REVERSION ? "" : "none";
    }
    if (elements.breakoutSection) {
        elements.breakoutSection.style.display = strategy === STRATEGY_BREAKOUT ? "" : "none";
    }
    if (elements.strategySelect && elements.strategySelect.tagName !== "SELECT") {
        elements.strategySelect.value = strategy;
    }
}

async function loadAvailableSymbols() {
    if (!elements.availableSymbolsSelect) {
        return;
    }
    elements.availableSymbolsSelect.innerHTML = "";
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
        elements.availableSymbolsSelect.append(option);
    });
}

function gatherSymbols(formData) {
    const selected = elements.availableSymbolsSelect
        ? Array.from(elements.availableSymbolsSelect.selectedOptions).map((option) => option.value.toUpperCase())
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

async function runBacktest(event) {
    if (event) {
        event.preventDefault();
    }

    const formData = new FormData(elements.form);
    const symbols = gatherSymbols(formData);
    if (!symbols.length) {
        setStatus("Please select at least one symbol to optimize.", "error");
        return;
    }

    const strategy = getSelectedStrategy();

    const payload = {
        strategy,
        symbols,
        initial_cash: parseFloatOr(formData.get("initial_cash"), 10000),
        limit: parseIntOr(formData.get("limit"), 250),
        auto_fetch: formData.has("auto_fetch"),
        paper_days: parseIntOr(formData.get("paper_days"), 360),
        training_years: parseFloatOr(formData.get("training_years"), 2),
    };

    if (strategy === STRATEGY_MEAN_REVERSION) {
        payload.parameter_spec = buildMeanReversionSpec(formData);
    } else {
        payload.breakout_spec = buildBreakoutSpec(formData);
    }

    setStatus("Searching for optimal parameters...", "info");
    disableRunButton(true);
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
        disableRunButton(false);
    }
}

function disableRunButton(disabled) {
    if (elements.runButton) {
        elements.runButton.disabled = disabled;
    }
}

function buildMeanReversionSpec(formData) {
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
        lot_size: toInt("lot_size", 1),
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

function buildBreakoutSpec(formData) {
    const toInt = (name, fallback) => parseIntOr(formData.get(name), fallback);
    const toFloat = (name, fallback) => parseFloatOr(formData.get(name), fallback);
    const toPercent = (name, fallback) => parseFloatOr(formData.get(name), fallback) / 100;

    const patternsSelect = document.getElementById("breakout_patterns");
    const patterns = patternsSelect
        ? Array.from(patternsSelect.selectedOptions).map((option) => option.value)
        : ["twenty_day_high"];

    const spec = {
        patterns: patterns.length ? patterns : ["twenty_day_high"],
        lookback_days: {
            minimum: toInt("breakout_lookback_min", 20),
            maximum: toInt("breakout_lookback_max", 60),
            step: toInt("breakout_lookback_step", 20),
        },
        breakout_buffer_pct: {
            minimum: toPercent("breakout_buffer_min", 0),
            maximum: toPercent("breakout_buffer_max", 1),
            step: toPercent("breakout_buffer_step", 0.5),
        },
        volume_ratio_threshold: {
            minimum: toFloat("breakout_volume_ratio_min", 1),
            maximum: toFloat("breakout_volume_ratio_max", 1.5),
            step: toFloat("breakout_volume_ratio_step", 0.5),
        },
        volume_lookback_days: {
            minimum: toInt("breakout_volume_lookback_min", 20),
            maximum: toInt("breakout_volume_lookback_max", 20),
            step: toInt("breakout_volume_lookback_step", 1),
        },
        max_hold_days: formData.has("breakout_hold_only_infinite")
            ? {
                minimum: 0,
                maximum: 0,
                step: 1,
                include_infinite: true,
                only_infinite: true,
            }
            : {
                minimum: toInt("breakout_hold_min", 10),
                maximum: toInt("breakout_hold_max", 20),
                step: toInt("breakout_hold_step", 10),
                include_infinite: formData.has("breakout_hold_infinite"),
            },
        target_position_pct: {
            minimum: toInt("breakout_target_min", 10),
            maximum: toInt("breakout_target_max", 20),
            step: toInt("breakout_target_step", 10),
        },
        stop_loss_pct: null,
        trailing_stop_pct: null,
        profit_target_pct: null,
        include_no_stop_loss: formData.has("breakout_include_no_stop_loss"),
        include_no_trailing_stop: formData.has("breakout_include_no_trailing_stop"),
        include_no_profit_target: formData.has("breakout_include_no_profit_target"),
        lot_size: toInt("breakout_lot_size", 1),
    };

    if (formData.has("breakout_use_stop_range")) {
        spec.stop_loss_pct = {
            minimum: toPercent("breakout_stop_min", 5),
            maximum: toPercent("breakout_stop_max", 5),
            step: toPercent("breakout_stop_step", 1),
        };
    }

    if (formData.has("breakout_use_trailing_range")) {
        spec.trailing_stop_pct = {
            minimum: toPercent("breakout_trailing_min", 8),
            maximum: toPercent("breakout_trailing_max", 8),
            step: toPercent("breakout_trailing_step", 1),
        };
    }

    if (formData.has("breakout_use_profit_range")) {
        spec.profit_target_pct = {
            minimum: toPercent("breakout_profit_min", 15),
            maximum: toPercent("breakout_profit_max", 15),
            step: toPercent("breakout_profit_step", 1),
        };
    }

    return spec;
}

function updateChartSymbolOptions(symbols) {
    elements.symbolSelect.innerHTML = "";
    symbols.forEach((symbol) => {
        const option = document.createElement("option");
        option.value = symbol;
        option.textContent = symbol;
        elements.symbolSelect.append(option);
    });
    if (symbols.length) {
        const active = symbols.includes(elements.symbolSelect.value) ? elements.symbolSelect.value : symbols[0];
        elements.symbolSelect.value = active;
        elements.chartTitle.textContent = `${active} Candlestick`;
    } else {
        elements.chartTitle.textContent = "Candlestick Chart";
    }
}

function getActiveSymbol() {
    if (!latestData || !Array.isArray(latestData.symbols) || !latestData.symbols.length) {
        return null;
    }
    const candidate = elements.symbolSelect.value;
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
        Plotly.purge(elements.chart);
        Plotly.purge(elements.equityChart);
        elements.paperMetrics.innerHTML = "";
        elements.trainingMetrics.innerHTML = "";
        elements.parameterMetrics.innerHTML = "";
        elements.rankingTableBody.innerHTML = "";
        elements.paperWindow.textContent = "";
        elements.trainingWindow.textContent = "";
        setStatus("Run an optimization to see results.", "info");
        return;
    }

    const symbol = getActiveSymbol();
    if (!symbol) {
        Plotly.purge(elements.chart);
        Plotly.purge(elements.equityChart);
        setStatus("No symbols returned from optimization.", "warning");
        return;
    }

    const result = getSymbolResult(symbol);
    renderCandles(symbol, result);
    renderEquity(result);
    renderOptimization(result);
    updateStatusForSymbol(symbol, result);
}

function renderCandles(symbol, result) {
    if (!result || !Array.isArray(result.candles) || !result.candles.length) {
        Plotly.purge(elements.chart);
        elements.chartTitle.textContent = symbol ? `${symbol} Candlestick (no data)` : "Candlestick Chart";
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
    Plotly.newPlot(elements.chart, traces, layout, config);
    elements.chartTitle.textContent = `${symbol} Candlestick`;
}

function renderEquity(result) {
    if (!result || !Array.isArray(result.equity_curve) || !result.equity_curve.length) {
        Plotly.purge(elements.equityChart);
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

    Plotly.newPlot(elements.equityChart, [trace], layout, { responsive: true, displaylogo: false });
}

function renderOptimization(result) {
    if (!result || !result.optimization) {
        elements.paperMetrics.innerHTML = "";
        elements.trainingMetrics.innerHTML = "";
        elements.parameterMetrics.innerHTML = "";
        elements.rankingTableBody.innerHTML = "";
        elements.paperWindow.textContent = "";
        elements.trainingWindow.textContent = "";
        if (result?.metrics) {
            renderMetricCards(elements.paperMetrics, buildMetricRows(result.metrics));
        }
        return;
    }

    const { paper, training, best_parameters: bestParams, rankings } = result.optimization;
    const strategy = result.strategy || latestData.strategy || getSelectedStrategy();

    renderMetricCards(elements.paperMetrics, buildMetricRows(paper?.metrics || result.metrics));
    renderMetricCards(elements.trainingMetrics, buildMetricRows(training?.metrics));
    renderParameterCards(bestParams, strategy);
    renderRankingTable(rankings || []);

    elements.paperWindow.textContent = `Window: ${formatRange(paper)}`;
    elements.trainingWindow.textContent = `Window: ${formatRange(training)}`;
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

function renderParameterCards(params, strategy) {
    if (!elements.parameterMetrics) {
        return;
    }
    if (!params) {
        elements.parameterMetrics.innerHTML = "<p class=\"metric-empty\">No parameters selected</p>";
        return;
    }

    const entries = Object.entries(params).map(([key, value]) => ({
        label: formatParameterLabel(key),
        value: formatParameterValue(key, value, strategy),
    }));

    elements.parameterMetrics.innerHTML = entries
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
    if (!elements.rankingTableBody) {
        return;
    }
    if (!rows.length) {
        elements.rankingTableBody.innerHTML = "<tr><td colspan=\"4\">No candidate parameters evaluated</td></tr>";
        return;
    }

    elements.rankingTableBody.innerHTML = rows
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

function updateStatusForSymbol(symbol, result) {
    if (!symbol || !result) {
        setStatus(`No optimization output for ${symbol || "selected symbol"}.`, "warning");
        return;
    }
    const optimization = result.optimization || {};
    const trainingRange = formatRange(optimization.training);
    const paperRange = formatRange(optimization.paper);
    const strategy = formatStrategyName(result.strategy || latestData.strategy || getSelectedStrategy());
    let message = `Optimized ${latestData.symbols?.length || 0} symbol(s) with ${strategy}. Showing ${symbol} — Train ${trainingRange}, Paper ${paperRange}.`;
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

function setStatus(message, level = "info") {
    if (!elements.status) {
        return;
    }
    elements.status.textContent = message;
    elements.status.className = `status ${level}`;
}

function formatParameterLabel(key) {
    return key
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function formatParameterValue(key, value, strategy) {
    if (value === null || value === undefined) {
        return "None";
    }
    if (typeof value === "number") {
        if (key.includes("_pct")) {
            return formatPercent(value);
        }
        if (key.includes("_ratio") || key.includes("ratio")) {
            return formatNumber(value);
        }
        if (key.includes("hold_days") && value === 0) {
            return "Infinite";
        }
        if (Number.isInteger(value)) {
            return formatNumber(value, 0);
        }
        return formatNumber(value);
    }
    if (Array.isArray(value)) {
        return value.map((item) => formatParameterValue(key, item, strategy)).join(", ");
    }
    if (typeof value === "string") {
        if (key === "pattern" || key.includes("pattern")) {
            return formatStrategyName(value);
        }
        return value
            .split("_")
            .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
            .join(" ");
    }
    return value;
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

function getSelectedStrategy() {
    if (elements.strategySelect && elements.strategySelect.value) {
        return elements.strategySelect.value;
    }
    if (pageId === "breakout") {
        return STRATEGY_BREAKOUT;
    }
    return STRATEGY_MEAN_REVERSION;
}

function formatStrategyName(value) {
    const normalized = value || "";
    return normalized
        .toString()
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

document.addEventListener("DOMContentLoaded", initialize);
