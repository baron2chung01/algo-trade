const DEFAULT_SYMBOLS = ["AAPL", "MSFT"];
const STRATEGY_MEAN_REVERSION = "mean_reversion";
const STRATEGY_BREAKOUT = "breakout";
const STRATEGY_VCP = "vcp";

const pageId = document.body?.dataset?.page || "";

const elements = {
    form: document.getElementById("control-form"),
    scanForm: document.getElementById("scan-form"),
    status: document.getElementById("status"),
    scanStatus: document.getElementById("scan-status"),
    paperMetrics: document.getElementById("paper-metrics"),
    trainingMetrics: document.getElementById("training-metrics"),
    parameterMetrics: document.getElementById("parameter-metrics"),
    rankingTableBody: document.querySelector("#ranking-table tbody"),
    scanTableBody: document.getElementById("scan-table-body"),
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
    vcpSection: document.getElementById("vcp-params"),
    scanRunButton: document.getElementById("scan-run-button"),
    scanTimeframe: document.getElementById("scan-timeframe"),
    scanMaxCandidates: document.getElementById("scan-max-candidates"),
    scanSymbolsInput: document.getElementById("scan-symbols"),
    scanCriteriaInputs: document.querySelectorAll("[data-scan-criterion]"),
    scanSummaryTimeframe: document.getElementById("scan-summary-timeframe"),
    scanSummaryParams: document.getElementById("scan-summary-params"),
    scanSummarySymbols: document.getElementById("scan-summary-symbols"),
    scanSummaryTimestamp: document.getElementById("scan-summary-timestamp"),
    scanWarningsGroup: document.getElementById("scan-warnings-group"),
    scanWarningsList: document.getElementById("scan-warnings"),
    globalStatus: document.getElementById("global-status"),
    fetchTechButton: document.getElementById("fetch-tech-button"),
    vcpTestForm: document.getElementById("vcp-test-form"),
    vcpTestStatus: document.getElementById("vcp-test-status"),
    vcpTestRunButton: document.getElementById("vcp-test-run"),
    vcpTestSymbols: document.getElementById("vcp-test-symbols"),
    vcpTestTimeframe: document.getElementById("vcp-test-timeframe"),
    vcpTestLookback: document.getElementById("vcp-test-lookback"),
    vcpTestMaxDetections: document.getElementById("vcp-test-max-detections"),
    vcpTestSummaryWindow: document.getElementById("vcp-test-summary-window"),
    vcpTestSummaryDetections: document.getElementById("vcp-test-summary-detections"),
    vcpTestSummaryTimeframe: document.getElementById("vcp-test-summary-timeframe"),
    vcpTestSummaryParameters: document.getElementById("vcp-test-summary-parameters"),
    vcpTestTableBody: document.getElementById("vcp-test-table-body"),
    vcpTestWarningsGroup: document.getElementById("vcp-test-warnings-group"),
    vcpTestWarningsList: document.getElementById("vcp-test-warnings"),
    vcpTestSymbolSelect: document.getElementById("vcp-test-symbol-select"),
    vcpTestDetectionSelect: document.getElementById("vcp-test-detection-select"),
    vcpTestChart: document.getElementById("vcp-test-chart"),
    vcpTestChartTitle: document.getElementById("vcp-test-chart-title"),
    vcpTestVolumeChart: document.getElementById("vcp-test-volume-chart"),
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

const vcpControls = {
    trailingToggle: document.getElementById("vcp_use_trailing_range"),
    trailingInputs: document.querySelectorAll("[data-vcp-trailing-range]"),
    holdInfiniteToggle: document.getElementById("vcp_hold_infinite"),
    holdOnlyInfiniteToggle: document.getElementById("vcp_hold_only_infinite"),
    holdRangeInputs: document.querySelectorAll("[data-vcp-hold-range]"),
    searchStrategy: document.getElementById("vcp_search_strategy"),
    annealingInputs: document.querySelectorAll("#vcp-annealing-settings [data-annealing-input]"),
    annealingSettings: document.getElementById("vcp-annealing-settings"),
};

const SCAN_TABLE_COLUMNS = 10;
const DEFAULT_SCAN_CRITERIA = ["flag_vcp", "minervini", "qullamagie"];
const SCAN_CRITERIA_LABELS = {
    flag_vcp: "Flag/VCP",
    minervini: "Minervini Criteria",
    qullamagie: "Qullamagie Criteria",
};

let latestData = null;
let latestScanResult = null;
let latestVcpTest = null;

function initialize() {
    setupTechDataFetch();

    if (pageId === "vcp-scan") {
        initializeScanPage();
        return;
    }

    if (!elements.form) {
        return;
    }
    setupMeanReversionControls();
    setupBreakoutControls();
    setupVcpControls();
    setupVcpTestSection();
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

    loadAvailableSymbols();
}

function setupTechDataFetch() {
    clearGlobalStatus();

    const button = elements.fetchTechButton;
    if (!button) {
        return;
    }
    if (!button.dataset.originalLabel) {
        button.dataset.originalLabel = button.textContent?.trim() || "Fetch US Liquidity";
    }
    button.addEventListener("click", handleTechDataFetch);
}

async function handleTechDataFetch(event) {
    if (event) {
        event.preventDefault();
    }

    const button = elements.fetchTechButton;
    if (!button || button.disabled) {
        return;
    }

    const originalLabel = button.dataset.originalLabel || button.textContent?.trim() || "Fetch US Liquidity";
    button.dataset.originalLabel = originalLabel;

    button.disabled = true;
    button.textContent = "Fetching…";

    const forceRefresh = !(event?.shiftKey);
    const statusMessage = forceRefresh
        ? "Refreshing liquidity-filtered US universe and downloading Polygon history…"
        : "Downloading Polygon history for cached US liquidity universe…";
    setGlobalStatus(statusMessage, "info");

    try {
        const response = await fetch("/api/vcp/universe/fetch", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ force_refresh_universe: forceRefresh }),
        });

        if (!response.ok) {
            const errorMessage = await readErrorMessage(response);
            let errorWarnings = [];
            try {
                const data = await response.clone().json();
                const detail = typeof data === "object" && data !== null ? data.detail : null;
                if (Array.isArray(detail?.warnings)) {
                    errorWarnings = detail.warnings
                        .map((message) => (typeof message === "string" ? message.trim() : ""))
                        .filter(Boolean);
                }
            } catch (parseError) {
                // ignore JSON parse failures for error detail extraction
            }
            const error = new Error(errorMessage);
            if (errorWarnings.length) {
                error.warnings = errorWarnings;
            }
            throw error;
        }

        const data = await response.json();
        renderTechFetchResult(data);

        if (pageId === "vcp-scan" && elements.scanSymbolsInput && !elements.scanSymbolsInput.value.trim()) {
            prefillScanSymbols();
        }
    } catch (error) {
        console.error("Failed to refresh US liquidity universe", error);
        const warnings = Array.isArray(error?.warnings) ? error.warnings : [];
        let message = error?.message || "Failed to refresh US liquidity universe.";
        if (warnings.length) {
            message = `${message} — ${warnings.join(" | ")}`;
        }
        setGlobalStatus(message, "error");
    } finally {
        button.disabled = false;
        button.textContent = button.dataset.originalLabel || originalLabel;
    }
}

function renderTechFetchResult(result) {
    const totalSymbols = typeof result?.total_symbols === "number" && Number.isFinite(result.total_symbols)
        ? result.total_symbols
        : 0;
    const updatedRows = typeof result?.updated_count === "number" && Number.isFinite(result.updated_count)
        ? result.updated_count
        : 0;
    const updatedMap = result?.updated_symbols && typeof result.updated_symbols === "object"
        ? result.updated_symbols
        : {};
    const updatedSymbols = Object.keys(updatedMap);
    const updatedSymbolCount = updatedSymbols.length;
    const skippedSymbols = Array.isArray(result?.skipped_symbols) ? result.skipped_symbols : [];
    const skippedCount = skippedSymbols.length;
    const warnings = Array.isArray(result?.warnings)
        ? result.warnings.filter((message) => typeof message === "string" && message.trim())
        : [];

    const summaryParts = [];
    if (totalSymbols) {
        summaryParts.push(`Universe size: ${totalSymbols} symbol${totalSymbols === 1 ? "" : "s"}.`);
    }

    if (updatedRows > 0) {
        const previewSymbols = updatedSymbols.slice(0, 5).join(", ");
        const previewSuffix = updatedSymbols.length > 5 ? "…" : "";
        const preview = previewSymbols ? ` (${previewSymbols}${previewSuffix})` : "";
        summaryParts.push(
            `Added ${updatedRows} new daily row${updatedRows === 1 ? "" : "s"} across ${updatedSymbolCount} symbol${updatedSymbolCount === 1 ? "" : "s"}${preview}.`,
        );
    } else {
        summaryParts.push("All tracked symbols are already up to date.");
    }

    if (skippedCount) {
        summaryParts.push(`${skippedCount} symbol${skippedCount === 1 ? "" : "s"} already current.`);
    }

    if (warnings.length) {
        summaryParts.push(`Warnings: ${warnings.join(" | ")}`);
    }

    const level = warnings.length ? "warning" : updatedRows > 0 ? "success" : "info";
    const message = summaryParts.join(" ").trim();
    setGlobalStatus(message || "US liquidity universe refresh complete.", level);
}

function setGlobalStatus(message, level = "info") {
    const status = elements.globalStatus;
    if (!status) {
        return;
    }
    if (!message) {
        status.textContent = "";
        status.hidden = true;
        status.className = "global-status";
        return;
    }
    status.textContent = message;
    status.className = `global-status ${level}`;
    status.hidden = false;
}

function clearGlobalStatus() {
    setGlobalStatus("");
}

function initializeScanPage() {
    if (!elements.scanForm) {
        return;
    }
    if (elements.scanRunButton && !elements.scanRunButton.dataset.originalLabel) {
        elements.scanRunButton.dataset.originalLabel = elements.scanRunButton.textContent?.trim() || "Run Scan";
    }
    elements.scanForm.addEventListener("submit", runScan);
    resetScanResults();
    setScanStatus("Choose a preset and run the scan to discover active VCP breakouts.", "info");
    prefillScanSymbols();
}

function resetScanResults() {
    latestScanResult = null;
    if (elements.scanSummaryTimeframe) {
        elements.scanSummaryTimeframe.textContent = "–";
    }
    if (elements.scanSummaryParams) {
        elements.scanSummaryParams.textContent = "–";
    }
    if (elements.scanSummarySymbols) {
        elements.scanSummarySymbols.textContent = "0";
    }
    if (elements.scanSummaryTimestamp) {
        elements.scanSummaryTimestamp.textContent = "–";
    }
    if (elements.scanTableBody) {
        elements.scanTableBody.innerHTML = `<tr><td colspan="${SCAN_TABLE_COLUMNS}" class="metric-empty">Run the scan to populate candidates.</td></tr>`;
    }
    renderScanWarnings([]);
}

function setScanStatus(message, level = "info") {
    if (!elements.scanStatus) {
        return;
    }
    elements.scanStatus.textContent = message;
    elements.scanStatus.className = `status ${level}`;
}

async function runScan(event) {
    if (event) {
        event.preventDefault();
    }
    if (!elements.scanForm) {
        return;
    }

    const button = elements.scanRunButton;
    if (button) {
        button.disabled = true;
        if (!button.dataset.originalLabel) {
            button.dataset.originalLabel = button.textContent?.trim() || "Run Scan";
        }
        button.textContent = "Scanning…";
    }

    setScanStatus("Scanning for VCP breakouts...", "info");

    const payload = buildScanRequest();

    try {
        const response = await fetch("/api/vcp/scan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorMessage = await readErrorMessage(response);
            throw new Error(errorMessage);
        }

        const data = await response.json();
        renderScanResults(data);
    } catch (error) {
        console.error("Failed to run VCP scan", error);
        setScanStatus(error?.message || "Failed to run scan.", "error");
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = button.dataset.originalLabel || "Run Scan";
        }
    }
}

function buildScanRequest() {
    const timeframe = elements.scanTimeframe?.value || "medium";
    let maxCandidates = parseInt(elements.scanMaxCandidates?.value || "0", 10);
    if (!Number.isFinite(maxCandidates) || maxCandidates < 0) {
        maxCandidates = 0;
    }
    const request = {
        timeframe,
        max_candidates: maxCandidates,
    };
    const symbols = collectScanSymbols();
    if (symbols.length) {
        request.symbols = symbols;
    }
    const criteria = collectScanCriteria();
    if (criteria.length) {
        request.criteria = criteria;
    }
    return request;
}

function collectScanSymbols() {
    const raw = elements.scanSymbolsInput?.value || "";
    return Array.from(new Set(parseSymbolList(raw)));
}

function collectScanCriteria() {
    const inputs = Array.from(elements.scanCriteriaInputs || []);
    if (!inputs.length) {
        return [...DEFAULT_SCAN_CRITERIA];
    }
    const selected = inputs
        .filter((input) => input.checked)
        .map((input) => input.value || input.dataset.value)
        .filter((value) => typeof value === "string" && value.trim());
    if (selected.length) {
        return selected;
    }
    inputs.forEach((input, index) => {
        if (index < DEFAULT_SCAN_CRITERIA.length) {
            input.checked = true;
        }
    });
    return inputs
        .map((input) => input.value || input.dataset.value)
        .filter((value) => typeof value === "string" && value.trim())
        .slice(0, DEFAULT_SCAN_CRITERIA.length);
}

async function prefillScanSymbols() {
    if (!elements.scanSymbolsInput || elements.scanSymbolsInput.value.trim()) {
        return;
    }
    try {
        const response = await fetch("/api/vcp/universe");
        if (!response.ok) {
            return;
        }
        const payload = await response.json();
        if (Array.isArray(payload.symbols) && payload.symbols.length) {
            const formatted = payload.symbols.join(", ");
            elements.scanSymbolsInput.value = formatted;
            elements.scanSymbolsInput.dataset.prefilled = "true";
        }

        const initialWarnings = [];
        if (Array.isArray(payload.missing) && payload.missing.length) {
            initialWarnings.push(
                `Missing cached data for ${payload.missing.length} liquid-universe symbol${payload.missing.length === 1 ? "" : "s"}.`,
            );
        }
        if (Array.isArray(payload.warnings)) {
            payload.warnings.forEach((message) => {
                if (typeof message === "string" && message.trim()) {
                    initialWarnings.push(message.trim());
                }
            });
        }
        if (initialWarnings.length) {
            renderScanWarnings(initialWarnings);
        }
    } catch (error) {
        console.warn("Unable to prefill VCP scan symbols", error);
    }
}

function parseSymbolList(raw) {
    if (typeof raw !== "string" || !raw.trim()) {
        return [];
    }
    return raw
        .split(/[\s,]+/)
        .map((token) => token.trim().toUpperCase())
        .filter(Boolean);
}

function setupVcpTestSection() {
    const form = elements.vcpTestForm;
    if (!form) {
        return;
    }
    if (elements.vcpTestRunButton && !elements.vcpTestRunButton.dataset.originalLabel) {
        elements.vcpTestRunButton.dataset.originalLabel =
            elements.vcpTestRunButton.textContent?.trim() || "Find Patterns";
    }
    form.addEventListener("submit", runVcpTest);
    if (elements.vcpTestSymbolSelect) {
        elements.vcpTestSymbolSelect.addEventListener("change", () => {
            updateVcpTestDetectionOptions();
            renderVcpTestCharts();
        });
    }
    if (elements.vcpTestDetectionSelect) {
        elements.vcpTestDetectionSelect.addEventListener("change", () => {
            renderVcpTestCharts();
        });
    }
    if (elements.vcpTestTableBody) {
        elements.vcpTestTableBody.addEventListener("click", handleVcpTestTableClick);
    }
}

function buildVcpTestRequest() {
    let symbols = parseSymbolList(elements.vcpTestSymbols?.value || "");
    if (!symbols.length && elements.form) {
        try {
            symbols = gatherSymbols(new FormData(elements.form));
        } catch (error) {
            console.warn("Failed to gather optimization symbols for VCP testing fallback", error);
        }
    }
    if (!symbols.length) {
        symbols = DEFAULT_SYMBOLS;
    }
    const timeframe = (elements.vcpTestTimeframe?.value || "medium").toString().trim().toLowerCase();
    let lookback = parseFloat(elements.vcpTestLookback?.value || "3");
    if (!Number.isFinite(lookback) || lookback <= 0) {
        lookback = 3;
    }
    let maxDetections = parseInt(elements.vcpTestMaxDetections?.value || "8", 10);
    if (!Number.isFinite(maxDetections) || maxDetections <= 0) {
        maxDetections = 8;
    }
    return {
        symbols,
        timeframe,
        lookback_years: lookback,
        max_detections: maxDetections,
    };
}

function setVcpTestStatus(message, level = "info") {
    const status = elements.vcpTestStatus;
    if (!status) {
        return;
    }
    status.textContent = message;
    status.className = `status ${level}`;
}

async function runVcpTest(event) {
    if (event) {
        event.preventDefault();
    }
    if (!elements.vcpTestForm) {
        return;
    }

    let payload;
    try {
        payload = buildVcpTestRequest();
    } catch (error) {
        setVcpTestStatus(error?.message || "Unable to build request.", "error");
        return;
    }

    const button = elements.vcpTestRunButton;
    if (button) {
        if (!button.dataset.originalLabel) {
            button.dataset.originalLabel = button.textContent?.trim() || "Find Patterns";
        }
        button.disabled = true;
        button.textContent = "Scanning…";
    }

    setVcpTestStatus("Scanning historical bars for VCP patterns…", "info");

    try {
        const response = await fetch("/api/vcp/testing", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorMessage = await readErrorMessage(response);
            throw new Error(errorMessage);
        }

        const data = await response.json();
        const totalDetections = renderVcpTestResults(data);
        if (totalDetections > 0) {
            const plural = totalDetections === 1 ? "" : "s";
            setVcpTestStatus(`Found ${totalDetections} VCP pattern${plural}.`, "success");
        } else {
            setVcpTestStatus("No VCP patterns detected in the selected window.", "warning");
        }
    } catch (error) {
        console.error("Failed to run VCP testing scan", error);
        setVcpTestStatus(error?.message || "Failed to scan for VCP patterns.", "error");
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = button.dataset.originalLabel || "Find Patterns";
        }
    }
}

function renderVcpTestResults(result) {
    latestVcpTest = result;
    const symbols = getVcpResultSymbols(result);
    updateVcpTestSymbolOptions(symbols);
    const totalDetections = renderVcpTestSummary(result);
    renderVcpTestTable(result);
    renderVcpTestWarnings(result);
    updateVcpTestDetectionOptions();
    renderVcpTestCharts();
    return totalDetections;
}

function getVcpResultSymbols(result) {
    if (!result) {
        return [];
    }
    if (Array.isArray(result.symbols) && result.symbols.length) {
        return result.symbols;
    }
    if (result.results && typeof result.results === "object") {
        return Object.keys(result.results);
    }
    return [];
}

function updateVcpTestSymbolOptions(symbolsOverride) {
    const symbols = Array.isArray(symbolsOverride) ? symbolsOverride : getVcpResultSymbols(latestVcpTest);
    const select = elements.vcpTestSymbolSelect;
    if (!select) {
        return;
    }
    const previous = select.value;
    select.innerHTML = "";
    if (!symbols.length) {
        select.disabled = true;
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No symbols";
        select.append(option);
        return;
    }
    select.disabled = false;
    symbols.forEach((symbol) => {
        const option = document.createElement("option");
        option.value = symbol;
        option.textContent = symbol;
        select.append(option);
    });
    const active = symbols.includes(previous) ? previous : symbols[0];
    select.value = active;
}

function getActiveVcpTestSymbol() {
    const symbols = getVcpResultSymbols(latestVcpTest);
    if (!symbols.length) {
        return null;
    }
    const candidate = elements.vcpTestSymbolSelect?.value;
    if (candidate && symbols.includes(candidate)) {
        return candidate;
    }
    if (elements.vcpTestSymbolSelect) {
        elements.vcpTestSymbolSelect.value = symbols[0];
    }
    return symbols[0];
}

function getVcpTestSymbolResult(symbol) {
    if (!latestVcpTest || !symbol) {
        return null;
    }
    return latestVcpTest.results?.[symbol] || null;
}

function updateVcpTestDetectionOptions(symbolOverride) {
    const select = elements.vcpTestDetectionSelect;
    if (!select) {
        return;
    }
    const symbol = symbolOverride || getActiveVcpTestSymbol();
    if (elements.vcpTestSymbolSelect && symbol) {
        elements.vcpTestSymbolSelect.value = symbol;
    }
    const result = getVcpTestSymbolResult(symbol);
    const detections = Array.isArray(result?.detections) ? result.detections : [];

    select.innerHTML = "";
    if (!detections.length) {
        select.disabled = true;
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "No detections";
        select.append(option);
        return;
    }

    select.disabled = false;
    detections.forEach((detection, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        const label = formatDate(detection.breakout_timestamp);
        const reward = formatNumber(detection.reward_to_risk ?? 0, 2);
        option.textContent = `${label} · ${reward}R`;
        select.append(option);
    });
    select.value = "0";
}

function getActiveVcpTestDetectionIndex(symbolResult) {
    if (!symbolResult || !Array.isArray(symbolResult.detections) || !symbolResult.detections.length) {
        return -1;
    }
    const raw = elements.vcpTestDetectionSelect?.value ?? "0";
    const index = Number.parseInt(raw, 10);
    if (Number.isFinite(index) && index >= 0 && index < symbolResult.detections.length) {
        return index;
    }
    return 0;
}

function renderVcpTestSummary(result) {
    const symbols = getVcpResultSymbols(result);
    let totalDetections = 0;
    let earliest = null;
    let latest = null;
    let parameters = null;

    symbols.forEach((symbol) => {
        const series = result.results?.[symbol];
        if (!series) {
            return;
        }
        if (Array.isArray(series.detections)) {
            totalDetections += series.detections.length;
        }
        if (!parameters && series.parameters) {
            parameters = series.parameters;
        }
        const window = series.analysis_window;
        if (window?.start) {
            const startDate = new Date(window.start);
            if (!Number.isNaN(startDate.getTime())) {
                earliest = earliest === null ? startDate.getTime() : Math.min(earliest, startDate.getTime());
            }
        }
        if (window?.end) {
            const endDate = new Date(window.end);
            if (!Number.isNaN(endDate.getTime())) {
                latest = latest === null ? endDate.getTime() : Math.max(latest, endDate.getTime());
            }
        }
    });

    if (elements.vcpTestSummaryDetections) {
        elements.vcpTestSummaryDetections.textContent = formatNumber(totalDetections, 0);
    }
    if (elements.vcpTestSummaryTimeframe) {
        const timeframe = (result?.timeframe || "").toString().toUpperCase();
        elements.vcpTestSummaryTimeframe.textContent = timeframe || "–";
    }
    if (elements.vcpTestSummaryWindow) {
        if (earliest !== null && latest !== null) {
            elements.vcpTestSummaryWindow.textContent = `${formatDate(new Date(earliest).toISOString())} → ${formatDate(
                new Date(latest).toISOString(),
            )}`;
        } else {
            elements.vcpTestSummaryWindow.textContent = "–";
        }
    }
    if (elements.vcpTestSummaryParameters) {
        elements.vcpTestSummaryParameters.textContent = formatVcpTestParameterSummary(parameters);
    }

    return totalDetections;
}

function formatVcpTestParameterSummary(parameters) {
    if (!parameters) {
        return "–";
    }
    const parts = [];
    if (Number.isFinite(parameters.base_lookback_days)) {
        parts.push(`${parameters.base_lookback_days}d base`);
    }
    if (Number.isFinite(parameters.pivot_lookback_days)) {
        parts.push(`${parameters.pivot_lookback_days}d pivot`);
    }
    if (Number.isFinite(parameters.min_contractions)) {
        parts.push(`≥${parameters.min_contractions} contractions`);
    }
    if (Number.isFinite(parameters.max_contraction_pct)) {
        parts.push(`${formatPercent(parameters.max_contraction_pct, 1)} max drop`);
    }
    if (Number.isFinite(parameters.breakout_volume_ratio)) {
        parts.push(`${formatNumber(parameters.breakout_volume_ratio, 1)}× breakout vol`);
    }
    return parts.length ? parts.join(" • ") : "–";
}

function renderVcpTestTable(result) {
    if (!elements.vcpTestTableBody) {
        return;
    }
    const rows = [];
    const symbols = getVcpResultSymbols(result);
    symbols.forEach((symbol) => {
        const series = result.results?.[symbol];
        if (!series) {
            return;
        }
        const detections = Array.isArray(series.detections) ? series.detections : [];
        detections.forEach((detection, index) => {
            const breakout = formatDateTime(detection.breakout_timestamp);
            const entry = formatCurrency(detection.entry_price);
            const rewardValue = detection.reward_to_risk;
            const reward = rewardValue === null || rewardValue === undefined || Number.isNaN(rewardValue)
                ? "–"
                : `${formatNumber(rewardValue, 2)}R`;
            const baseRange = `${formatDate(detection.base_start)} → ${formatDate(detection.base_end)}`;
            rows.push(
                `<tr data-symbol="${symbol}" data-index="${index}">
                    <td>${symbol}</td>
                    <td>${breakout}</td>
                    <td>${entry}</td>
                    <td>${reward}</td>
                    <td>${baseRange}</td>
                </tr>`,
            );
        });
    });

    if (!rows.length) {
        elements.vcpTestTableBody.innerHTML =
            '<tr><td colspan="5" class="metric-empty">Run the test to populate detections.</td></tr>';
    } else {
        elements.vcpTestTableBody.innerHTML = rows.join("");
    }
    syncVcpTestTableSelection();
}

function renderVcpTestWarnings(result) {
    if (!elements.vcpTestWarningsGroup || !elements.vcpTestWarningsList) {
        return;
    }
    const collected = [];
    if (Array.isArray(result?.warnings)) {
        collected.push(...result.warnings);
    }
    if (Array.isArray(result?.missing)) {
        result.missing.forEach((item) => {
            if (!item || !item.symbol) {
                return;
            }
            const reason = item.reason ? ` (${item.reason})` : "";
            collected.push(`Missing data for ${item.symbol}${reason}`);
        });
    }
    if (result?.results) {
        Object.entries(result.results).forEach(([symbol, series]) => {
            if (Array.isArray(series?.warnings)) {
                series.warnings.forEach((message) => {
                    if (typeof message === "string" && message.trim()) {
                        collected.push(`${symbol}: ${message.trim()}`);
                    }
                });
            }
        });
    }

    if (!collected.length) {
        elements.vcpTestWarningsList.innerHTML = "";
        elements.vcpTestWarningsGroup.hidden = true;
        return;
    }

    const uniqueWarnings = Array.from(new Set(collected));
    elements.vcpTestWarningsList.innerHTML = uniqueWarnings.map((warning) => `<li>${warning}</li>`).join("");
    elements.vcpTestWarningsGroup.hidden = false;
}

function handleVcpTestTableClick(event) {
    const row = event.target?.closest("tr[data-symbol]");
    if (!row) {
        return;
    }
    const symbol = row.dataset.symbol;
    const index = row.dataset.index;
    if (!symbol) {
        return;
    }
    if (elements.vcpTestSymbolSelect) {
        elements.vcpTestSymbolSelect.value = symbol;
    }
    updateVcpTestDetectionOptions(symbol);
    if (elements.vcpTestDetectionSelect && index !== undefined) {
        elements.vcpTestDetectionSelect.value = index;
    }
    renderVcpTestCharts();
}

function renderVcpTestCharts() {
    const symbol = getActiveVcpTestSymbol();
    if (!elements.vcpTestChart || !elements.vcpTestVolumeChart) {
        return;
    }
    if (!latestVcpTest || !symbol) {
        Plotly.purge(elements.vcpTestChart);
        Plotly.purge(elements.vcpTestVolumeChart);
        if (elements.vcpTestChartTitle) {
            elements.vcpTestChartTitle.textContent = "VCP Candlestick";
        }
        syncVcpTestTableSelection();
        return;
    }

    const result = getVcpTestSymbolResult(symbol);
    if (!result || !Array.isArray(result.candles) || !result.candles.length) {
        Plotly.purge(elements.vcpTestChart);
        Plotly.purge(elements.vcpTestVolumeChart);
        if (elements.vcpTestChartTitle) {
            elements.vcpTestChartTitle.textContent = `${symbol} Candlestick (no data)`;
        }
        syncVcpTestTableSelection();
        return;
    }

    const detectionIndex = getActiveVcpTestDetectionIndex(result);
    const detections = Array.isArray(result.detections) ? result.detections : [];
    const detection = detectionIndex >= 0 && detectionIndex < detections.length ? detections[detectionIndex] : null;

    if (elements.vcpTestChartTitle) {
        elements.vcpTestChartTitle.textContent = detection
            ? `${symbol} Candlestick · Breakout ${formatDate(detection.breakout_timestamp)}`
            : `${symbol} Candlestick`;
    }

    const timestamps = result.candles.map((candle) => candle.timestamp);
    const opens = result.candles.map((candle) => candle.open);
    const highs = result.candles.map((candle) => candle.high);
    const lows = result.candles.map((candle) => candle.low);
    const closes = result.candles.map((candle) => candle.close);
    const volumes = result.candles.map((candle) => candle.volume);

    const traces = [
        {
            type: "candlestick",
            name: `${symbol} price`,
            x: timestamps,
            open: opens,
            high: highs,
            low: lows,
            close: closes,
            increasing: { line: { color: "#2ecc71" } },
            decreasing: { line: { color: "#e74c3c" } },
        },
    ];

    const shapes = [];
    const annotations = [];

    if (detection) {
        shapes.push({
            type: "rect",
            xref: "x",
            yref: "paper",
            x0: detection.base_start,
            x1: detection.base_end,
            y0: 0,
            y1: 1,
            fillcolor: "rgba(99, 102, 241, 0.18)",
            line: { width: 0 },
            layer: "below",
        });

        traces.push({
            type: "scatter",
            mode: "markers",
            name: "Breakout",
            x: [detection.breakout_timestamp],
            y: [detection.breakout_price],
            marker: {
                color: "#f59e0b",
                size: 10,
                symbol: "star",
                line: { color: "#0f172a", width: 1 },
            },
        });

        const xExtent = [timestamps[0], timestamps[timestamps.length - 1]];
        traces.push(createLevelTrace("Entry", detection.entry_price, "#22c55e", xExtent));
        traces.push(createLevelTrace("Stop", detection.stop_price, "#ef4444", xExtent, "dash"));
        traces.push(createLevelTrace("Target", detection.target_price, "#3b82f6", xExtent, "dot"));

        annotations.push({
            x: detection.breakout_timestamp,
            y: detection.entry_price,
            text: `${formatNumber(detection.reward_to_risk ?? 0, 2)}R`,
            showarrow: true,
            arrowhead: 4,
            ax: 0,
            ay: -60,
            bgcolor: "rgba(15, 23, 42, 0.85)",
            bordercolor: "#6366f1",
        });
    }

    const chartLayout = {
        margin: { t: 40, r: 10, b: 50, l: 60 },
        xaxis: { title: "Date", rangeslider: { visible: false } },
        yaxis: { title: "Price" },
        legend: { orientation: "h", x: 0, y: 1.1 },
        dragmode: "pan",
        hovermode: "x unified",
    };
    if (shapes.length) {
        chartLayout.shapes = shapes;
    }
    if (annotations.length) {
        chartLayout.annotations = annotations;
    }

    const config = { responsive: true, displaylogo: false };
    Plotly.newPlot(elements.vcpTestChart, traces, chartLayout, config);

    const volumeTrace = {
        type: "bar",
        name: "Volume",
        x: timestamps,
        y: volumes,
        marker: {
            color: detection ? highlightVolumeColors(result.candles, detection) : "rgba(148, 163, 184, 0.7)",
        },
    };
    const volumeLayout = {
        margin: { t: 40, r: 10, b: 50, l: 60 },
        xaxis: { title: "Date" },
        yaxis: { title: "Volume" },
        hovermode: "x unified",
        bargap: 0,
    };
    if (detection) {
        volumeLayout.shapes = [
            {
                type: "rect",
                xref: "x",
                yref: "paper",
                x0: detection.base_start,
                x1: detection.base_end,
                y0: 0,
                y1: 1,
                fillcolor: "rgba(99, 102, 241, 0.18)",
                line: { width: 0 },
                layer: "below",
            },
        ];
    }

    Plotly.newPlot(elements.vcpTestVolumeChart, [volumeTrace], volumeLayout, config);
    syncVcpTestTableSelection();
}

function createLevelTrace(name, value, color, xExtent, dash = "solid") {
    return {
        type: "scatter",
        mode: "lines",
        name,
        x: xExtent,
        y: [value, value],
        line: { color, width: 2, dash },
        hoverinfo: "skip",
        showlegend: true,
    };
}

function highlightVolumeColors(candles, detection) {
    const baseStart = new Date(detection.base_start).getTime();
    const baseEnd = new Date(detection.base_end).getTime();
    return candles.map((candle) => {
        const ts = new Date(candle.timestamp).getTime();
        if (!Number.isNaN(ts) && ts >= baseStart && ts <= baseEnd) {
            return "rgba(99, 102, 241, 0.8)";
        }
        return "rgba(148, 163, 184, 0.6)";
    });
}

function syncVcpTestTableSelection() {
    if (!elements.vcpTestTableBody) {
        return;
    }
    const symbol = getActiveVcpTestSymbol();
    const result = getVcpTestSymbolResult(symbol);
    const index = getActiveVcpTestDetectionIndex(result);
    elements.vcpTestTableBody.querySelectorAll("tr[data-symbol]").forEach((row) => {
        const rowIndex = Number.parseInt(row.dataset.index ?? "-1", 10);
        if (row.dataset.symbol === symbol && rowIndex === index) {
            row.classList.add("is-active");
        } else {
            row.classList.remove("is-active");
        }
    });
}

async function readErrorMessage(response) {
    try {
        const data = await response.clone().json();
        if (typeof data === "string") {
            return data;
        }
        if (typeof data?.detail === "string") {
            return data.detail;
        }
        if (data?.detail && typeof data.detail === "object") {
            if (typeof data.detail.message === "string") {
                return data.detail.message;
            }
            return JSON.stringify(data.detail);
        }
    } catch (error) {
        // ignore JSON parse errors
    }
    try {
        const text = await response.text();
        if (text) {
            return text;
        }
    } catch (error) {
        // ignore text parse errors
    }
    return response.statusText || `Request failed with status ${response.status}`;
}

function renderScanResults(result) {
    latestScanResult = result;
    renderScanSummary(result);
    renderScanCandidates(result?.candidates);
    renderScanWarnings(result?.warnings);
    const candidateCount = Array.isArray(result?.candidates) ? result.candidates.length : 0;
    const symbolsScanned = Number.isFinite(result?.symbols_scanned) ? result.symbols_scanned : 0;
    const message = candidateCount
        ? `Found ${candidateCount} VCP candidate${candidateCount === 1 ? "" : "s"} across ${symbolsScanned} symbol${symbolsScanned === 1 ? "" : "s"}.`
        : `No VCP breakouts found across ${symbolsScanned} symbol${symbolsScanned === 1 ? "" : "s"}.`;
    setScanStatus(message, candidateCount ? "success" : "warning");
}

function renderScanSummary(summary) {
    if (!summary) {
        return;
    }
    if (elements.scanSummaryTimeframe) {
        elements.scanSummaryTimeframe.textContent = formatStrategyName(summary.timeframe) || "–";
    }
    if (elements.scanSummaryParams) {
        elements.scanSummaryParams.textContent = summarizeScanParameters(summary.parameters);
    }
    if (elements.scanSummarySymbols) {
        const count = Number.isFinite(summary.symbols_scanned) ? summary.symbols_scanned : 0;
        elements.scanSummarySymbols.textContent = formatNumber(count, 0);
    }
    if (elements.scanSummaryTimestamp) {
        elements.scanSummaryTimestamp.textContent = formatDateTime(summary.analysis_timestamp);
    }
}

function summarizeScanParameters(parameters) {
    if (!parameters) {
        return "–";
    }
    const criteria = Array.isArray(parameters.criteria) ? parameters.criteria : [];
    if (criteria.length) {
        const labels = criteria
            .map((key) => SCAN_CRITERIA_LABELS[key] || key)
            .filter((label) => typeof label === "string" && label.trim());
        if (labels.length) {
            return labels.join(" + ");
        }
    }
    if (typeof parameters.rule_set === "string" && parameters.rule_set.trim()) {
        return parameters.rule_set;
    }
    return "All criteria";
}

function renderScanCandidates(candidates) {
    if (!elements.scanTableBody) {
        return;
    }
    if (!Array.isArray(candidates) || !candidates.length) {
        elements.scanTableBody.innerHTML = `<tr><td colspan="${SCAN_TABLE_COLUMNS}" class="metric-empty">No candidates matched the scan.</td></tr>`;
        return;
    }

    const rows = candidates
        .map((candidate) => {
            const dollarVolume = formatCurrency(candidate.monthly_dollar_volume);
            const rsPercentile = candidate.rs_percentile === null || candidate.rs_percentile === undefined
                ? "–"
                : formatNumber(candidate.rs_percentile, 1);
            return `
                <tr>
                    <td>${candidate.symbol || ""}</td>
                    <td>${formatCurrency(candidate.close_price)}</td>
                    <td>${dollarVolume}</td>
                    <td>${rsPercentile}</td>
                    <td>${candidate.flag_vcp_pass ? "Yes" : "No"}</td>
                    <td>${candidate.weekly_contraction ? "Yes" : "No"}</td>
                    <td>${candidate.monthly_contraction ? "Yes" : "No"}</td>
                    <td>${candidate.minervini_pass ? "Yes" : "No"}</td>
                    <td>${formatNumber(candidate.qullamagie_score, 2)}</td>
                    <td>${candidate.qullamagie_pass ? "Yes" : "No"}</td>
                </tr>
            `;
        })
        .join("");
    elements.scanTableBody.innerHTML = rows;
}

function renderScanWarnings(warnings) {
    if (!elements.scanWarningsGroup || !elements.scanWarningsList) {
        return;
    }
    if (!Array.isArray(warnings) || !warnings.length) {
        elements.scanWarningsList.innerHTML = "";
        elements.scanWarningsGroup.hidden = true;
        return;
    }
    elements.scanWarningsList.innerHTML = warnings.map((warning) => `<li>${warning}</li>`).join("");
    elements.scanWarningsGroup.hidden = false;
}

function formatDateTime(value) {
    if (!value) {
        return "–";
    }
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) {
        return value.toString();
    }
    return new Intl.DateTimeFormat("en-US", {
        year: "numeric",
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "UTC",
        hour12: false,
    }).format(parsed);
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

function setupVcpControls() {
    const controls = vcpControls;
    createRangeToggle(controls.trailingToggle, controls.trailingInputs);
    applyHoldState(controls.holdOnlyInfiniteToggle, controls.holdInfiniteToggle, controls.holdRangeInputs);
    if (controls.holdOnlyInfiniteToggle) {
        controls.holdOnlyInfiniteToggle.addEventListener("change", () => {
            applyHoldState(controls.holdOnlyInfiniteToggle, controls.holdInfiniteToggle, controls.holdRangeInputs);
        });
    }

    initializeVcpSearchControls();
}

function initializeVcpSearchControls() {
    const controls = vcpControls;
    if (!controls.searchStrategy) {
        return;
    }

    const annealingInputs = Array.from(controls.annealingInputs || []);
    const updateVisibility = () => {
        const strategy = controls.searchStrategy.value;
        const isAnnealing = strategy === "annealing";
        if (controls.annealingSettings) {
            controls.annealingSettings.style.display = isAnnealing ? "grid" : "none";
        }
        annealingInputs.forEach((input) => {
            input.disabled = !isAnnealing;
            if (!isAnnealing) {
                input.dataset.previousValue = input.value;
                const defaultValue = input.dataset.defaultValue;
                if (defaultValue !== undefined) {
                    input.value = defaultValue;
                } else if (input.type === "number") {
                    input.value = "";
                }
            } else if (input.dataset.previousValue !== undefined) {
                const restored = input.dataset.previousValue;
                if (restored !== undefined && restored !== null && restored !== "") {
                    input.value = restored;
                } else if (input.dataset.defaultValue !== undefined) {
                    input.value = input.dataset.defaultValue;
                }
                delete input.dataset.previousValue;
            }
        });
    };

    controls.searchStrategy.addEventListener("change", updateVisibility);
    updateVisibility();
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
    if (elements.vcpSection) {
        elements.vcpSection.style.display = strategy === STRATEGY_VCP ? "" : "none";
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

function ensureStepIncrement(value, increment, label) {
    if (!Number.isFinite(value)) {
        throw new Error(`${label} must be a number.`);
    }
    if (value <= 0) {
        throw new Error(`${label} must be greater than zero.`);
    }
    const scaled = value / increment;
    const rounded = Math.round(scaled * 1e6) / 1e6;
    if (Math.abs(rounded - Math.round(rounded)) > 1e-6) {
        throw new Error(`${label} must be in ${increment.toFixed(1)} increments.`);
    }
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
    } else if (strategy === STRATEGY_BREAKOUT) {
        payload.breakout_spec = buildBreakoutSpec(formData);
    } else if (strategy === STRATEGY_VCP) {
        payload.vcp_spec = buildVcpSpec(formData);
        Object.assign(payload, buildVcpSearchSettings(formData));
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

function buildVcpSpec(formData) {
    const toInt = (name, fallback) => parseIntOr(formData.get(name), fallback);
    const toFloat = (name, fallback) => parseFloatOr(formData.get(name), fallback);
    const toPercent = (name, fallback) => parseFloatOr(formData.get(name), fallback) / 100;

    const bufferStepRaw = parseFloatOr(formData.get("vcp_buffer_step"), 0.2);
    ensureStepIncrement(bufferStepRaw, 0.1, "Breakout buffer step");

    const squeezeStepRaw = parseFloatOr(formData.get("vcp_squeeze_step"), 0.2);
    ensureStepIncrement(squeezeStepRaw, 0.1, "Volume squeeze step");

    const breakoutVolumeStepRaw = parseFloatOr(formData.get("vcp_breakout_volume_step"), 0.3);
    ensureStepIncrement(breakoutVolumeStepRaw, 0.1, "Breakout volume step");

    const stopStepRaw = parseFloatOr(formData.get("vcp_stop_r_step"), 0.2);
    ensureStepIncrement(stopStepRaw, 0.1, "Stop-loss step");

    const targetStepRaw = parseFloatOr(formData.get("vcp_target_r_step"), 0.5);
    ensureStepIncrement(targetStepRaw, 0.1, "Profit target step");

    const trailingStepRaw = parseFloatOr(formData.get("vcp_trailing_r_step"), 0.1);
    ensureStepIncrement(trailingStepRaw, 0.1, "Trailing stop step");

    const holdOnlyInfinite = formData.has("vcp_hold_only_infinite");
    const holdRange = holdOnlyInfinite
        ? {
            minimum: 0,
            maximum: 0,
            step: 1,
            include_infinite: true,
            only_infinite: true,
        }
        : {
            minimum: toInt("vcp_hold_min", 0),
            maximum: toInt("vcp_hold_max", 0),
            step: toInt("vcp_hold_step", 1),
            include_infinite: formData.has("vcp_hold_infinite"),
        };

    const trailingEnabled = formData.has("vcp_use_trailing_range");
    const trailingRange = trailingEnabled
        ? {
            minimum: toFloat("vcp_trailing_r_min", 1.5),
            maximum: toFloat("vcp_trailing_r_max", 1.5),
            step: trailingStepRaw,
        }
        : null;

    const spec = {
        base_lookback_days: {
            minimum: toInt("vcp_base_lookback_min", 45),
            maximum: toInt("vcp_base_lookback_max", 60),
            step: toInt("vcp_base_lookback_step", 15),
        },
        pivot_lookback_days: {
            minimum: toInt("vcp_pivot_lookback_min", 4),
            maximum: toInt("vcp_pivot_lookback_max", 6),
            step: toInt("vcp_pivot_lookback_step", 2),
        },
        min_contractions: {
            minimum: toInt("vcp_min_contractions_min", 3),
            maximum: toInt("vcp_min_contractions_max", 3),
            step: toInt("vcp_min_contractions_step", 1),
        },
        max_contraction_pct: {
            minimum: toPercent("vcp_max_contraction_min", 12),
            maximum: toPercent("vcp_max_contraction_max", 16),
            step: toPercent("vcp_max_contraction_step", 4),
        },
        contraction_decay: {
            minimum: toPercent("vcp_decay_min", 60),
            maximum: toPercent("vcp_decay_max", 80),
            step: toPercent("vcp_decay_step", 20),
        },
        breakout_buffer_pct: {
            minimum: toPercent("vcp_buffer_min", 0.1),
            maximum: toPercent("vcp_buffer_max", 0.3),
            step: bufferStepRaw / 100,
        },
        volume_squeeze_ratio: {
            minimum: toFloat("vcp_squeeze_min", 0.65),
            maximum: toFloat("vcp_squeeze_max", 0.85),
            step: squeezeStepRaw,
        },
        breakout_volume_ratio: {
            minimum: toFloat("vcp_breakout_volume_min", 1.8),
            maximum: toFloat("vcp_breakout_volume_max", 2.1),
            step: breakoutVolumeStepRaw,
        },
        volume_lookback_days: {
            minimum: toInt("vcp_volume_lookback_min", 18),
            maximum: toInt("vcp_volume_lookback_max", 24),
            step: toInt("vcp_volume_lookback_step", 6),
        },
        trend_ma_period: {
            minimum: toInt("vcp_trend_ma_min", 45),
            maximum: toInt("vcp_trend_ma_max", 60),
            step: toInt("vcp_trend_ma_step", 15),
        },
        stop_loss_r_multiple: {
            minimum: toFloat("vcp_stop_r_min", 0.9),
            maximum: toFloat("vcp_stop_r_max", 1.1),
            step: stopStepRaw,
        },
        profit_target_r_multiple: {
            minimum: toFloat("vcp_target_r_min", 2.0),
            maximum: toFloat("vcp_target_r_max", 2.5),
            step: targetStepRaw,
        },
        trailing_stop_r_multiple: trailingRange,
        include_no_trailing_stop: formData.has("vcp_include_no_trailing"),
        max_hold_days: holdRange,
        target_position_pct: {
            minimum: toInt("vcp_target_pct_min", 15),
            maximum: toInt("vcp_target_pct_max", 15),
            step: toInt("vcp_target_pct_step", 1),
        },
        lot_size: toInt("vcp_lot_size", 1),
        cash_reserve_pct: Math.max(0, Math.min(toPercent("vcp_cash_reserve", 10), 0.95)),
    };

    return spec;
}

function buildVcpSearchSettings(formData) {
    const rawStrategy = (formData.get("vcp_search_strategy") || "grid").toString().trim().toLowerCase();
    const iterations = Math.max(1, parseIntOr(formData.get("vcp_search_iterations"), 150));

    if (rawStrategy !== "annealing") {
        // Only override defaults for annealing; grid search parameters are ignored by the backend.
        return {};
    }

    const initialTemp = Math.max(0.0001, parseFloatOr(formData.get("vcp_initial_temperature"), 1.0));
    const coolingRate = parseFloatOr(formData.get("vcp_cooling_rate"), 0.95);
    const seedValue = formData.get("vcp_random_seed");
    const hasSeed = typeof seedValue === "string" && seedValue.trim() !== "";
    const seed = hasSeed ? parseIntOr(seedValue, 0) : null;

    const settings = {
        vcp_search_strategy: "annealing",
        vcp_search_iterations: iterations,
        vcp_initial_temperature: initialTemp,
        vcp_cooling_rate: Number.isFinite(coolingRate) && coolingRate > 0 && coolingRate < 1 ? coolingRate : 0.95,
    };

    if (hasSeed && Number.isFinite(seed) && seed >= 0) {
        settings.vcp_random_seed = seed;
    }

    return settings;
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
    const annotations = Array.isArray(result.annotations) ? result.annotations : [];

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

    const annotationTraces = buildAnnotationTraces(annotations);
    const shapes = buildAnnotationShapes(annotations, candles);

    const layout = {
        margin: { t: 40, r: 10, b: 50, l: 60 },
        xaxis: { title: "Date", rangeslider: { visible: false } },
        yaxis: { title: "Price" },
        legend: { orientation: "h", x: 0, y: 1.1 },
        dragmode: "pan",
        hovermode: "x unified",
    };
    if (shapes.length) {
        layout.shapes = shapes;
    }

    const config = { responsive: true, displaylogo: false };
    const traces = [traceCandles];
    if (buys.length) {
        traces.push(traceBuys);
    }
    annotationTraces.forEach((trace) => {
        if (trace.x.length) {
            traces.push(trace);
        }
    });
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

function formatPercent(value, digits = 2) {
    if (value === undefined || value === null || Number.isNaN(value)) {
        return "–";
    }
    const formatter = new Intl.NumberFormat("en-US", {
        style: "percent",
        maximumFractionDigits: digits,
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
    if (pageId === "vcp") {
        return STRATEGY_VCP;
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

function buildAnnotationTraces(annotations) {
    const entryTrace = {
        type: "scatter",
        mode: "markers",
        name: "Plan Entry",
        x: [],
        y: [],
        marker: { color: "#f4d03f", size: 9, symbol: "diamond", line: { width: 1, color: "#1f2a44" } },
        hovertemplate: "Entry %{y:.2f}<extra></extra>",
    };
    const stopTrace = {
        type: "scatter",
        mode: "markers",
        name: "Plan Stop",
        x: [],
        y: [],
        marker: { color: "#e74c3c", size: 8, symbol: "x", line: { width: 1, color: "#1f2a44" } },
        hovertemplate: "Stop %{y:.2f}<extra></extra>",
    };
    const targetTrace = {
        type: "scatter",
        mode: "markers",
        name: "Plan Target",
        x: [],
        y: [],
        marker: { color: "#27ae60", size: 8, symbol: "triangle-up", line: { width: 1, color: "#1f2a44" } },
        hovertemplate: "Target %{y:.2f}<extra></extra>",
    };

    annotations.forEach((note) => {
        const ts = note.timestamp;
        if (!ts) {
            return;
        }
        if (note.entry !== null && note.entry !== undefined) {
            entryTrace.x.push(ts);
            entryTrace.y.push(note.entry);
        }
        if (note.stop !== null && note.stop !== undefined) {
            stopTrace.x.push(ts);
            stopTrace.y.push(note.stop);
        }
        if (note.target !== null && note.target !== undefined) {
            targetTrace.x.push(ts);
            targetTrace.y.push(note.target);
        }
    });

    return [entryTrace, stopTrace, targetTrace];
}

function buildAnnotationShapes(annotations, candles) {
    if (!Array.isArray(annotations) || !annotations.length || !Array.isArray(candles) || !candles.length) {
        return [];
    }
    const lastTimestamp = candles[candles.length - 1]?.timestamp;
    const shapes = [];

    annotations.forEach((note) => {
        const ts = note.timestamp;
        if (!ts) {
            return;
        }
        const x1 = lastTimestamp && lastTimestamp > ts ? lastTimestamp : ts;
        if (note.entry !== null && note.entry !== undefined) {
            shapes.push(
                horizontalShape(ts, x1, note.entry, "#f4d03f", "solid"),
            );
        }
        if (note.stop !== null && note.stop !== undefined) {
            shapes.push(
                horizontalShape(ts, x1, note.stop, "#e74c3c", "dot"),
            );
        }
        if (note.target !== null && note.target !== undefined) {
            shapes.push(
                horizontalShape(ts, x1, note.target, "#27ae60", "dot"),
            );
        }
        if (note.resistance !== null && note.resistance !== undefined) {
            shapes.push(
                horizontalShape(ts, x1, note.resistance, "#8e44ad", "dash"),
            );
        }
        if (note.base_low !== null && note.base_low !== undefined) {
            shapes.push(
                horizontalShape(ts, x1, note.base_low, "#2c3e50", "dash"),
            );
        }
    });

    return shapes;
}

function horizontalShape(x0, x1, y, color, dash) {
    return {
        type: "line",
        x0,
        x1,
        y0: y,
        y1: y,
        xref: "x",
        yref: "y",
        line: {
            color,
            width: 1.5,
            dash,
        },
        opacity: 0.85,
    };
}

document.addEventListener("DOMContentLoaded", initialize);
