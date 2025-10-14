const DEFAULT_SYMBOLS = ["AAPL", "MSFT"];
const STRATEGY_MEAN_REVERSION = "mean_reversion";
const STRATEGY_BREAKOUT = "breakout";
const STRATEGY_VCP = "vcp";
const STRATEGY_MOMENTUM = "momentum";
const MOMENTUM_MODE_BACKTEST = "backtest";
const MOMENTUM_MODE_OPTIMIZE = "optimize";
const MOMENTUM_PAPER_DEFAULT_PARAMETERS = {
    lookback_days: 126,
    skip_days: 0,
    rebalance_days: 5,
    max_positions: 10,
    lot_size: 1,
    cash_reserve_pct: 0.05,
    min_momentum: 0.04,
    volatility_window: 20,
    volatility_exponent: 1.25,
};

const MOMENTUM_LIVE_STATUS_POLL_MS = 5000;
const MOMENTUM_LIVE_HISTORY_POLL_MS = 12000;
const MOMENTUM_LIVE_MIN_INTERVAL_MINUTES = 1;
const MOMENTUM_LIVE_MAX_INTERVAL_MINUTES = 24 * 60;

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
    useNasdaqButton: document.getElementById("use-nasdaq-universe"),
    useSnp100Button: document.getElementById("use-snp100-universe"),
    universeImportInput: document.getElementById("universe-import"),
    universeMessage: document.getElementById("universe-message"),
    strategySelect: document.getElementById("strategy"),
    meanReversionSection: document.getElementById("mean-reversion-params"),
    breakoutSection: document.getElementById("breakout-params"),
    vcpSection: document.getElementById("vcp-params"),
    scanRunButton: document.getElementById("scan-run-button"),
    scanExportButton: document.getElementById("scan-export-button"),
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

const momentumElements = {
    form: document.getElementById("momentum-form"),
    status: document.getElementById("momentum-status"),
    runButton: document.getElementById("momentum-run"),
    parameterList: document.getElementById("momentum-parameter-list"),
    parameterTemplate: document.getElementById("momentum-parameter-template"),
    addParameterButton: document.getElementById("add-parameter-set"),
    paperMetrics: document.getElementById("momentum-paper-metrics"),
    trainingMetrics: document.getElementById("momentum-training-metrics"),
    paperWindow: document.getElementById("momentum-paper-window"),
    trainingWindow: document.getElementById("momentum-training-window"),
    rankingTableBody: document.querySelector("#momentum-ranking-table tbody"),
    warningContainer: document.getElementById("momentum-warnings"),
    warningList: document.getElementById("momentum-warning-list"),
    resultSelect: document.getElementById("momentum-result-select"),
    trainingChart: document.getElementById("momentum-training-chart"),
    paperChart: document.getElementById("momentum-paper-chart"),
    tradeTableBody: document.querySelector("#momentum-trade-table tbody"),
    tradesSection: document.getElementById("momentum-trades-section"),
    modeRadios: document.querySelectorAll("input[name='momentum_mode']"),
    modeMessage: document.getElementById("momentum-mode-message"),
    optimizeSection: document.getElementById("momentum-optimize-section"),
    parameterSection: document.getElementById("momentum-parameter-section"),
};

const momentumPaperElements = {
    form: document.getElementById("momentum-paper-form"),
    status: document.getElementById("paper-status"),
    runButton: document.getElementById("paper-run"),
    symbolsInput: document.getElementById("paper-symbols"),
    autoFetchToggle: document.getElementById("paper-auto-fetch"),
    executeToggle: document.getElementById("paper-execute-orders"),
    initialCashInput: document.getElementById("paper-initial-cash"),
    trainingStartInput: document.getElementById("paper-training-start"),
    trainingEndInput: document.getElementById("paper-training-end"),
    paperStartInput: document.getElementById("paper-paper-start"),
    paperEndInput: document.getElementById("paper-paper-end"),
    parameterGrid: document.getElementById("paper-parameter-grid"),
    metrics: document.getElementById("paper-metrics"),
    trainingMetrics: document.getElementById("paper-training-metrics"),
    paperWindowLabel: document.getElementById("paper-window-label"),
    trainingWindowLabel: document.getElementById("paper-training-window-label"),
    tradeTableBody: document.querySelector("#paper-trade-table tbody"),
    actionsGroup: document.getElementById("paper-actions-group"),
    actionsList: document.getElementById("paper-action-list"),
    warningsGroup: document.getElementById("paper-warnings-group"),
    warningsList: document.getElementById("paper-warning-list"),
    trainingChart: document.getElementById("paper-training-chart"),
    paperChart: document.getElementById("paper-paper-chart"),
};

const momentumLiveElements = {
    form: document.getElementById("momentum-live-form"),
    status: document.getElementById("live-status"),
    startButton: document.getElementById("live-start"),
    stopButton: document.getElementById("live-stop"),
    intervalInput: document.getElementById("live-interval"),
    symbolsInput: document.getElementById("live-symbols"),
    autoFetchToggle: document.getElementById("live-auto-fetch"),
    executeToggle: document.getElementById("live-execute-orders"),
    parameterGrid: document.getElementById("live-parameter-grid"),
    statusBadge: document.getElementById("live-status-badge"),
    iterations: document.getElementById("live-iterations"),
    lastRun: document.getElementById("live-last-run"),
    nextRun: document.getElementById("live-next-run"),
    configCash: document.getElementById("live-config-cash"),
    historyTableBody: document.querySelector("#live-history-table tbody"),
    actionsGroup: document.getElementById("live-actions-group"),
    actionsList: document.getElementById("live-actions"),
    warningsGroup: document.getElementById("live-warnings-group"),
    warningsList: document.getElementById("live-warnings"),
    tradeDate: document.getElementById("live-trade-date"),
    tradeTableBody: document.querySelector("#live-trade-table tbody"),
    portfolioGroup: document.getElementById("live-portfolio-group"),
    portfolioCash: document.getElementById("live-portfolio-cash"),
    portfolioPositionCount: document.getElementById("live-portfolio-position-count"),
    portfolioTableBody: document.getElementById("live-portfolio-positions-body"),
    tradeHistoryTableBody: document.querySelector("#live-trade-history-table tbody"),
    historyResetButton: document.getElementById("live-history-reset"),
    equityChart: document.getElementById("live-equity-chart"),
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

const SCAN_TABLE_COLUMNS = 11;
const DEFAULT_SCAN_CRITERIA = ["liquidity", "uptrend_breakout", "higher_lows", "volume_contraction"];
const SCAN_CRITERIA_LABELS = {
    liquidity: "Liquidity Filter",
    uptrend_breakout: "Uptrend Nearing Breakout",
    higher_lows: "Higher Lows",
    volume_contraction: "Volume Contracting",
};

let latestData = null;
let latestScanResult = null;
let latestVcpTest = null;
let latestMomentum = null;
let latestMomentumPaper = null;
let latestMomentumLive = null;
let momentumLiveStatusTimer = null;
let momentumLiveHistoryTimer = null;
let currentMomentumMode = MOMENTUM_MODE_BACKTEST;
let currentUniverseSource = "cache";
let currentUniverseSymbols = [...DEFAULT_SYMBOLS];

function initialize() {
    setupTechDataFetch();

    if (pageId === "vcp-scan") {
        initializeScanPage();
        return;
    }

    if (pageId === "momentum") {
        initializeMomentumPage();
        return;
    }

    if (pageId === "momentum-paper") {
        initializeMomentumPaperPage();
        return;
    }

    if (pageId === "momentum-live") {
        initializeMomentumLivePage();
        return;
    }

    if (!elements.form) {
        return;
    }
    setupMeanReversionControls();
    setupBreakoutControls();
    setupVcpControls();
    setupVcpTestSection();
    setupUniverseControls();
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
    if (elements.scanExportButton && !elements.scanExportButton.dataset.originalLabel) {
        elements.scanExportButton.dataset.originalLabel =
            elements.scanExportButton.textContent?.trim() || "Export to IBKR CSV";
    }
    if (elements.scanExportButton) {
        elements.scanExportButton.addEventListener("click", handleScanExport);
    }
    elements.scanForm.addEventListener("submit", runScan);
    resetScanResults();
    setScanStatus("Choose a preset and run the scan to discover active VCP breakouts.", "info");
    prefillScanSymbols();
}

function initializeMomentumPage() {
    if (!momentumElements.form) {
        return;
    }

    setupUniverseControls();
    loadAvailableSymbols();
    prefillMomentumDates();
    setMomentumStatus("Configure the experiment window and run to evaluate momentum rotations.", "info");

    if (momentumElements.runButton && !momentumElements.runButton.dataset.backtestLabel) {
        const defaultLabel = momentumElements.runButton.textContent?.trim() || "Run Momentum Backtest";
        momentumElements.runButton.dataset.backtestLabel = defaultLabel;
        momentumElements.runButton.dataset.optimizeLabel =
            momentumElements.runButton.dataset.optimizeLabel || "Optimize Momentum Parameters";
    }

    momentumElements.form.addEventListener("submit", runMomentumExperiment);

    if (momentumElements.addParameterButton) {
        momentumElements.addParameterButton.addEventListener("click", () => {
            addMomentumParameterSet();
        });
    }

    if (momentumElements.resultSelect) {
        momentumElements.resultSelect.addEventListener("change", (event) => {
            renderMomentumSelection(event.target.value);
        });
    }

    if (momentumElements.modeRadios && momentumElements.modeRadios.length) {
        momentumElements.modeRadios.forEach((radio) => {
            radio.addEventListener("change", () => {
                applyMomentumMode(readMomentumMode());
            });
        });
    }

    applyMomentumMode(readMomentumMode());
}

function initializeMomentumPaperPage() {
    if (!momentumPaperElements.form) {
        return;
    }

    populatePaperDefaults();
    momentumPaperElements.form.addEventListener("submit", runMomentumPaperTrade);
    setPaperStatus("Configure parameters and run to simulate IBKR paper trading signals.", "info");
}

function initializeMomentumLivePage() {
    if (!momentumLiveElements.form) {
        return;
    }

    momentumLiveElements.form.addEventListener("submit", startMomentumLiveTrading);
    if (momentumLiveElements.stopButton) {
        momentumLiveElements.stopButton.addEventListener("click", stopMomentumLiveTrading);
    }
    if (momentumLiveElements.historyResetButton) {
        momentumLiveElements.historyResetButton.addEventListener("click", resetMomentumLiveHistory);
    }

    disableLiveStartButton(false);
    disableLiveStopButton(true);

    setLiveStatus("Configure the live daemon and start to begin scheduled IBKR paper trades.", "info");

    refreshMomentumLiveStatus();
    refreshMomentumLiveHistory();
    startMomentumLivePolling();

    window.addEventListener("beforeunload", stopMomentumLivePolling);
}

function populatePaperDefaults() {
    prefillPaperDates();
    if (!momentumPaperElements.parameterGrid) {
        return;
    }
    const inputs = momentumPaperElements.parameterGrid.querySelectorAll("input[name]");
    inputs.forEach((input) => {
        const key = input.name;
        if (Object.prototype.hasOwnProperty.call(MOMENTUM_PAPER_DEFAULT_PARAMETERS, key)) {
            const value = MOMENTUM_PAPER_DEFAULT_PARAMETERS[key];
            if (input.name === "cash_reserve_pct") {
                input.value = Number((value * 100).toFixed(2));
            } else {
                input.value = value;
            }
        }
    });
}

function prefillPaperDates() {
    const { trainingStartInput, trainingEndInput, paperStartInput, paperEndInput } = momentumPaperElements;
    if (!trainingStartInput || !trainingEndInput || !paperStartInput || !paperEndInput) {
        return;
    }
    if (trainingStartInput.value && trainingEndInput.value && paperStartInput.value && paperEndInput.value) {
        return;
    }

    const today = new Date();
    const paperEnd = today;
    const paperStart = new Date(paperEnd);
    paperStart.setDate(paperStart.getDate() - 365);

    const trainingEnd = new Date(paperStart);
    trainingEnd.setDate(trainingEnd.getDate() - 1);
    const trainingStart = new Date(trainingEnd);
    trainingStart.setDate(trainingStart.getDate() - 730);

    trainingStartInput.valueAsDate = trainingStart;
    trainingEndInput.valueAsDate = trainingEnd;
    paperStartInput.valueAsDate = paperStart;
    paperEndInput.valueAsDate = paperEnd;
}

function prefillMomentumDates() {
    const trainingStartInput = document.getElementById("training-start");
    const trainingEndInput = document.getElementById("training-end");
    const paperStartInput = document.getElementById("paper-start");
    const paperEndInput = document.getElementById("paper-end");

    if (!trainingStartInput || !trainingEndInput || !paperStartInput || !paperEndInput) {
        return;
    }

    if (trainingStartInput.value && trainingEndInput.value && paperStartInput.value && paperEndInput.value) {
        return;
    }

    const today = new Date();
    const paperEnd = today;
    const paperStart = new Date(paperEnd);
    paperStart.setDate(paperStart.getDate() - 365);

    const trainingEnd = new Date(paperStart);
    trainingEnd.setDate(trainingEnd.getDate() - 1);

    const trainingStart = new Date(trainingEnd);
    trainingStart.setDate(trainingStart.getDate() - 730);

    if (!trainingStartInput.value) {
        trainingStartInput.value = formatDateForInput(trainingStart);
    }
    if (!trainingEndInput.value) {
        trainingEndInput.value = formatDateForInput(trainingEnd);
    }
    if (!paperStartInput.value) {
        paperStartInput.value = formatDateForInput(paperStart);
    }
    if (!paperEndInput.value) {
        paperEndInput.value = formatDateForInput(paperEnd);
    }
}

function formatDateForInput(date) {
    if (!(date instanceof Date) || Number.isNaN(date.getTime())) {
        return "";
    }
    return date.toISOString().split("T")[0];
}

function readMomentumMode(formData = null) {
    if (formData instanceof FormData) {
        const value = formData.get("momentum_mode");
        return value === MOMENTUM_MODE_OPTIMIZE ? MOMENTUM_MODE_OPTIMIZE : MOMENTUM_MODE_BACKTEST;
    }
    const radios = momentumElements.modeRadios ? Array.from(momentumElements.modeRadios) : [];
    const checked = radios.find((radio) => radio.checked);
    return checked?.value === MOMENTUM_MODE_OPTIMIZE ? MOMENTUM_MODE_OPTIMIZE : MOMENTUM_MODE_BACKTEST;
}

function updateMomentumRunButtonLabel(mode) {
    if (!momentumElements.runButton) {
        return;
    }
    const backtestLabel = momentumElements.runButton.dataset.backtestLabel || "Run Momentum Backtest";
    const optimizeLabel = momentumElements.runButton.dataset.optimizeLabel || "Optimize Momentum Parameters";
    momentumElements.runButton.textContent = mode === MOMENTUM_MODE_OPTIMIZE ? optimizeLabel : backtestLabel;
}

function updateMomentumModeStatus(mode) {
    if (!momentumElements.modeMessage) {
        return;
    }
    if (mode === MOMENTUM_MODE_OPTIMIZE) {
        momentumElements.modeMessage.textContent =
            "Optimizer samples smart ranges across the S&P 100 (or your overrides) and surfaces the top-performing configuration.";
    } else {
        momentumElements.modeMessage.textContent =
            "Define one or more parameter sets to backtest directly against your selected universe.";
    }
}

function applyMomentumMode(mode) {
    currentMomentumMode = mode === MOMENTUM_MODE_OPTIMIZE ? MOMENTUM_MODE_OPTIMIZE : MOMENTUM_MODE_BACKTEST;
    const isOptimize = currentMomentumMode === MOMENTUM_MODE_OPTIMIZE;

    toggleMomentumSection(momentumElements.optimizeSection, isOptimize);
    toggleMomentumSection(momentumElements.parameterSection, !isOptimize);
    if (momentumElements.addParameterButton) {
        momentumElements.addParameterButton.hidden = isOptimize;
    }

    if (!isOptimize) {
        ensureMomentumParameterSet();
    }

    updateMomentumRunButtonLabel(currentMomentumMode);
    updateMomentumModeStatus(currentMomentumMode);

    if (currentMomentumMode === MOMENTUM_MODE_OPTIMIZE) {
        setMomentumStatus("Ready to optimize S&P 100 momentum parameters.", "info");
    } else {
        setMomentumStatus("Configure parameter sets and run to evaluate momentum rotations.", "info");
    }
}

function toggleMomentumSection(section, shouldShow) {
    if (!section) {
        return;
    }
    section.hidden = !shouldShow;
    section.classList.toggle("is-hidden", !shouldShow);
}

function ensureMomentumParameterSet() {
    if (!momentumElements.parameterList) {
        return;
    }
    const existing = momentumElements.parameterList.querySelectorAll("[data-parameter-card]");
    if (!existing.length) {
        addMomentumParameterSet();
    } else {
        updateMomentumParameterIndexes();
    }
}

function addMomentumParameterSet(prefill = null) {
    if (!momentumElements.parameterTemplate || !momentumElements.parameterList) {
        return null;
    }
    const fragment = momentumElements.parameterTemplate.content.cloneNode(true);
    const card = fragment.querySelector("[data-parameter-card]");
    if (!card) {
        return null;
    }

    card.dataset.parameterId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const inputs = card.querySelectorAll("[data-field]");
    inputs.forEach((input) => {
        const field = input.dataset.field;
        if (prefill && Object.prototype.hasOwnProperty.call(prefill, field)) {
            input.value = prefill[field];
        }
    });

    const removeButton = card.querySelector("[data-remove-parameter]");
    if (removeButton) {
        removeButton.addEventListener("click", () => {
            card.remove();
            updateMomentumParameterIndexes();
        });
    }

    momentumElements.parameterList.append(card);
    updateMomentumParameterIndexes();
    return card;
}

function updateMomentumParameterIndexes() {
    if (!momentumElements.parameterList) {
        return;
    }
    const cards = Array.from(momentumElements.parameterList.querySelectorAll("[data-parameter-card]"));
    cards.forEach((card, index) => {
        const indexLabel = card.querySelector("[data-parameter-index]");
        if (indexLabel) {
            indexLabel.textContent = index + 1;
        }
        const removeButton = card.querySelector("[data-remove-parameter]");
        if (removeButton) {
            removeButton.hidden = cards.length <= 1;
        }
    });
}

function collectMomentumParameters() {
    if (!momentumElements.parameterList) {
        return [];
    }
    const cards = Array.from(momentumElements.parameterList.querySelectorAll("[data-parameter-card]"));
    if (!cards.length) {
        addMomentumParameterSet();
        return collectMomentumParameters();
    }

    const labels = {
        lookback_days: "Lookback days",
        skip_days: "Skip days",
        rebalance_days: "Rebalance days",
        max_positions: "Max positions",
        lot_size: "Lot size",
        cash_reserve_pct: "Cash reserve",
        min_momentum: "Minimum momentum",
        volatility_window: "Volatility window",
        volatility_exponent: "Volatility exponent",
    };

    const results = cards.map((card) => {
        const getInput = (name) => card.querySelector(`[data-field="${name}"]`);
        const getLabel = (name) => labels[name] || name;

        const readInt = (name, fallback, min = Number.NEGATIVE_INFINITY) => {
            const input = getInput(name);
            const value = parseIntOr(input?.value ?? "", fallback);
            if (!Number.isFinite(value)) {
                throw new Error(`${getLabel(name)} must be a number.`);
            }
            if (value < min) {
                throw new Error(`${getLabel(name)} must be at least ${min}.`);
            }
            return value;
        };

        const readFloat = (name, fallback, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY) => {
            const input = getInput(name);
            const value = parseFloatOr(input?.value ?? "", fallback);
            if (!Number.isFinite(value)) {
                throw new Error(`${getLabel(name)} must be a number.`);
            }
            if (value < min) {
                throw new Error(`${getLabel(name)} must be at least ${min}.`);
            }
            if (value > max) {
                throw new Error(`${getLabel(name)} must be at most ${max}.`);
            }
            return value;
        };

        const lookback = readInt("lookback_days", 126, 1);
        const skip = readInt("skip_days", 21, 0);
        const rebalance = readInt("rebalance_days", 21, 1);
        const maxPositions = readInt("max_positions", 5, 1);
        const lotSize = readInt("lot_size", 1, 1);
        const reservePercent = readFloat("cash_reserve_pct", 5, 0, 95);
        const minMomentum = readFloat("min_momentum", 0);
        const volatilityWindow = readInt("volatility_window", 20, 1);
        const volatilityExponent = readFloat("volatility_exponent", 1, 0);

        const cashReservePct = Math.max(0, Math.min(reservePercent / 100, 0.95));

        return {
            lookback_days: lookback,
            skip_days: skip,
            rebalance_days: rebalance,
            max_positions: maxPositions,
            lot_size: lotSize,
            cash_reserve_pct: Number(cashReservePct.toFixed(4)),
            min_momentum: minMomentum,
            volatility_window: volatilityWindow,
            volatility_exponent: Number(volatilityExponent.toFixed(4)),
        };
    });

    return results;
}

function readMomentumBaseConfig(formData) {
    const initialCash = parseFloatOr(formData.get("initial_cash"), 100000);
    if (!Number.isFinite(initialCash) || initialCash <= 0) {
        throw new Error("Initial cash must be a positive number.");
    }

    const trainingStartRaw = formData.get("training_start");
    const trainingEndRaw = formData.get("training_end");
    const paperStartRaw = formData.get("paper_start");
    const paperEndRaw = formData.get("paper_end");

    if (!trainingStartRaw || !trainingEndRaw || !paperStartRaw || !paperEndRaw) {
        throw new Error("Please provide complete training and paper windows.");
    }

    const trainingStart = new Date(trainingStartRaw);
    const trainingEnd = new Date(trainingEndRaw);
    const paperStart = new Date(paperStartRaw);
    const paperEnd = new Date(paperEndRaw);

    if (Number.isNaN(trainingStart.getTime()) || Number.isNaN(trainingEnd.getTime())) {
        throw new Error("Training window contains invalid dates.");
    }

    if (Number.isNaN(paperStart.getTime()) || Number.isNaN(paperEnd.getTime())) {
        throw new Error("Paper window contains invalid dates.");
    }

    if (trainingStart >= trainingEnd) {
        throw new Error("Training start must be before training end.");
    }

    if (paperStart >= paperEnd) {
        throw new Error("Paper start must be before paper end.");
    }

    if (trainingEnd >= paperStart) {
        throw new Error("Training window must end before the paper window starts.");
    }

    const storePath = (formData.get("store_path") || "").toString().trim();

    return {
        initialCash,
        trainingStart,
        trainingEnd,
        paperStart,
        paperEnd,
        autoFetch: formData.has("auto_fetch"),
        storePath,
    };
}

function buildMomentumBacktestRequest(formData) {
    const base = readMomentumBaseConfig(formData);
    const symbols = gatherSymbols(formData);
    if (!symbols.length) {
        throw new Error("Please select at least one symbol.");
    }

    const parameters = collectMomentumParameters();
    if (!parameters.length) {
        throw new Error("At least one parameter set is required.");
    }

    const payload = {
        symbols,
        initial_cash: base.initialCash,
        training_window: [formatDateForInput(base.trainingStart), formatDateForInput(base.trainingEnd)],
        paper_window: [formatDateForInput(base.paperStart), formatDateForInput(base.paperEnd)],
        parameters,
        auto_fetch: base.autoFetch,
    };

    if (base.storePath) {
        payload.store_path = base.storePath;
    }

    return payload;
}

function buildMomentumOptimizationSpec(formData) {
    const readIntField = (name, fallback, min = Number.NEGATIVE_INFINITY) => {
        const value = parseIntOr(formData.get(name), fallback);
        if (!Number.isFinite(value)) {
            throw new Error(`${name.replace(/_/g, " ")} must be a number.`);
        }
        if (value < min) {
            throw new Error(`${name.replace(/_/g, " ")} must be at least ${min}.`);
        }
        return value;
    };

    const readFloatField = (name, fallback, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY) => {
        const value = parseFloatOr(formData.get(name), fallback);
        if (!Number.isFinite(value)) {
            throw new Error(`${name.replace(/_/g, " ")} must be a number.`);
        }
        if (value < min) {
            throw new Error(`${name.replace(/_/g, " ")} must be at least ${min}.`);
        }
        if (value > max) {
            throw new Error(`${name.replace(/_/g, " ")} must be less than or equal to ${max}.`);
        }
        return value;
    };

    const buildIntRange = (prefix, defaults, label, minValue = 1) => {
        const minimum = readIntField(`${prefix}_min`, defaults.min, minValue);
        const maximum = readIntField(`${prefix}_max`, defaults.max, minValue);
        const step = readIntField(`${prefix}_step`, defaults.step, 1);
        if (maximum < minimum) {
            throw new Error(`${label} max must be greater than or equal to min.`);
        }
        return { minimum, maximum, step };
    };

    const buildFloatRange = (prefix, defaults, label, minValue, maxValue) => {
        const minimum = readFloatField(`${prefix}_min`, defaults.min, minValue, maxValue);
        const maximum = readFloatField(`${prefix}_max`, defaults.max, minValue, maxValue);
        const step = readFloatField(`${prefix}_step`, defaults.step, 1e-6, maxValue - minValue + defaults.step);
        if (maximum < minimum) {
            throw new Error(`${label} max must be greater than or equal to min.`);
        }
        if (step <= 0) {
            throw new Error(`${label} step must be positive.`);
        }
        return { minimum, maximum, step };
    };

    return {
        lookback_days: buildIntRange("opt_lookback", { min: 84, max: 189, step: 21 }, "Lookback days"),
        skip_days: buildIntRange("opt_skip", { min: 5, max: 21, step: 8 }, "Skip days", 0),
        rebalance_days: buildIntRange("opt_rebalance", { min: 10, max: 30, step: 10 }, "Rebalance days", 1),
        max_positions: buildIntRange("opt_positions", { min: 4, max: 10, step: 2 }, "Max positions", 1),
        lot_size: buildIntRange("opt_lot", { min: 1, max: 1, step: 1 }, "Lot size", 1),
        cash_reserve_pct: buildFloatRange("opt_cash", { min: 0.05, max: 0.15, step: 0.05 }, "Cash reserve", 0, 0.95),
        min_momentum: buildFloatRange("opt_min_momentum", { min: 0.0, max: 0.15, step: 0.05 }, "Minimum momentum", -10, 10),
        volatility_window: buildIntRange("opt_vol_window", { min: 15, max: 35, step: 5 }, "Volatility window", 1),
        volatility_exponent: buildFloatRange("opt_vol_exp", { min: 0.8, max: 1.2, step: 0.2 }, "Volatility exponent", 0, 5),
        max_combinations: readIntField("opt_max_combinations", 60, 1),
    };
}

function buildMomentumOptimizeRequest(formData) {
    const base = readMomentumBaseConfig(formData);
    const spec = buildMomentumOptimizationSpec(formData);
    const lookbackYears = parseFloatOr(formData.get("opt_lookback_years"), 3.0);
    if (!Number.isFinite(lookbackYears) || lookbackYears <= 0) {
        throw new Error("Lookback years must be a positive number.");
    }

    const { symbols, explicit } = gatherExplicitSymbols(formData);

    const payload = {
        initial_cash: base.initialCash,
        training_window: [formatDateForInput(base.trainingStart), formatDateForInput(base.trainingEnd)],
        paper_window: [formatDateForInput(base.paperStart), formatDateForInput(base.paperEnd)],
        auto_fetch: base.autoFetch,
        lookback_years: Number(lookbackYears.toFixed(3)),
        parameter_spec: {
            ...spec,
            max_combinations: spec.max_combinations,
        },
    };

    if (base.storePath) {
        payload.store_path = base.storePath;
    }

    if (explicit) {
        payload.symbols = symbols;
        payload.use_snp100 = false;
    } else {
        payload.use_snp100 = true;
    }

    return payload;
}

function buildMomentumRequest(mode, formData) {
    if (mode === MOMENTUM_MODE_OPTIMIZE) {
        return {
            endpoint: "/api/momentum/optimize",
            payload: buildMomentumOptimizeRequest(formData),
        };
    }
    return {
        endpoint: "/api/momentum/run",
        payload: buildMomentumBacktestRequest(formData),
    };
}

async function runMomentumExperiment(event) {
    if (event) {
        event.preventDefault();
    }

    if (!momentumElements.form) {
        return;
    }

    const formData = new FormData(momentumElements.form);

    const mode = readMomentumMode(formData);
    let request;
    try {
        request = buildMomentumRequest(mode, formData);
    } catch (error) {
        setMomentumStatus(error?.message || "Unable to build momentum request.", "error");
        return;
    }

    const runningMessage = mode === MOMENTUM_MODE_OPTIMIZE
        ? "Optimizing momentum parameters…"
        : "Running momentum experiment…";
    setMomentumStatus(runningMessage, "info");
    disableMomentumRunButton(true);

    try {
        const response = await fetch(request.endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(request.payload),
        });

        if (!response.ok) {
            const errorMessage = await readErrorMessage(response);
            throw new Error(errorMessage);
        }

        const data = await response.json();
        renderMomentumResult(data);

        if (mode === MOMENTUM_MODE_OPTIMIZE) {
            const evaluatedCount = Array.isArray(data?.results) ? data.results.length : data?.evaluated_count ?? 0;
            const bestLabel = data?.best?.label || data?.rankings?.[0]?.label || "best";
            setMomentumStatus(
                `Evaluated ${evaluatedCount} candidate set${evaluatedCount === 1 ? "" : "s"}. Top configuration: ${bestLabel}.`,
                "success",
            );
        } else {
            const symbolCount = Array.isArray(data?.symbols) ? data.symbols.length : request.payload.symbols.length;
            const parameterCount = Array.isArray(data?.results) ? data.results.length : request.payload.parameters.length;
            setMomentumStatus(
                `Evaluated ${parameterCount} parameter set${parameterCount === 1 ? "" : "s"} across ${symbolCount} symbol${symbolCount === 1 ? "" : "s"}.`,
                "success",
            );
        }
    } catch (error) {
        console.error("Failed to run momentum experiment", error);
        setMomentumStatus(error?.message || "Failed to run momentum experiment.", "error");
    } finally {
        disableMomentumRunButton(false);
    }
}

function setMomentumStatus(message, level = "info") {
    if (!momentumElements.status) {
        return;
    }
    momentumElements.status.textContent = message;
    momentumElements.status.className = `status ${level}`;
}

function disableMomentumRunButton(disabled) {
    if (momentumElements.runButton) {
        momentumElements.runButton.disabled = disabled;
    }
}

function renderMomentumResult(data) {
    const resultList = Array.isArray(data?.results) ? data.results : [];
    const resultMap = new Map();
    resultList.forEach((item) => {
        if (item?.label) {
            resultMap.set(item.label, item);
        }
    });

    latestMomentum = {
        ...data,
        results: resultList,
        resultMap,
    };

    const paperWindow = data?.paper_window || null;
    const trainingWindow = data?.training_window || null;

    if (momentumElements.paperWindow) {
        momentumElements.paperWindow.textContent = paperWindow ? `Window: ${formatRange(paperWindow)}` : "";
    }
    if (momentumElements.trainingWindow) {
        momentumElements.trainingWindow.textContent = trainingWindow ? `Window: ${formatRange(trainingWindow)}` : "";
    }

    updateMomentumResultOptions(resultList);
    renderMomentumRankingTable(Array.isArray(data?.rankings) ? data.rankings : []);
    renderMomentumWarnings(data?.warnings, data?.fetched_symbols);

    const defaultLabel = data?.best?.label
        || (data?.rankings && data.rankings[0]?.label)
        || (resultList[0]?.label ?? "");
    renderMomentumSelection(defaultLabel);
}

function updateMomentumResultOptions(results) {
    if (!momentumElements.resultSelect) {
        return;
    }
    const options = Array.isArray(results) ? results : [];
    if (!options.length) {
        momentumElements.resultSelect.innerHTML = "";
        return;
    }
    momentumElements.resultSelect.innerHTML = options
        .map((item) => `<option value="${item.label}">${item.label}</option>`)
        .join("");
}

function renderMomentumSelection(label) {
    if (!latestMomentum || !latestMomentum.results?.length) {
        renderMomentumMetrics(null);
        renderMomentumCharts(null);
        renderMomentumTrades(null);
        if (momentumElements.resultSelect) {
            momentumElements.resultSelect.value = "";
        }
        return;
    }

    const selected = label && latestMomentum.resultMap?.has(label)
        ? latestMomentum.resultMap.get(label)
        : latestMomentum.results[0];

    if (!selected) {
        renderMomentumMetrics(null);
        renderMomentumCharts(null);
        renderMomentumTrades(null);
        return;
    }

    if (momentumElements.resultSelect && label && momentumElements.resultSelect.value !== label) {
        momentumElements.resultSelect.value = label;
    }

    renderMomentumMetrics(selected);
    renderMomentumCharts(selected);
    renderMomentumTrades(selected);
}

function renderMomentumMetrics(result) {
    const trainingMetrics = result?.training_metrics || null;
    const paperMetrics = result?.paper_metrics || null;

    renderMetricCards(momentumElements.paperMetrics, buildMetricRows(paperMetrics));
    renderMetricCards(momentumElements.trainingMetrics, buildMetricRows(trainingMetrics));
}

function renderMomentumWarnings(warnings, fetchedSymbols) {
    if (!momentumElements.warningContainer || !momentumElements.warningList) {
        return;
    }
    const warningItems = [];

    if (Array.isArray(fetchedSymbols) && fetchedSymbols.length) {
        warningItems.push(`Auto-downloaded Polygon data for: ${fetchedSymbols.join(", ")}.`);
    }

    if (Array.isArray(warnings)) {
        warnings.forEach((warning) => {
            if (!warning) {
                return;
            }
            if (typeof warning === "string") {
                warningItems.push(warning);
            } else if (warning.label && warning.reason) {
                warningItems.push(`${warning.label}: ${warning.reason}`);
            } else if (warning.reason) {
                warningItems.push(warning.reason);
            }
        });
    }

    if (!warningItems.length) {
        momentumElements.warningList.innerHTML = "";
        momentumElements.warningContainer.hidden = true;
        return;
    }

    momentumElements.warningList.innerHTML = warningItems
        .map((message) => `<li>${message}</li>`)
        .join("");
    momentumElements.warningContainer.hidden = false;
}

function renderMomentumRankingTable(rankings) {
    if (!momentumElements.rankingTableBody) {
        return;
    }
    if (!Array.isArray(rankings) || !rankings.length) {
        momentumElements.rankingTableBody.innerHTML =
            '<tr><td colspan="5" class="metric-empty">No parameter sets evaluated yet.</td></tr>';
        return;
    }

    momentumElements.rankingTableBody.innerHTML = rankings
        .map((row) => {
            const sharpe = formatNumber(row.paper_sharpe ?? row.paper_metrics?.sharpe_ratio ?? 0, 2);
            const totalReturn = formatPercent(row.paper_total_return ?? 0);
            const cagr = formatPercent(row.paper_cagr ?? 0);
            const drawdown = formatPercent(-(Math.abs(row.paper_max_drawdown ?? 0)));
            return `
        <tr>
          <td>${row.label || "Set"}</td>
          <td>${sharpe}</td>
          <td>${totalReturn}</td>
          <td>${cagr}</td>
          <td>${drawdown}</td>
        </tr>
      `;
        })
        .join("");
}

function renderMomentumCharts(result) {
    if (!momentumElements.trainingChart || !momentumElements.paperChart) {
        return;
    }

    const trainingCurve = Array.isArray(result?.training_equity_curve) ? result.training_equity_curve : [];
    const paperCurve = Array.isArray(result?.paper_equity_curve) ? result.paper_equity_curve : [];

    if (!trainingCurve.length) {
        Plotly.purge(momentumElements.trainingChart);
    } else {
        const trainingTrace = {
            type: "scatter",
            mode: "lines",
            name: "Training Equity",
            x: trainingCurve.map((point) => point.timestamp),
            y: trainingCurve.map((point) => point.equity),
            line: { color: "#6366f1", width: 3 },
        };
        Plotly.newPlot(
            momentumElements.trainingChart,
            [trainingTrace],
            {
                margin: { t: 40, r: 10, b: 50, l: 60 },
                xaxis: { title: "Date" },
                yaxis: { title: "Equity" },
                hovermode: "x unified",
            },
            { responsive: true, displaylogo: false },
        );
    }

    if (!paperCurve.length) {
        Plotly.purge(momentumElements.paperChart);
    } else {
        const paperTrace = {
            type: "scatter",
            mode: "lines",
            name: "Paper Equity",
            x: paperCurve.map((point) => point.timestamp),
            y: paperCurve.map((point) => point.equity),
            line: { color: "#22c55e", width: 3 },
        };
        Plotly.newPlot(
            momentumElements.paperChart,
            [paperTrace],
            {
                margin: { t: 40, r: 10, b: 50, l: 60 },
                xaxis: { title: "Date" },
                yaxis: { title: "Equity" },
                hovermode: "x unified",
            },
            { responsive: true, displaylogo: false },
        );
    }
}

function renderMomentumTrades(result) {
    if (!momentumElements.tradeTableBody) {
        return;
    }

    const trades = [];
    if (Array.isArray(result?.training_trades)) {
        trades.push(...result.training_trades);
    }
    if (Array.isArray(result?.paper_trades)) {
        trades.push(...result.paper_trades);
    }

    trades.sort((a, b) => {
        const tsA = new Date(a.timestamp || 0).getTime();
        const tsB = new Date(b.timestamp || 0).getTime();
        return tsA - tsB;
    });

    if (!trades.length) {
        momentumElements.tradeTableBody.innerHTML =
            '<tr><td colspan="5" class="metric-empty">No trades recorded for the selected set.</td></tr>';
        return;
    }

    momentumElements.tradeTableBody.innerHTML = trades
        .map((trade) => {
            const timestamp = formatDateTime(trade.timestamp);
            const symbol = (trade.symbol || "").toUpperCase();
            const quantity = formatNumber(trade.quantity, 0);
            const price = formatCurrency(trade.price);
            const cashAfter = formatCurrency(trade.cash_after);
            return `
        <tr>
          <td>${timestamp}</td>
          <td>${symbol}</td>
          <td>${quantity}</td>
          <td>${price}</td>
          <td>${cashAfter}</td>
        </tr>
      `;
        })
        .join("");
}

async function runMomentumPaperTrade(event) {
    if (event) {
        event.preventDefault();
    }

    if (!momentumPaperElements.form) {
        return;
    }

    const formData = new FormData(momentumPaperElements.form);
    let request;
    try {
        request = buildMomentumPaperRequest(formData);
    } catch (error) {
        console.error("Failed to build momentum paper request", error);
        setPaperStatus(error?.message || "Unable to build paper trading request.", "error");
        return;
    }

    setPaperStatus("Running momentum paper trade simulation…", "info");
    disablePaperRunButton(true);

    try {
        const response = await fetch(request.endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(request.payload),
        });

        if (!response.ok) {
            const errorMessage = await readErrorMessage(response);
            throw new Error(errorMessage || `Request failed with status ${response.status}`);
        }

        const data = await response.json();
        renderMomentumPaperResult(data);

        const tradeCount = Array.isArray(data?.paper_trades) ? data.paper_trades.length : 0;
        const actionCount = Array.isArray(data?.actions) ? data.actions.length : 0;
        const autoFetchLabel = request.payload.auto_fetch ? " with auto-fetch" : "";
        setPaperStatus(
            `Paper trade simulation complete${autoFetchLabel}. Signals generated: ${tradeCount}. Actions recorded: ${actionCount}.`,
            "success",
        );
    } catch (error) {
        console.error("Failed to run momentum paper trade", error);
        setPaperStatus(error?.message || "Failed to run momentum paper trade.", "error");
    } finally {
        disablePaperRunButton(false);
    }
}

function buildMomentumPaperRequest(formData) {
    const payload = {};

    const initialCash = parseFloatOr(formData.get("initial_cash"), 10000);
    if (!Number.isFinite(initialCash) || initialCash <= 0) {
        throw new Error("Initial cash must be a positive number.");
    }
    payload.initial_cash = Number(initialCash.toFixed(2));

    const symbolInput = formData.get("symbols") || "";
    const customSymbols = parseSymbolList(symbolInput);
    if (customSymbols.length) {
        payload.symbols = Array.from(new Set(customSymbols));
    }

    const autoFetch = formData.has("auto_fetch");
    payload.auto_fetch = autoFetch;

    const executeOrders = formData.has("execute_orders");
    payload.execute_orders = executeOrders;

    const trainingStart = (formData.get("training_start") || "").toString().trim();
    const trainingEnd = (formData.get("training_end") || "").toString().trim();
    const paperStart = (formData.get("paper_start") || "").toString().trim();
    const paperEnd = (formData.get("paper_end") || "").toString().trim();

    const hasTrainingDates = trainingStart && trainingEnd;
    const hasPaperDates = paperStart && paperEnd;
    if ((trainingStart && !trainingEnd) || (!trainingStart && trainingEnd)) {
        throw new Error("Both training start and end dates must be provided.");
    }
    if ((paperStart && !paperEnd) || (!paperStart && paperEnd)) {
        throw new Error("Both paper start and end dates must be provided.");
    }

    if (hasTrainingDates) {
        const startDate = new Date(trainingStart);
        const endDate = new Date(trainingEnd);
        if (!(startDate instanceof Date && !Number.isNaN(startDate.valueOf()))) {
            throw new Error("Training start date is invalid.");
        }
        if (!(endDate instanceof Date && !Number.isNaN(endDate.valueOf()))) {
            throw new Error("Training end date is invalid.");
        }
        if (startDate >= endDate) {
            throw new Error("Training start must be before training end.");
        }
        payload.training_window = [trainingStart, trainingEnd];
    }

    if (hasPaperDates) {
        const startDate = new Date(paperStart);
        const endDate = new Date(paperEnd);
        if (!(startDate instanceof Date && !Number.isNaN(startDate.valueOf()))) {
            throw new Error("Paper start date is invalid.");
        }
        if (!(endDate instanceof Date && !Number.isNaN(endDate.valueOf()))) {
            throw new Error("Paper end date is invalid.");
        }
        if (startDate >= endDate) {
            throw new Error("Paper start must be before paper end.");
        }
        payload.paper_window = [paperStart, paperEnd];
    }

    const parameters = buildPaperMomentumParameters(formData);
    payload.parameters = [parameters];

    return {
        endpoint: "/api/momentum/paper-trade",
        payload,
    };
}

function buildPaperMomentumParameters(formData) {
    const getInt = (name, fallback, min = Number.NEGATIVE_INFINITY) => {
        const value = parseIntOr(formData.get(name), fallback);
        if (!Number.isFinite(value)) {
            throw new Error(`${formatParameterLabel(name)} must be a number.`);
        }
        if (value < min) {
            throw new Error(`${formatParameterLabel(name)} must be at least ${min}.`);
        }
        return value;
    };

    const getFloat = (name, fallback, min = Number.NEGATIVE_INFINITY, max = Number.POSITIVE_INFINITY) => {
        const value = parseFloatOr(formData.get(name), fallback);
        if (!Number.isFinite(value)) {
            throw new Error(`${formatParameterLabel(name)} must be a number.`);
        }
        if (value < min) {
            throw new Error(`${formatParameterLabel(name)} must be at least ${min}.`);
        }
        if (value > max) {
            throw new Error(`${formatParameterLabel(name)} must be at most ${max}.`);
        }
        return value;
    };

    const lookbackDays = getInt("lookback_days", MOMENTUM_PAPER_DEFAULT_PARAMETERS.lookback_days, 1);
    const skipDays = getInt("skip_days", MOMENTUM_PAPER_DEFAULT_PARAMETERS.skip_days, 0);
    const rebalanceDays = getInt("rebalance_days", MOMENTUM_PAPER_DEFAULT_PARAMETERS.rebalance_days, 1);
    const maxPositions = getInt("max_positions", MOMENTUM_PAPER_DEFAULT_PARAMETERS.max_positions, 1);
    const lotSize = getInt("lot_size", MOMENTUM_PAPER_DEFAULT_PARAMETERS.lot_size, 1);
    const cashReservePct = getFloat("cash_reserve_pct", MOMENTUM_PAPER_DEFAULT_PARAMETERS.cash_reserve_pct * 100, 0, 95);
    const minMomentum = getFloat("min_momentum", MOMENTUM_PAPER_DEFAULT_PARAMETERS.min_momentum, -10, 10);
    const volatilityWindow = getInt("volatility_window", MOMENTUM_PAPER_DEFAULT_PARAMETERS.volatility_window, 1);
    const volatilityExponent = getFloat("volatility_exponent", MOMENTUM_PAPER_DEFAULT_PARAMETERS.volatility_exponent, 0, 10);

    if (skipDays >= lookbackDays) {
        throw new Error("Skip days must be less than lookback days.");
    }

    return {
        lookback_days: lookbackDays,
        skip_days: skipDays,
        rebalance_days: rebalanceDays,
        max_positions: maxPositions,
        lot_size: lotSize,
        cash_reserve_pct: Math.max(0, Math.min(cashReservePct / 100, 0.95)),
        min_momentum: Number(minMomentum.toFixed(4)),
        volatility_window: volatilityWindow,
        volatility_exponent: Number(volatilityExponent.toFixed(4)),
    };
}

function renderMomentumPaperResult(data) {
    latestMomentumPaper = data;

    const evaluation = data?.evaluation || null;
    const paperWindow = data?.paper_window || null;
    const trainingWindow = data?.training_window || null;

    if (momentumPaperElements.paperWindowLabel) {
        momentumPaperElements.paperWindowLabel.textContent = paperWindow ? `Window: ${formatRange(paperWindow)}` : "";
    }
    if (momentumPaperElements.trainingWindowLabel) {
        momentumPaperElements.trainingWindowLabel.textContent = trainingWindow ? `Window: ${formatRange(trainingWindow)}` : "";
    }

    renderMetricCards(momentumPaperElements.metrics, buildMetricRows(evaluation?.paper_metrics));
    renderMetricCards(momentumPaperElements.trainingMetrics, buildMetricRows(evaluation?.training_metrics));
    renderPaperTrades(data);
    renderPaperActions(data?.actions || []);
    renderPaperWarnings(data?.warnings || [], data?.fetched_symbols || []);
    renderPaperCharts(evaluation);
}

function renderPaperTrades(data) {
    if (!momentumPaperElements.tradeTableBody) {
        return;
    }
    const trades = Array.isArray(data?.paper_trades) ? data.paper_trades : [];
    if (!trades.length) {
        momentumPaperElements.tradeTableBody.innerHTML =
            '<tr><td colspan="5" class="metric-empty">No trade signals generated.</td></tr>';
        return;
    }

    const rows = trades
        .map((trade) => {
            const quantity = Number(trade?.quantity || 0);
            const direction = quantity > 0 ? "Buy" : quantity < 0 ? "Sell" : "Hold";
            return `
        <tr>
          <td>${trade?.timestamp || ""}</td>
          <td>${trade?.symbol || ""}</td>
          <td>${direction}</td>
          <td>${Math.abs(quantity)}</td>
          <td>${formatCurrency(trade?.price)}</td>
        </tr>
      `;
        })
        .join("");
    momentumPaperElements.tradeTableBody.innerHTML = rows;
}

async function startMomentumLiveTrading(event) {
    if (event) {
        event.preventDefault();
    }

    if (!momentumLiveElements.form) {
        return;
    }

    let payload;
    try {
        payload = buildMomentumLiveRequest(new FormData(momentumLiveElements.form));
    } catch (error) {
        setLiveStatus(error?.message || "Unable to build live trading request.", "error");
        return;
    }

    setLiveStatus("Starting live trading daemon…", "info");
    disableLiveStartButton(true);

    try {
        const response = await fetch("/api/momentum/live/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const message = await readErrorMessage(response);
            throw new Error(message || `Request failed with status ${response.status}`);
        }

        const data = await response.json();
        renderMomentumLiveStatus(data);
        setLiveStatus("Live trading daemon is running.", "success");
        disableLiveStopButton(false);
        refreshMomentumLiveHistory();
    } catch (error) {
        console.error("Failed to start momentum live trading", error);
        setLiveStatus(error?.message || "Failed to start live trading.", "error");
        disableLiveStartButton(false);
    }
}

async function stopMomentumLiveTrading(event) {
    if (event) {
        event.preventDefault();
    }

    disableLiveStopButton(true);
    setLiveStatus("Stopping live trading daemon…", "info");

    try {
        const response = await fetch("/api/momentum/live/stop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        });

        if (!response.ok) {
            const message = await readErrorMessage(response);
            throw new Error(message || `Request failed with status ${response.status}`);
        }

        const data = await response.json();
        renderMomentumLiveStatus(data);
        setLiveStatus("Live trading daemon stopped.", "success");
    } catch (error) {
        console.error("Failed to stop momentum live trading", error);
        setLiveStatus(error?.message || "Failed to stop live trading.", "error");
    } finally {
        disableLiveStartButton(false);
        disableLiveStopButton(true);
    }
}

function buildMomentumLiveRequest(formData) {
    const payload = {};

    const intervalMinutes = parseIntOr(formData.get("interval_minutes"), MOMENTUM_LIVE_MIN_INTERVAL_MINUTES);
    if (!Number.isFinite(intervalMinutes) || intervalMinutes < MOMENTUM_LIVE_MIN_INTERVAL_MINUTES) {
        throw new Error(`Interval must be at least ${MOMENTUM_LIVE_MIN_INTERVAL_MINUTES} minute(s).`);
    }
    if (intervalMinutes > MOMENTUM_LIVE_MAX_INTERVAL_MINUTES) {
        throw new Error("Interval is too large. Choose a shorter cadence.");
    }
    payload.interval_seconds = intervalMinutes * 60;

    const symbols = parseSymbolList(formData.get("symbols") || "");
    if (symbols.length) {
        payload.symbols = Array.from(new Set(symbols));
    }

    payload.auto_fetch = formData.has("auto_fetch");
    payload.execute_orders = formData.has("execute_orders");

    payload.parameters = [readLiveMomentumParameters(formData)];

    return payload;
}

function readLiveMomentumParameters(formData) {
    if (!momentumLiveElements.parameterGrid) {
        return MOMENTUM_PAPER_DEFAULT_PARAMETERS;
    }
    const parameterInputs = momentumLiveElements.parameterGrid.querySelectorAll("input[name]");
    const values = {};
    parameterInputs.forEach((input) => {
        const name = input.name;
        const raw = formData.get(name) ?? input.value;
        const number = Number.parseFloat(raw);
        if (!Number.isFinite(number)) {
            return;
        }
        if (name === "cash_reserve_pct") {
            values[name] = Math.max(0, Math.min(number / 100, 0.95));
        } else if (["lookback_days", "rebalance_days", "max_positions", "lot_size", "volatility_window"].includes(name)) {
            values[name] = Math.max(1, Math.round(number));
        } else if (name === "skip_days") {
            values[name] = Math.max(0, Math.round(number));
        } else {
            values[name] = Number(number.toFixed(4));
        }
    });

    return { ...MOMENTUM_PAPER_DEFAULT_PARAMETERS, ...values };
}

function renderPaperActions(actions) {
    if (!momentumPaperElements.actionsGroup || !momentumPaperElements.actionsList) {
        return;
    }
    const entries = Array.isArray(actions)
        ? actions
            .map((action) => describePaperAction(action))
            .filter((message) => typeof message === "string" && message.trim())
        : [];

    if (!entries.length) {
        momentumPaperElements.actionsGroup.hidden = true;
        momentumPaperElements.actionsList.innerHTML = "";
        return;
    }

    momentumPaperElements.actionsList.innerHTML = entries.map((item) => `<li>${item}</li>`).join("");
    momentumPaperElements.actionsGroup.hidden = false;
}

function renderPaperWarnings(warnings, fetchedSymbols) {
    if (!momentumPaperElements.warningsGroup || !momentumPaperElements.warningsList) {
        return;
    }

    const messages = [];
    if (Array.isArray(fetchedSymbols) && fetchedSymbols.length) {
        messages.push(`Auto-downloaded Polygon data for: ${fetchedSymbols.join(", ")}.`);
    }
    if (Array.isArray(warnings)) {
        warnings.forEach((warning) => {
            if (!warning) {
                return;
            }
            if (typeof warning === "string") {
                messages.push(warning);
            } else if (warning.label && warning.reason) {
                messages.push(`${warning.label}: ${warning.reason}`);
            } else if (warning.reason) {
                messages.push(warning.reason);
            }
        });
    }

    if (!messages.length) {
        momentumPaperElements.warningsGroup.hidden = true;
        momentumPaperElements.warningsList.innerHTML = "";
        return;
    }

    momentumPaperElements.warningsList.innerHTML = messages.map((item) => `<li>${item}</li>`).join("");
    momentumPaperElements.warningsGroup.hidden = false;
}

function renderPaperCharts(result) {
    if (!momentumPaperElements.trainingChart || !momentumPaperElements.paperChart) {
        return;
    }

    const trainingCurve = Array.isArray(result?.training_equity_curve) ? result.training_equity_curve : [];
    const paperCurve = Array.isArray(result?.paper_equity_curve) ? result.paper_equity_curve : [];

    if (!trainingCurve.length) {
        Plotly.purge(momentumPaperElements.trainingChart);
    } else {
        const trainingTrace = {
            type: "scatter",
            mode: "lines",
            name: "Training Equity",
            x: trainingCurve.map((point) => point.timestamp),
            y: trainingCurve.map((point) => point.equity),
            line: { color: "#2563eb", width: 3 },
        };
        Plotly.newPlot(
            momentumPaperElements.trainingChart,
            [trainingTrace],
            {
                margin: { t: 40, r: 10, b: 50, l: 60 },
                xaxis: { title: "Date" },
                yaxis: { title: "Equity" },
                hovermode: "x unified",
            },
            { responsive: true, displaylogo: false },
        );
    }

    if (!paperCurve.length) {
        Plotly.purge(momentumPaperElements.paperChart);
    } else {
        const paperTrace = {
            type: "scatter",
            mode: "lines",
            name: "Paper Equity",
            x: paperCurve.map((point) => point.timestamp),
            y: paperCurve.map((point) => point.equity),
            line: { color: "#22c55e", width: 3 },
        };
        Plotly.newPlot(
            momentumPaperElements.paperChart,
            [paperTrace],
            {
                margin: { t: 40, r: 10, b: 50, l: 60 },
                xaxis: { title: "Date" },
                yaxis: { title: "Equity" },
                hovermode: "x unified",
            },
            { responsive: true, displaylogo: false },
        );
    }
}

function renderMomentumLiveStatus(status) {
    latestMomentumLive = {
        ...(latestMomentumLive || {}),
        status,
    };

    const running = Boolean(status?.running);
    disableLiveStartButton(running);
    disableLiveStopButton(!running);

    if (momentumLiveElements.statusBadge) {
        const current = String(status?.status || "idle");
        const label = current.charAt(0).toUpperCase() + current.slice(1);
        momentumLiveElements.statusBadge.textContent = label;
        let badgeClass = "badge";
        if (running && current === "running") {
            badgeClass += " badge--done";
        } else if (!running) {
            badgeClass += " badge--todo";
        }
        momentumLiveElements.statusBadge.className = badgeClass;
    }

    if (momentumLiveElements.iterations) {
        momentumLiveElements.iterations.textContent = formatNumber(status?.iterations, 0);
    }
    if (momentumLiveElements.lastRun) {
        momentumLiveElements.lastRun.textContent = formatDateTime(status?.last_run_at);
    }
    if (momentumLiveElements.nextRun) {
        momentumLiveElements.nextRun.textContent = formatDateTime(status?.next_run_at);
    }

    const config = status?.config;
    if (config) {
        if (momentumLiveElements.intervalInput) {
            momentumLiveElements.intervalInput.value = Math.round((config.interval_seconds || 0) / 60) || 30;
        }
        if (momentumLiveElements.autoFetchToggle) {
            momentumLiveElements.autoFetchToggle.checked = Boolean(config.auto_fetch);
        }
        if (momentumLiveElements.executeToggle) {
            momentumLiveElements.executeToggle.checked = Boolean(config.execute_orders);
        }
        if (Array.isArray(config.symbols) && momentumLiveElements.symbolsInput) {
            momentumLiveElements.symbolsInput.value = config.symbols.join(", ");
        }
        if (momentumLiveElements.configCash) {
            const cash = Number(config.initial_cash ?? Number.NaN);
            momentumLiveElements.configCash.textContent = Number.isFinite(cash) ? formatCurrency(cash) : "–";
        }
    } else if (momentumLiveElements.configCash) {
        momentumLiveElements.configCash.textContent = "–";
    }

    if (momentumLiveElements.tradeDate && status?.config?.trade_date) {
        momentumLiveElements.tradeDate.textContent = formatDate(status.config.trade_date);
    }
}

function renderMomentumLiveHistory(history) {
    latestMomentumLive = {
        ...(latestMomentumLive || {}),
        history,
    };

    const runs = Array.isArray(history?.runs) ? history.runs : [];
    const tradeHistory = Array.isArray(history?.trade_history) ? history.trade_history : [];
    const equityPoints = Array.isArray(history?.equity_curve) ? history.equity_curve : [];
    if (momentumLiveElements.historyTableBody) {
        if (!runs.length) {
            momentumLiveElements.historyTableBody.innerHTML =
                '<tr><td colspan="5" class="metric-empty">No runs recorded yet.</td></tr>';
        } else {
            momentumLiveElements.historyTableBody.innerHTML = runs
                .map((run) => {
                    const status = (run.status || "").toString();
                    const badgeClass = status === "completed" ? "badge--done" : status === "error" ? "badge--todo" : "";
                    const statusBadge = `<span class="badge ${badgeClass}">${status.toUpperCase()}</span>`;
                    return `
            <tr>
              <td>${run.run_id || "–"}</td>
              <td>${formatDateTime(run.started_at)}</td>
              <td>${statusBadge}</td>
              <td>${formatNumber(run.trade_count, 0)}</td>
              <td>${formatNumber(run.order_count, 0)}</td>
            </tr>
          `;
                })
                .join("");
        }
    }

    const latestRun = runs[runs.length - 1];
    if (latestRun) {
        renderMomentumLiveTrades(latestRun);
        renderMomentumLiveWarnings(latestRun);
        renderMomentumLivePortfolio(latestRun);
    } else {
        renderMomentumLiveTrades(null);
        renderMomentumLiveWarnings(null);
        renderMomentumLivePortfolio(null);
    }

    renderMomentumLiveTradeHistoryTable(tradeHistory);
    renderMomentumLiveEquityCurve(equityPoints);
    if (momentumLiveElements.historyResetButton) {
        momentumLiveElements.historyResetButton.disabled = !tradeHistory.length && !runs.length;
    }
}

function renderMomentumLiveWarnings(run) {
    const warnings = run?.warnings || [];
    const actions = run?.actions || [];
    if (momentumLiveElements.warningsGroup && momentumLiveElements.warningsList) {
        const warningMessages = Array.isArray(warnings)
            ? warnings
                .map((warning) => (typeof warning === "string" ? warning : warning?.reason || ""))
                .filter(Boolean)
            : [];

        const runError = typeof run?.error === "string" && run.error.trim() ? run.error.trim() : null;
        if (runError) {
            warningMessages.unshift(`Live iteration failed: ${runError}`);
        }

        if (warningMessages.length) {
            momentumLiveElements.warningsList.innerHTML = warningMessages.map((message) => `<li>${message}</li>`).join("");
            momentumLiveElements.warningsGroup.hidden = false;
        } else {
            momentumLiveElements.warningsGroup.hidden = true;
            momentumLiveElements.warningsList.innerHTML = "";
        }
    }

    if (momentumLiveElements.actionsGroup && momentumLiveElements.actionsList) {
        const actionMessages = Array.isArray(actions)
            ? actions
                .map((action) => describePaperAction(action))
                .filter((message) => typeof message === "string" && message.trim())
            : [];

        if (actionMessages.length) {
            momentumLiveElements.actionsList.innerHTML = actionMessages.map((message) => `<li>${message}</li>`).join("");
            momentumLiveElements.actionsGroup.hidden = false;
        } else {
            momentumLiveElements.actionsGroup.hidden = true;
            momentumLiveElements.actionsList.innerHTML = "";
        }
    }
}

function renderMomentumLiveTrades(run) {
    if (!momentumLiveElements.tradeTableBody) {
        return;
    }

    const trades = Array.isArray(run?.paper_trades) ? run.paper_trades : [];
    if (!trades.length) {
        momentumLiveElements.tradeTableBody.innerHTML =
            '<tr><td colspan="6" class="metric-empty">No trades submitted this run.</td></tr>';
    } else {
        momentumLiveElements.tradeTableBody.innerHTML = trades
            .map((trade) => {
                const timestamp = formatDateTime(trade.timestamp);
                const symbol = (trade.symbol || "").toString().toUpperCase();
                const requestedQuantity = Number(trade.quantity || 0);
                const executionQuantityRaw = typeof trade.execution_filled_quantity === "number"
                    ? Number(trade.execution_filled_quantity)
                    : Number.NaN;
                const hasExecutionQuantity = Number.isFinite(executionQuantityRaw);
                const quantityValue = hasExecutionQuantity ? executionQuantityRaw : requestedQuantity;
                const direction = quantityValue > 0
                    ? "Buy"
                    : quantityValue < 0
                        ? "Sell"
                        : requestedQuantity > 0
                            ? "Buy"
                            : requestedQuantity < 0
                                ? "Sell"
                                : "Hold";
                const displayQuantity = formatNumber(Math.abs(quantityValue || requestedQuantity), 0);
                const executionPriceRaw = typeof trade.execution_price === "number" && Number.isFinite(trade.execution_price)
                    ? trade.execution_price
                    : Number(trade.price);
                const hasExecutionPrice = Number.isFinite(executionPriceRaw);
                const paperPrice = typeof trade.paper_price === "number" && Number.isFinite(trade.paper_price)
                    ? trade.paper_price
                    : null;
                let priceDisplay = hasExecutionPrice ? formatCurrency(executionPriceRaw) : "–";
                if (paperPrice !== null && hasExecutionPrice && Math.abs(executionPriceRaw - paperPrice) > 0.01) {
                    priceDisplay = `${priceDisplay} (sim ${formatCurrency(paperPrice)})`;
                }
                const cashAfter = formatCurrency(trade.cash_after);
                return `
        <tr>
          <td>${timestamp}</td>
          <td>${symbol}</td>
          <td>${direction}</td>
          <td>${displayQuantity}</td>
          <td>${priceDisplay}</td>
          <td>${cashAfter}</td>
        </tr>
      `;
            })
            .join("");
    }

    if (momentumLiveElements.tradeDate) {
        const tradeDate = run?.trade_date || null;
        momentumLiveElements.tradeDate.textContent = tradeDate ? formatDate(tradeDate) : "–";
    }
}

function renderMomentumLivePortfolio(run) {
    const portfolio = run?.portfolio && typeof run.portfolio === "object" ? run.portfolio : null;
    const cashValue = typeof portfolio?.cash === "number" ? portfolio.cash : null;
    const positions = Array.isArray(portfolio?.positions) ? portfolio.positions : [];

    if (momentumLiveElements.portfolioCash) {
        momentumLiveElements.portfolioCash.textContent = cashValue !== null ? formatCurrency(cashValue) : "–";
    }

    if (momentumLiveElements.portfolioPositionCount) {
        momentumLiveElements.portfolioPositionCount.textContent = formatNumber(positions.length, 0);
    }

    if (momentumLiveElements.portfolioTableBody) {
        if (!positions.length) {
            momentumLiveElements.portfolioTableBody.innerHTML =
                '<tr><td colspan="3" class="metric-empty">No open positions.</td></tr>';
        } else {
            momentumLiveElements.portfolioTableBody.innerHTML = positions
                .map((position) => {
                    const symbol = (position.symbol || "").toString().toUpperCase();
                    const quantity = formatNumber(position.quantity || 0, 0);
                    const price = formatCurrency(position.avg_price || 0);
                    return `
        <tr>
          <td>${symbol}</td>
          <td>${quantity}</td>
          <td>${price}</td>
        </tr>
      `;
                })
                .join("");
        }
    }

    if (momentumLiveElements.portfolioGroup) {
        momentumLiveElements.portfolioGroup.hidden = false;
    }
}

function renderMomentumLiveTradeHistoryTable(trades) {
    const body = momentumLiveElements.tradeHistoryTableBody;
    if (!body) {
        return;
    }

    if (!Array.isArray(trades) || !trades.length) {
        body.innerHTML = '<tr><td colspan="8" class="metric-empty">No recorded trades yet.</td></tr>';
        return;
    }

    const coerceNumber = (value) => {
        if (typeof value === "number") {
            return Number.isFinite(value) ? value : null;
        }
        const parsed = parseFloat(value);
        return Number.isFinite(parsed) ? parsed : null;
    };

    const rows = trades
        .slice()
        .sort((a, b) => {
            const timeA = new Date(a?.timestamp || a?.trade_date || 0).getTime();
            const timeB = new Date(b?.timestamp || b?.trade_date || 0).getTime();
            return timeA - timeB;
        })
        .map((trade) => {
            const tradeDate = trade?.trade_date || trade?.timestamp;
            const direction = (trade?.direction || "").toString();
            const quantityValue = coerceNumber(trade?.quantity ?? trade?.signed_quantity);
            const priceCandidates = [trade?.price, trade?.execution_price, trade?.paper_price];
            const executionPrice = priceCandidates.reduce((acc, value) => {
                if (acc !== null) {
                    return acc;
                }
                return coerceNumber(value);
            }, null);
            const paperPrice = coerceNumber(trade?.paper_price);
            let priceDisplay = executionPrice !== null ? formatCurrency(executionPrice) : "–";
            if (
                paperPrice !== null &&
                executionPrice !== null &&
                Math.abs(executionPrice - paperPrice) > 0.01
            ) {
                priceDisplay = `${priceDisplay} (sim ${formatCurrency(paperPrice)})`;
            }

            const cashAfter = coerceNumber(trade?.cash_after);
            const runId = (trade?.run_id || "–").toString();

            return `
        <tr>
          <td>${formatDate(tradeDate)}</td>
          <td>${formatDateTime(trade?.timestamp)}</td>
          <td>${(trade?.symbol || "").toString().toUpperCase()}</td>
          <td>${direction ? direction.charAt(0).toUpperCase() + direction.slice(1) : ""}</td>
          <td>${formatNumber(quantityValue, 0)}</td>
          <td>${priceDisplay}</td>
          <td>${formatCurrency(cashAfter)}</td>
          <td>${runId}</td>
        </tr>
      `;
        })
        .join("");

    body.innerHTML = rows;
}

function renderMomentumLiveEquityCurve(points) {
    const chart = momentumLiveElements.equityChart;
    if (!chart) {
        return;
    }

    if (!Array.isArray(points) || !points.length) {
        if (typeof Plotly !== "undefined") {
            Plotly.purge(chart);
        }
        chart.innerHTML = '<div class="metric-empty">No equity data recorded yet.</div>';
        return;
    }

    chart.innerHTML = "";

    if (typeof Plotly === "undefined") {
        return;
    }

    const sorted = points
        .slice()
        .filter((point) => point && point.timestamp)
        .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

    if (!sorted.length) {
        Plotly.purge(chart);
        chart.innerHTML = '<div class="metric-empty">No equity data recorded yet.</div>';
        return;
    }

    const timestamps = sorted.map((point) => point.timestamp);
    const totalAssets = sorted.map((point) => (Number.isFinite(point.total_assets) ? point.total_assets : null));
    const cashAvailable = sorted.map((point) =>
        Number.isFinite(point.cash_available) ? point.cash_available : null,
    );

    const traces = [
        {
            type: "scatter",
            mode: "lines",
            name: "Total Assets",
            x: timestamps,
            y: totalAssets,
            line: { color: "#2563eb", width: 3 },
        },
    ];

    if (cashAvailable.some((value) => value !== null)) {
        traces.push({
            type: "scatter",
            mode: "lines",
            name: "Cash Available",
            x: timestamps,
            y: cashAvailable,
            line: { color: "#22c55e", width: 2, dash: "dot" },
        });
    }

    Plotly.newPlot(
        chart,
        traces,
        {
            margin: { t: 40, r: 10, b: 50, l: 60 },
            xaxis: { title: "Timestamp" },
            yaxis: { title: "Value (USD)", tickprefix: "$" },
            hovermode: "x unified",
            legend: { orientation: "h" },
        },
        { responsive: true, displaylogo: false },
    );
}

async function resetMomentumLiveHistory(event) {
    if (event) {
        event.preventDefault();
    }

    const button = momentumLiveElements.historyResetButton;
    if (!button) {
        return;
    }
    if (!button.dataset.originalLabel) {
        button.dataset.originalLabel = button.textContent?.trim() || "Hard Reset";
    }
    if (button.disabled) {
        return;
    }

    const confirmReset = window.confirm(
        "This will erase all recorded live trades and equity history. Continue?",
    );
    if (!confirmReset) {
        return;
    }

    button.disabled = true;
    button.textContent = "Resetting…";
    setLiveStatus("Resetting live trade history…", "info");

    try {
        const response = await fetch("/api/momentum/live/history/reset", {
            method: "POST",
        });
        if (!response.ok) {
            const errorMessage = await readErrorMessage(response);
            throw new Error(errorMessage);
        }
        const data = await response.json();
        const warnings = Array.isArray(data?.warnings)
            ? data.warnings
                .map((message) => (typeof message === "string" ? message.trim() : ""))
                .filter(Boolean)
            : [];
        const statusMessage = warnings.length
            ? `History reset with warnings: ${warnings.join(" | ")}`
            : "Live trade history reset.";
        setLiveStatus(statusMessage, warnings.length ? "warning" : "success");
        refreshMomentumLiveHistory();
    } catch (error) {
        console.error("Failed to reset live history", error);
        setLiveStatus(error?.message || "Failed to reset live history.", "error");
    } finally {
        button.disabled = false;
        button.textContent = button.dataset.originalLabel || "Hard Reset";
    }
}

function describePaperAction(action) {
    if (!action || typeof action !== "object") {
        return "";
    }
    const type = action.type;
    switch (type) {
        case "universe_loaded": {
            const source = action.source || "universe";
            const count = action.symbol_count || (Array.isArray(action.symbols) ? action.symbols.length : 0);
            return `Loaded ${count} symbols from ${source}.`;
        }
        case "parameters_prepared":
            return "Momentum parameters prepared for execution.";
        case "history_fetched": {
            const symbols = Array.isArray(action.symbols) ? action.symbols.join(", ") : "requested universe";
            return `Fetched Polygon history for ${symbols}.`;
        }
        case "experiment_run": {
            const training = action.training_window ? formatRange(action.training_window) : "training window";
            const paper = action.paper_window ? formatRange(action.paper_window) : "paper window";
            return `Backtested over ${training} and paper tested over ${paper}.`;
        }
        case "trade_signal": {
            if (action.status === "no_trades") {
                return "No trades generated for the paper window.";
            }
            const direction = action.direction || "trade";
            const symbol = action.symbol || "symbol";
            const quantity = Math.abs(action.quantity || 0);
            return `${direction === "buy" ? "Buy" : "Sell"} ${quantity} ${symbol} @ ${formatCurrency(action.price)}.`;
        }
        case "ibkr_connect":
            return `Connecting to IBKR TWS (client ${action.client_id ?? ""})…`;
        case "ibkr_order_submitted": {
            const direction = action.direction || "order";
            const symbol = action.symbol || "symbol";
            const quantity = action.quantity || 0;
            return `Submitted ${direction} order for ${quantity} ${symbol} to IBKR.`;
        }
        case "ibkr_order_status": {
            const symbol = action.symbol || "symbol";
            const status = (action.status || "unknown").toString();
            const fillCount = Number.isFinite(action.fill_count) ? action.fill_count : action.filled ? 1 : 0;
            const filled = Boolean(action.filled);
            const orderId = action.order_id ?? "?";
            const fillLabel = filled ? "filled" : `pending (${fillCount} fills)`;
            const quantity = Number.isFinite(action.filled_quantity) ? Math.abs(action.filled_quantity) : null;
            const avgPrice = typeof action.avg_price === "number" && Number.isFinite(action.avg_price)
                ? formatCurrency(action.avg_price)
                : null;
            const details = [];
            if (quantity !== null) {
                details.push(`${formatNumber(quantity, 0)} shares`);
            }
            if (avgPrice) {
                details.push(`@ ${avgPrice}`);
            }
            const suffix = details.length ? ` (${details.join(", ")})` : "";
            return `Order ${orderId} for ${symbol} ${fillLabel}${suffix} [status: ${status}].`;
        }
        case "ibkr_execution_recorded": {
            const symbol = action.symbol || "symbol";
            const executionPrice = typeof action.execution_price === "number" && Number.isFinite(action.execution_price)
                ? formatCurrency(action.execution_price)
                : "latest IBKR fill";
            const previousPrice = typeof action.previous_price === "number" && Number.isFinite(action.previous_price)
                ? formatCurrency(action.previous_price)
                : null;
            const quantity = Number.isFinite(action.execution_filled_quantity)
                ? formatNumber(Math.abs(action.execution_filled_quantity), 0)
                : null;
            const parts = [];
            if (quantity) {
                parts.push(`${quantity} shares`);
            }
            if (previousPrice) {
                parts.push(`sim ${previousPrice}`);
            }
            const extra = parts.length ? ` (${parts.join(", ")})` : "";
            const status = action.status ? ` [${action.status}]` : "";
            return `Recorded IBKR fill for ${symbol}: ${executionPrice}${extra}.${status}`;
        }
        case "polygon_quote_fetch": {
            const status = action.status || "unknown";
            const count = Number.isFinite(action.symbol_count) ? Number(action.symbol_count) : null;
            const reason = action.reason ? ` (${action.reason})` : "";
            const countLabel = count !== null ? `${formatNumber(count, 0)} symbols` : "symbols";
            return `Fetching Polygon quotes for ${countLabel}: ${status}${reason}.`;
        }
        case "polygon_quote": {
            const symbol = action.symbol || "symbol";
            const status = action.status || "unknown";
            if (status === "ok") {
                const price = typeof action.price === "number" && Number.isFinite(action.price)
                    ? formatCurrency(action.price)
                    : "n/a";
                const timestamp = action.timestamp ? formatDateTime(action.timestamp) : null;
                const tsLabel = timestamp ? ` @ ${timestamp}` : "";
                return `Polygon quote ${symbol}: ${price}${tsLabel}.`;
            }
            const message = action.message ? ` (${action.message})` : "";
            return `Polygon quote ${symbol} ${status}.${message}`;
        }
        case "ibkr_execution_skipped":
            return "IBKR execution skipped.";
        case "ibkr_execution_failed":
            return `IBKR execution failed: ${action.message || "Unknown error"}.`;
        case "ibkr_disconnect":
            return "Disconnected from IBKR.";
        default:
            return JSON.stringify(action);
    }
}

function setPaperStatus(message, level = "info") {
    if (!momentumPaperElements.status) {
        return;
    }
    momentumPaperElements.status.textContent = message;
    momentumPaperElements.status.className = `status ${level}`;
}

function setLiveStatus(message, level = "info") {
    if (!momentumLiveElements.status) {
        return;
    }
    momentumLiveElements.status.textContent = message;
    momentumLiveElements.status.className = `status ${level}`;
}

function disableLiveStartButton(disabled) {
    if (momentumLiveElements.startButton) {
        momentumLiveElements.startButton.disabled = disabled;
        momentumLiveElements.startButton.classList.toggle("button--disabled", disabled);
    }
}

function disableLiveStopButton(disabled) {
    if (momentumLiveElements.stopButton) {
        momentumLiveElements.stopButton.disabled = disabled;
        momentumLiveElements.stopButton.classList.toggle("button--disabled", disabled);
    }
}

function startMomentumLivePolling() {
    stopMomentumLivePolling();
    momentumLiveStatusTimer = window.setInterval(refreshMomentumLiveStatus, MOMENTUM_LIVE_STATUS_POLL_MS);
    momentumLiveHistoryTimer = window.setInterval(refreshMomentumLiveHistory, MOMENTUM_LIVE_HISTORY_POLL_MS);
}

function stopMomentumLivePolling() {
    if (momentumLiveStatusTimer) {
        window.clearInterval(momentumLiveStatusTimer);
        momentumLiveStatusTimer = null;
    }
    if (momentumLiveHistoryTimer) {
        window.clearInterval(momentumLiveHistoryTimer);
        momentumLiveHistoryTimer = null;
    }
}

async function refreshMomentumLiveStatus() {
    try {
        const response = await fetch("/api/momentum/live/status", { cache: "no-store" });
        if (!response.ok) {
            throw new Error(await readErrorMessage(response));
        }
        const data = await response.json();
        renderMomentumLiveStatus(data);
    } catch (error) {
        console.warn("Failed to refresh live status", error);
        setLiveStatus(error?.message || "Unable to refresh live status.", "warning");
    }
}

async function refreshMomentumLiveHistory() {
    try {
        const response = await fetch("/api/momentum/live/history", { cache: "no-store" });
        if (!response.ok) {
            throw new Error(await readErrorMessage(response));
        }
        const data = await response.json();
        renderMomentumLiveHistory(data);
    } catch (error) {
        console.warn("Failed to refresh live history", error);
    }
}

function disablePaperRunButton(disabled) {
    if (momentumPaperElements.runButton) {
        momentumPaperElements.runButton.disabled = disabled;
    }
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
    updateScanExportAvailability(null);
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
    updateScanExportAvailability(null);

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
    updateScanExportAvailability(result);
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
            const marketCap = candidate.market_cap === null || candidate.market_cap === undefined
                ? "–"
                : formatCurrency(candidate.market_cap);
            const dollarVolume = candidate.monthly_dollar_volume === null || candidate.monthly_dollar_volume === undefined
                ? "–"
                : formatCurrency(candidate.monthly_dollar_volume);
            const rsPercentile = candidate.rs_percentile === null || candidate.rs_percentile === undefined
                ? "–"
                : formatNumber(candidate.rs_percentile, 1);
            const dailyDistance = candidate.daily_breakout_distance_pct === null || candidate.daily_breakout_distance_pct === undefined
                ? "–"
                : formatPercent(candidate.daily_breakout_distance_pct, 2);
            const weeklyDistance = candidate.weekly_breakout_distance_pct === null || candidate.weekly_breakout_distance_pct === undefined
                ? "–"
                : formatPercent(candidate.weekly_breakout_distance_pct, 2);
            return `
                <tr>
                    <td>${candidate.symbol || ""}</td>
                    <td>${formatCurrency(candidate.close_price)}</td>
                    <td>${marketCap}</td>
                    <td>${dollarVolume}</td>
                    <td>${rsPercentile}</td>
                    <td>${candidate.liquidity_pass ? "Yes" : "No"}</td>
                    <td>${candidate.uptrend_breakout_pass ? "Yes" : "No"}</td>
                    <td>${candidate.higher_lows_pass ? "Yes" : "No"}</td>
                    <td>${candidate.volume_contraction_pass ? "Yes" : "No"}</td>
                    <td>${dailyDistance}</td>
                    <td>${weeklyDistance}</td>
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

function updateScanExportAvailability(result) {
    const button = elements.scanExportButton;
    if (!button) {
        return;
    }
    const symbols = collectCandidateSymbols(result?.candidates);
    const hasCandidates = symbols.length > 0;
    button.disabled = !hasCandidates;
    button.dataset.symbolCount = hasCandidates ? String(symbols.length) : "";
    if (!hasCandidates && button.dataset.originalLabel) {
        button.textContent = button.dataset.originalLabel;
    }
}

function collectCandidateSymbols(candidates) {
    if (!Array.isArray(candidates)) {
        return [];
    }
    const unique = new Set();
    candidates.forEach((candidate) => {
        const symbol = typeof candidate?.symbol === "string" ? candidate.symbol.trim().toUpperCase() : "";
        if (symbol) {
            unique.add(symbol);
        }
    });
    return Array.from(unique);
}

function buildScanWatchlistName(result) {
    const timeframeLabel = formatStrategyName(result?.timeframe) || "Scan";
    return `VCP ${timeframeLabel} Candidates`;
}

function buildScanExportPayload(result) {
    const symbols = collectCandidateSymbols(result?.candidates);
    if (!symbols.length) {
        throw new Error("No breakout candidates available to export.");
    }
    const payload = { symbols };
    if (typeof result?.timeframe === "string" && result.timeframe.trim()) {
        payload.timeframe = result.timeframe;
    }
    const watchlistName = buildScanWatchlistName(result);
    if (watchlistName) {
        payload.watchlist_name = watchlistName;
    }
    return payload;
}

function sanitizeFilename(name, fallbackBase = "watchlist") {
    let base = typeof name === "string" ? name.trim() : "";
    if (!base) {
        base = fallbackBase;
    }
    if (base.toLowerCase().endsWith(".csv")) {
        base = base.slice(0, -4);
    }
    const sanitized = base.replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
    return `${sanitized || fallbackBase}.csv`;
}

function extractFilenameFromDisposition(headerValue, fallbackBase = "watchlist") {
    if (typeof headerValue !== "string" || !headerValue.trim()) {
        return sanitizeFilename("", fallbackBase);
    }
    const starMatch = headerValue.match(/filename\*=([^;]+)/i);
    if (starMatch && starMatch[1]) {
        let value = starMatch[1].trim();
        if (value.toLowerCase().startsWith("utf-8''")) {
            value = value.slice(7);
        }
        try {
            value = decodeURIComponent(value);
        } catch (error) {
            // ignore decode failures
        }
        return sanitizeFilename(value, fallbackBase);
    }
    const plainMatch = headerValue.match(/filename=([^;]+)/i);
    if (plainMatch && plainMatch[1]) {
        let value = plainMatch[1].trim();
        if (value.startsWith("\"") && value.endsWith("\"")) {
            value = value.slice(1, -1);
        }
        return sanitizeFilename(value, fallbackBase);
    }
    return sanitizeFilename("", fallbackBase);
}

async function handleScanExport(event) {
    if (event) {
        event.preventDefault();
    }
    const button = elements.scanExportButton;
    if (!button || button.disabled) {
        return;
    }
    if (!latestScanResult) {
        setScanStatus("Run the scan to populate candidates before exporting.", "warning");
        return;
    }

    let payload;
    try {
        payload = buildScanExportPayload(latestScanResult);
    } catch (error) {
        setScanStatus(error?.message || "No candidates available to export.", "warning");
        return;
    }

    const originalLabel = button.dataset.originalLabel || button.textContent?.trim() || "Export to IBKR CSV";
    button.disabled = true;
    button.textContent = "Exporting…";

    const fallbackBase = payload.watchlist_name || "vcp_scan";

    try {
        const response = await fetch("/api/vcp/scan/export", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorMessage = await readErrorMessage(response);
            throw new Error(errorMessage);
        }

        const blob = await response.blob();
        const filename = extractFilenameFromDisposition(response.headers.get("Content-Disposition"), fallbackBase);
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = filename;
        document.body.append(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);

        const exportedCount = payload.symbols.length;
        setScanStatus(
            `Exported ${exportedCount} candidate${exportedCount === 1 ? "" : "s"} to IBKR CSV.`,
            "success",
        );
    } catch (error) {
        console.error("Failed to export IBKR watchlist", error);
        setScanStatus(error?.message || "Failed to export watchlist.", "error");
    } finally {
        button.disabled = false;
        button.textContent = originalLabel;
    }
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

function setupUniverseControls() {
    if (elements.useNasdaqButton && !elements.useNasdaqButton.dataset.bound) {
        elements.useNasdaqButton.addEventListener("click", handleUseNasdaqUniverse);
        elements.useNasdaqButton.dataset.bound = "true";
    }
    if (elements.useSnp100Button && !elements.useSnp100Button.dataset.bound) {
        elements.useSnp100Button.addEventListener("click", handleUseSnp100Universe);
        elements.useSnp100Button.dataset.bound = "true";
    }
    if (elements.universeImportInput && !elements.universeImportInput.dataset.bound) {
        elements.universeImportInput.addEventListener("change", handleUniverseImport);
        elements.universeImportInput.dataset.bound = "true";
    }
}

function setUniverseMessage(message, level = "info") {
    const helper = elements.universeMessage;
    if (!helper) {
        return;
    }
    if (!message) {
        helper.textContent = "";
        helper.dataset.level = "";
        helper.hidden = true;
        return;
    }
    helper.textContent = message;
    helper.dataset.level = level;
    helper.hidden = false;
}

function updateAvailableSymbolOptions(symbols, options = {}) {
    const select = elements.availableSymbolsSelect;
    if (!select) {
        currentUniverseSymbols = [];
        return;
    }

    const uniqueSymbols = Array.isArray(symbols)
        ? Array.from(
            new Set(
                symbols
                    .map((symbol) => (typeof symbol === "string" ? symbol.trim().toUpperCase() : ""))
                    .filter(Boolean),
            ),
        )
        : [];

    select.innerHTML = "";
    currentUniverseSymbols = uniqueSymbols;

    const selectAll = Boolean(options.selectAll);
    const selectedList = Array.isArray(options.selected)
        ? options.selected.map((symbol) => symbol.toUpperCase())
        : [];
    const selectedSet = new Set(selectAll ? uniqueSymbols : selectedList);

    uniqueSymbols.forEach((symbol) => {
        const option = document.createElement("option");
        option.value = symbol;
        option.textContent = symbol;
        if (selectAll || selectedSet.has(symbol) || (!selectAll && !selectedSet.size && DEFAULT_SYMBOLS.includes(symbol))) {
            option.selected = true;
        }
        select.append(option);
    });

    if (options.source) {
        select.dataset.source = options.source;
    }
}

function applyUniverseSymbols(symbols, { selectAll = false, source = "cache" } = {}) {
    updateAvailableSymbolOptions(symbols, { selectAll });
    currentUniverseSource = source;
    if (elements.availableSymbolsSelect) {
        elements.availableSymbolsSelect.dataset.source = source;
    }
}

async function handleUseNasdaqUniverse(event) {
    if (event) {
        event.preventDefault();
    }

    const button = elements.useNasdaqButton;
    if (button && button.disabled) {
        return;
    }

    setUniverseMessage("Loading NASDAQ universe…", "info");

    let restoreLabel = "";
    if (button) {
        restoreLabel = button.dataset.originalLabel || button.textContent?.trim() || "Use NASDAQ Universe";
        button.dataset.originalLabel = restoreLabel;
        button.disabled = true;
        button.textContent = "Loading…";
    }

    try {
        const response = await fetch("/api/universe/nasdaq");
        if (!response.ok) {
            const detail = await readErrorMessage(response);
            throw new Error(detail || `Request failed with status ${response.status}`);
        }
        const payload = await response.json();
        const symbols = Array.isArray(payload?.symbols) ? payload.symbols : [];
        if (!symbols.length) {
            throw new Error("NASDAQ universe returned no symbols.");
        }
        applyUniverseSymbols(symbols, { selectAll: true, source: "nasdaq" });
        if (elements.extraSymbolsInput) {
            elements.extraSymbolsInput.value = "";
        }
        const missingCount = Array.isArray(payload?.missing) ? payload.missing.length : 0;
        const warnings = Array.isArray(payload?.warnings) ? payload.warnings : [];
        const summaryParts = [`Loaded ${symbols.length} NASDAQ symbol${symbols.length === 1 ? "" : "s"}.`];
        if (missingCount) {
            summaryParts.push(
                `${missingCount} symbol${missingCount === 1 ? "" : "s"} will be fetched from Polygon when the test runs.`,
            );
        }
        warnings.forEach((warning) => {
            if (typeof warning === "string" && warning.trim()) {
                summaryParts.push(warning.trim());
            }
        });
        const level = missingCount > 0 || warnings.length > 0 ? "warning" : "success";
        setUniverseMessage(summaryParts.join(" "), level);
    } catch (error) {
        console.error("Failed to load NASDAQ universe", error);
        setUniverseMessage(error?.message || "Unable to load NASDAQ universe.", "error");
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = restoreLabel || "Use NASDAQ Universe";
        }
    }
}

async function handleUseSnp100Universe(event) {
    if (event) {
        event.preventDefault();
    }

    const button = elements.useSnp100Button;
    if (button && button.disabled) {
        return;
    }

    setUniverseMessage("Loading S&P 100 universe…", "info");

    let restoreLabel = "";
    if (button) {
        restoreLabel = button.dataset.originalLabel || button.textContent?.trim() || "Use S&P 100 Universe";
        button.dataset.originalLabel = restoreLabel;
        button.disabled = true;
        button.textContent = "Loading…";
    }

    try {
        const response = await fetch("/api/universe/snp100");
        if (!response.ok) {
            const detail = await readErrorMessage(response);
            throw new Error(detail || `Request failed with status ${response.status}`);
        }
        const payload = await response.json();
        const symbols = Array.isArray(payload?.symbols) ? payload.symbols : [];
        if (!symbols.length) {
            throw new Error("S&P 100 universe returned no symbols.");
        }
        applyUniverseSymbols(symbols, { selectAll: true, source: "snp100" });
        if (elements.extraSymbolsInput) {
            elements.extraSymbolsInput.value = "";
        }
        const missingCount = Array.isArray(payload?.missing) ? payload.missing.length : 0;
        const warnings = Array.isArray(payload?.warnings) ? payload.warnings : [];
        const summaryParts = [`Loaded ${symbols.length} S&P 100 symbol${symbols.length === 1 ? "" : "s"}.`];
        if (missingCount) {
            summaryParts.push(
                `${missingCount} symbol${missingCount === 1 ? "" : "s"} will be fetched from Polygon when the test runs.`,
            );
        }
        warnings.forEach((warning) => {
            if (typeof warning === "string" && warning.trim()) {
                summaryParts.push(warning.trim());
            }
        });
        const level = missingCount > 0 || warnings.length > 0 ? "warning" : "success";
        setUniverseMessage(summaryParts.join(" "), level);
    } catch (error) {
        console.error("Failed to load S&P 100 universe", error);
        setUniverseMessage(error?.message || "Unable to load S&P 100 universe.", "error");
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = restoreLabel || "Use S&P 100 Universe";
        }
    }
}

async function handleUniverseImport(event) {
    const input = event?.target;
    if (!input || input.files?.length !== 1) {
        return;
    }

    const file = input.files[0];
    setUniverseMessage(`Importing symbols from ${file.name}…`, "info");

    try {
        const text = await file.text();
        const symbols = parseUniverseCsv(text);
        if (!symbols.length) {
            throw new Error("No symbols were detected in the uploaded CSV.");
        }
        applyUniverseSymbols(symbols, { selectAll: true, source: "import" });
        if (elements.extraSymbolsInput) {
            elements.extraSymbolsInput.value = "";
        }

        const activeForm = momentumElements.form || elements.form || null;
        let storePath = "";
        if (activeForm) {
            const storeInput = activeForm.querySelector("[name='store_path']");
            if (storeInput && typeof storeInput.value === "string") {
                storePath = storeInput.value.trim();
            }
        }

        setUniverseMessage(`Imported ${symbols.length} symbol${symbols.length === 1 ? "" : "s"} from ${file.name}. Checking historical coverage…`, "info");

        try {
            const fetchPayload = {
                symbols,
                lookback_years: 3.0,
            };
            if (storePath) {
                fetchPayload.store_path = storePath;
            }
            const response = await fetch("/api/universe/import/fetch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(fetchPayload),
            });
            if (!response.ok) {
                const detail = await readErrorMessage(response);
                throw new Error(detail || `Request failed with status ${response.status}`);
            }
            const payload = await response.json();
            const fetched = Array.isArray(payload?.fetched) ? payload.fetched : [];
            const missing = Array.isArray(payload?.missing) ? payload.missing : [];
            const warnings = Array.isArray(payload?.warnings) ? payload.warnings : [];
            const summaryParts = [
                `Imported ${symbols.length} symbol${symbols.length === 1 ? "" : "s"} from ${file.name}.`,
            ];
            if (fetched.length) {
                summaryParts.push(
                    `Fetched ${fetched.length} symbol${fetched.length === 1 ? "" : "s"} from Polygon to fill missing history.`,
                );
            } else {
                summaryParts.push("All symbols already had cached history.");
            }
            if (missing.length) {
                summaryParts.push(
                    `${missing.length} symbol${missing.length === 1 ? "" : "s"} still lack historical bars. They will be fetched automatically when tests run.`,
                );
            }
            warnings.forEach((warning) => {
                if (typeof warning === "string" && warning.trim()) {
                    summaryParts.push(warning.trim());
                }
            });
            const level = missing.length || warnings.length ? "warning" : "success";
            setUniverseMessage(summaryParts.join(" "), level);
        } catch (fetchError) {
            console.error("Failed to backfill universe data", fetchError);
            setUniverseMessage(
                fetchError?.message || "Imported symbols but failed to ensure historical coverage.",
                "error",
            );
        }
    } catch (error) {
        console.error("Failed to import universe CSV", error);
        setUniverseMessage(error?.message || "Unable to import symbols from CSV.", "error");
    } finally {
        input.value = "";
    }
}

function parseUniverseCsv(csvText) {
    if (typeof csvText !== "string" || !csvText.trim()) {
        return [];
    }
    const lines = csvText.split(/\r?\n/);
    const symbols = [];
    lines.forEach((line) => {
        const trimmed = line.trim();
        if (!trimmed) {
            return;
        }
        const columns = splitCsvLine(trimmed);
        if (!columns.length) {
            return;
        }
        let candidate = columns[0];
        if (columns[0].toUpperCase() === "SYM" && columns.length > 1) {
            candidate = columns[1];
        } else if (columns[0].toUpperCase() === "SYMBOL" && columns.length > 1) {
            candidate = columns[1];
        }
        if (typeof candidate !== "string") {
            return;
        }
        let token = candidate.split("·")[0];
        token = token.split(" ")[0];
        token = token.replace(/[^A-Za-z0-9._-]/g, "");
        token = token.trim().toUpperCase();
        if (token && token !== "SYM" && token !== "SYMBOL" && token !== "TYPE") {
            symbols.push(token);
        }
    });
    return Array.from(new Set(symbols));
}

function splitCsvLine(line) {
    const result = [];
    let current = "";
    let insideQuotes = false;

    for (let i = 0; i < line.length; i += 1) {
        const char = line[i];
        if (char === '"') {
            if (insideQuotes && line[i + 1] === '"') {
                current += '"';
                i += 1;
            } else {
                insideQuotes = !insideQuotes;
            }
            continue;
        }
        if (char === "," && !insideQuotes) {
            result.push(current.trim());
            current = "";
        } else {
            current += char;
        }
    }
    result.push(current.trim());
    return result;
}

async function loadAvailableSymbols() {
    if (!elements.availableSymbolsSelect) {
        return;
    }
    setUniverseMessage("Loading cached symbols…", "info");
    let symbols = [...DEFAULT_SYMBOLS];
    let messageLevel = "info";
    let messageText = "";
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
        messageLevel = "warning";
        messageText = "Unable to load cached symbols; using defaults.";
    }

    applyUniverseSymbols(symbols, { source: "cache" });
    if (!messageText && symbols.length) {
        messageText = `Loaded ${symbols.length} cached symbol${symbols.length === 1 ? "" : "s"}.`;
    }
    setUniverseMessage(messageText, messageLevel);
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

function gatherExplicitSymbols(formData) {
    const selected = elements.availableSymbolsSelect
        ? Array.from(elements.availableSymbolsSelect.selectedOptions).map((option) => option.value.toUpperCase()).filter(Boolean)
        : [];
    const manualRaw = (formData.get("extra_symbols") || "").toString();
    const manual = manualRaw
        .split(",")
        .map((value) => value.trim().toUpperCase())
        .filter(Boolean);
    const combined = [...selected, ...manual];
    const unique = Array.from(new Set(combined));
    const explicit = selected.length > 0 || manualRaw.trim().length > 0;
    return { symbols: unique, explicit };
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
