/**
 * Ghar ka Daam (घर का दाम) - Hierarchical Valuation Frontend
 * Supports State -> City -> Area cascading selection, 3-tier confidence ladder,
 * multi-level benchmarks, and local area comparison.
 */

let METADATA = null;
let CURRENT_AREAS = [];

// DOM Elements
const stateSelect = document.getElementById("state-select");
const locationSelect = document.getElementById("location-select");
const areaSelect = document.getElementById("area-select");
const areaCountTag = document.getElementById("area-count-tag");
const areaFilterInput = document.getElementById("area-filter-input");

const areaInput = document.getElementById("area-input");
const areaSlider = document.getElementById("area-slider");
const areaValDisplay = document.getElementById("area-val-display");
const priceForm = document.getElementById("price-form");
const calculateBtn = document.getElementById("calculate-btn");
const btnText = document.getElementById("btn-text");
const btnSpinner = document.getElementById("btn-spinner");

// Result Elements
const resPrice = document.getElementById("res-price");
const resPriceWords = document.getElementById("res-price-words");
const rangeLowLabel = document.getElementById("range-low-label");
const rangeHighLabel = document.getElementById("range-high-label");
const rangePin = document.getElementById("range-pin");

const confidenceBadge = document.getElementById("confidence-badge");
const confidenceNote = document.getElementById("confidence-note");
const valuationTitle = document.getElementById("valuation-title");
const latencyTag = document.getElementById("latency-tag");

// Benchmark Elements
const resPpsf = document.getElementById("res-ppsf");
const resAreaName = document.getElementById("res-area-name");
const resCityMedian = document.getElementById("res-city-median");
const resBenchmarkDiff = document.getElementById("res-benchmark-diff");
const resStateMedian = document.getElementById("res-state-median");
const resStateDiff = document.getElementById("res-state-diff");

// Comparison Elements
const tabAreasBtn = document.getElementById("tab-areas-btn");
const tabMetrosBtn = document.getElementById("tab-metros-btn");
const compareSubtext = document.getElementById("compare-subtext");
const compareAreasList = document.getElementById("compare-areas-list");
const compareMetrosList = document.getElementById("compare-metros-list");

// Blueprint Elements
const bpLivingDim = document.getElementById("bp-living-dim");
const bpMasterDim = document.getElementById("bp-master-dim");
const bpBed2Dim = document.getElementById("bp-bed2-dim");
const bpKitchenDim = document.getElementById("bp-kitchen-dim");
const bpBathDim = document.getElementById("bp-bath-dim");

// History Drawer Elements
const historyToggleBtn = document.getElementById("history-toggle-btn");
const historyDrawer = document.getElementById("history-drawer");
const closeHistoryBtn = document.getElementById("close-history-btn");
const historyList = document.getElementById("history-list");
const historyCount = document.getElementById("history-count");
const clearHistoryBtn = document.getElementById("clear-history-btn");


/** Initialize page: load hierarchical metadata */
async function init() {
    try {
        const resp = await fetch("/metadata");
        METADATA = await resp.json();

        populateStates(METADATA.states);
        setupEventListeners();
        renderHistory();

        // Default selection: Uttar Pradesh -> Greater Noida -> Sector 1
        selectDefaultHierarchy("Uttar Pradesh", "greater-noida", "sector 1");
    } catch (err) {
        console.error("Failed to initialize metadata:", err);
    }
}

/** Populate states dropdown */
function populateStates(states) {
    stateSelect.innerHTML = '<option value="" disabled selected>Select State...</option>';
    states.forEach(state => {
        const opt = document.createElement("option");
        opt.value = state;
        opt.textContent = state;
        stateSelect.appendChild(opt);
    });
}

/** Populate cities dropdown based on chosen state */
function populateCities(state) {
    const cities = METADATA.state_to_cities[state] || [];
    locationSelect.innerHTML = '<option value="" disabled selected>Select City...</option>';
    cities.forEach(city => {
        const opt = document.createElement("option");
        opt.value = city;
        opt.textContent = formatCityName(city);
        locationSelect.appendChild(opt);
    });

    if (cities.length > 0) {
        locationSelect.value = cities[0];
        populateAreas(state, cities[0]);
    }
}

/** Populate areas dropdown based on chosen state & city */
function populateAreas(state, city) {
    const cityData = METADATA.hierarchy?.[state]?.[city] || {};
    CURRENT_AREAS = cityData.areas || [];

    renderAreaOptions(CURRENT_AREAS);
    areaCountTag.textContent = `${CURRENT_AREAS.length} localities`;
}

/** Renders area <option> elements, supporting search filtering */
function renderAreaOptions(areasList) {
    areaSelect.innerHTML = "";
    
    // Default option: Whole City
    const wholeOpt = document.createElement("option");
    wholeOpt.value = "other";
    wholeOpt.textContent = "⭐ Whole City / General Area";
    areaSelect.appendChild(wholeOpt);

    areasList.forEach(a => {
        if (a.id === "other") return;
        const opt = document.createElement("option");
        opt.value = a.id;
        opt.textContent = `${a.name} (${a.count} listings)`;
        areaSelect.appendChild(opt);
    });
}

/** Helper: Set default initial hierarchy */
function selectDefaultHierarchy(state, city, area) {
    if (METADATA.states.includes(state)) {
        stateSelect.value = state;
        populateCities(state);
        locationSelect.value = city;
        populateAreas(state, city);
        if (area) areaSelect.value = area;
        triggerValuation();
    }
}

function formatCityName(str) {
    return str.split("-").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

/** Set up event listeners */
function setupEventListeners() {
    // 1. State change -> refill cities
    stateSelect.addEventListener("change", (e) => {
        populateCities(e.target.value);
    });

    // 2. City change -> refill areas
    locationSelect.addEventListener("change", (e) => {
        const state = stateSelect.value;
        populateAreas(state, e.target.value);
    });

    // 3. Area search filter
    areaFilterInput.addEventListener("input", (e) => {
        const q = e.target.value.toLowerCase().trim();
        if (!q) {
            renderAreaOptions(CURRENT_AREAS);
        } else {
            const filtered = CURRENT_AREAS.filter(a => a.name.toLowerCase().includes(q) || a.id.toLowerCase().includes(q));
            renderAreaOptions(filtered);
        }
    });

    // Area slider & input synchronization
    areaInput.addEventListener("input", () => {
        const val = areaInput.value;
        areaSlider.value = val;
        areaValDisplay.textContent = `${Number(val).toLocaleString()} sq ft`;
        updateBlueprint();
    });

    areaSlider.addEventListener("input", () => {
        const val = areaSlider.value;
        areaInput.value = val;
        areaValDisplay.textContent = `${Number(val).toLocaleString()} sq ft`;
        updateBlueprint();
    });

    // Preset area chips
    document.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", () => {
            document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
            chip.classList.add("active");
            const area = chip.dataset.area;
            areaInput.value = area;
            areaSlider.value = area;
            areaValDisplay.textContent = `${Number(area).toLocaleString()} sq ft`;
            updateBlueprint();
        });
    });

    // BHK / Bath radio pills
    document.querySelectorAll("input[name='bhk'], input[name='bathrooms']").forEach(input => {
        input.addEventListener("change", updateBlueprint);
    });

    // Tab buttons for comparison
    tabAreasBtn.addEventListener("click", () => {
        tabAreasBtn.classList.add("active");
        tabMetrosBtn.classList.remove("active");
        compareAreasList.classList.remove("hidden");
        compareMetrosList.classList.add("hidden");
        compareSubtext.textContent = "How this property's price compares to other localities in the same city";
    });

    tabMetrosBtn.addEventListener("click", () => {
        tabMetrosBtn.classList.add("active");
        tabAreasBtn.classList.remove("active");
        compareMetrosList.classList.remove("hidden");
        compareAreasList.classList.add("hidden");
        compareSubtext.textContent = "How this property's price compares across India's top metropolitan areas";
    });

    // Form submission
    priceForm.addEventListener("submit", (e) => {
        e.preventDefault();
        triggerValuation();
    });

    // History drawer controls
    historyToggleBtn.addEventListener("click", () => historyDrawer.classList.toggle("hidden"));
    closeHistoryBtn.addEventListener("click", () => historyDrawer.classList.add("hidden"));
    clearHistoryBtn.addEventListener("click", clearHistory);
}

/** Updates the blueprint room dimensions proportionally */
function updateBlueprint() {
    const totalArea = parseFloat(areaInput.value) || 1000;
    const bhk = parseInt(document.querySelector("input[name='bhk']:checked")?.value || "2");
    const baths = parseInt(document.querySelector("input[name='bathrooms']:checked")?.value || "2");

    const livingArea = Math.round(totalArea * 0.38);
    const masterArea = Math.round(totalArea * 0.26);
    const bed2Area = bhk >= 2 ? Math.round(totalArea * 0.18) : 0;
    const kitchenArea = Math.round(totalArea * 0.11);
    const bathArea = Math.round(totalArea * 0.07);

    bpLivingDim.textContent = `~${livingArea} sq ft`;
    bpMasterDim.textContent = `~${masterArea} sq ft`;
    bpKitchenDim.textContent = `~${kitchenArea} sq ft`;
    bpBathDim.textContent = `~${bathArea} sq ft (${baths} Bath)`;

    const bed2El = document.getElementById("bp-bed2");
    if (bhk === 1) {
        bed2El.style.display = "none";
    } else {
        bed2El.style.display = "flex";
        bed2Dim.textContent = `~${bed2Area} sq ft`;
    }
}

/** Triggers hierarchical valuation API */
async function triggerValuation() {
    const formData = new FormData(priceForm);
    const payload = {
        state: formData.get("state"),
        location: formData.get("location"),
        area: formData.get("area") || "other",
        area_sqft: parseFloat(formData.get("area_sqft")),
        bhk: parseFloat(formData.get("bhk")),
        bathrooms: parseFloat(formData.get("bathrooms")),
        Furnishing: formData.get("Furnishing"),
        Transaction: formData.get("Transaction"),
    };

    const floorNo = formData.get("floor_no");
    if (floorNo !== "" && floorNo !== null) payload.floor_no = parseFloat(floorNo);

    const totalFloors = formData.get("total_floors");
    if (totalFloors !== "" && totalFloors !== null) payload.total_floors = parseFloat(totalFloors);

    // UI loading state
    calculateBtn.disabled = true;
    btnSpinner.classList.remove("hidden");
    btnText.textContent = "Valuating...";

    try {
        // Run prediction, area comparison, and metro comparison concurrently
        const [predResp, areaCompResp, metroCompResp] = await Promise.all([
            fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }),
            fetch("/compare-areas", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }),
            fetch("/compare", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }),
        ]);

        if (!predResp.ok) {
            const errData = await predResp.json();
            alert("Error: " + (errData.error || "Prediction request failed"));
            return;
        }

        const predData = await predResp.json();
        updateResultsUI(predData, payload);

        if (areaCompResp.ok) {
            const areaCompData = await areaCompResp.json();
            renderAreaComparison(areaCompData.areas || [], predData.price);
        }

        if (metroCompResp.ok) {
            const metroCompData = await metroCompResp.json();
            renderMetroComparison(metroCompData.comparisons || [], predData.price);
        }

        saveToHistory(predData, payload);
    } catch (err) {
        console.error("Valuation request error:", err);
    } finally {
        calculateBtn.disabled = false;
        btnSpinner.classList.add("hidden");
        btnText.textContent = "Estimate Price & Range ✨";
    }
}

/** Updates main results display */
function updateResultsUI(data, input) {
    resPrice.textContent = `₹ ${data.formatted.price}`;
    resPriceWords.textContent = inrToWords(data.price);

    rangeLowLabel.textContent = `Low: ₹ ${data.formatted.low}`;
    rangeHighLabel.textContent = `High: ₹ ${data.formatted.high}`;

    // Position range pin
    const low = data.low;
    const high = data.high;
    const current = data.price;
    let pinPct = 50;
    if (high > low) {
        pinPct = Math.min(92, Math.max(8, ((current - low) / (high - low)) * 100));
    }
    rangePin.style.left = `${pinPct}%`;

    // Confidence badge
    confidenceBadge.className = `confidence-badge confidence-${data.confidence.toLowerCase()}`;
    if (data.confidence === "High") {
        confidenceBadge.textContent = "🟢 High Confidence";
    } else if (data.confidence === "Medium") {
        confidenceBadge.textContent = "🟡 Medium Confidence";
    } else {
        confidenceBadge.textContent = "⚪ City Fallback";
    }
    confidenceNote.textContent = data.confidence_label;
    valuationTitle.textContent = `${data.area_used !== 'Other' ? data.area_used + ', ' : ''}${data.city}`;

    // Multi-level benchmarks
    resPpsf.textContent = data.formatted.per_sqft;
    resAreaName.textContent = data.area_used !== "Other" ? `${data.area_used} Rate` : "Property Rate";

    const bm = data.benchmarks || {};
    if (bm.city) {
        resCityMedian.textContent = bm.city.median_formatted;
        resBenchmarkDiff.textContent = `${bm.city.diff_pct > 0 ? '+' : ''}${bm.city.diff_pct}% vs ${bm.city.name} Median`;
        resBenchmarkDiff.style.color = bm.city.diff_pct > 0 ? "#b45309" : "#1e5c41";
    }
    if (bm.state) {
        resStateMedian.textContent = bm.state.median_formatted;
        resStateDiff.textContent = `${bm.state.diff_pct > 0 ? '+' : ''}${bm.state.diff_pct}% vs ${bm.state.name} Median`;
        resStateDiff.style.color = bm.state.diff_pct > 0 ? "#b45309" : "#1e5c41";
    }

    latencyTag.textContent = `⚡ ${data.latency_ms} ms`;
}

/** Render neighbouring areas comparison in chosen city */
function renderAreaComparison(areas, currentPrice) {
    compareAreasList.innerHTML = "";
    if (!areas || areas.length === 0) {
        compareAreasList.innerHTML = '<p class="subtext">No other recorded areas in this city with >= 5 listings.</p>';
        return;
    }

    const maxPrice = Math.max(...areas.map(a => a.price));

    areas.forEach(item => {
        const widthPct = Math.round((item.price / maxPrice) * 100);
        const row = document.createElement("div");
        row.className = `compare-row ${item.is_current ? "current-city" : ""}`;

        row.innerHTML = `
            <div class="compare-city-name">
                ${item.area}
                ${item.is_current ? '<span class="current-tag">Selected</span>' : ''}
            </div>
            <div class="compare-bar-wrapper">
                <div class="compare-bar-track">
                    <div class="compare-bar-fill" style="width: ${widthPct}%;"></div>
                </div>
            </div>
            <div>
                <div class="compare-price-val">₹ ${item.formatted}</div>
                <div class="compare-sub-rate">₹ ${item.per_sqft.toLocaleString()}/sq ft • ${item.sample_count} listings</div>
            </div>
        `;
        compareAreasList.appendChild(row);
    });
}

/** Render cross-metro comparison cards */
function renderMetroComparison(comparisons, currentPrice) {
    compareMetrosList.innerHTML = "";
    if (!comparisons || comparisons.length === 0) return;

    const maxPrice = Math.max(...comparisons.map(c => c.price));

    comparisons.forEach(item => {
        const widthPct = Math.round((item.price / maxPrice) * 100);
        const row = document.createElement("div");
        row.className = `compare-row ${item.is_current ? "current-city" : ""}`;

        row.innerHTML = `
            <div class="compare-city-name">
                ${item.city}
                ${item.is_current ? '<span class="current-tag">Selected</span>' : ''}
            </div>
            <div class="compare-bar-wrapper">
                <div class="compare-bar-track">
                    <div class="compare-bar-fill" style="width: ${widthPct}%;"></div>
                </div>
            </div>
            <div>
                <div class="compare-price-val">₹ ${item.formatted}</div>
                <div class="compare-sub-rate">₹ ${item.per_sqft.toLocaleString()}/sq ft</div>
            </div>
        `;
        compareMetrosList.appendChild(row);
    });
}

/** Save prediction to localStorage history */
function saveToHistory(data, input) {
    let history = getHistory();
    const entry = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        state: input.state,
        location: input.location,
        area: input.area,
        area_sqft: input.area_sqft,
        bhk: input.bhk,
        confidence: data.confidence,
        price_formatted: data.formatted.price,
        range: data.formatted.range,
        payload: input,
    };
    history.unshift(entry);
    if (history.length > 20) history.pop();
    localStorage.setItem("ghar_ka_daam_history_v2", JSON.stringify(history));
    renderHistory();
}

function getHistory() {
    try {
        return JSON.parse(localStorage.getItem("ghar_ka_daam_history_v2") || "[]");
    } catch {
        return [];
    }
}

function clearHistory() {
    localStorage.removeItem("ghar_ka_daam_history_v2");
    renderHistory();
}

/** Render history list inside drawer */
function renderHistory() {
    const history = getHistory();
    historyCount.textContent = history.length;
    historyList.innerHTML = "";

    if (history.length === 0) {
        historyList.innerHTML = '<p class="empty-msg">No valuations performed yet.</p>';
        return;
    }

    history.forEach(item => {
        const card = document.createElement("div");
        card.className = "history-card";
        const localityText = item.area && item.area !== 'other' ? `${item.area.toUpperCase()}, ` : '';
        card.innerHTML = `
            <div class="history-card-top">
                <span>${localityText}${formatCityName(item.location)}</span>
                <span class="history-card-price">₹ ${item.price_formatted}</span>
            </div>
            <div class="history-card-sub">
                ${item.state} • ${item.bhk} BHK • ${item.area_sqft} sq ft • [${item.confidence}] • ${item.timestamp}
            </div>
        `;
        card.addEventListener("click", () => {
            loadHistoryItem(item.payload);
            historyDrawer.classList.add("hidden");
        });
        historyList.appendChild(card);
    });
}

/** Load a historical item back into the form */
function loadHistoryItem(payload) {
    stateSelect.value = payload.state;
    populateCities(payload.state);
    locationSelect.value = payload.location;
    populateAreas(payload.state, payload.location);
    areaSelect.value = payload.area || "other";

    areaInput.value = payload.area_sqft;
    areaSlider.value = payload.area_sqft;
    areaValDisplay.textContent = `${payload.area_sqft} sq ft`;

    const bhkRadio = document.querySelector(`input[name='bhk'][value='${payload.bhk}']`);
    if (bhkRadio) bhkRadio.checked = true;

    const bathRadio = document.querySelector(`input[name='bathrooms'][value='${payload.bathrooms}']`);
    if (bathRadio) bathRadio.checked = true;

    document.getElementById("furnishing-select").value = payload.Furnishing;
    document.getElementById("transaction-select").value = payload.Transaction;

    updateBlueprint();
    triggerValuation();
}

/** Converts Rupee amount to readable Indian words */
function inrToWords(num) {
    if (num >= 1e7) {
        const cr = (num / 1e7).toFixed(2);
        return `Approx. ${cr} Crore Rupees`;
    } else if (num >= 1e5) {
        const lac = (num / 1e5).toFixed(2);
        return `Approx. ${lac} Lakh Rupees`;
    }
    return `Approx. ₹ ${Math.round(num).toLocaleString('en-IN')}`;
}

// Kick off initialization on DOM ready
document.addEventListener("DOMContentLoaded", init);
