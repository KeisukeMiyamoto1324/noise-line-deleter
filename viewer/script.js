const state = {
  data: null,
  filteredSamples: [],
  selectedIndex: 0,
  threshold: 0.5,
  predictedOnly: false,
  showActual: true,
};

const elements = {
  datasetLabel: document.querySelector("#datasetLabel"),
  sampleSearch: document.querySelector("#sampleSearch"),
  thresholdInput: document.querySelector("#thresholdInput"),
  thresholdValue: document.querySelector("#thresholdValue"),
  predictedOnlyInput: document.querySelector("#predictedOnlyInput"),
  actualInput: document.querySelector("#actualInput"),
  sampleList: document.querySelector("#sampleList"),
  sampleMeta: document.querySelector("#sampleMeta"),
  documentTitle: document.querySelector("#documentTitle"),
  lineCount: document.querySelector("#lineCount"),
  predictedCount: document.querySelector("#predictedCount"),
  actualCount: document.querySelector("#actualCount"),
  lineList: document.querySelector("#lineList"),
};

async function main() {
  const response = await fetch("../samples/predictions.json");
  state.data = await response.json();
  state.filteredSamples = state.data.samples;
  elements.datasetLabel.textContent = `${state.data.metadata.dataset_name} / ${state.data.metadata.split}`;
  bindEvents();
  renderSampleList();
  renderDocument();
}

function bindEvents() {
  elements.sampleSearch.addEventListener("input", () => {
    const query = elements.sampleSearch.value.trim().toLowerCase();
    state.filteredSamples = state.data.samples.filter((sample) => matchesSample(sample, query));
    state.selectedIndex = 0;
    renderSampleList();
    renderDocument();
  });

  elements.thresholdInput.addEventListener("input", () => {
    state.threshold = Number(elements.thresholdInput.value);
    elements.thresholdValue.textContent = state.threshold.toFixed(2);
    renderSampleList();
    renderDocument();
  });

  elements.predictedOnlyInput.addEventListener("change", () => {
    state.predictedOnly = elements.predictedOnlyInput.checked;
    renderDocument();
  });

  elements.actualInput.addEventListener("change", () => {
    state.showActual = elements.actualInput.checked;
    renderDocument();
  });
}

function matchesSample(sample, query) {
  if (query === "") {
    return true;
  }
  return `${sample.sample_index} ${sample.document_id} ${sample.text_file}`.toLowerCase().includes(query);
}

function renderSampleList() {
  elements.sampleList.replaceChildren();
  const fragment = document.createDocumentFragment();

  state.filteredSamples.forEach((sample, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `sample-button${index === state.selectedIndex ? " is-active" : ""}`;
    button.addEventListener("click", () => {
      state.selectedIndex = index;
      renderSampleList();
      renderDocument();
    });

    const label = document.createElement("div");
    const name = document.createElement("div");
    name.className = "sample-name";
    name.textContent = `#${String(sample.sample_index).padStart(3, "0")} ${sample.document_id}`;

    const detail = document.createElement("div");
    detail.className = "sample-detail";
    detail.textContent = `${sample.line_count} lines / ${sample.text_file}`;
    label.append(name, detail);

    const count = document.createElement("div");
    count.className = "sample-count";
    count.textContent = String(countPredicted(sample));

    button.append(label, count);
    fragment.append(button);
  });

  elements.sampleList.append(fragment);
}

function renderDocument() {
  const sample = state.filteredSamples[state.selectedIndex];
  elements.lineList.replaceChildren();

  if (!sample) {
    elements.sampleMeta.textContent = "0 samples";
    elements.documentTitle.textContent = "No sample";
    elements.lineCount.textContent = "0";
    elements.predictedCount.textContent = "0";
    elements.actualCount.textContent = "0";
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No matching samples";
    elements.lineList.append(empty);
    return;
  }

  const predictedCount = countPredicted(sample);
  elements.sampleMeta.textContent = `sample ${sample.sample_index} / ${sample.text_file}`;
  elements.documentTitle.textContent = sample.document_id;
  elements.lineCount.textContent = String(sample.line_count);
  elements.predictedCount.textContent = String(predictedCount);
  elements.actualCount.textContent = String(sample.actual_delete_count);

  const fragment = document.createDocumentFragment();
  sample.lines.forEach((line) => {
    const predicted = line.noise_probability >= state.threshold;
    if (state.predictedOnly && !predicted) {
      return;
    }

    const row = document.createElement("article");
    row.className = [
      "line-row",
      predicted ? "is-predicted" : "",
      state.showActual && line.actual_delete ? "is-actual" : "",
    ]
      .filter(Boolean)
      .join(" ");

    const number = document.createElement("div");
    number.className = "line-number";
    number.textContent = String(line.number);

    const text = document.createElement("div");
    text.className = "line-text";
    text.textContent = line.text === "" ? " " : line.text;

    const probability = document.createElement("div");
    probability.className = "line-probability";
    probability.textContent = line.noise_probability.toFixed(3);

    row.append(number, text, probability);
    fragment.append(row);
  });

  elements.lineList.append(fragment);
}

function countPredicted(sample) {
  return sample.lines.reduce((count, line) => count + Number(line.noise_probability >= state.threshold), 0);
}

main().catch((error) => {
  elements.documentTitle.textContent = "Load error";
  elements.lineList.textContent = error instanceof Error ? error.message : String(error);
});
