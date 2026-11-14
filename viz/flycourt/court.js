const form = document.querySelector('#case-form');
const summary = document.querySelector('#case');
const swap = document.querySelector('#swap');
const jury = document.querySelector('#jury');
const verdict = document.querySelector('#verdict');
const evidence = [
  { label: 'Evidence for clearing (sugar)', detail: 'An explanation, alibi, or permission.', demo: 76 },
  { label: 'Evidence for responsibility (bitter)', detail: 'Observations or records supporting the claim.', demo: 46 },
  { label: 'Perceived harm (looming)', detail: 'How serious the alleged consequences were.', demo: 17 },
];
const perspectives = [
  [1.25, .55, .30], [1.12, .75, .48], [.95, .80, .28],
  [.76, 1.15, 1.25], [.65, 1.35, 1.50], [.55, 1.05, 1.65],
  [1.05, .91, .56], [.86, 1.05, .85], [1.20, .60, .40],
  [.93, .68, .68], [.85, 1.12, 1.10], [1.06, .85, .75],
];

const sliders = evidence.map(({ label, detail }, index) => {
  const field = document.createElement('fieldset');
  field.innerHTML = `<label for="sense-${index}">${label} <output for="sense-${index}">0</output></label>
    <input id="sense-${index}" type="range" min="0" max="100" value="0" aria-describedby="hint-${index}">
    <small id="hint-${index}">${detail}</small>`;
  document.querySelector('#evidence').append(field);
  return field.querySelector('input');
});

perspectives.forEach((_, index) => swap.add(new Option(`Juror ${index + 1}`, index)));
let heard = false;

function interpolate(grid, levels) {
  const position = levels.map(level => level * (grid.length - 1));
  const lower = position.map(Math.floor);
  const fraction = position.map((value, axis) => value - lower[axis]);
  const rates = [0, 0];
  for (let mask = 0; mask < 8; mask++) {
    const offsets = lower.map((_, axis) => (mask >> axis) & 1);
    const weight = offsets.reduce((product, step, axis) =>
      product * (step ? fraction[axis] : 1 - fraction[axis]), 1);
    if (!weight) continue;
    const [sugar, bitter, looming] = lower.map((value, axis) => value + offsets[axis]);
    grid[sugar][bitter][looming].forEach((rate, index) => { rates[index] += rate * weight; });
  }
  return rates;
}

const peaks = [0, 1].map(index => Math.max(...responseGrid.intact.flat(2).map(pair => pair[index])));
const labels = { clear: 'Clear', responsible: 'Hold responsible', abstain: 'Abstain' };

function judge(levels, shuffled) {
  const [feed, escape] = interpolate(responseGrid[shuffled ? 'shuffle' : 'intact'], levels);
  const difference = feed / peaks[0] - escape / peaks[1];
  const vote = Math.abs(difference) < 1e-9 ? 'abstain' : difference > 0 ? 'clear' : 'responsible';
  return { feed, escape, vote };
}

function tally(results) {
  const counts = { clear: 0, responsible: 0, abstain: 0 };
  results.forEach(({ vote }) => { counts[vote]++; });
  return {
    label: counts.clear >= 7 ? 'Cleared' : counts.responsible >= 7 ? 'Held responsible' : 'No verdict',
    count: `${counts.clear} clear · ${counts.responsible} responsible · ${counts.abstain} abstain`,
  };
}

function render() {
  sliders.forEach(slider => { slider.parentElement.querySelector('output').value = slider.value; });
  const selected = swap.value === '' ? -1 : Number(swap.value);
  const inputs = perspectives.map(weights => weights.map((weight, axis) =>
    Math.min(1, Number(sliders[axis].value) / 100 * weight)));
  const intact = inputs.map(levels => judge(levels, false));
  const results = intact.map((result, index) => index === selected ? judge(inputs[index], true) : result);
  jury.replaceChildren(...results.map(({ feed, escape, vote }, index) => {
    const card = document.createElement('li');
    const shuffled = index === selected;
    card.classList.toggle('shuffled', shuffled);
    if (heard) card.dataset.vote = vote;
    const [sugar, bitter, looming] = inputs[index].map(level => Math.round(level * 100));
    card.innerHTML = `<b>Juror ${index + 1}</b><small>${shuffled ? 'Shuffled wiring' : 'Intact wiring'}</small>`;
    if (heard) {
      card.insertAdjacentHTML('beforeend', `<b>${labels[vote]}</b>
        <small>Sugar ${sugar}% · Bitter ${bitter}% · Looming ${looming}%</small>
        <small>Feed ${feed.toFixed(1)} Hz · Escape ${escape.toFixed(1)} Hz</small>`);
    }
    return card;
  }));
  if (!heard) {
    verdict.textContent = 'Enter a case or load the demo to hear the jury.';
    return;
  }
  const current = tally(results);
  const baseline = tally(intact);
  verdict.innerHTML = `<strong>${current.label}</strong><p>${current.count}</p>`;
  if (selected >= 0) {
    const effect = current.label === baseline.label ? 'Verdict unchanged' : 'Verdict changed';
    verdict.insertAdjacentHTML('beforeend', `<p>All-intact baseline · ${baseline.label} (${baseline.count})</p>
      <p>${effect}. Juror ${selected + 1} · ${labels[intact[selected].vote]} → ${labels[results[selected].vote]}</p>`);
  }
}

function loadDemo() {
  summary.setCustomValidity("");
  summary.value = 'At a fruit stand, Pip took the last peach from a crate. The owner says it was stock for sale. Pip says it sat beside a sign offering free samples. A witness saw Pip take it but could not read the sign.';
  sliders.forEach((slider, index) => { slider.value = evidence[index].demo; });
  swap.value = '';
  heard = true;
  render();
}

form.addEventListener('submit', event => {
  event.preventDefault();
  if (!summary.value.trim()) {
    summary.setCustomValidity('Enter a short case summary.');
    summary.reportValidity();
    return;
  }
  heard = true;
  render();
});
summary.addEventListener('input', () => {
  summary.setCustomValidity('');
  if (heard) {
    heard = false;
    render();
  }
});
sliders.forEach(slider => slider.addEventListener('input', render));
swap.addEventListener('change', render);
document.querySelector('#demo').addEventListener('click', loadDemo);
render();

const analyze = document.querySelector('#analyze');
const analysisStatus = document.querySelector('#analysis-status');
const analysisNotes = document.querySelector('#analysis-notes');
let caseRevision = 0;

function clearAnalysis() {
  caseRevision++;
  analysisStatus.textContent = '';
  analysisNotes.hidden = true;
  analysisNotes.replaceChildren();
}

summary.addEventListener('input', clearAnalysis);
document.querySelector('#demo').addEventListener('click', clearAnalysis);
analyze.addEventListener('click', async () => {
  const caseText = summary.value.trim();
  if (!caseText) {
    summary.setCustomValidity('Enter a short case summary.');
    summary.reportValidity();
    return;
  }
  clearAnalysis();
  if (location.protocol === 'file:') {
    analysisStatus.textContent = 'Start the Flyguys server and open /flycourt/ there to analyze text.';
    return;
  }
  const revision = caseRevision;
  analyze.disabled = true;
  analysisStatus.textContent = 'Reading the case with the local model…';
  try {
    const response = await fetch('/api/flycourt/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ summary: caseText }),
      signal: AbortSignal.timeout(95000),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : '');
    if (revision !== caseRevision) return;
    const titles = {
      clearing: 'Clearing', responsibility: 'Responsibility', harm: 'Harm', uncertainty: 'Uncertainty',
    };
    analysisNotes.replaceChildren(...Object.entries(titles).map(([key, title]) => {
      const item = document.createElement('li');
      item.textContent = `${title} · ${result[key]}`;
      return item;
    }));
    analysisNotes.hidden = false;
    ['sugar', 'bitter', 'looming'].forEach((key, index) => { sliders[index].value = result[key]; });
    heard = false;
    render();
    analysisStatus.textContent = 'Review or adjust the suggested scores, then select Hear the case.';
  } catch (error) {
    if (revision === caseRevision) {
      analysisStatus.textContent = error instanceof Error && error.message
        ? error.message : 'Case analysis failed. You can set evidence scores manually.';
    }
  } finally {
    analyze.disabled = false;
  }
});
