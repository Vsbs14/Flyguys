const scenarios = {
  product: [[85, 22, 8], 'Three customers asked for a waterproof sensor. The prototype passed testing. Should we budget for production tooling?'],
  demand: [[20, 70, 65], 'Two customers reduced next quarter’s orders. Should we cut the production run before we buy materials?'],
  supplier: [[55, 45, 4], 'A second supplier could shorten lead times. Qualification would take six weeks and tie up an engineer. Should we start?'],
  warehouse: [[38, 48, 6], 'The lease is up in 90 days. The landlord wants a three-year renewal at a 7% increase. Should we sign?'],
  custom: [[50, 50, 0], 'Pitch an ordinary business decision. Set the three sliders and call the vote.'],
};

const senses = ['sugar', 'bitter', 'looming'].map(id => document.getElementById(id));
const scenario = document.getElementById('scenario');

function updateLabels() {
  for (const sense of senses) {
    document.getElementById(sense.id + '-value').textContent = sense.value + '%';
  }
}

function loadScenario() {
  const [values, brief] = scenarios[scenario.value];
  senses.forEach((sense, index) => { sense.value = values[index]; });
  document.getElementById('brief').textContent = brief;
  updateLabels();
}

// Trilinear interpolation over the three cached stimulus axes.
function estimate(table, inputs) {
  const coordinates = inputs.map(input => input * (table.length - 1));
  const lower = coordinates.map(Math.floor);
  const rates = [0, 0, 0, 0, 0];
  for (let corner = 0; corner < 8; corner++) {
    const offset = lower.map((_, axis) => (corner >> axis) & 1);
    const weight = offset.reduce((product, bit, axis) => {
      const fraction = coordinates[axis] - lower[axis];
      return product * (bit ? fraction : 1 - fraction);
    }, 1);
    if (!weight) continue;
    const [x, y, z] = lower.map((index, axis) => index + offset[axis]);
    table[x][y][z].forEach((rate, index) => { rates[index] += weight * rate; });
  }
  return rates;
}

function decide([feed, , retreat, advance, escape]) {
  if (escape >= 0.5) return ['Act now.', 'Put it first on the agenda. Yes, ahead of lunch.'];
  if (feed >= 25 && retreat < 10) return ['Ship it.', 'Approved. Please send the purchase order and a slice of banana.'];
  if (retreat >= 10) return ['Pull back.', 'Please remove the words “all in” from the minutes.'];
  if (advance >= 5) return ['Fund a pilot.', 'We’re moving fast. Procurement says six to eight weeks.'];
  return ['Table it.', 'I propose a working group to investigate the need for a working group.'];
}

const outputs = ['MN9', 'GF', 'MDN', 'DNp09'];
const outputLabels = ['Feed', 'Escape', 'Retreat', 'Advance'];
const source = document.getElementById('source');
const circuit = document.getElementById('circuit');
const live = document.getElementById('live');
let revision = 0;

function paintNode(name, label, hz) {
  const node = document.getElementById('path-' + name);
  node.textContent = label + (hz === undefined ? '' : ' · ' + hz.toFixed(1) + ' Hz');
  node.classList.toggle('active', hz > 0);
}

function resetVote() {
  revision++;
  circuit.classList.remove('firing');
  source.textContent = 'Call a vote to see which output groups fired.';
  senses.forEach(sense => paintNode(sense.id, sense.id));
  outputs.forEach((name, index) => paintNode(name, name + ' · ' + outputLabels[index]));
  document.getElementById('verdict').textContent = 'The chair is buzzing.';
  document.getElementById('quote').textContent = 'Call the vote to decide this agenda item.';
  document.getElementById('rates').textContent = '';
  document.getElementById('shuffle').textContent = '—';
}

function showVote(inputs, rates, fresh) {
  const hasSpikes = rates.slice(0, 4).some(rate => rate > 0);
  circuit.classList.toggle('firing', hasSpikes);
  senses.forEach((sense, index) => paintNode(sense.id, sense.id, inputs[index] * 200));
  outputs.forEach((name, index) => paintNode(name, name + ' · ' + outputLabels[index], rates[index]));
  source.textContent = fresh
    ? 'Fresh intact simulation. Shuffled comparison uses cached activity.'
    : 'Cached simulation activity. Lit outputs indicate firing above zero.';

  const [verdict, quote] = decide(rates);
  document.getElementById('verdict').textContent = verdict;
  document.getElementById('quote').textContent = quote;
  document.getElementById('rates').textContent = outputs
    .map((name, index) => name + ' ' + rates[index].toFixed(1) + ' Hz').join(' · ');
  document.getElementById('shuffle').textContent = decide(estimate(responses.shuffle, inputs))[0];
}

scenario.addEventListener('change', () => { loadScenario(); resetVote(); });
senses.forEach(sense => sense.addEventListener('input', () => { updateLabels(); resetVote(); }));
const readInputs = () => senses.map(sense => Number(sense.value) / 100);
document.getElementById('vote').addEventListener('click', () => {
  revision++;
  const inputs = readInputs();
  showVote(inputs, estimate(responses.intact, inputs), false);
});

live.addEventListener('click', async () => {
  const request = ++revision;
  const inputs = readInputs();
  const [sugar, bitter, looming] = inputs;
  live.disabled = true;
  source.textContent = 'Running four fresh simulated brains…';
  try {
    const response = await fetch('/api/decide', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({sugar, bitter, looming}),
    });
    if (!response.ok) throw new Error('Simulation request failed');
    const result = await response.json();
    const rates = outputs.map(name => result.rates[name].mean_hz);
    rates.push(result.gf_spiked / result.brains);
    if (request === revision) showVote(inputs, rates, true);
  } catch {
    if (request === revision) source.textContent = 'Fresh simulation unavailable. Call the board vote for the cached run.';
  } finally {
    live.disabled = false;
  }
});

if (location.protocol !== 'file:') {
  fetch('/api/health').then(response => response.json()).then(health => {
    live.hidden = health.ok !== true;
  }).catch(() => {});
}
loadScenario();
