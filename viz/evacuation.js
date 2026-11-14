(() => {
  const scenes = [...document.querySelectorAll('.comparison .evacuation')];
  const outcomes = [...document.querySelectorAll('.outcome-text')];
  const trigger = document.querySelector('#trigger');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');

  function show(index, phase, text) {
    scenes[index].dataset.phase = phase;
    outcomes[index].textContent = text;
  }

  async function load() {
    const response = await fetch('evacuation-trials.json');
    if (!response.ok) throw new Error('Recording unavailable');
    const { trials } = await response.json();
    trigger.disabled = false;
    trigger.addEventListener('click', () => {
      trigger.disabled = true;
      trials.forEach((_, index) => show(index, 'ready', 'Ship in orbit'));
      setTimeout(() => {
        trials.forEach(({ spike_ms }, index) => {
          show(index, 'warning', 'Star swelling…');
          setTimeout(() => {
            if (spike_ms !== null) show(index, 'escaped', 'Neuron fired → ship escaped');
          }, reduced.matches ? 30 : 650 + (spike_ms ?? 0) * 20);
        });
        setTimeout(() => {
          trials.forEach(({ spike_ms }, index) => {
            if (spike_ms === null) show(index, 'lost', 'Neuron silent → ship destroyed');
          });
          trigger.textContent = 'Replay both ships';
          trigger.disabled = false;
        }, reduced.matches ? 150 : 1900);
      }, reduced.matches ? 20 : 300);
    });
  }
  load().catch(() => {
    document.querySelector('#evacReadout').textContent = 'Brain recording unavailable. Reload from the demo server.';
  });
})();
