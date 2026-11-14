# Port validation: PyTorch simulator vs Shiu's Brian2 model

Same connectome, same parameters, 1000 ms per trial. Values are mean ± SD over trials (2 Brian2, 8 PyTorch).

| Condition | Measure | Brian2 (reference) | PyTorch port | Difference |
|---|---|---|---|---|
| sugar 200 Hz | feed rate (Hz) | 75.2 ± 0.2 | 72.2 ± 2.9 | -4.1% |
| sugar 200 Hz | escape rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz | retreat rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz | advance rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz | whole-brain spikes | 19957.5 ± 15.5 | 19541.4 ± 163.8 | -2.1% |
| sugar 200 Hz | neurons that spiked | 445.0 ± 1.0 | 439.4 ± 4.0 | -1.3% |
| sugar 200 Hz + bitter 200 Hz | feed rate (Hz) | 1.2 ± 0.2 | 3.1 ± 1.4 | +150.0% |
| sugar 200 Hz + bitter 200 Hz | escape rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz + bitter 200 Hz | retreat rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz + bitter 200 Hz | advance rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz + bitter 200 Hz | whole-brain spikes | 23163.0 ± 133.0 | 23391.5 ± 128.3 | +1.0% |
| sugar 200 Hz + bitter 200 Hz | neurons that spiked | 339.0 ± 7.0 | 342.8 ± 9.5 | +1.1% |
| looming 150 Hz | feed rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| looming 150 Hz | escape rate (Hz) | 185.5 ± 0.5 | 185.1 ± 1.9 | -0.2% |
| looming 150 Hz | retreat rate (Hz) | 12.4 ± 0.1 | 9.6 ± 0.7 | -22.5% |
| looming 150 Hz | advance rate (Hz) | 0.0 ± 0.0 | 0.2 ± 0.2 | reference is 0 |
| looming 150 Hz | whole-brain spikes | 85066.0 ± 34.0 | 85392.5 ± 379.1 | +0.4% |
| looming 150 Hz | neurons that spiked | 1342.0 ± 6.0 | 1377.9 ± 21.3 | +2.7% |
