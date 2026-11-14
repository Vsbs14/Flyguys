# Port validation: PyTorch simulator vs Shiu's Brian2 model

Same connectome, same parameters, 1000 ms per trial. Values are mean ± SD over trials (2 Brian2, 4 PyTorch).

| Condition | Measure | Brian2 (reference) | PyTorch port | Difference |
|---|---|---|---|---|
| sugar 200 Hz | feed rate (Hz) | 75.2 ± 0.2 | 73.2 ± 3.3 | -2.7% |
| sugar 200 Hz | escape rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz | retreat rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz | advance rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz | whole-brain spikes | 19957.5 ± 15.5 | 19642.8 ± 152.1 | -1.6% |
| sugar 200 Hz | neurons that spiked | 445.0 ± 1.0 | 441.0 ± 5.1 | -0.9% |
| sugar 200 Hz + bitter 200 Hz | feed rate (Hz) | 1.2 ± 0.2 | 2.8 ± 0.2 | +120.0% |
| sugar 200 Hz + bitter 200 Hz | escape rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz + bitter 200 Hz | retreat rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz + bitter 200 Hz | advance rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| sugar 200 Hz + bitter 200 Hz | whole-brain spikes | 23163.0 ± 133.0 | 23196.5 ± 61.7 | +0.1% |
| sugar 200 Hz + bitter 200 Hz | neurons that spiked | 339.0 ± 7.0 | 335.0 ± 15.9 | -1.2% |
| looming 150 Hz | feed rate (Hz) | 0.0 ± 0.0 | 0.0 ± 0.0 | both 0 |
| looming 150 Hz | escape rate (Hz) | 185.5 ± 0.5 | 185.1 ± 1.2 | -0.2% |
| looming 150 Hz | retreat rate (Hz) | 12.4 ± 0.1 | 12.9 ± 2.0 | +4.0% |
| looming 150 Hz | advance rate (Hz) | 0.0 ± 0.0 | 0.2 ± 0.2 | reference is 0 |
| looming 150 Hz | whole-brain spikes | 85066.0 ± 34.0 | 85241.0 ± 172.6 | +0.2% |
| looming 150 Hz | neurons that spiked | 1342.0 ± 6.0 | 1349.5 ± 21.1 | +0.6% |
