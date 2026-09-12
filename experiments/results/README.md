# Recorded evaluation runs

Two runs of the same pipeline, same configs, same seed (1337), on the same 180-image test
split. They disagree about whether DCCAN or the source-only baseline comes second.

| Model | run 1 AP50 | run 2 AP50 | spread |
|---|---|---|---|
| raw (in-domain ceiling) | 0.1619 | 0.1448 | 0.0171 |
| DCCAN | 0.1299 | 0.1279 | 0.0020 |
| source-only | 0.1141 | 0.1310 | 0.0169 |
| DANN | 0.1131 | 0.1141 | 0.0010 |

The spread is larger than every effect in the table. The configs leave `deterministic` off, so
cuDNN picks algorithms by timing and GPU atomics accumulate differently between runs. Any
future comparison should set `deterministic: true` and run several seeds per model.
