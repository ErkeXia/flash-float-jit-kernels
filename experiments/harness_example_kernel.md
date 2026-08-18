# Harness Report — `example_kernel`

- **File**: `tools/example_kernel.py`
- **Timestamp**: 2026-08-18T16:16:15
- **Correctness**: PASS (stage: n/a)
- **Reference compared**: yes

## Correctness (5-stage)

| Stage | Status |
|-------|--------|
| smoke | pass |
| shape_sweep | pass |
| stability | pass |
| determinism | pass |
| edge_cases | pass |

## Performance (triton.testing.do_bench, quantiles [0.5, 0.2, 0.8])

- **Input size**: N = 4096
- **Time (median)**: 5.1 us (min 4.9 / max 5.3)
- **Reference time**: 5.2 us (speedup 1.01x)
- **SOL time**: 0.0 us
- **SOL gap**: 349.17x
- **Classification**: memory-bound (compute 0.0 us / mem 0.0 us)

| Metric | Value |
|--------|-------|
| median (us) | 5.1 |
| min (us) | 4.9 |
| max (us) | 5.3 |
| reference (us) | 5.2 |
| speedup vs ref | 1.01x |
