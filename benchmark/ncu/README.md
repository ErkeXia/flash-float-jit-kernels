# Nsight Compute Targets

This folder is separate from `benchmark/profiling`.

`benchmark/profiling` runs the in-repo CUPTI Activity/Range collectors and
generates `report.html`. The files here are intentionally smaller: they launch a
single provider so Nsight Compute can collect source/SASS-level data without
another CUPTI profiler attached.

Typical first run from the repository root:

```bash
bash benchmark/ncu/run_ncu_examples.sh
```

The script profiles:

- `cuda_warm`
- `triton_symm_native_warm`

Both use `benchmark/ncu/ncu_target.py`, which performs warmup first and then
uses `cudaProfilerStart/Stop`. The `ncu` commands therefore pass
`--profile-from-start off`, so warmup and Triton compilation are not profiled.

Useful Nsight Compute sections to inspect:

- GPU Speed Of Light
- Launch Statistics
- Occupancy
- Scheduler Statistics
- Warp State Statistics
- Memory Workload Analysis
- Source Counters
- Instruction Statistics
