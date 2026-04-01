# Session: TurboQuant CUDA Build on RTX 5090 Blackwell — 2026-03-31

## What We Did
Built the first public TurboQuant CUDA binary for RTX 5090 (Blackwell SM120). Cloned seanrasch/llama-cpp-turboquant, fixed 3 Windows DLL linking issues, compiled 569 files with CUDA 12.8 + SM120a + MSVC 2022.

## Build Fixes (Windows DLL)
1. `#define _USE_MATH_DEFINES` in `ggml-turbo-quant.c` (MSVC M_PI)
2. Local `turbo3_cpu_wht_group_size` definition in `ops.cpp` (cross-DLL extern fix)
3. Replaced `#ifdef GGML_USE_CUDA` extern block in `llama-kv-cache.cpp` with local innerq stubs

## Benchmark Results (Qwen3.5-27B Q4_K_M on RTX 5090)

### turbo3 Prefill (tok/s)
- 512: 3,541 (f16: 3,534) → 1.00x
- 2K: 3,575 (f16: 3,516) → 1.02x
- 8K: 3,470 (f16: 3,291) → 1.05x
- 32K: 3,068 (f16: 2,482) → 1.24x FASTER
- 64K: 2,498 (f16: OOM)
- 131K: 1,731 (f16: OOM)

### Generation (tg128)
- f16: 70.22 tok/s
- turbo3: 67.77 tok/s (0.965x)
- turbo2: 71.1 tok/s (1.01x — FASTER than f16!)

### Extreme Context
- turbo2 at 1M context: 69.3 tok/s gen, 122-130 tok/s prefill
- turbo2 at 1.5M context: CONFIRMED WORKING (1.4 tok/s gen at 1.5M positions)
- f16 OOMs at ~232K

## Key Files
- Build binaries: `build/bin/llama-server.exe` (4.7 MB), `build/bin/llama-cli.exe` (2.6 MB)
- Model: Qwen3.5-27B-Q4_K_M.gguf (15.6 GB)

## Theoretical Context Limits (Qwen3.5-27B on 5090)
- f16: 232K | q8_0: 436K | q4_0: 823K | turbo3: 1.1M | turbo2: 1.5M
- KVTC 20x theoretical: 4.6M tokens

## Build Sources
- Repo: seanrasch/llama-cpp-turboquant fork
- Also tested: Aaryan-Kapoor fork (turboquant-tq3_0 branch), TheTom's Python experiments

## Next Steps
- Post benchmarks to llama.cpp discussion #20969
- Post to r/LocalLLaMA and X
- BUILD KVTC (NVIDIA's 20x compression) — theoretical 4.6M context
- PR Windows fixes back to seanrasch fork
