# ParallelCRM

**Parallelizing the Circumcentered-Reflection Method**

[![arXiv](https://img.shields.io/badge/arXiv-2505.17258-b31b1b.svg)](https://arxiv.org/abs/2505.17258)

This repository contains the Julia implementation and reproducible numerical experiments for the paper:

> **Parallelizing the Circumcentered-Reflection Method**  
> Pablo Barros, Roger Behling, Vincent Guigues, and Luiz-Rafael Santos  
> [arXiv:2505.17258](https://arxiv.org/abs/2505.17258)

---

## Overview

`ParallelCRM` is a self-contained [DrWatson.jl](https://juliadynamics.github.io/DrWatson.jl/stable/) scientific project implementing projection and reflection methods for solving the **Linear Best Approximation Problem (LBAP)** over a finite intersection of affine subspaces $S = \bigcap_{i=1}^{\ell} U_i \subset \mathbb{R}^n$:

- **`pCRM` (Parallelized Circumcentered-Reflection Method / P-CRM)**: Evaluates reflections $R_{U_1}(x_k), \dots, R_{U_\ell}(x_k)$ independently (in parallel via `ThreadsX.jl` when multithreading is active) and computes the next iterate as the circumcenter $\text{circ}(x_k, R_{U_1}(x_k), \dots, R_{U_\ell}(x_k))$.
- **`CRM` (Sequential Circumcentered-Reflection Method)**: Evaluates successive reflections $R_{U_\ell} \cdots R_{U_1}(x_k)$ sequentially and computes their circumcenter.
- **`FSPM` / `SPM` (Framework for the Simultaneous Projection Method)**: Computes simultaneous projections $P_{U_1}(x_k), \dots, P_{U_\ell}(x_k)$ (in parallel or serially) and forms their convex combination (including Cimmino's uniform-weight method $p_0 = 0, p_i = 1/\ell$).

### Hardware-Aware Linear Algebra Acceleration

The module automatically detects the host architecture via `configure_accelerator!`:
- On **Apple Silicon (`aarch64` macOS, e.g., Apple M1/M2/M3/M4)**, it loads [`AppleAccelerate.jl`](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl).
- On **x86_64 (`Intel` / `AMD`, e.g., AMD Ryzen / EPYC)**, it loads [`MKL.jl`](https://github.com/JuliaLinearAlgebra/MKL.jl).
- To isolate block-level parallelism in `ThreadsX.map` without BLAS thread oversubscription, BLAS threads are fixed to `1` (`BLAS.set_num_threads(1)`) by default during the experiments.

---

## Repository Structure

```text
ParallelCRM/
├── Project.toml                  # Lightweight Julia environment (DrWatson, BenchmarkTools, ThreadsX, etc.)
├── README.md                     # Documentation and reproduction instructions
├── src/
│   ├── ParallelCRM.jl            # Core module: balanced partitions, QR projectors, circumcenter, CRM, P-CRM, F-SPM
│   └── reporting.jl              # Table 1 validation, statistical aggregation, and LaTeX/CSV formatting
├── scripts/
│   ├── run_experiments.jl        # Benchmark runner using DrWatson.jl and BenchmarkTools.jl
│   └── make_table1.jl            # CLI entry point for Table 1 aggregation
├── test/
│   └── runtests.jl               # Unit tests for partitions, stopping criteria, QR projectors, and multithreading
└── data/
    └── table1/                   # Output directory for raw CSVs, TOML metadata, and generated Table 1
```

---

## Reproducing Table 1

Use Julia **1.10+** and run all commands from the repository root.

### 1. Instantiate the environment and run unit tests

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --threads=1 --project=. test/runtests.jl
julia --startup-file=no --threads=auto --project=. test/runtests.jl
```

### 2. Optional smoke test (fast verification)

```bash
julia --startup-file=no --threads=1 --project=. scripts/run_experiments.jl --smoke --samples=3 --output-dir=data/table1_smoke
julia --startup-file=no --threads=auto --project=. scripts/run_experiments.jl --smoke --samples=3 --output-dir=data/table1_smoke
```

### 3. Run the full Table 1 benchmark

Run the single-threaded process and the multithreaded process **sequentially** on the same idle machine (replace `N` in `--threads=N` with the desired thread count, e.g., `10` on Apple M4, `8`, or `32` on AMD Ryzen 9 7950X3D):

```bash
julia --startup-file=no --threads=1 --project=. scripts/run_experiments.jl --seed=42 --samples=5 --blas-threads=1 --output-dir=data/table1
julia --startup-file=no --threads=N --project=. scripts/run_experiments.jl --seed=42 --samples=5 --blas-threads=1 --output-dir=data/table1
```

### 4. Generate Table 1 (CSV, LaTeX, and assessment summary)

```bash
julia --startup-file=no --project=. scripts/make_table1.jl
```

By default, `make_table1.jl` reads all `data/table1/raw_threads*.csv` files, uses the per-instance minimum time (`--stat=min`, equivalent to `BenchmarkTools.@belapsed`) or median time (`--stat=median`), averages across the three coherence levels $c \in \{0.0, 0.1, 0.2\}$, displays the table in the terminal, and writes:
- `data/table1/table1.csv` and `data/table1/table1_instances.csv`
- `data/table1/table1.tex` (with automatic `\textbf{...}` highlights on the best Iteration count, Projection count, and Time per problem size)
- `data/table1/table1_assessment.txt` (win/tie counts comparing `CRM`, `P-CRM`, and `F-SPM`)

You can also pass explicit file paths and statistics:

```bash
julia --startup-file=no --project=. scripts/make_table1.jl --stat=min --unit=ms data/table1/table1 data/table1/raw_threads1.csv data/table1/raw_threadsN.csv
```

---

## Experimental Protocol Summary

- **Problem instances**: 36 synthetic affine feasibility instances generated in `Float64` using `MersenneTwister(42)` across $m \in \{5000, 7500, 10000, 12500\}$, $n \in \{100, 250, 500\}$, and coherence $c \in \{0.0, 0.1, 0.2\}$, with $A = (1 - c)G + c\,\mathbf{1}_{[m,n]}$, $x^\star = A^\top w / \|A^\top w\|_2$, and $b = Ax^\star$.
- **Partitioning**: Each matrix $A$ is partitioned into $\ell = \lfloor m/n \rfloor + 1$ contiguous, nonempty balanced row blocks of size $\lceil m/\ell \rceil$ or $\lfloor m/\ell \rfloor$.
- **Stopping rule**: Starting from $x_0 = 0$, each method terminates when $\frac{\|x_k - x^\star\|_2}{\|x^\star\|_2} \le 10^{-5}$ or after $10^4$ full iterations. Each full iteration evaluates $\ell$ block projections.
- **Timing**: Timings are measured with `BenchmarkTools.@benchmark` (`evals=1`, `gcsample=true`) after an untimed reference/warm-up solve. Preprocessing (matrix generation, partitioning, and QR factorizations) is excluded from solver timings.
