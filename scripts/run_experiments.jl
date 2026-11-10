using DrWatson
@quickactivate "ParallelCRM"

using BenchmarkTools
using CSV
using DataFrames
using Dates
using LinearAlgebra
using ParallelCRM
using Random
using SHA
using Statistics
using TOML

function parse_options(args)
    options = Dict{String,Any}(
        "smoke" => false,
        "large" => false,
        "reuse-full" => true,
        "serial" => false,
        "both" => true,
        "seed" => 42,
        "samples" => 5,
        "seconds" => 300.0,
        "blas-threads" => 1,
        "accelerator" => "auto",
        "stat" => "min",
        "overwrite" => true,
        "output-dir" => datadir("table1")
    )
    for arg in args
        if arg == "--smoke"
            options["smoke"] = true
        elseif arg == "--large"
            options["large"] = true
        elseif arg == "--reuse-full"
            options["reuse-full"] = true
        elseif arg == "--no-reuse-full"
            options["reuse-full"] = false
        elseif arg == "--serial"
            options["serial"] = true
            options["both"] = false
        elseif arg == "--parallel-only"
            options["serial"] = false
            options["both"] = false
        elseif arg == "--both"
            options["both"] = true
        elseif arg == "--no-overwrite"
            options["overwrite"] = false
        elseif arg == "--help"
            println("""
            Usage: julia --project=. --threads=N scripts/run_experiments.jl [options]

            Options:
              --smoke                 Run a quick smoke test on (m=5000, n=250)
              --large                 Run the full grid plus larger configurations (m up to 30000, n up to 1000)
              --no-reuse-full         Re-run base full configurations from scratch even when --large is passed
              --both                  Run both serial (threads=1) and parallel (threads=N) passes in one session (default when N > 1)
              --serial                Force only the serial (1-thread) pass even if Julia has N > 1 threads
              --parallel-only         Run only the parallel (N-thread) pass
              --seed=42               MersenneTwister random seed (default: 42)
              --samples=5             Number of BenchmarkTools samples per method/instance (default: 5)
              --repetitions=5         Alias for --samples
              --seconds=300           Maximum BenchmarkTools budget in seconds per trial (default: 300)
              --blas-threads=1        Number of BLAS threads (default: 1)
              --accelerator=auto      BLAS backend: auto, accelerate, mkl, or openblas (default: auto)
              --stat=min              Aggregation statistic: min (@belapsed) or median (default: min)
              --output-dir=PATH       Directory for CSV and TOML outputs (default: data/table1)
              --no-overwrite          Fail if raw_threadsN.csv already exists in output-dir
            """)
            return nothing
        else
            startswith(arg, "--") && occursin('=', arg) || error("Unknown option: $arg")
            key, value = split(arg[3:end], '='; limit=2)
            if key == "repetitions"
                options["samples"] = parse(Int, value)
            elseif key in ("seed", "samples", "blas-threads")
                options[key] = parse(Int, value)
            elseif key == "seconds"
                options[key] = parse(Float64, value)
            elseif key in ("output-dir", "accelerator", "stat")
                options[key] = String(value)
            elseif key in ("overwrite", "serial", "both", "large", "reuse-full")
                options[key] = parse(Bool, value)
            else
                error("Unknown option: $arg")
            end
        end
    end
    options["samples"] > 0 || error("samples/repetitions must be positive")
    options["blas-threads"] >= 1 || error("blas-threads must be >= 1")
    return options
end

function file_fingerprint(paths)
    io = IOBuffer()
    for path in paths
        isfile(path) || error("Missing reproducibility file: $path")
        write(io, relpath(path, projectdir()), '\0', read(path), '\0')
    end
    return bytes2hex(sha256(take!(io)))
end

function instance_fingerprint(A, b, solution, ranges)
    io = IOBuffer()
    write(io, Int64[size(A)...], A, b, solution)
    for r in ranges
        write(io, Int64[first(r), last(r)])
    end
    return bytes2hex(sha256(take!(io)))
end

function git_information()
    root = projectdir()
    paths = [
        "src/ParallelCRM.jl",
        "src/reporting.jl",
        "scripts/run_experiments.jl",
        "scripts/make_table1.jl",
        "test/runtests.jl",
        "Project.toml"
    ]
    try
        head = strip(read(pipeline(`git -C $root rev-parse HEAD`; stderr=devnull), String))
        changes = strip(read(pipeline(`git -C $root status --porcelain -- $paths`; stderr=devnull), String))
        tracked = all(paths) do path
            success(pipeline(`git -C $root ls-files --error-unmatch $path`; stdout=devnull, stderr=devnull))
        end
        return head, !isempty(changes) || !tracked
    catch
        return "not-a-git-checkout", true
    end
end

function machine_information(active_accelerator::String)
    cpu = Sys.cpu_info()
    return Dict{String,Any}(
        "cpu_models" => sort(unique(c.model for c in cpu)),
        "logical_cpus_visible" => length(cpu),
        "cpu_threads" => Sys.CPU_THREADS,
        "memory_bytes" => Sys.total_memory(),
        "machine" => Sys.MACHINE,
        "arch" => string(Sys.ARCH),
        "kernel" => string(Sys.KERNEL),
        "accelerator" => active_accelerator,
        "blas_configuration" => sprint(show, BLAS.get_config())
    )
end

function find_reusable_full_data(outdir::String, nt::Int, seed::Int, samples::Int)
    all_raw = sort(filter(f -> startswith(basename(f), "raw_") && endswith(f, "threads=$(nt).csv"), readdir(outdir; join=true)))
    for path in reverse(all_raw)
        try
            df = CSV.read(path, DataFrame)
            if nrow(df) > 0 && all(df.seed .== seed) && all(df.repetitions .== samples) && any(df.mode .== "full")
                return df, path
            end
        catch
        end
    end
    return nothing, nothing
end

function run_single_pass(nt::Int, run_tag::String, options::Dict{String,Any}, active_accelerator::String)
    outdir = abspath(options["output-dir"])
    mkpath(outdir)
    params = (date=run_tag, threads=nt)
    raw_filename = savename("raw", params, "csv"; sort=false)
    meta_filename = savename("metadata", params, "toml"; sort=false)
    raw_path = joinpath(outdir, raw_filename)
    metadata_path = joinpath(outdir, meta_filename)
    if !options["overwrite"] && any(isfile, [raw_path, metadata_path])
        error("Outputs already exist in $outdir; choose a new --output-dir or omit --no-overwrite")
    end

    samples = options["samples"]
    max_seconds = options["seconds"]
    smoke = options["smoke"]
    large = options["large"]
    mode = smoke ? "smoke" : (large ? "large" : "full")

    base_pairs = [(m, n) for m in [5000, 7500, 10000, 12500] for n in [100, 250, 500]]
    extra_large_pairs = vcat(
        [(m, 1000) for m in [5000, 7500, 10000, 12500]],
        [(m, n) for m in [20000, 30000] for n in [250, 500, 1000]]
    )
    size_pairs = smoke ? [(5000, 250)] : (large ? vcat(base_pairs, extra_large_pairs) : base_pairs)
    row_counts = sort(unique(first.(size_pairs)))
    col_counts = sort(unique(last.(size_pairs)))
    coherences = [0.0, 0.1, 0.2]
    tolerance = 1e-5
    iteration_limit = 10_000

    sources = [
        srcdir("ParallelCRM.jl"),
        srcdir("reporting.jl"),
        scriptsdir("run_experiments.jl"),
        scriptsdir("make_table1.jl"),
        projectdir("test", "runtests.jl"),
        projectdir("Project.toml")
    ]
    source_hash = file_fingerprint(sources)
    machine = machine_information(active_accelerator)
    env_buffer = IOBuffer()
    TOML.print(env_buffer, machine; sorted=true)
    environment_hash = bytes2hex(sha256(take!(env_buffer)))
    commit, dirty = git_information()

    metadata = Dict{String,Any}(
        "started_utc" => string(now(UTC)),
        "complete" => false,
        "source_sha256" => source_hash,
        "environment_sha256" => environment_hash,
        "source_commit" => commit,
        "source_dirty_or_untracked" => dirty,
        "julia_version" => string(VERSION),
        "julia_threads" => nt,
        "julia_process_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "accelerator" => active_accelerator,
        "seed" => options["seed"],
        "rng" => "MersenneTwister (private; regenerated identically in each pass)",
        "mode" => mode,
        "rows" => row_counts,
        "columns" => col_counts,
        "coherences" => coherences,
        "tolerance" => tolerance,
        "iteration_limit" => iteration_limit,
        "samples" => samples,
        "benchmark_engine" => "BenchmarkTools (@benchmark with evals=1, gcsample=true)",
        "timing_statistic" => "configurable: minimum (@belapsed) or median across samples, then arithmetic mean over three coherences",
        "preprocessing_included" => false,
        "preprocessing_excluded" => "matrix/RHS generation, partition, QR factorizations, reference/warm-up run, explicit pre-sample GC, validation and output",
        "gc_policy" => "BenchmarkTools gcsample=true (GC.gc() before each timed sample)",
        "method_order" => "cyclic rotation by instance; runs are sequential",
        "machine" => machine
    )

    function save_metadata()
        open(metadata_path, "w") do io
            TOML.print(io, metadata; sorted=true)
        end
    end
    save_metadata()

    rng = MersenneTwister(options["seed"])
    raw = DataFrame()

    reusable_df, reusable_path = (large && options["reuse-full"]) ?
        find_reusable_full_data(outdir, nt, options["seed"], samples) : (nothing, nothing)
    if !isnothing(reusable_df)
        println("[$mode] Reusing base full instances (threads=$nt) from $(basename(reusable_path))")
    end

    # The serial run (threads=1) executes CRM, P-CRM, and F-SPM with threaded=false.
    # Multithreaded runs (threads>1) execute P-CRM and F-SPM in parallel across blocks.
    methods = nt == 1 ? [:pCRM, :CRM, :SPM] : [:pCRM, :SPM]
    labels = Dict(:pCRM => "P-CRM", :CRM => "CRM", :SPM => "F-SPM (uniform weights)")
    instance_number = 0

    for (m, n) in size_pairs, c in coherences
        instance_number += 1
        preprocessing_start = time_ns()
        A = (1 - c) .* randn(rng, m, n) .+ c
        w = randn(rng, m)
        xSol = normalize!(A' * w)
        b = A * xSol
        x0 = zeros(n)
        blocks = fld(m, n) + 1
        ranges = balanced_row_ranges(m, blocks)

        if !isnothing(reusable_df)
            instance_hash = instance_fingerprint(A, b, xSol, ranges)
            cached = reusable_df[(reusable_df.rows .== m) .& (reusable_df.columns .== n) .&
                                 (reusable_df.coherence .== c) .& (reusable_df.instance_sha256 .== instance_hash), :]
            if nrow(cached) == length(methods) * samples
                cached_copy = copy(cached)
                cached_copy.mode .= mode
                cached_copy.source_sha256 .= source_hash
                cached_copy.environment_sha256 .= environment_hash
                cached_copy.source_commit .= commit
                cached_copy.source_dirty_or_untracked .= dirty
                append!(raw, cached_copy; cols=:union)
                wsave(raw_path, raw)
                println("[$mode] Reused cached size=($m,$n), blocks=$blocks, c=$c (threads=$nt)")
                flush(stdout)
                continue
            end
        end

        projections = projection_block_QR(A, b, blocks)
        preprocessing_seconds = (time_ns() - preprocessing_start) / 1e9
        instance_hash = instance_fingerprint(A, b, xSol, ranges)

        for method in circshift(methods, -(instance_number - 1))
            threaded = nt > 1 && method != :CRM

            # Reference solve outside timing: warms up JIT and records exact deterministic counters & accuracy
            reference = solve(method, x0, projections;
                xSol=xSol, itmax=iteration_limit, EPSVAL=tolerance, threaded=threaded)
            measured_error = norm(reference.x - xSol) / norm(xSol)
            isequal(measured_error, reference.relative_error) || error("Incorrect reported error")

            # BenchmarkTools timing with evals=1 and pre-sample GC
            trial = @benchmark solve($method, $x0, $projections;
                xSol=$xSol, itmax=$iteration_limit, EPSVAL=$tolerance, threaded=$threaded) samples=samples evals=1 seconds=max_seconds gcsample=true

            n_recorded = length(trial.times)
            for repetition in 1:n_recorded
                push!(raw, (
                    method = labels[method],
                    threads = nt,
                    blocks = blocks,
                    rows = m,
                    columns = n,
                    coherence = c,
                    iterations = reference.iterations,
                    projections = reference.projections,
                    cholesky_fallbacks = reference.cholesky_fallbacks,
                    relative_error = measured_error,
                    converged = reference.converged,
                    status = reference.status,
                    repetition = repetition,
                    repetitions = n_recorded,
                    time_seconds = trial.times[repetition] / 1e9,
                    gc_seconds = trial.gctimes[repetition] / 1e9,
                    allocated_bytes = trial.memory,
                    preprocessing_seconds = preprocessing_seconds,
                    seed = options["seed"],
                    tolerance = tolerance,
                    iteration_limit = iteration_limit,
                    blas_threads = BLAS.get_num_threads(),
                    accelerator = active_accelerator,
                    julia_version = string(VERSION),
                    mode = mode,
                    source_sha256 = source_hash,
                    instance_sha256 = instance_hash,
                    environment_sha256 = environment_hash,
                    source_commit = commit,
                    source_dirty_or_untracked = dirty,
                    threaded = threaded
                ); cols=:union)
            end

            wsave(raw_path, raw)
            min_ms = (minimum(trial.times) / 1e6)
            med_ms = (median(trial.times) / 1e6)
            println("[$mode] $(labels[method]) (threads=$nt), size=($m,$n), blocks=$blocks, c=$c: ",
                    "iter=$(reference.iterations), proj=$(reference.projections), fallbacks=$(reference.cholesky_fallbacks), ",
                    "min=$(round(min_ms, digits=3)) ms, median=$(round(med_ms, digits=3)) ms, ",
                    "err=$(round(reference.relative_error, sigdigits=3))")
            flush(stdout)
        end
    end

    stat_sym = Symbol(options["stat"])
    summary_prefix = joinpath(outdir, savename("table1", params; sort=false))
    aggregate_table1([raw_path], summary_prefix; require_full=false, stat=stat_sym, show_table=true)
    metadata["complete"] = true
    metadata["finished_utc"] = string(now(UTC))
    metadata["timed_solves"] = nrow(raw)
    save_metadata()
    println("Saved $raw_path and $metadata_path")
    return raw_path
end

function run_experiments(args=ARGS)
    options = parse_options(args)
    isnothing(options) && return

    active_accelerator = configure_accelerator!(
        accelerator = Symbol(options["accelerator"]),
        blas_threads = options["blas-threads"],
        verbose = true
    )

    nt_proc = Threads.nthreads()
    if options["serial"] || nt_proc == 1
        thread_passes = [1]
    elseif options["both"]
        thread_passes = [1, nt_proc]
    else
        thread_passes = [nt_proc]
    end

    run_tag = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    raw_paths = String[]
    for nt in thread_passes
        println("\n=== Running pass with threads = $nt (date=$run_tag) ===")
        push!(raw_paths, run_single_pass(nt, run_tag, options, active_accelerator))
    end

    if length(raw_paths) == 2
        outdir = abspath(options["output-dir"])
        stat_sym = Symbol(options["stat"])
        dated_prefix = joinpath(outdir, savename("table1", (date=run_tag,); sort=false))
        latest_prefix = joinpath(outdir, "table1")
        println("\n=== Generating combined Table 1 (date=$run_tag, threads = 1 and threads = $nt_proc) ===")
        aggregate_table1(raw_paths, dated_prefix; require_full=!options["smoke"], stat=stat_sym, show_table=true)
        aggregate_table1(raw_paths, latest_prefix; require_full=!options["smoke"], stat=stat_sym, show_table=false)
        println("Combined Table 1 written to:")
        println("  - $(dated_prefix).tex / $(dated_prefix).csv")
        println("  - $(latest_prefix).tex / $(latest_prefix).csv (latest alias)")
    end
    return raw_paths
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    run_experiments()
end
