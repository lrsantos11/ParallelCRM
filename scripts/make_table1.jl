using DrWatson
@quickactivate "ParallelCRM"

using ParallelCRM

function make_table1(args=ARGS)
    stat_opt = :min
    unit_opt = :ms
    positional = String[]
    for arg in args
        if startswith(arg, "--stat=")
            stat_opt = Symbol(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--unit=")
            unit_opt = Symbol(split(arg, '='; limit=2)[2])
        else
            push!(positional, arg)
        end
    end

    if isempty(positional)
        default_dir = datadir("table1")
        csv_files = sort(filter(f -> startswith(basename(f), "raw_threads") && endswith(f, ".csv"), readdir(default_dir; join=true)))
        isempty(csv_files) && error("Usage: julia --project=. scripts/make_table1.jl [--stat=min|median] [--unit=ms|s] [OUTPUT_PREFIX RAW_THREADS1.csv RAW_THREADSN.csv...]")
        prefix = datadir("table1", "table1")
        aggregate_table1(csv_files, prefix; stat=stat_opt, time_unit=unit_opt)
        println("Generated ", prefix, ".csv, .tex, _instances.csv and _assessment.txt")
    else
        length(positional) >= 2 || error("Usage: OUTPUT_PREFIX RAW_THREADS1.csv [RAW_THREADSN.csv...]")
        require_full = length(positional) >= 3
        aggregate_table1(positional[2:end], positional[1]; require_full=require_full, stat=stat_opt, time_unit=unit_opt)
        println("Generated ", positional[1], ".csv, .tex, _instances.csv and _assessment.txt")
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    make_table1(ARGS)
end
