using DrWatson
@quickactivate "ParallelCRM"

using ParallelCRM

function make_table1(args=ARGS)
    stat_opt = :min
    unit_opt = :ms
    date_opt = nothing
    positional = String[]
    for arg in args
        if startswith(arg, "--stat=")
            stat_opt = Symbol(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--unit=")
            unit_opt = Symbol(split(arg, '='; limit=2)[2])
        elseif startswith(arg, "--date=")
            date_opt = String(split(arg, '='; limit=2)[2])
        else
            push!(positional, arg)
        end
    end

    if isempty(positional)
        default_dir = datadir("table1")
        all_raw = sort(filter(f -> startswith(basename(f), "raw_") && endswith(f, ".csv"), readdir(default_dir; join=true)))
        isempty(all_raw) && error("No raw_*.csv files found in $default_dir. Run scripts/run_experiments.jl first.")

        # Use DrWatson.parse_savename to group by `date` if present, selecting either --date=... or the most recent date
        dated_groups = Dict{String,Vector{String}}()
        undated_files = String[]
        for f in all_raw
            try
                _, params, _ = parse_savename(basename(f))
                if haskey(params, "date")
                    d = string(params["date"])
                    push!(get!(dated_groups, d, String[]), f)
                else
                    push!(undated_files, f)
                end
            catch
                push!(undated_files, f)
            end
        end

        if !isnothing(date_opt)
            haskey(dated_groups, date_opt) || error("No raw CSV files found for date=$date_opt in $default_dir")
            csv_files = sort(dated_groups[date_opt])
            selected_tag = date_opt
        elseif !isempty(dated_groups)
            selected_tag = maximum(keys(dated_groups))
            csv_files = sort(dated_groups[selected_tag])
        else
            selected_tag = nothing
            csv_files = sort(undated_files)
        end

        prefix = isnothing(selected_tag) ? datadir("table1", "table1") : datadir("table1", savename("table1", (date=selected_tag,); sort=false))
        aggregate_table1(csv_files, prefix; require_full=false, stat=stat_opt, time_unit=unit_opt)
        if !isnothing(selected_tag)
            latest_prefix = datadir("table1", "table1")
            aggregate_table1(csv_files, latest_prefix; require_full=false, stat=stat_opt, time_unit=unit_opt, show_table=false)
        end
        println("Generated ", prefix, ".csv, .tex, _instances.csv and _assessment.txt from: ", join(basename.(csv_files), ", "))
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
