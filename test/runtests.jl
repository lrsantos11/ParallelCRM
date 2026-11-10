using Test
using LinearAlgebra
using Random
using CSV
using DataFrames
using ParallelCRM

include(joinpath(dirname(@__DIR__), "scripts", "make_table1.jl"))
using .Table1Reporting

configure_accelerator!(blas_threads=1, verbose=false)

@testset "ParallelCRM Test Suite" begin

    @testset "Balanced, nonempty partitions" begin
        for (rows, blocks) in [(10000, 101), (12500, 126), (5000, 11), (7, 7), (13, 1)]
            ranges = balanced_row_ranges(rows, blocks)
            @test length(ranges) == blocks
            @test all(!isempty, ranges)
            @test reduce(vcat, collect.(ranges)) == collect(1:rows)
            @test maximum(length.(ranges)) - minimum(length.(ranges)) <= 1
            @test all(length(r) in (fld(rows, blocks), cld(rows, blocks)) for r in ranges)
        end
        @test_throws ArgumentError balanced_row_ranges(10, 0)
        @test_throws ArgumentError balanced_row_ranges(10, 11)
    end

    @testset "Relative-error stopping and actual counters" begin
        for method in [:CRM, :pCRM, :SPM]
            calls = Threads.Atomic{Int}(0)
            projections = [x -> (Threads.atomic_add!(calls, 1); zeros(length(x))) for _ in 1:3]
            result = solve(method, zeros(2), projections; xSol=ones(2), itmax=4)
            @test result.iterations == 4
            @test result.projections == calls[] == 12
            @test !result.converged
            @test result.status == "iteration_limit"
            @test result.relative_error == 1.0

            exact = solve(method, ones(2), projections; xSol=ones(2))
            @test exact.iterations == exact.projections == 0
            @test exact.converged

            zero_limit = solve(method, zeros(2), projections; xSol=ones(2), itmax=0)
            @test zero_limit.iterations == zero_limit.projections == 0
            @test !zero_limit.converged
        end
    end

    @testset "Uniform weights, Results wrappers, and common starting point" begin
        projections = [x -> [1.0, x[2]], x -> [x[1], 2.0]]
        x0 = zeros(2)
        for threaded in [false, true]
            single = solve(:pCRM, [0.0], [x -> [1.0]]; xSol=[1.0], itmax=1, threaded=threaded)
            @test single.x == [1.0] # Omitting the common point x would return [2.0].

            res_spm = FSPM(x0, projections; xSol=[1.0, 2.0], itmax=1, threaded=threaded)
            @test res_spm.xApprox ≈ [0.5, 1.0]
            @test res_spm.proj_total == 2
            @test res_spm.iter_total == 1

            res_pcrm = pCRM(x0, projections; xSol=[1.0, 2.0], itmax=1, threaded=threaded)
            @test res_pcrm.xApprox ≈ [1.0, 2.0]
            @test res_pcrm.converged
        end
        res_crm = CRM(x0, projections; xSol=[1.0, 2.0], itmax=1)
        @test res_crm.xApprox ≈ [1.0, 2.0]
        @test x0 == zeros(2)
    end

    @testset "QR projections and serial/threaded agreement" begin
        rng = MersenneTwister(123)
        A = randn(rng, 30, 10)
        solution = normalize!(A' * randn(rng, 30))
        b = A * solution
        blocks = 4
        projections = projection_block_QR(A, b, blocks)
        x = randn(rng, 10)
        for (project, r) in zip(projections, balanced_row_ranges(30, blocks))
            expected = x - pinv(A[r, :]) * (A[r, :] * x - b[r])
            @test project(x) ≈ expected atol=1e-11
            @test A[r, :] * project(x) ≈ b[r] atol=1e-11
            @test project(project(x)) ≈ project(x) atol=1e-11
        end
        for method in [:CRM, :pCRM, :SPM]
            serial = solve(method, zeros(10), projections; xSol=solution, EPSVAL=1e-8)
            @test serial.converged
            @test serial.projections == blocks * serial.iterations
            @test norm(serial.x - solution) / norm(solution) <= 1e-8
            if method != :CRM
                parallel = solve(method, zeros(10), projections; xSol=solution, EPSVAL=1e-8, threaded=true)
                @test parallel.iterations == serial.iterations
                @test parallel.x ≈ serial.x atol=1e-12
            end
        end
    end

    @testset "Parallel work reaches multiple Julia threads" begin
        if Threads.nthreads() > 1
            for method in [:pCRM, :SPM]
                worker_ids = zeros(Int, 32)
                projections = [let i=i
                    x -> begin
                        worker_ids[i] = Threads.threadid()
                        sleep(0.002)
                        ones(length(x))
                    end
                end for i in eachindex(worker_ids)]
                result = solve(method, zeros(2), projections; xSol=ones(2), itmax=1, threaded=true)
                @test length(unique(worker_ids)) > 1
                @test all(>(0), worker_ids)
                @test result.projections == length(projections)
                @test result.converged
            end
        else
            @test_skip Threads.nthreads() > 1
        end
    end

    @testset "Aggregation uses raw counters and configurable statistic" begin
        mktempdir() do dir
            data = DataFrame()
            for (i, c) in enumerate([0.0, 0.1, 0.2]), repetition in 1:3
                push!(data, (
                    method="F-SPM (uniform weights)", threads=1, blocks=7,
                    rows=120, columns=20, coherence=c, iterations=3, projections=21,
                    time_seconds=2.0 * i + (repetition - 1), repetition=repetition,
                    repetitions=3, relative_error=i == 3 ? 0.1 : 1e-8,
                    tolerance=1e-5, converged=i != 3,
                    status=i == 3 ? "iteration_limit" : "converged",
                    source_sha256="test-source", instance_sha256="instance-$i",
                    environment_sha256="test-environment", seed=42,
                    iteration_limit=3, blas_threads=1, julia_version=string(VERSION),
                    mode="smoke"
                ); cols=:union)
            end
            path = joinpath(dir, "raw.csv")
            CSV.write(path, data)
            summary_min = aggregate_table1([path], joinpath(dir, "table_min"); require_full=false, stat=:min, show_table=false)
            @test only(summary_min.iterations) == 3
            @test only(summary_min.projections) == 21
            @test only(summary_min.time_seconds) == 4.0 # min per instance: 2, 4, 6 -> mean = 4.0
            @test only(summary_min.failed_instances) == 1
            @test occursin("\\textsuperscript{*}", read(joinpath(dir, "table_min.tex"), String))

            summary_med = aggregate_table1([path], joinpath(dir, "table_med"); require_full=false, stat=:median, show_table=false)
            @test only(summary_med.time_seconds) == 5.0 # median per instance: 3, 5, 7 -> mean = 5.0
        end
    end

end

println("ParallelCRM tests completed with $(Threads.nthreads()) Julia thread(s), BLAS=$(BLAS.get_num_threads()).")
