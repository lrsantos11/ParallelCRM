module ParallelCRM

using Dates
using LinearAlgebra
using Random
using Statistics
using ThreadsX

import Base: @kwdef

export Results,
       configure_accelerator!,
       balanced_row_ranges,
       form_Affine_blocks,
       projection_factory,
       projection_block_QR,
       find_circumcenter!,
       circumcenter,
       step,
       solve,
       CRM,
       pCRM,
       FSPM,
       SPM

"""
    Results

Standard result structure compatible with the `CRM-CFP` ecosystem and Table 1 scripts.
"""
@kwdef mutable struct Results
    iter_total::Int = 0
    proj_total::Int = 0
    final_tol::Float64 = 0.0
    xApprox::AbstractArray = Float64[]
    method::Symbol = :None
    converged::Bool = false
    status::String = "uninitialized"
    date::DateTime = Dates.now()
end

"""
    configure_accelerator!(; accelerator::Symbol = :auto, blas_threads::Int = 1, verbose::Bool = true)

Configures the hardware-appropriate BLAS/LAPACK backend (`AppleAccelerate` on Apple Silicon
macOS, `MKL` on x86_64 Intel/AMD architectures, or default `OpenBLAS`) and sets the number
of BLAS threads.

When parallelizing across affine blocks with `ThreadsX.map` (`Threads.nthreads() > 1`),
setting `blas_threads = 1` prevents thread oversubscription between Julia worker threads
and internal BLAS threads.
"""
function configure_accelerator!(;
    accelerator::Symbol = :auto,
    blas_threads::Int = 1,
    verbose::Bool = true
)
    blas_threads >= 1 || throw(ArgumentError("blas_threads must be >= 1"))
    active_backend = "OpenBLAS"

    if accelerator == :accelerate || (accelerator == :auto && Sys.isapple() && Sys.ARCH === :aarch64)
        try
            @eval Main using AppleAccelerate
            active_backend = "AppleAccelerate"
        catch e
            verbose && @warn "Could not load AppleAccelerate; falling back to default BLAS." exception=e
        end
    elseif accelerator == :mkl || (accelerator == :auto && Sys.ARCH === :x86_64)
        try
            @eval Main using MKL
            active_backend = "MKL"
        catch e
            verbose && @warn "Could not load MKL; falling back to default BLAS." exception=e
        end
    elseif accelerator == :openblas
        active_backend = "OpenBLAS"
    elseif accelerator != :auto
        throw(ArgumentError("Unknown accelerator option: $accelerator. Use :auto, :accelerate, :mkl, or :openblas."))
    end

    BLAS.set_num_threads(blas_threads)
    if verbose
        cpu_name = isempty(Sys.cpu_info()) ? "unknown CPU" : Sys.cpu_info()[1].model
        @info "Linear algebra accelerator configured" backend=active_backend blas_config=sprint(show, BLAS.get_config()) cpu=cpu_name arch=Sys.ARCH julia_threads=Threads.nthreads() blas_threads=BLAS.get_num_threads()
    end
    return active_backend
end

"""
    balanced_row_ranges(m::Int, blocks::Int) -> Vector{UnitRange{Int}}

Partitions `1:m` into `blocks` contiguous, nonempty row ranges whose sizes differ by at most 1.
The first `m % blocks` ranges have `cld(m, blocks)` rows and the remaining ranges have `fld(m, blocks)` rows.
"""
function balanced_row_ranges(m::Int, blocks::Int)
    1 <= blocks <= m || throw(ArgumentError("require 1 <= blocks <= rows (got blocks=$blocks, rows=$m)"))
    q, r = divrem(m, blocks)
    ranges = Vector{UnitRange{Int}}(undef, blocks)
    first_row = 1
    for i in 1:blocks
        count = q + (i <= r ? 1 : 0)
        ranges[i] = first_row:(first_row + count - 1)
        first_row += count
    end
    return ranges
end

"""
    form_Affine_blocks(A, b, num_blocks::Int, FuncProj)

Partitions `(A, b)` into `num_blocks` balanced row blocks and applies `FuncProj(A_i, b_i)` to each block.
"""
function form_Affine_blocks(A::AbstractMatrix, b::AbstractVector, num_blocks::Int, FuncProj)
    m = size(A, 1)
    length(b) == m || throw(DimensionMismatch("A and b row counts differ"))
    ranges = balanced_row_ranges(m, num_blocks)
    return [FuncProj(A[r, :], b[r]) for r in ranges]
end

"""
    projection_factory(A, b)

Creates a closure computing the orthogonal projection onto the affine subspace `{x : A*x = b}`
using a precomputed column-pivoted QR factorization of `A'` and block-private work buffers.
Because each closure owns its buffers, distinct blocks can be evaluated concurrently across threads.
"""
function projection_factory(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    rows, cols = size(A)
    length(b) == rows || throw(DimensionMismatch("A and b row counts differ"))
    rows <= cols || throw(ArgumentError("projection_factory requires blocks with rows <= cols (got ($rows, $cols))"))
    F = qr(transpose(A), ColumnNorm())
    rhs = similar(b)
    correction = similar(b, cols)
    coordinates = similar(b, rows)

    function project(x::AbstractVector{T})
        iszero(x) ? fill!(rhs, zero(T)) : mul!(rhs, A, x)
        axpy!(-one(T), b, rhs)
        # Since A'[:, F.p] = Q * R, applying A'(A A')^{-1} reduces to Q * (R')^{-1} * rhs[F.p].
        @views rhspivoted = rhs[F.p]
        ldiv!(coordinates, LowerTriangular(F.R'), rhspivoted)
        mul!(correction, F.Q, coordinates)
        return x - correction
    end
    return project
end

"""
    projection_block_QR(A, b, blocks::Int)

Partitions `(A, b)` into `blocks` balanced row blocks and returns a vector of QR-based
projection closures onto the corresponding affine subspaces `U_i = {x : A_i x = b_i}`.
"""
function projection_block_QR(A::AbstractMatrix, b::AbstractVector, blocks::Int)
    return form_Affine_blocks(A, b, blocks, projection_factory)
end

"""
    find_circumcenter!(CC, points)

Computes the circumcenter of the points in `points` and stores the result in `CC`.
`points[1]` is used as the anchor point to construct the Gram matrix.
"""
function find_circumcenter!(CC::AbstractVector{T}, points) where {T<:AbstractFloat}
    m = length(points)
    m >= 1 || throw(ArgumentError("points collection must be nonempty"))
    if m == 1
        copyto!(CC, points[1])
        return CC
    elseif m == 2
        @. CC = 0.5 * (points[1] + points[2])
        return CC
    end

    dimG = m - 1
    directions = [points[i] - points[1] for i in 2:m]
    b = Vector{T}(undef, dimG)
    for i in 1:dimG
        b[i] = dot(directions[i], directions[i])
    end

    G = diagm(b)
    for i in 1:(dimG - 1)
        for j in (i + 1):dimG
            val = dot(directions[i], directions[j])
            G[i, j] = val
            G[j, i] = val
        end
    end

    coefficients = try
        cholesky(Symmetric(G)) \ b
    catch exception
        exception isa PosDefException || rethrow()
        pinv(G) * b
    end

    copyto!(CC, points[1])
    for i in 1:dimG
        axpy!(T(0.5) * coefficients[i], directions[i], CC)
    end
    return CC
end

"""
    circumcenter(points)

Returns a newly allocated vector containing the circumcenter of `points`.
"""
function circumcenter(points)
    return find_circumcenter!(similar(points[1]), points)
end

"""
    step(method::Symbol, x, projections, threaded::Bool; weights=nothing, p0::Real=0.0)

Performs one full iteration of `:CRM` (sequential reflections + circumcenter),
`:pCRM` (parallel/simultaneous reflections + circumcenter), or `:SPM` / `:FSPM`
(simultaneous projections + convex combination).
"""
function step(
    method::Symbol,
    x::AbstractVector{T},
    projections,
    threaded::Bool;
    weights::Union{Nothing,AbstractVector} = nothing,
    p0::Real = zero(T)
) where {T<:AbstractFloat}
    num_blocks = length(projections)
    if method == :CRM
        points = Vector{ Vector{T} }(undef, num_blocks + 1)
        points[1] = copy(x)
        reflected = x
        for (i, project) in enumerate(projections)
            reflected = 2 * project(reflected) - reflected
            points[i + 1] = reflected
        end
        return circumcenter(points)
    elseif method == :pCRM
        reflect = project -> 2 * project(x) - x
        reflected = threaded ? ThreadsX.map(reflect, projections; basesize=1) : map(reflect, projections)
        # Keep the current iterate x at index 1 as the anchor of the affine hull W_x.
        return circumcenter(vcat([x], reflected))
    elseif method in (:SPM, :FSPM)
        projected = threaded ? ThreadsX.map(project -> project(x), projections; basesize=1) :
                               map(project -> project(x), projections)
        combined = zeros(T, length(x))
        if p0 != 0
            axpy!(T(p0), x, combined)
        end
        if isnothing(weights)
            w = (one(T) - T(p0)) / num_blocks
            for point in projected
                axpy!(w, point, combined)
            end
        else
            length(weights) == num_blocks || throw(DimensionMismatch("weights length must match projections length"))
            for (w, point) in zip(weights, projected)
                axpy!(T(w), point, combined)
            end
        end
        return combined
    end
    throw(ArgumentError("unknown method: $method (expected :CRM, :pCRM, :SPM, or :FSPM)"))
end

"""
    solve(method::Symbol, x0, projections; xSol, itmax=10_000, EPSVAL=1e-5, threaded=false, weights=nothing, p0=0.0)

Solves the affine best approximation problem starting from `x0` using `method` (`:CRM`, `:pCRM`, or `:SPM`/`:FSPM`).
Stops when `norm(x - xSol) / norm(xSol) <= EPSVAL` or after `itmax` full iterations.
"""
function solve(
    method::Symbol,
    x0::AbstractVector{T},
    projections;
    xSol::AbstractVector{T},
    itmax::Int = 10_000,
    EPSVAL::Float64 = 1e-5,
    threaded::Bool = false,
    weights::Union{Nothing,AbstractVector} = nothing,
    p0::Real = zero(T)
) where {T<:AbstractFloat}
    method in (:CRM, :pCRM, :SPM, :FSPM) || throw(ArgumentError("unknown method: $method"))
    method == :CRM && threaded && throw(ArgumentError("CRM reflections are sequential; threaded=true is not supported for :CRM"))
    isempty(projections) && throw(ArgumentError("at least one block is required"))
    length(x0) == length(xSol) || throw(DimensionMismatch("x0 and xSol lengths differ"))
    itmax >= 0 || throw(ArgumentError("itmax must be nonnegative"))
    isfinite(EPSVAL) && EPSVAL > 0 || throw(ArgumentError("EPSVAL must be finite and positive"))
    norm_solution = norm(xSol)
    isfinite(norm_solution) && norm_solution > 0 || throw(ArgumentError("xSol must have finite, positive norm"))

    x = copy(x0)
    relative_error = norm(x - xSol) / norm_solution
    iterations = 0
    projections_used = 0
    num_blocks = length(projections)

    while isfinite(relative_error) && relative_error > EPSVAL && iterations < itmax
        x = step(method, x, projections, threaded; weights=weights, p0=p0)
        iterations += 1
        projections_used += num_blocks
        relative_error = norm(x - xSol) / norm_solution
    end

    converged = isfinite(relative_error) && relative_error <= EPSVAL
    status = converged ? "converged" : (!isfinite(relative_error) ? "nonfinite" : "iteration_limit")

    return (
        x = x,
        iterations = iterations,
        projections = projections_used,
        relative_error = relative_error,
        converged = converged,
        status = status,
        iter_total = iterations,
        proj_total = projections_used,
        final_tol = relative_error,
        xApprox = x,
        method = method
    )
end

"""
    CRM(x0, projections, [num_blocks]; xSol, itmax=10_000, EPSVAL=1e-5, kwargs...) -> Results

Sequential Circumcentered-Reflection Method over a collection of affine block projections.
"""
function CRM(
    x0::AbstractVector,
    projections,
    num_blocks::Int = length(projections);
    xSol::AbstractVector,
    itmax::Int = 10_000,
    EPSVAL::Float64 = 1e-5,
    kwargs...
)
    num_blocks == length(projections) || throw(ArgumentError("num_blocks ($num_blocks) must equal length(projections) ($(length(projections)))"))
    res = solve(:CRM, x0, projections; xSol=xSol, itmax=itmax, EPSVAL=EPSVAL, threaded=false)
    return Results(
        iter_total = res.iterations,
        proj_total = res.projections,
        final_tol = res.relative_error,
        xApprox = res.x,
        method = :CRM,
        converged = res.converged,
        status = res.status
    )
end

"""
    pCRM(x0, projections, [num_blocks]; xSol, itmax=10_000, EPSVAL=1e-5, threaded=(Threads.nthreads() > 1), kwargs...) -> Results

Parallelized Circumcentered-Reflection Method (P-CRM) over a collection of affine block projections.
"""
function pCRM(
    x0::AbstractVector,
    projections,
    num_blocks::Int = length(projections);
    xSol::AbstractVector,
    itmax::Int = 10_000,
    EPSVAL::Float64 = 1e-5,
    threaded::Bool = (Threads.nthreads() > 1),
    kwargs...
)
    num_blocks == length(projections) || throw(ArgumentError("num_blocks ($num_blocks) must equal length(projections) ($(length(projections)))"))
    res = solve(:pCRM, x0, projections; xSol=xSol, itmax=itmax, EPSVAL=EPSVAL, threaded=threaded)
    return Results(
        iter_total = res.iterations,
        proj_total = res.projections,
        final_tol = res.relative_error,
        xApprox = res.x,
        method = :pCRM,
        converged = res.converged,
        status = res.status
    )
end

"""
    FSPM(x0, projections, [num_blocks]; xSol, itmax=10_000, EPSVAL=1e-5, threaded=(Threads.nthreads() > 1), weights=nothing, p0=0.0, kwargs...) -> Results

Framework for the Simultaneous Projection Method (F-SPM). By default (`weights=nothing, p0=0.0`),
uses Cimmino's uniform weights `p_i = 1 / length(projections)`.
"""
function FSPM(
    x0::AbstractVector,
    projections,
    num_blocks::Int = length(projections);
    xSol::AbstractVector,
    itmax::Int = 10_000,
    EPSVAL::Float64 = 1e-5,
    threaded::Bool = (Threads.nthreads() > 1),
    weights::Union{Nothing,AbstractVector} = nothing,
    p0::Real = 0.0,
    kwargs...
)
    num_blocks == length(projections) || throw(ArgumentError("num_blocks ($num_blocks) must equal length(projections) ($(length(projections)))"))
    res = solve(:SPM, x0, projections; xSol=xSol, itmax=itmax, EPSVAL=EPSVAL, threaded=threaded, weights=weights, p0=p0)
    return Results(
        iter_total = res.iterations,
        proj_total = res.projections,
        final_tol = res.relative_error,
        xApprox = res.x,
        method = :SPM,
        converged = res.converged,
        status = res.status
    )
end

const SPM = FSPM

end # module ParallelCRM
