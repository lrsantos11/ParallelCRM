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
       SPM,
       aggregate_table1

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
    BlockQRProjection{T,MT,VT}

Callable struct computing the orthogonal projection onto the affine subspace `{x : A*x = b}`
using a precomputed column-pivoted thin QR factorization of `A'` and block-private work buffers.
Supports both out-of-place `P(x)` and zero-allocation in-place `P(out, x)` evaluation.
"""
struct BlockQRProjection{T<:AbstractFloat,MT<:AbstractMatrix{T},VT<:AbstractVector{T}}
    A::MT
    b_piv::VT
    Q::Matrix{T}
    Rt::Matrix{T}
    piv::Vector{Int}
    rhs::Vector{T}
    correction::Vector{T}
    coordinates::Vector{T}
end

@inline function (P::BlockQRProjection{T})(out::AbstractVector{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    iszero(x) ? fill!(P.rhs, zero(T)) : mul!(P.rhs, P.A, x)
    @inbounds @simd for k in eachindex(P.coordinates, P.piv, P.b_piv)
        P.coordinates[k] = P.rhs[P.piv[k]] - P.b_piv[k]
    end
    ldiv!(LowerTriangular(P.Rt), P.coordinates)
    mul!(P.correction, P.Q, P.coordinates)
    @inbounds @simd for j in eachindex(out, x, P.correction)
        out[j] = x[j] - P.correction[j]
    end
    return out
end

@inline function (P::BlockQRProjection{T})(x::AbstractVector{T}) where {T<:AbstractFloat}
    return P(similar(x), x)
end

@inline function apply_projection!(out::AbstractVector{T}, project::BlockQRProjection{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    return project(out, x)
end

@inline function apply_projection!(out::AbstractVector{T}, project, x::AbstractVector{T}) where {T<:AbstractFloat}
    res = project(x)
    copyto!(out, res)
    return out
end

@inline function apply_reflection!(out::AbstractVector{T}, project::BlockQRProjection{T}, x::AbstractVector{T}) where {T<:AbstractFloat}
    project(out, x)
    @inbounds @simd for j in eachindex(out, x)
        out[j] = T(2) * out[j] - x[j]
    end
    return out
end

@inline function apply_reflection!(out::AbstractVector{T}, project, x::AbstractVector{T}) where {T<:AbstractFloat}
    res = project(x)
    @inbounds @simd for j in eachindex(out, res, x)
        out[j] = T(2) * res[j] - x[j]
    end
    return out
end

"""
    projection_factory(A, b)

Creates a `BlockQRProjection` callable struct computing the orthogonal projection onto the
affine subspace `{x : A*x = b}` using a precomputed column-pivoted thin QR factorization of `A'`
and block-private work buffers. Because each instance owns its buffers and uses explicit dense
`Q` and `R'` factors, distinct blocks can be evaluated concurrently across threads using pure
BLAS-2 operations with zero heap allocations.
"""
function projection_factory(A::AbstractMatrix{T}, b::AbstractVector{T}) where {T<:AbstractFloat}
    rows, cols = size(A)
    length(b) == rows || throw(DimensionMismatch("A and b row counts differ"))
    rows <= cols || throw(ArgumentError("projection_factory requires blocks with rows <= cols (got ($rows, $cols))"))
    A_mat = Matrix{T}(A)
    F = qr(transpose(A_mat), ColumnNorm())
    piv = Vector{Int}(F.p)
    b_piv = Vector{T}(b[piv])
    Q_thin = Matrix{T}(F.Q)[:, 1:rows]
    Rt = Matrix{T}(transpose(F.R))
    rhs = Vector{T}(undef, rows)
    correction = Vector{T}(undef, cols)
    coordinates = Vector{T}(undef, rows)
    return BlockQRProjection{T,Matrix{T},Vector{T}}(A_mat, b_piv, Q_thin, Rt, piv, rhs, correction, coordinates)
end

"""
    projection_block_QR(A, b, blocks::Int)

Partitions `(A, b)` into `blocks` balanced row blocks and returns a vector of QR-based
projection closures onto the corresponding affine subspaces `U_i = {x : A_i x = b_i}`.
"""
function projection_block_QR(A::AbstractMatrix, b::AbstractVector, blocks::Int)
    return form_Affine_blocks(A, b, blocks, projection_factory)
end

function symmetric_pinv_solve!(G_mat::Matrix{T}, b_vec::Vector{T}) where {T<:AbstractFloat}
    dimG = length(b_vec)
    try
        F = eigen!(Symmetric(G_mat))
        max_ev = maximum(abs, F.values)
        if !(isfinite(max_ev) && max_ev > zero(T))
            return fill(one(T) / dimG, dimG)
        end
        cutoff = dimG * eps(T) * max_ev
        vt_b = F.vectors' * b_vec
        @inbounds for i in eachindex(vt_b, F.values)
            λ = F.values[i]
            vt_b[i] = λ > cutoff ? (vt_b[i] / λ) : zero(T)
        end
        return F.vectors * vt_b
    catch
        try
            return qr(G_mat, ColumnNorm()) \ b_vec
        catch
            # Fallback to Cimmino step (average of projections: coefficients = 1 / dimG)
            return fill(one(T) / dimG, dimG)
        end
    end
end

"""
    find_circumcenter!(CC, points; directions=nothing, G=nothing, b=nothing, D_mat=nothing, coeffs=nothing, threaded::Bool=false, fallback_counter=nothing)

Computes the circumcenter of the points in `points` and stores the result in `CC`.
`points[1]` is used as the anchor point to construct the Gram matrix. Optional preallocated
buffers `directions`, `G`, `b`, `D_mat`, and `coeffs` avoid heap allocations inside iterative loops.
"""
function find_circumcenter!(
    CC::AbstractVector{T},
    points;
    directions::Union{Nothing,Vector{Vector{T}}} = nothing,
    G::Union{Nothing,Matrix{T}} = nothing,
    b::Union{Nothing,Vector{T}} = nothing,
    D_mat::Union{Nothing,Matrix{T}} = nothing,
    coeffs::Union{Nothing,Vector{T}} = nothing,
    threaded::Bool = false,
    fallback_counter::Union{Nothing,Base.RefValue{Int}} = nothing
) where {T<:AbstractFloat}
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
    p1 = points[1]
    n_dim = length(p1)
    b_vec = isnothing(b) ? Vector{T}(undef, dimG) : b
    D = (!isnothing(D_mat) && size(D_mat) == (n_dim, dimG)) ? D_mat : Matrix{T}(undef, n_dim, dimG)
    c_vec = (!isnothing(coeffs) && length(coeffs) == dimG) ? coeffs : Vector{T}(undef, dimG)

    if threaded && dimG * n_dim >= 16_000
        Threads.@threads :static for i in 1:dimG
            @inbounds begin
                pi = points[i + 1]
                s = zero(T)
                @simd for k in 1:n_dim
                    val = pi[k] - p1[k]
                    D[k, i] = val
                    s += val * val
                end
                b_vec[i] = s
            end
        end
    else
        @inbounds for i in 1:dimG
            pi = points[i + 1]
            s = zero(T)
            @simd for k in 1:n_dim
                val = pi[k] - p1[k]
                D[k, i] = val
                s += val * val
            end
            b_vec[i] = s
        end
    end

    if dimG > n_dim
        # Overdetermined direction set (m - 1 > n): instead of forming and diagonalizing a singular
        # dimG × dimG Gram matrix in O(dimG^3), solve the minimum-norm system D' * δ = 0.5 * b_vec
        # directly in R^n via column-pivoted QR of D' (size dimG × n_dim, cost O(dimG * n_dim^2)).
        !isnothing(fallback_counter) && (fallback_counter[] += 1)
        try
            @inbounds @simd for i in 1:dimG
                c_vec[i] = T(0.5) * b_vec[i]
            end
            delta = qr(transpose(D), ColumnNorm()) \ c_vec
            @inbounds @simd for k in 1:n_dim
                CC[k] = p1[k] + delta[k]
            end
            return CC
        catch
        end
    end

    G_mat = (!isnothing(G) && size(G) == (dimG, dimG)) ? G : Matrix{T}(undef, dimG, dimG)
    # Fast BLAS Level-3 symmetric rank-k update: G_mat = D' * D (upper triangle)
    BLAS.syrk!('U', 'T', one(T), D, zero(T), G_mat)

    if dimG <= n_dim
        # Backup upper triangle into lower triangle before cholesky! overwrites upper triangle
        @inbounds for j in 2:dimG
            for i in 1:(j - 1)
                G_mat[j, i] = G_mat[i, j]
            end
        end
        copyto!(c_vec, b_vec)
        _, info = LAPACK.potrf!('U', G_mat)
        if info == 0
            LAPACK.potrs!('U', G_mat, c_vec)
            copyto!(CC, p1)
            mul!(CC, D, c_vec, T(0.5), one(T))
            return CC
        end
        !isnothing(fallback_counter) && (fallback_counter[] += 1)
        # Reconstruct G_mat upper triangle from lower triangle backup
        @inbounds for i in 1:dimG
            G_mat[i, i] = b_vec[i]
            for j in (i + 1):dimG
                G_mat[i, j] = G_mat[j, i]
            end
        end
    end

    sol = symmetric_pinv_solve!(G_mat, b_vec)
    copyto!(CC, p1)
    mul!(CC, D, sol, T(0.5), one(T))
    return CC
end

"""
    circumcenter(points)

Returns a newly allocated vector containing the circumcenter of `points`.
"""
function circumcenter(points)
    return find_circumcenter!(similar(points[1]), points)
end

function step!(
    x_next::AbstractVector{T},
    method::Symbol,
    x::AbstractVector{T},
    projections,
    threaded::Bool,
    points::Vector{Vector{T}},
    directions::Vector{Vector{T}},
    G::Matrix{T},
    b_gram::Vector{T},
    D_mat::Union{Nothing,Matrix{T}} = nothing,
    coeffs::Union{Nothing,Vector{T}} = nothing;
    weights::Union{Nothing,AbstractVector} = nothing,
    p0::Real = zero(T),
    fallback_counter::Union{Nothing,Base.RefValue{Int}} = nothing
) where {T<:AbstractFloat}
    num_blocks = length(projections)
    if method == :CRM
        copyto!(points[1], x)
        @inbounds for i in 1:num_blocks
            apply_reflection!(points[i + 1], projections[i], points[i])
        end
        return find_circumcenter!(x_next, points; directions=directions, G=G, b=b_gram, D_mat=D_mat, coeffs=coeffs, threaded=false, fallback_counter=fallback_counter)
    elseif method == :pCRM
        copyto!(points[1], x)
        if threaded
            Threads.@threads :static for i in 1:num_blocks
                @inbounds apply_reflection!(points[i + 1], projections[i], x)
            end
        else
            @inbounds for i in 1:num_blocks
                apply_reflection!(points[i + 1], projections[i], x)
            end
        end
        return find_circumcenter!(x_next, points; directions=directions, G=G, b=b_gram, D_mat=D_mat, coeffs=coeffs, threaded=threaded, fallback_counter=fallback_counter)
    elseif method in (:SPM, :FSPM)
        if threaded
            Threads.@threads :static for i in 1:num_blocks
                @inbounds apply_projection!(points[i], projections[i], x)
            end
        else
            @inbounds for i in 1:num_blocks
                apply_projection!(points[i], projections[i], x)
            end
        end
        fill!(x_next, zero(T))
        if p0 != 0
            axpy!(T(p0), x, x_next)
        end
        if isnothing(weights)
            w = (one(T) - T(p0)) / num_blocks
            @inbounds for i in 1:num_blocks
                axpy!(w, points[i], x_next)
            end
        else
            length(weights) == num_blocks || throw(DimensionMismatch("weights length must match projections length"))
            @inbounds for i in 1:num_blocks
                axpy!(T(weights[i]), points[i], x_next)
            end
        end
        return x_next
    end
    throw(ArgumentError("unknown method: $method (expected :CRM, :pCRM, :SPM, or :FSPM)"))
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
    n = length(x)
    points = [Vector{T}(undef, n) for _ in 1:(num_blocks + 1)]
    directions = [Vector{T}(undef, n) for _ in 1:num_blocks]
    G = Matrix{T}(undef, num_blocks, num_blocks)
    b_gram = Vector{T}(undef, num_blocks)
    D_mat = Matrix{T}(undef, n, num_blocks)
    coeffs = Vector{T}(undef, num_blocks)
    x_next = similar(x)
    return step!(x_next, method, x, projections, threaded, points, directions, G, b_gram, D_mat, coeffs; weights=weights, p0=p0)
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

    n = length(x0)
    num_blocks = length(projections)
    x = copy(x0)
    x_next = similar(x)
    diff = similar(x)
    points = [Vector{T}(undef, n) for _ in 1:(num_blocks + 1)]
    directions = [Vector{T}(undef, n) for _ in 1:num_blocks]
    G = Matrix{T}(undef, num_blocks, num_blocks)
    b_gram = Vector{T}(undef, num_blocks)
    D_mat = Matrix{T}(undef, n, num_blocks)
    coeffs = Vector{T}(undef, num_blocks)

    @. diff = x - xSol
    relative_error = norm(diff) / norm_solution
    iterations = 0
    projections_used = 0
    fallback_counter = Ref(0)

    while isfinite(relative_error) && relative_error > EPSVAL && iterations < itmax
        step!(x_next, method, x, projections, threaded, points, directions, G, b_gram, D_mat, coeffs;
              weights=weights, p0=p0, fallback_counter=fallback_counter)
        copyto!(x, x_next)
        iterations += 1
        projections_used += num_blocks
        @. diff = x - xSol
        relative_error = norm(diff) / norm_solution
    end

    converged = isfinite(relative_error) && relative_error <= EPSVAL
    status = converged ? "converged" : (!isfinite(relative_error) ? "nonfinite" : "iteration_limit")

    return (
        x = x,
        iterations = iterations,
        projections = projections_used,
        cholesky_fallbacks = fallback_counter[],
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

include("reporting.jl")

end # module ParallelCRM
