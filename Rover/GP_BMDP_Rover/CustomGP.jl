mutable struct GaussianProcess
    m # mean
    mXq # mean function at query points
    k # covariance function
    X # design points
    X_query # query points (assuming these always stay the same)
    y # objective values
    ν # noise variance
    KXX # K(X,X) the points we have measured
    KXqX # K(Xq,X) the points we are querying and we have measured
    KXqXq
end

μ(X, m) = [m(x) for x in X]
# μ(X::Vector{Int64}, m) = reshape([m(x) for x in X][1], length(X))
# μ(X::Vector{Vector{Int64}}, m) = reshape([m(x) for x in X][1], length(X))
# μ(X, m::Interpolations.Extrapolation{Float64, 2, ScaledInterpolation{Float64, 2, Interpolations.BSplineInterpolation{Float64, 2, Matrix{Float64}, BSpline{Linear{Throw{OnGrid}}}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}, BSpline{Linear{Throw{OnGrid}}}, Tuple{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}}}, BSpline{Linear{Throw{OnGrid}}}, Throw{Nothing}}) = reshape([m(x) for x in X][1], length(X))

# Σ(X, k) = [k(x,x′) for x in X, x′ in X]
# K(X, X′, k) = [k(x,x′) for x in X, x′ in X′]
Σ(X, k) = kernelmatrix(k, X, X)
K(X, X′, k) = kernelmatrix(k, X, X′)
# ν(X, K) = [variance(x, X, K) for x in X]

function mvnrand(rng, μ, Σ, inflation=1e-6)
    N = MvNormal(μ, Σ + inflation*I)
    #N = MvNormal(μ, Σ)
    return rand(rng, N)
end
Base.rand(rng, GP, X) = mvnrand(rng, μ(X, GP.m), Σ(X, GP.k))
Base.rand(rng, GP, μ_calc, Σ_calc) = mvnrand(rng, μ_calc, Σ_calc)

function query_no_data(GP::GaussianProcess)
    μₚ = GP.mXq
    S = GP.KXqXq
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ)
end

function query(GP::GaussianProcess)
    # tmp = GP.KXqX / (GP.KXX + diagm(GP.ν .+ 1e-6))
    tmp = GP.KXqX / (GP.KXX + Diagonal(GP.ν .+ 1e-6))
    μₚ = GP.mXq + tmp*(GP.y - μ(GP.X, GP.m))
    S = GP.KXqXq - tmp*GP.KXqX'
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ, S)
end

function posterior(GP::GaussianProcess, X_samp, y_samp, ν_samp)
    if GP.X == []
        #KXX = [GP.k(x,x′) for x in X_samp, x′ in X_samp]
        #KXqX = [GP.k(x,x′) for x in GP.X_query, x′ in X_samp]

        KXX = kernelmatrix(GP.k, X_samp, X_samp)
        KXqX = kernelmatrix(GP.k, GP.X_query, X_samp)

        return GaussianProcess(GP.m, GP.mXq, GP.k, X_samp, GP.X_query, y_samp, ν_samp, KXX, KXqX, GP.KXqXq)
    else
        # a = K(GP.X, X_samp, k)
        # KXX = [GP.KXX a; a' I]
        #
        # KXqX = hcat(GP.KXqX, [GP.k(x,x′) for x in GP.X_query, x′ in X_samp])


        # KXX = zeros((size(GP.KXX)[1]+1,  size(GP.KXX)[2]+1))
        # KXX[1:(size(GP.KXX)[1]+1), 1:(size(GP.KXX)[1]+1)] = KXX
        # KXX[1:end-1, end] = a
        # KXX[end, 1:end-1] = a'
        # KXX[end,end] = 1.0
        #KXX = kernelmatrix(k, [GP.X; X_samp], [GP.X; X_samp])

        #KXqX = kernelmatrix(k, GP.X_query, [GP.X; X_samp])

        # Recomputing kernel every time with kernelmatrix(): 21.203073 seconds (80.14 M allocations: 8.581 GiB, 28.29% gc time)
        # Using hvcat: 5.571435 seconds (4.60 M allocations: 4.458 GiB, 19.92% gc time)
        # Using [ ] : 5.094872 seconds (4.60 M allocations: 4.458 GiB, 18.46% gc time)

        a = kernelmatrix(GP.k, GP.X, X_samp)
        KXX = [GP.KXX a; a' 1.0] #KXX = [GP.KXX a; a' I]
        KXqX = [GP.KXqX kernelmatrix(GP.k, GP.X_query, X_samp)]#hcat(GP.KXqX, kernelmatrix(k, GP.X_query, X_samp))



        return GaussianProcess(GP.m, GP.mXq, GP.k, [GP.X; X_samp], GP.X_query, [GP.y; y_samp], [GP.ν; ν_samp], KXX, KXqX, GP.KXqXq)

    end
end
