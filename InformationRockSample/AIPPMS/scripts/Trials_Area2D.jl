using MultimodalIPP
using Graphs
using Random
using BasicPOMCP
using POMDPs
using JSON
using Distributions



function solver_test_area2d(;seed::Int64=1, num_graph_trials=40)

    NUM_NODES = 20
    naive_rewards = Vector{Float64}(undef, 0)
    pomcp_gcb_rewards = Vector{Float64}(undef, 0)
    pomcp_basic_rewards = Vector{Float64}(undef, 0)

    @show seed

    i = 1
    idx = 1
    while idx <= num_graph_trials
        @show i
        rng = MersenneTwister(seed+i)

        rad_thresh = rand(rng, Uniform(0.25,0.35))
        area_2d_env = AreaCoverage2DEnv(NUM_NODES, rad_thresh, rng)
        total_budget = 75.0*rad_thresh

        pomdp = AreaCoverage2DPOMDP(area_2d_env, total_budget)
        ns = NaiveSolver(area_2d_env, total_budget)

        pomcp_isterminal(s) = POMDPs.isterminal(pomdp, s)
        naive_isterminal(s) = MultimodalIPP.isterminal_naive(ns, s)

        pomcp_gcb_policy = get_pomcp_gcb_policy(area_2d_env, pomdp, total_budget, rng)
        pomcp_basic_policy = get_pomcp_basic_policy(area_2d_env, pomdp, total_budget, rng)
        naive_policy = get_naive_policy(area_2d_env, ns, total_budget)

        pomcp_gcb_reward = 0.0
        pomcp_basic_reward = 0.0
        naive_reward = 0.0

        try
            pomcp_gcb_reward = graph_trial(rng, pomdp, pomcp_gcb_policy, pomcp_isterminal)
            @show pomcp_gcb_reward
            pomcp_basic_reward = graph_trial(rng, pomdp, pomcp_basic_policy, pomcp_isterminal)
            @show pomcp_basic_reward
            naive_reward = graph_trial(rng, pomdp, naive_policy, naive_isterminal)
            @show naive_reward
        catch y
            if isa(y, InterruptException)
                throw(InterruptException)
            end
            pomcp_gcb_reward = 0.0
            pomcp_basic_reward = 0.0
            naive_reward = 0.0
            i = i+1
            continue
        end

        i = i+1
        idx = idx+1

        push!(pomcp_gcb_rewards, pomcp_gcb_reward)
        push!(pomcp_basic_rewards, pomcp_basic_reward)
        push!(naive_rewards, naive_reward)
    end

    outfile_pomcp_gcb = string("pomcp-gcb-res-seed-",seed,".json")
    open(outfile_pomcp_gcb,"w") do f
        JSON.print(f,Dict("rewards"=>pomcp_gcb_rewards),2)
    end

    outfile_pomcp_basic = string("pomcp-basic-res-seed-",seed,".json")
    open(outfile_pomcp_basic,"w") do f
        JSON.print(f,Dict("rewards"=>pomcp_basic_rewards),2)
    end

    outfile_naive = string("naive-res-seed-",seed,".json")
    open(outfile_naive,"w") do f
        JSON.print(f,Dict("rewards"=>naive_rewards),2)
    end
end
