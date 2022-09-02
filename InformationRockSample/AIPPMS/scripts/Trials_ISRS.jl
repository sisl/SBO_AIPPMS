# using MultimodalIPP
include("/Users/joshuaott/icra2022/GP_AIPPMS/InformationRockSample/AIPPMS/src/MultimodalIPP.jl")
#include("/home/jott2/GP_AIPPMS/InformationRockSample/AIPPMS/src/MultimodalIPP.jl")

using Graphs
using Random
using BasicPOMCP
using MCTS
using POMDPs
using JSON
using Statistics
using Distributions
using Plots
using DelimitedFiles
using KernelFunctions
using ParticleFilters
include("CustomGP.jl")
include("plot_ISRS.jl")
# include("pomcpdpw.jl")



function solver_test_isrs(pref::String;good_prob::Float64=0.5, num_rocks::Int64=10, num_beacons::Int64=25,
                          seed::Int64=1234, num_graph_trials=50, total_budget = 100.0, use_ssh_dir=false, plot_results=false, log_trace_rmse = false)

    isrs_map_size = (10, 10)

    pos_dist = 1:10

    pomcp_gcb_rewards = Vector{Float64}(undef, 0)
    pomcp_basic_rewards = Vector{Float64}(undef, 0)
	pomcpow_rewards = Vector{Float64}(undef, 0)

	rmse_hist_gcb = []
	trace_hist_gcb = []
	rmse_hist_basic = []
	trace_hist_basic = []
	rmse_hist_pomcpow = []
	trace_hist_pomcpow = []


	total_planning_time_gcb = 0
	total_plans_gcb = 0

	total_planning_time_basic = 0
	total_plans_basic = 0

	total_planning_time_pomcpow = 0
	total_plans_pomcpow = 0


    i = 1
    idx = 1
    while idx <= num_graph_trials
        @show i
        # @show idx
        rng = MersenneTwister(seed+i)

        rocks_positions = ISRSPos[]
        rocks = ISRS_STATE[]
        beacon_positions = ISRSPos[]

        n = 1
        while n <= num_beacons
            beac_pos = (rand(rng, pos_dist), rand(rng, pos_dist))
            if findfirst(isequal(beac_pos), rocks_positions) == nothing &&
                findfirst(isequal(beac_pos), beacon_positions) == nothing
                push!(beacon_positions, beac_pos)
                n = n + 1
            end
        end

        n = 1
        while n <= num_rocks
            rock_pos = (rand(rng, pos_dist), rand(rng, pos_dist))

            if rock_pos != (1, 1) && findfirst(isequal(rock_pos), rocks_positions) == nothing

                push!(rocks_positions, rock_pos)

                if rand(rng) < good_prob
                    rock_state = RSGOOD
                else
                    rock_state = RSBAD
                end
                push!(rocks, rock_state)
                n = n + 1
            end
        end


        pomdp = setup_isrs_pomdp(isrs_map_size, rocks_positions, rocks, beacon_positions, total_budget)
        isrs_env = pomdp.env
        # println(rocks_positions)

        ns = NaiveSolver(isrs_env, total_budget)

        pomcp_isterminal(s) = POMDPs.isterminal(pomdp, s)
        naive_isterminal(s) = MultimodalIPP.isterminal_naive(ns, s)

		depth = 5
        pomcp_gcb_policy = get_pomcp_gcb_policy(isrs_env, pomdp, total_budget, rng, depth, 100)
        pomcp_basic_policy = get_pomcp_basic_policy(isrs_env, pomdp, total_budget, rng, depth, 100)
		pomcpow_policy = get_pomcpow_policy(isrs_env, pomdp, total_budget, rng, depth, 100)
		# pomcpdpw_policy = get_pomcpdpw_policy(isrs_env, pomdp, total_budget, rng, depth, 100)

        pomcp_gcb_reward = 0.0
        pomcp_basic_reward = 0.0
		pomcpow_reward = 0.0
		# pomcpdpw_reward = 0.0


		# pomcpdpw_reward, state_hist, location_states_hist, action_hist, obs_hist, reward_hist, total_reward_hist, planning_time, num_plans = graph_trial(rng, pomdp, pomcpdpw_policy, pomcp_isterminal)
		# total_planning_time_pomcpdpw += planning_time
		# total_plans_pomcpdpw += num_plans
		# if log_trace_rmse
		# 	rmse_hist_pomcpdpw = vcat(rmse_hist_pomcpdpw, [calculate_rmse_along_traj(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])
		# 	trace_hist_pomcpdpw = vcat(trace_hist_pomcpdpw, [calculate_trace_Σ(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])		
		# end
		# if plot_results
		# 	plot_trial(state_hist, location_states_hist, action_hist, total_reward_hist, i, "gcb")
		# end
		# @show pomcpdpw_reward

		pomcpow_reward, state_hist, location_states_hist, action_hist, obs_hist, reward_hist, total_reward_hist, planning_time, num_plans = graph_trial(rng, pomdp, pomcpow_policy, pomcp_isterminal)
		total_planning_time_pomcpow += planning_time
		total_plans_pomcpow += num_plans
		if log_trace_rmse
				rmse_hist_pomcpow = vcat(rmse_hist_pomcpow, [calculate_rmse_along_traj(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])
				trace_hist_pomcpow = vcat(trace_hist_pomcpow, [calculate_trace_Σ(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])		
		end
		if plot_results
			plot_trial(state_hist, location_states_hist, action_hist, total_reward_hist, i, "pomcpow")
		end
		@show pomcpow_reward

		try
			pomcp_gcb_reward, state_hist, location_states_hist, action_hist, obs_hist, reward_hist, total_reward_hist, planning_time, num_plans = graph_trial(rng, pomdp, pomcp_gcb_policy, pomcp_isterminal)
			total_planning_time_gcb += planning_time
			total_plans_gcb += num_plans
			if log_trace_rmse
				rmse_hist_gcb = vcat(rmse_hist_gcb, [calculate_rmse_along_traj(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])
				trace_hist_gcb = vcat(trace_hist_gcb, [calculate_trace_Σ(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])		
			end
			if plot_results
				plot_trial(state_hist, location_states_hist, action_hist, reward_hist, i, "gcb")
			end
			@show pomcp_gcb_reward


			pomcp_basic_reward, state_hist, location_states_hist, action_hist, obs_hist, reward_hist, total_reward_hist, planning_time, num_plans = graph_trial(rng, pomdp, pomcp_basic_policy, pomcp_isterminal)
			total_planning_time_basic += planning_time
			total_plans_basic += num_plans
			if log_trace_rmse 
				rmse_hist_basic = vcat(rmse_hist_basic, [calculate_rmse_along_traj(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])
				trace_hist_basic = vcat(trace_hist_basic, [calculate_trace_Σ(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])		
			end
			if plot_results
				plot_trial(state_hist, location_states_hist, action_hist, reward_hist, i, "basic")
			end
			@show pomcp_basic_reward
		
		catch y
			if isa(y, InterruptException)
                throw(InterruptException)
            end
            pomcp_gcb_reward = 0.0
            pomcp_basic_reward = 0.0
            i = i+1
            continue
		end



        i = i+1
        idx = idx+1

        push!(pomcp_gcb_rewards, pomcp_gcb_reward)
        push!(pomcp_basic_rewards, pomcp_basic_reward)
		# push!(pomcpdpw_rewards, pomcpdpw_reward)
		push!(pomcpow_rewards, pomcpow_reward)

    end

	println("POMCP GCB average planning time: ", total_planning_time_gcb/total_plans_gcb)
	println("POMCP Basic average planning time: ", total_planning_time_basic/total_plans_basic)
	# println("POMCPOW average planning time: ", total_planning_time_pomcpow/total_plans_pomcpow)


    @show mean(pomcp_gcb_rewards)
    @show mean(pomcp_basic_rewards)
	@show mean(pomcpow_rewards)
	# @show mean(pomcpdpw_rewards)

	if log_trace_rmse
		if use_ssh_dir
			writedlm( "/home/jott2/figures/rmse_hist_gcb_ISRS.csv",  rmse_hist_gcb, ',')
			writedlm( "/home/jott2/figures/rmse_hist_basic_ISRS.csv",  rmse_hist_basic, ',')
			writedlm( "/home/jott2/figures/rmse_hist_pomcpow_ISRS.csv",  rmse_hist_pomcpow, ',')
			writedlm( "/home/jott2/figures/trace_hist_gcb_ISRS.csv",  trace_hist_gcb, ',')
			writedlm( "/home/jott2/figures/trace_hist_basic_ISRS.csv",  trace_hist_basic, ',')
			writedlm( "/home/jott2/figures/trace_hist_pomcpow_ISRS.csv",  trace_hist_pomcpow, ',')
		else
			writedlm( "/Users/joshuaott/icra2022/rmse_hist_gcb_ISRS.csv",  rmse_hist_gcb, ',')
			writedlm( "/Users/joshuaott/icra2022/rmse_hist_basic_ISRS.csv",  rmse_hist_basic, ',')
			writedlm( "/Users/joshuaott/icra2022/trace_hist_gcb_ISRS.csv",  trace_hist_gcb, ',')
			writedlm( "/Users/joshuaott/icra2022/trace_hist_basic_ISRS.csv",  trace_hist_basic, ',')
		end
	end
end


solver_test_isrs("test", good_prob=0.5, num_rocks=10, num_beacons=25, total_budget = 100.0, use_ssh_dir=true, plot_results=false, log_trace_rmse = true)
