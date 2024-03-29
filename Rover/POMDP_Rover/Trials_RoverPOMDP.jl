include("CustomGP.jl") # only used for RMSE comparison in POMDP_Rover dir
include("rover_pomdp.jl")
include("pomcp_policies.jl")
include("plot_RoverPOMDP.jl")
using Random
using BasicPOMCP
using POMDPs
using Statistics
using Distributions
using Plots
using KernelFunctions
using DelimitedFiles

################################################################################
# Map Building
################################################################################

function get_neighbors(idx::Int, map_size::Tuple{Int, Int})
	pos = [CartesianIndices(map_size)[idx].I[1], CartesianIndices(map_size)[idx].I[2]]
	neighbors = [pos+[0,1], pos+[0,-1], pos+[1,0], pos+[-1,0]]
	bounds_neighbors = []
	for i in 1:length(neighbors)
		if inbounds(map_size, RoverPos(neighbors[i][1], neighbors[i][2]))
			append!(bounds_neighbors, [neighbors[i]])
		end
	end

	bounds_neighbors_idx = [LinearIndices(map_size)[bounds_neighbors[i][1], bounds_neighbors[i][2]] for i in 1:length(bounds_neighbors)]
	return bounds_neighbors_idx
end

function inbounds(map_size::Tuple{Int, Int}, pos::RoverPos)
    if map_size[1] >= pos[1] > 0 && map_size[2] >= pos[2] > 0
        # i = abs(s[2] - pomdp.map_size[1]) + 1
        # j = s[1]
        return true
    else
        return false
    end
end

function build_map(rng::RNG, number_of_sample_types::Int, map_size::Tuple{Int, Int}) where {RNG<:AbstractRNG}
	sample_types = collect(0:(1/number_of_sample_types):(1-1/number_of_sample_types))
	init_map = rand(rng, sample_types, map_size[1], map_size[2])
	new_map = zeros(map_size)

	p_neighbors = 0.95

	for i in 1:(map_size[1]*map_size[2])
		if i == 1
			continue
		else
			if rand(rng) < p_neighbors
				neighbor_values = init_map[get_neighbors(i, map_size)]
				new_map[i] = round(mean(neighbor_values),digits=1)
				#true_map[i] = true_map[i-1]
			else
				continue
			end
		end
	end

	return new_map
end

################################################################################
# Running Trials
################################################################################

function run_rover_pomdp(rng::RNG, pomdp::POMDPs.POMDP, policy, isterminal::Function) where {RNG<:AbstractRNG}


    state = initialstate(pomdp)
    belief_state = initial_belief_state(pomdp, rng)
	state_hist = [deepcopy(state.pos)]
	belief_hist = [deepcopy(belief_state.location_belief)]
	action_hist = []
	obs_hist = []
	reward_hist = []
	total_reward_hist = []
	total_planning_time = 0


    total_reward = 0.0
    while true
		a, t = @timed policy(belief_state)
		total_planning_time += t

        if isterminal(state)
            break
        end

		# println("State: ", convert_pos_idx_2_pos_coord(pomdp, state.pos))
		# println("Cost Expended: ", state.cost_expended)
		# println("Actions available: ", actions(pomdp, state))
		# println("Action: ", a)

        new_state = generate_s(pomdp, state, a, rng)
        loc_reward = reward(pomdp, state, a, new_state)
        obs = generate_o(pomdp, state, a, new_state, rng)

		# println("Reward: ", loc_reward)
		# println("Observation: ", obs)
		# println("True Value: ", pomdp.true_map[new_state.pos])
		# println("Drill Samples: ", new_state.drill_samples)
		#
		# println("Particles: ", belief_state.location_belief.particles[:, new_state.pos])
		# println("Weights: ", belief_state.location_belief.weights[:, new_state.pos])
		# println("")

        belief_state = update_belief(pomdp, belief_state, a, obs, rng)

		# println("New Particles: ", belief_state.location_belief.particles[:, new_state.pos])
		# println("New Weights: ", belief_state.location_belief.weights[:, new_state.pos])
		# println("")

        total_reward += loc_reward
        state = new_state

        if isterminal(state)
            break
        end

		state_hist = vcat(state_hist, deepcopy(state.pos))
		belief_hist = vcat(belief_hist, deepcopy(belief_state.location_belief))
		action_hist = vcat(action_hist, deepcopy(a))
		obs_hist = vcat(obs_hist, deepcopy(obs))
		reward_hist = vcat(reward_hist, deepcopy(loc_reward))
		total_reward_hist = vcat(total_reward_hist, deepcopy(total_reward))


    end

    return total_reward, state_hist, belief_hist, action_hist, obs_hist, reward_hist, total_reward_hist, total_planning_time, length(reward_hist)

end

function solver_test_RoverPOMDP(pref::String; number_of_sample_types::Int=10, map_size::Tuple{Int, Int}=(10,10), seed::Int64=1234, num_graph_trials=50, total_budget = 100.0, use_ssh_dir=false, plot_results=true)



    pomcp_gcb_rewards = Vector{Float64}(undef, 0)
    pomcp_basic_rewards = Vector{Float64}(undef, 0)
	gp_mcts_rewards = Vector{Float64}(undef, 0)

	rmse_hist_gcb= []
	trace_hist_gcb= []
	total_planning_time_gcb = 0
	total_plans_gcb = 0

	rmse_hist_basic= []
	trace_hist_basic= []
	total_planning_time_basic = 0
	total_plans_basic = 0


    i = 1
    idx = 1
    while idx <= num_graph_trials
        @show i
        rng = MersenneTwister(seed+i)

        true_map = build_map(rng, number_of_sample_types, map_size)
        pomdp = RoverPOMDP(true_map=true_map, cost_budget=total_budget, sample_types=sample_types = collect(0:(1/number_of_sample_types):(1-1/number_of_sample_types)), rng=rng)

        # ns = NaiveSolver(isrs_env, total_budget)

        pomcp_isterminal(s) = POMDPs.isterminal(pomdp, s)
        # naive_isterminal(s) = MultimodalIPP.isterminal_naive(ns, s)

		depth = 5
        pomcp_gcb_policy = get_pomcp_gcb_policy(pomdp, rng, depth, 100)
        pomcp_basic_policy = get_pomcp_basic_policy(pomdp, rng, depth, 100)

        pomcp_gcb_reward = 0.0
        pomcp_basic_reward = 0.0

		# GCB
		pomcp_gcb_reward, state_hist, belief_hist, action_hist, obs_hist, reward_hist, total_reward_hist, planning_time, num_plans = run_rover_pomdp(rng, pomdp, pomcp_gcb_policy, pomcp_isterminal)
		total_planning_time_gcb += planning_time
		total_plans_gcb += num_plans
		rmse_hist_gcb = vcat(rmse_hist_gcb, [calculate_rmse_along_traj(pomdp, true_map, state_hist, belief_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])
		trace_hist_gcb = vcat(trace_hist_gcb, [calculate_trace_Σ(pomdp, true_map, state_hist, belief_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])
		if plot_results
			plot_trial(pomdp.true_map, state_hist, belief_hist, action_hist, total_reward_hist, reward_hist, i, "gcb", use_ssh_dir)
			plot_trial_with_mean(pomdp.true_map, state_hist, belief_hist, action_hist, total_reward_hist, reward_hist, i, "gcb", use_ssh_dir)
		end
		@show pomcp_gcb_reward
		println("average planning time: ", planning_time/num_plans)

		# Basic
		pomcp_basic_reward, state_hist, belief_hist, action_hist, obs_hist, reward_hist, total_reward_hist, planning_time, num_plans = run_rover_pomdp(rng, pomdp, pomcp_basic_policy, pomcp_isterminal)
		total_planning_time_basic += planning_time
		total_plans_basic += num_plans
		rmse_hist_basic = vcat(rmse_hist_basic, [calculate_rmse_along_traj(pomdp, true_map, state_hist, belief_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])
		trace_hist_basic = vcat(trace_hist_basic, [calculate_trace_Σ(pomdp, true_map, state_hist, belief_hist, action_hist, obs_hist, total_reward_hist, reward_hist, i)])
		if plot_results
			plot_trial(pomdp.true_map, state_hist, belief_hist, action_hist, total_reward_hist, reward_hist, i, "basic", use_ssh_dir)
			plot_trial_with_mean(pomdp.true_map, state_hist, belief_hist, action_hist, total_reward_hist, reward_hist, i, "basic", use_ssh_dir)
		end
		@show pomcp_basic_reward
		println("average planning time: ", planning_time/num_plans)



        i = i+1
        idx = idx+1

        push!(pomcp_gcb_rewards, pomcp_gcb_reward)
        push!(pomcp_basic_rewards, pomcp_basic_reward)
    end

	if plot_results
		plot_RMSE_trajectory_history(rmse_hist_gcb, "gcb", use_ssh_dir)
		plot_RMSE_trajectory_history(rmse_hist_basic, "basic", use_ssh_dir)
		plot_trace_Σ_history(trace_hist_gcb, "gcb", use_ssh_dir)
		plot_trace_Σ_history(trace_hist_basic, "basic", use_ssh_dir)
	end

	if use_ssh_dir
		writedlm( "/home/jott2/figures/rmse_hist_gcb.csv",  rmse_hist_gcb, ',')
		writedlm( "/home/jott2/figures/rmse_hist_basic.csv",  rmse_hist_basic, ',')
		writedlm( "/home/jott2/figures/trace_hist_gcb.csv",  trace_hist_gcb, ',')
		writedlm( "/home/jott2/figures/trace_hist_basic.csv",  trace_hist_basic, ',')
	else
		# writedlm( "/Users/joshuaott/icra2022/rmse_hist_gcb.csv",  rmse_hist_gcb, ',')
		# writedlm( "/Users/joshuaott/icra2022/rmse_hist_basic.csv",  rmse_hist_basic, ',')
		# writedlm( "/Users/joshuaott/icra2022/trace_hist_gcb.csv",  trace_hist_gcb, ',')
		# writedlm( "/Users/joshuaott/icra2022/trace_hist_basic.csv",  trace_hist_basic, ',')
	end

	println("POMCP GCB average planning time: ", total_planning_time_gcb/total_plans_gcb)
	println("POMCP Basic average planning time: ", total_planning_time_basic/total_plans_basic)

	@show mean(pomcp_gcb_rewards)
    @show mean(pomcp_basic_rewards)


end


solver_test_RoverPOMDP("test", number_of_sample_types=10, total_budget = 60.0, use_ssh_dir=false, plot_results=false)
