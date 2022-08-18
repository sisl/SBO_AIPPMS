include("CustomGP.jl")
include("rover_pomdp.jl")
include("belief_mdp.jl")
include("plot_RoverBMDP.jl")
using Random
using BasicPOMCP
using POMDPs
using Statistics
using Distributions
using Plots
using KernelFunctions
using MCTS


POMDPs.isterminal(bmdp::BeliefMDP, b::RoverBelief) = isterminal(bmdp.pomdp, b)

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

function run_rover_bmdp(rng::RNG, bmdp::BeliefMDP, policy, isterminal::Function) where {RNG<:AbstractRNG}

    belief_state = initialstate(bmdp).val

    # belief_state = initial_belief_state(bmdp, rng)
	state_hist = [deepcopy(belief_state.pos)]
	gp_hist = [deepcopy(belief_state.location_belief)]
	action_hist = []
	reward_hist = []
	total_reward_hist = []

    total_reward = 0.0
    while true
        a = policy(belief_state)

        if isterminal(belief_state)
            break
        end

		new_belief_state, sim_reward = POMDPs.gen(bmdp, belief_state, a, rng)

		# just use these to get the true reward NOT the simulated reward
		s = RoverState(belief_state.pos, belief_state.visited, bmdp.pomdp.true_map, belief_state.cost_expended, belief_state.drill_samples)
		sp = RoverState(new_belief_state.pos, new_belief_state.visited, bmdp.pomdp.true_map, new_belief_state.cost_expended, new_belief_state.drill_samples)
		true_reward = reward(bmdp.pomdp, s, a, sp)

		if a == :drill
			println("State: ", convert_pos_idx_2_pos_coord(bmdp.pomdp, belief_state.pos))
			println("Cost Expended: ", belief_state.cost_expended)
			println("Actions available: ", actions(bmdp.pomdp, belief_state))
			println("Action: ", a)
			println("True reward: ", true_reward)
			println("Sim reward: ", sim_reward)
			println("True Value: ", bmdp.pomdp.true_map[new_belief_state.pos])

			if belief_state.location_belief.X == []
	            μ_init, ν_init = query_no_data(belief_state.location_belief)
	        else
	            μ_init, ν_init, S_init = query(belief_state.location_belief)
	        end
			if new_belief_state.location_belief.X == []
				μ_post, ν_post = query_no_data(new_belief_state.location_belief)
			else
				μ_post, ν_post, S_post = query(new_belief_state.location_belief)
			end
			println("Mean Value Before Drill: ", μ_init[s.pos])
			println("Mean Value After Drill: ", μ_post[s.pos])

			println("Drill Samples: ", new_belief_state.drill_samples)
			println("")
		end


        # new_state = generate_s(pomdp, state, a, rng)
        # loc_reward = reward(pomdp, state, a, new_state)
        # obs = generate_o(pomdp, state, a, new_state, rng)
		#
		# # println("Reward: ", loc_reward)
		# # println("Observation: ", obs)
		# # println("True Value: ", pomdp.true_map[new_state.pos])
		# # println("Drill Samples: ", new_state.drill_samples)
		# #
		# # # println("Particles: ", belief_state.location_belief.particles[:, new_state.pos])
		# # # println("Weights: ", belief_state.location_belief.weights[:, new_state.pos])
		# # println("")
		#
        # belief_state = update_belief(pomdp, belief_state, a, obs, rng)

		# println("New Particles: ", belief_state.location_belief.particles[:, new_state.pos])
		# println("New Weights: ", belief_state.location_belief.weights[:, new_state.pos])
		# println("")

        total_reward += true_reward
        belief_state = new_belief_state

        if isterminal(belief_state)
            break
        end
		state_hist = vcat(state_hist, deepcopy(belief_state.pos))
		gp_hist = vcat(gp_hist, deepcopy(belief_state.location_belief))
		# # location_states_hist = vcat(location_states_hist, deepcopy(state.location_states))
		action_hist = vcat(action_hist, deepcopy(a))
		reward_hist = vcat(reward_hist, deepcopy(true_reward))
		total_reward_hist = vcat(total_reward_hist, deepcopy(total_reward))

    end

    return total_reward, state_hist, gp_hist, action_hist, reward_hist, total_reward_hist

end


function get_gp_bmdp_policy(bmdp, rng, max_depth=20, queries = 100)
	planner = solve(MCTS.DPWSolver(depth=max_depth, n_iterations=queries, rng=rng, k_state=0.5, k_action=10000.0, alpha_state=0.5), bmdp)
	# planner = solve(MCTSSolver(depth=max_depth, n_iterations=queries, rng=rng), bmdp)

	return b -> action(planner, b)
end

function solver_test_RoverBMDP(pref::String; number_of_sample_types::Int=10, map_size::Tuple{Int, Int}=(10,10), seed::Int64=1234, num_graph_trials=40, total_budget = 100.0)

	# k = with_lengthscale(SqExponentialKernel(), 1.0) + with_lengthscale(MaternKernel(), 1.0)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 1.0) # NOTE: check length scale
    m(x) = 0.5 # default to 0.5 in the middle of the sample spectrum
    X_query = [[i,j] for i = 1:10, j = 1:10]
    query_size = size(X_query)
    X_query = reshape(X_query, size(X_query)[1]*size(X_query)[2])
    KXqXq = K(X_query, X_query, k)
    GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);
    f_prior = GP


	gp_mcts_rewards = Vector{Float64}(undef, 0)


    i = 1
    idx = 1
    while idx <= num_graph_trials
        @show i
        rng = MersenneTwister(seed+i)

        true_map = build_map(rng, number_of_sample_types, map_size)
        pomdp = RoverPOMDP(true_map=true_map, f_prior=f_prior, cost_budget=total_budget, sample_types=sample_types = collect(0:(1/number_of_sample_types):(1-1/number_of_sample_types)), rng=rng)

		bmdp = BeliefMDP(pomdp, RoverBeliefUpdater(pomdp), belief_reward)

        gp_bmdp_isterminal(s) = POMDPs.isterminal(pomdp, s)

		depth = 5
		gp_bmdp_policy = get_gp_bmdp_policy(bmdp, rng, depth, 100)

		gp_mcts_reward = 0

		gp_mcts_reward, state_hist, gp_hist, action_hist, reward_hist, total_reward_hist = run_rover_bmdp(rng, bmdp, gp_bmdp_policy, gp_bmdp_isterminal)
		# plot_trial(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)
		# plot_trial_with_mean(pomdp.true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)
		@show gp_mcts_reward


        i = i+1
        idx = idx+1

        push!(gp_mcts_rewards, gp_mcts_reward)
    end

	@show mean(gp_mcts_rewards)

end


solver_test_RoverBMDP("test", number_of_sample_types=10, total_budget = 100.0)
