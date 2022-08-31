using Graphs
using Random
using BasicPOMCP
using POMDPs
using JSON
using Statistics
using Distributions
using KernelFunctions
using Plots
using MCTS
include("CustomGP.jl")
include("MultimodalIPP.jl")
include("belief_mdp.jl")
include("plot_ISRS.jl")

POMDPs.isterminal(bmdp::BeliefMDP, b::WorldBeliefState) = isterminal(bmdp.pomdp, b)

function POMDPs.actions(bmdp::BeliefMDP, b::ISRSBeliefState)
    possible_actions = actions_possible_from_current(bmdp.pomdp, b.current, b.cost_expended)
    if possible_actions == MultimodalIPPAction[]
        return [MultimodalIPPAction(b.current, nothing, b.current)]
    else
        return possible_actions
    end
end

function initialstate(pomdp::ISRSPOMDP)
    curr = LinearIndices(pomdp.map_size)[pomdp.init_pos[1], pomdp.init_pos[2]]
    return ISRSWorldState(curr,pomdp.env.location_states, 0.0)
end

function initialstate(bmdp::BeliefMDP)
    curr = LinearIndices(bmdp.pomdp.map_size)[bmdp.pomdp.init_pos[1], bmdp.pomdp.init_pos[2]]
    return ISRSBeliefState(curr, bmdp.pomdp.f_prior, 0.0)
end

function run_rock_sample_bmdp(rng::RNG, bmdp::BeliefMDP, policy, isterminal::Function) where {RNG<:AbstractRNG}

    belief_state = initialstate(bmdp)
	true_s = WorldState(belief_state.current, bmdp.pomdp.env.location_states, belief_state.cost_expended)

    # belief_state = initial_belief_state(bmdp, rng)
	state_hist = [deepcopy(belief_state.current)]
	gp_hist = [deepcopy(belief_state.gp)]
	location_states_hist = [deepcopy(true_s.location_states)]
	action_hist = []
	reward_hist = []
	total_reward_hist = []
	total_planning_time = 0

    total_reward = 0.0
    while true
        a, t = @timed policy(belief_state)

		total_planning_time += t

        if isterminal(belief_state)
            break
        end

		new_belief_state, sim_reward = POMDPs.gen(bmdp, belief_state, a, rng)

		# just use these to get the true reward NOT the simulated reward
		true_sp = generate_s(bmdp.pomdp, true_s, a, rng)
		true_reward = reward(bmdp.pomdp, true_s, a, true_sp)

		# println("State: ", convert_pos_idx_2_pos_coord(bmdp.pomdp, belief_state.current))
		# println("Cost Expended: ", belief_state.cost_expended)
		# println("Actions available: ", actions(bmdp.pomdp, belief_state))
		# println("Action: ", a)
		# println("True reward: ", true_reward)
		# println("Sim reward: ", sim_reward)
		# println("")

        total_reward += true_reward
        belief_state = new_belief_state
		true_s = true_sp

        if isterminal(belief_state)
            break
        end
		state_hist = vcat(state_hist, deepcopy(belief_state.current))
		gp_hist = vcat(gp_hist, deepcopy(belief_state.gp))
		location_states_hist = vcat(location_states_hist, deepcopy(true_s.location_states))
		action_hist = vcat(action_hist, deepcopy(a))
		reward_hist = vcat(reward_hist, deepcopy(true_reward))
		total_reward_hist = vcat(total_reward_hist, deepcopy(total_reward))


    end

    return total_reward, state_hist, location_states_hist, gp_hist, action_hist, reward_hist, total_reward_hist, total_planning_time, length(reward_hist)

end



function get_gp_bmdp_policy(bmdp, rng, max_depth=20, queries = 100)
	planner = solve(MCTS.DPWSolver(depth=max_depth, n_iterations=queries, rng=rng, k_state=0.5, k_action=10000.0, alpha_state=0.5), bmdp)
	# planner = solve(MCTSSolver(depth=max_depth, n_iterations=queries, rng=rng), bmdp)

	return b -> action(planner, b)
end



function solver_test_isrs(pref::String;good_prob::Float64=0.5, num_rocks::Int64=10, num_beacons::Int64=25,
                          seed::Int64=1234, num_graph_trials=50, total_budget = 100.0, use_ssh_dir=false, plot_results=false)

    isrs_map_size = (10, 10)
    pos_dist = 1:10


    # k = with_lengthscale(SqExponentialKernel(), 0.5) + with_lengthscale(MaternKernel(), 0.5)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
    m(x) = 0.0 # default to bad
    X_query = [[i,j] for i = 1:10, j = 1:10]
    query_size = size(X_query)
    X_query = reshape(X_query, size(X_query)[1]*size(X_query)[2])
    KXqXq = K(X_query, X_query, k)
    GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);
    f_prior = GP

    i = 1
    idx = 1

	gp_mcts_rewards = Vector{Float64}(undef, 0)
	rmse_hist_gp_mcts = []
	trace_hist_gp_mcts = []
	total_planning_time = 0
	total_plans = 0


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

			# neighbors = [[rock_pos...]+[0,1], [rock_pos...]+[0,-1], [rock_pos...]+[1,0], [rock_pos...]+[-1,0]]
			# for i in 1:length(neighbors)
			# 	for j in 1:length(rock_positions)
			# 		if neighbors[i] == rock_positions[j]
			# 			rock_state = rocks[j]
			# 		end
			# 	end
			# end

			# if rock_pos != (1, 1) && findfirst(isequal(rock_pos), rocks_positions) == nothing
			# Fix repeat rock positions
			if rock_pos != (1, 1) && findfirst([[rock_pos[1],rock_pos[2]] == rocks_positions[i] for i in 1:length(rocks_positions)]) == nothing

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


        pomdp = setup_isrs_pomdp(rng, isrs_map_size, rocks_positions, rocks, beacon_positions, total_budget, f_prior)
		bmdp = BeliefMDP(pomdp, MultimodalIPPBeliefUpdater(pomdp), belief_reward)

        gp_bmdp_isterminal(s) = POMDPs.isterminal(pomdp, s)
        isrs_env = pomdp.env
 
		depth = 5
		gp_bmdp_policy = get_gp_bmdp_policy(bmdp, rng, depth, 100)


        gp_mcts_reward = 0.0
		gp_mcts_reward, state_hist, location_states_hist, gp_hist, action_hist, reward_hist, total_reward_hist, planning_time, num_plans = run_rock_sample_bmdp(rng, bmdp, gp_bmdp_policy, gp_bmdp_isterminal)
		total_planning_time += planning_time
		total_plans += num_plans
		rmse_hist_gp_mcts = vcat(rmse_hist_gp_mcts, [calculate_rmse_along_traj(pomdp, location_states_hist, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist)])
		trace_hist_gp_mcts = vcat(trace_hist_gp_mcts, [calculate_trace_Σ(pomdp, location_states_hist, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)])
		if plot_results
        	plot_trial(state_hist, location_states_hist, gp_hist, action_hist,total_reward_hist, reward_hist,i, "gp_mcts_dpw",use_ssh_dir)
			plot_trial_with_mean(state_hist, location_states_hist, gp_hist, action_hist,total_reward_hist, reward_hist,i, "gp_mcts_dpw",use_ssh_dir)
			plot_RMSE_trajectory(pomdp, location_states_hist, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i, "gp_mcts_dpw", use_ssh_dir)
		end

        @show gp_mcts_reward

        i = i+1
        idx = idx+1

        push!(gp_mcts_rewards, gp_mcts_reward)
    end

	println("average planning time: ", total_planning_time/total_plans)
    @show mean(gp_mcts_rewards)

	if use_ssh_dir
		writedlm( "/home/jott2/figures/rmse_hist_gp_mcts_ISRS.csv",  rmse_hist_gp_mcts, ',')
		writedlm( "/home/jott2/figures/trace_hist_gp_mcts_ISRS.csv",  trace_hist_gp_mcts, ',')
	else
		writedlm( "/Users/joshuaott/icra2022/rmse_hist_gp_mcts_ISRS.csv",  rmse_hist_gp_mcts, ',')
		writedlm( "/Users/joshuaott/icra2022//trace_hist_gp_mcts_ISRS.csv",  trace_hist_gp_mcts, ',')
	end

end

solver_test_isrs("test", good_prob=0.5, total_budget = 100.0, use_ssh_dir=false, plot_results=false)
