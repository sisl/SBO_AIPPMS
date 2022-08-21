clim=(0,1)# using MultimodalIPP

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

POMDPs.isterminal(bmdp::BeliefMDP, b::WorldBeliefState) = isterminal(bmdp.pomdp, b)

function POMDPs.actions(bmdp::BeliefMDP, b::ISRSBeliefState)
    possible_actions = actions_possible_from_current(bmdp.pomdp, b.current, b.cost_expended)
    if possible_actions == MultimodalIPPAction[]
        return [MultimodalIPPAction(b.current, nothing, b.current)]
    else
        return possible_actions
    end
end

function plot_trial(state_hist, location_states_hist, gp_hist, action_hist, reward_hist, trial_num)

	# increase GP query resolution for plotting
	# k = with_lengthscale(SqExponentialKernel(), 0.5) + with_lengthscale(MaternKernel(), 0.5)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
	plot_scale = 1:0.18:10
	X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
    #GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

	loc_states = deepcopy(location_states_hist[1])

    anim = @animate for i = 1:length(state_hist)

		# increase GP query resolution for plotting
		if gp_hist[i].X == []
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
		else
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
		end


		good_rocks = findall(x-> x == RSGOOD, loc_states)
		good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

		bad_rocks = findall(x-> x == RSBAD, loc_states)
		bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

		beacons = findall(x-> x == RSBEACON, loc_states)
		beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


		# Display Total Reward
		if i == 1
			title = "Total Reward: $(reward_hist[1])"
		else
			title = "Total Reward: $(reward_hist[i-1])"
		end


		# Gaussian Process Variance
		if gp_hist[i].X == []
			contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) # xlims = (1, 10), ylims = (1, 10)
			# contourf(collect(1:0.3:9.7), collect(1:0.3:9.7), reshape(query_no_data(gp_hist[i])[2], (30,30))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
		else
	    	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		end


	    # Beacons
		scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect)

		# Rocks
		scatter!([good_rocks[i][1] for i in 1:length(good_rocks)], [good_rocks[i][2] for i in 1:length(good_rocks)], legend=false, color=:green)
		scatter!([bad_rocks[i][1] for i in 1:length(bad_rocks)], [bad_rocks[i][2] for i in 1:length(bad_rocks)], legend=false, color=:red)

		# Agent location
		scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title)

		# Update rock visit
		if loc_states[state_hist[i]] == RSGOOD
			# println(loc_states[state_hist[i]])
			loc_states[state_hist[i]] = RSBAD
			# println(loc_states[state_hist[i]])
		end
		# println(location_states_hist)
	end
	Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS/variance/GPISRS$(trial_num).gif", fps = 1)
	# Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).gif", fps = 1)


	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
	loc_states = deepcopy(location_states_hist[1])

	gp = GaussianProcess(gp_hist[end].m, μ(X_plot, gp_hist[end].m), k, gp_hist[end].X, X_plot, gp_hist[end].y, gp_hist[end].ν, gp_hist[end].KXX, K(X_plot, gp_hist[end].X, k), KXqXq);


	good_rocks = findall(x-> x == RSGOOD, loc_states)
	good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

	bad_rocks = findall(x-> x == RSBAD, loc_states)
	bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

	beacons = findall(x-> x == RSBEACON, loc_states)
	beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


	# Display Total Reward
	title = "Total Reward: $(reward_hist[end-1])"

	# Gaussian Process Variance
	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

	# Beacons
	scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect)

	# Rocks
	scatter!([good_rocks[i][1] for i in 1:length(good_rocks)], [good_rocks[i][2] for i in 1:length(good_rocks)], legend=false, color=:green)
	scatter!([bad_rocks[i][1] for i in 1:length(bad_rocks)], [bad_rocks[i][2] for i in 1:length(bad_rocks)], legend=false, color=:red)

	# Agent location
	plot!([CartesianIndices((10,10))[state_hist[i]].I[1] for i in 1:length(state_hist)],[CartesianIndices((10,10))[state_hist[i]].I[2] for i in 1:length(state_hist)],legend=false, color=:orchid1, linestyle=:dashdotdot, linewidth=2)
	savefig("/Users/joshuaott/icra2022/figures/GPISRS/variance/GPISRS$(trial_num).pdf")
	# savefig("/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).pdf")


end

function plot_trial_with_mean(state_hist, location_states_hist, gp_hist, action_hist, reward_hist, trial_num)

	# increase GP query resolution for plotting
	# k = with_lengthscale(SqExponentialKernel(), 0.5) + with_lengthscale(MaternKernel(), 0.5)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
	plot_scale = 1:0.18:10
	X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
    #GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

	loc_states = deepcopy(location_states_hist[1])

    anim = @animate for i = 1:length(state_hist)

		# increase GP query resolution for plotting
		if gp_hist[i].X == []
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
		else
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
		end


		good_rocks = findall(x-> x == RSGOOD, loc_states)
		good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

		bad_rocks = findall(x-> x == RSBAD, loc_states)
		bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

		beacons = findall(x-> x == RSBEACON, loc_states)
		beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


		# Display Total Reward
		if i == 1
			title = "Total Reward: $(reward_hist[1])"
		else
			title = "Total Reward: $(reward_hist[i-1])"
		end


		# Gaussian Process Variance
		if gp_hist[i].X == []
			contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) # xlims = (1, 10), ylims = (1, 10)
			# contourf(collect(1:0.3:9.7), collect(1:0.3:9.7), reshape(query_no_data(gp_hist[i])[2], (30,30))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
		else
	    	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
		end


	    # Beacons
		scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect)

		# Rocks
		scatter!([good_rocks[i][1] for i in 1:length(good_rocks)], [good_rocks[i][2] for i in 1:length(good_rocks)], legend=false, color=:green)
		scatter!([bad_rocks[i][1] for i in 1:length(bad_rocks)], [bad_rocks[i][2] for i in 1:length(bad_rocks)], legend=false, color=:red)

		# Agent location
		scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title)

		# Update rock visit
		if loc_states[state_hist[i]] == RSGOOD
			# println(loc_states[state_hist[i]])
			loc_states[state_hist[i]] = RSBAD
			# println(loc_states[state_hist[i]])
		end
		# println(location_states_hist)
	end
	Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS/mean/mean_GPISRS$(trial_num).gif", fps = 1)
	# Plots.gif(anim, "/Users/joshuaott/icra2022/figures/mean_GPISRS$(trial_num).gif", fps = 1)

	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
	loc_states = deepcopy(location_states_hist[1])

	gp = GaussianProcess(gp_hist[end].m, μ(X_plot, gp_hist[end].m), k, gp_hist[end].X, X_plot, gp_hist[end].y, gp_hist[end].ν, gp_hist[end].KXX, K(X_plot, gp_hist[end].X, k), KXqXq);


	good_rocks = findall(x-> x == RSGOOD, loc_states)
	good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

	bad_rocks = findall(x-> x == RSBAD, loc_states)
	bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

	beacons = findall(x-> x == RSBEACON, loc_states)
	beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


	# Display Total Reward
	title = "Total Reward: $(reward_hist[end-1])"

	# Gaussian Process Variance
	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

	# Beacons
	scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect)

	# Rocks
	scatter!([good_rocks[i][1] for i in 1:length(good_rocks)], [good_rocks[i][2] for i in 1:length(good_rocks)], legend=false, color=:green)
	scatter!([bad_rocks[i][1] for i in 1:length(bad_rocks)], [bad_rocks[i][2] for i in 1:length(bad_rocks)], legend=false, color=:red)

	# Agent location
	plot!([CartesianIndices((10,10))[state_hist[i]].I[1] for i in 1:length(state_hist)],[CartesianIndices((10,10))[state_hist[i]].I[2] for i in 1:length(state_hist)],legend=false, color=:orchid1, linestyle=:dashdotdot, linewidth=2)
	savefig("/Users/joshuaott/icra2022/figures/GPISRS/mean/mean_GPISRS$(trial_num).pdf")
	# savefig("/Users/joshuaott/icra2022/figures/mean_GPISRS$(trial_num).pdf")

end

function initialstate(pomdp::ISRSPOMDP)
    curr = LinearIndices(pomdp.map_size)[pomdp.init_pos[1], pomdp.init_pos[2]]
    return ISRSWorldState(curr, Set{Int}([curr]), pomdp.env.location_states, 0.0)
end

function initialstate(bmdp::BeliefMDP)
    curr = LinearIndices(bmdp.pomdp.map_size)[bmdp.pomdp.init_pos[1], bmdp.pomdp.init_pos[2]]
    return ISRSBeliefState(curr, Set{Int}(curr), bmdp.pomdp.f_prior, 0.0)
end

function run_rock_sample_bmdp(rng::RNG, bmdp::BeliefMDP, policy, isterminal::Function) where {RNG<:AbstractRNG}

    belief_state = initialstate(bmdp)
	true_s = WorldState(belief_state.current, belief_state.visited, bmdp.pomdp.env.location_states, belief_state.cost_expended)

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

		# if a == :drill
		# 	println("State: ", convert_pos_idx_2_pos_coord(bmdp.pomdp, belief_state.pos))
		# 	println("Cost Expended: ", belief_state.cost_expended)
		# 	println("Actions available: ", actions(bmdp.pomdp, belief_state))
		# 	println("Action: ", a)
		# 	println("True reward: ", true_reward)
		# 	println("Sim reward: ", sim_reward)
		# 	println("True Value: ", bmdp.pomdp.true_map[new_belief_state.pos])
		#
		# 	if belief_state.location_belief.X == []
	    #         μ_init, ν_init = query_no_data(belief_state.location_belief)
	    #     else
	    #         μ_init, ν_init, S_init = query(belief_state.location_belief)
	    #     end
		# 	if new_belief_state.location_belief.X == []
		# 		μ_post, ν_post = query_no_data(new_belief_state.location_belief)
		# 	else
		# 		μ_post, ν_post, S_post = query(new_belief_state.location_belief)
		# 	end
		# 	println("Mean Value Before Drill: ", μ_init[s.pos])
		# 	println("Mean Value After Drill: ", μ_post[s.pos])
		#
		# 	println("Drill Samples: ", new_belief_state.drill_samples)
		# 	println("")
		# end


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
                          seed::Int64=1234, num_graph_trials=40)

    isrs_map_size = (10, 10)
    total_budget = 100.

    pos_dist = 1:10

	gp_mcts_rewards = Vector{Float64}(undef, 0)


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

        # pomcp_isterminal(s) = POMDPs.isterminal(pomdp, s)

        # c = 1.0
        # U(s)=RandomRollout(rng, pomdp, s, depth)
        # π_target=MonteCarloTreeSearch(rng, pomdp, Dict(), Dict(), n_iter, depth, c, U)
        # policy = π_target
		depth = 5
		gp_bmdp_policy = get_gp_bmdp_policy(bmdp, rng, depth, 100)


        gp_mcts_reward = 0.0
		gp_mcts_reward, state_hist, location_states_hist, gp_hist, action_hist, reward_hist, total_reward_hist, planning_time, num_plans = run_rock_sample_bmdp(rng, bmdp, gp_bmdp_policy, gp_bmdp_isterminal)
		total_planning_time += planning_time
		total_plans += num_plans
		# gp_mcts_reward, state_hist, location_states_hist, gp_hist, action_hist, reward_hist = graph_trial(rng, pomdp, gp_bmdp_policy, gp_bmdp_isterminal)
        # plot_trial(state_hist, location_states_hist, gp_hist, action_hist, reward_hist, i)
		# plot_trial_with_mean(state_hist, location_states_hist, gp_hist, action_hist, reward_hist, i)

        @show gp_mcts_reward

        i = i+1
        idx = idx+1

        push!(gp_mcts_rewards, gp_mcts_reward)
    end

    @show mean(gp_mcts_rewards)

end

solver_test_isrs("test", good_prob=0.5)
