################################################################################
# POMCP Policy
################################################################################
struct MultimodalIPPGreedyPolicy{P<:POMDPs.POMDP, RNG<:AbstractRNG} <: Policy
    pomdp::P
    lambda::Float64
    rng::RNG
end

function get_belief_state_mode(b::RoverBelief)
	# NOTE: changed from belief state mode density to just belief state mode
	# all particles will have uniform mode density after resampling whereas the
	# mode itself will have changed so it tells us how much an observation changed
	# the mode rather than the mode density
	particles = b.location_belief.particles[:, b.pos]
	weights = b.location_belief.weights[:, b.pos]
    return particles[argmax(weights)]
end

# function average_mode_density(b::RoverBelief)
#
# 	# particles = b.location_belief.particles[:, b.pos]
# 	# weights = b.location_belief.weights[:, b.pos]
#
#     # sum_entropy = 0.0
# 	#
#     # for loc_belief_state in b.location_belief_states
#     #     sum_entropy += get_belief_state_mode_density(loc_belief_state)
#     # end
#
#     return get_belief_state_mode_density(b)#sum_entropy/length(curr_bel_state.location_belief_states)
# end

function exp_info_gain(pomdp::RoverPOMDP, b::RoverBelief, a::Symbol)
	particles = b.location_belief.particles[:, b.pos]
	weights = b.location_belief.weights[:, b.pos]

    # First sample an observed state for each location
	obs_location_state = StatsBase.sample(pomdp.rng, particles, Weights(weights), 1)[1]
    # obs_location_states = [rand(lbs) for lbs in curr_bel_state.location_belief_states]

    # Now obtain a new belief state
	bp = update_belief(pomdp, b, a, obs_location_state, pomdp.rng)

    information_gain = get_belief_state_mode(bp) - get_belief_state_mode(b)

    return information_gain
end

function expected_drill_utility(pomdp::RoverPOMDP, b::RoverBelief)
	particles = b.location_belief.particles[:, b.pos]
	weights = b.location_belief.weights[:, b.pos]
	exp_val = 0.0

	for i in 1:length(particles)
		if particles[i] in b.drill_samples
			exp_val += weights[i] * pomdp.repeat_sample_penalty
		else
			exp_val += weights[i] * pomdp.new_sample_reward
		end
	end

    return exp_val
end


function POMDPs.action(p::MultimodalIPPGreedyPolicy, b::RoverBelief)
    possible_actions = POMDPs.actions(p.pomdp, b)
    # shuffle!(p.rng, possible_actions)
    action_ratios = Vector{Float64}()

    ratio_sum = 0
    for a in possible_actions
		if a in [:NE, :NW, :SE, :SW]
			action_cost = sqrt(2*p.pomdp.step_size^2)
			exp_utility = p.lambda*exp_info_gain(p.pomdp, b, a)
		elseif a in [:up, :down, :left, :right, :wait]
			action_cost = 1.0*p.pomdp.step_size
			exp_utility = p.lambda*exp_info_gain(p.pomdp, b, a)
		elseif a == :drill
			action_cost = p.pomdp.drill_time
			exp_utility = p.lambda*expected_drill_utility(p.pomdp, b)

			# println("Expected Drill Utility: ", exp_utility)
		end

        action_ratio = exp(exp_utility/action_cost)
        push!(action_ratios, action_ratio)
        ratio_sum = ratio_sum + action_ratio
    end
	# @show action_ratios

    rand_num = rand(p.rng)
    running_sum = 0

    for idx=1:length(action_ratios)
        running_sum += action_ratios[idx]/ratio_sum
        if rand_num < running_sum
            return possible_actions[idx]
        end
    end

    return possible_actions[rand(p.rng, 1:length(possible_actions))]
end



function POMDPs.action(p::MultimodalIPPGreedyPolicy, s::RoverState)
    possible_actions = POMDPs.actions(p.pomdp, s)
    action_ratios = Vector{Float64}()

	for a in possible_actions
		if a in [:NE, :NW, :SE, :SW]
			action_cost = sqrt(2*p.pomdp.step_size^2)
			exp_utility = POMDPs.reward(p.pomdp, s, a) # same as reward(s,a)
		elseif a in [:up, :down, :left, :right, :wait]
			action_cost = 1.0*p.pomdp.step_size
			exp_utility = POMDPs.reward(p.pomdp, s, a, s) # same as reward(s,a)
		elseif a == :drill
			action_cost = p.pomdp.drill_time
			exp_utility = POMDPs.reward(p.pomdp, s, a) # same as reward(s,a)
		end

		push!(action_ratios, exp_utility/action_cost)
	end

    ratio_sum = sum([exp(r) for r in action_ratios])
    for idx=1:length(action_ratios)
        action_ratios[idx] = exp(action_ratios[idx]/ratio_sum)
    end

    rand_num = rand(p.rng)
    running_sum = 0

    for idx=1:length(action_ratios)
        running_sum += action_ratios[idx]
        if rand_num < running_sum
            return possible_actions[idx]
        end
    end

    return possible_actions[rand(p.rng, 1:length(possible_actions))]
end




function get_pomcp_gcb_policy(pomdp, rng,  max_depth=20, queries = 100, lambda=0.5)#0.00001)
    rollout_policy = MultimodalIPPGreedyPolicy(pomdp, lambda, rng)
    value_estimate = PORollout(rollout_policy, RoverBeliefUpdater(pomdp))
    solver = POMCPSolver(rng=rng, estimate_value=value_estimate, max_depth=max_depth, tree_queries = queries)
    pomcp_policy = solve(solver, pomdp)
    return b -> action(pomcp_policy, b)
end

function get_pomcp_basic_policy(pomdp, rng, max_depth=20, queries = 100)
	rollout_policy = RandomPolicy(pomdp, rng=rng, updater=RoverBeliefUpdater(pomdp))
    value_estimate = PORollout(rollout_policy, RoverBeliefUpdater(pomdp))
	solver = POMCPSolver(rng=rng, estimate_value=value_estimate, max_depth=max_depth, tree_queries = queries)


    # solver = POMCPSolver(rng=rng, max_depth=max_depth, tree_queries = queries)
    pomcp_policy = solve(solver, pomdp)
    return b -> action(pomcp_policy, b)
end
