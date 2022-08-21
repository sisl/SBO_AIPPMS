# All functions below are common for both environments/POMDPs and use templates
function get_sensors(env::Environment)
    return env.sensors
end

function get_energy_cost(env::Environment, s::Sensor)
    return get_energy_cost(s)
end

function get_location_graph(env::Environment)
    return env.location_graph
end


function average_mode_density(curr_bel_state::BS) where {E <: Environment, BS <: WorldBeliefState}
    sum_entropy = 0.0

    for loc_belief_state in curr_bel_state.gp
        sum_entropy += get_belief_state_mode_density(loc_belief_state)
    end

    return sum_entropy/length(curr_bel_state.gp)
end


function exp_info_gain(env::E, curr_bel_state::BS, s::SN) where {E <: Environment, BS <: WorldBeliefState, SN <: Sensor}

    # First sample an observed state for each location
    obs_location_states = [rand(lbs) for lbs in curr_bel_state.gp]

    # Now obtain a new belief state
    new_bel_state_set = belief_update_location_states_sensor(env, curr_bel_state.gp,
                                                             obs_location_states, curr_bel_state.current, s)
    new_bel_state = BS(curr_bel_state.current, curr_bel_state.visited, new_bel_state_set, curr_bel_state.cost_expended)

    information_gain = average_mode_density(new_bel_state) - average_mode_density(curr_bel_state)

    return information_gain
end


function generate_o(pomdp::P, s::S, a::MultimodalIPPAction, sp::S, rng::RNG) where {P <: POMDPs.POMDP, S <: WorldState, RNG <: AbstractRNG}
    O = obstype(P)
    if a.visit_location != nothing
        o = O(sp.current, sp.visited, sp.location_states, sp.cost_expended)
    else
        new_obs_location_states = sample_location_states(pomdp.env, s.current, a.sensing_action, rng)
        o = O(sp.current, sp.visited, new_obs_location_states, sp.cost_expended)
    end

    return o
end

# NOTE: Can be used for both simulated and true rewards - if sp is true new state
function POMDPs.reward(pomdp::P, s::S, a::MultimodalIPPAction) where {P <: POMDPs.POMDP, S <: WorldState}
    if a.visit_location != nothing
        return marginal_utility(pomdp.env, a.visit_location, s.visited, s.location_states)
    else
        return 0
    end
end

function belief_reward(pomdp::P, b::B, a::MultimodalIPPAction, bp::B) where {P <: POMDPs.POMDP, B <: WorldBeliefState}
    # NOTE: we don't use the TRUE location state any where here (except to check if its actual rock)
    # we only use the GP belief to assign expected rewards since this is called during tree search
    # graph trial reward is called after we have selected an action during a graph trial and we now want to know
    # what the actual reward for that action is
    r = 0.0
    next_visit = bp.current

    if a.visit_location != nothing
        # only check for actual rocks
        if pomdp.env.location_states[next_visit] == RSBAD || pomdp.env.location_states[next_visit] == RSGOOD

            # this is equivalent to ΣR(s,a)b(s) by looking at the mean
            expected_rock_value = b.gp.X == [] ? query_no_data(b.gp)[1][next_visit] : query(b.gp)[1][next_visit]

            # scale expected reward based on expected rock value
            r += expected_rock_value*(pomdp.env.good_rock_reward - pomdp.env.bad_rock_penalty) + pomdp.env.bad_rock_penalty

            # if expected_rock_value < 0.5 # RSBAD
            #     r += pomdp.env.bad_rock_penalty
            # else
            #     r += pomdp.env.good_rock_reward
            # end
        end

    else # sensing action. we don't want to reward for decreasing uncertainty just by visiting bad rocks!
        if b.gp.X == bp.gp.X
            r += 0
            #return marginal_utility(pomdp.env, sp.current, s.visited, s.location_states)
        else
            if b.gp.X == []
                ν_init = query_no_data(b.gp)[2]
            else
                ν_init = query(b.gp)[2]
            end

            if bp.gp.X == []
                ν_posterior = query_no_data(bp.gp)[2]
            else
                ν_posterior = query(bp.gp)[2]
            end

            r += 0.7*(sum(ν_init) - sum(ν_posterior))
            #return marginal_utility(pomdp.env, sp.current, s.visited, s.location_states) + 0.3*(sum(ν_init) - sum(ν_posterior))
        end
    end
    return r
end
# function POMDPs.reward(pomdp::P, s::S, a::MultimodalIPPAction, sp::S) where {P <: POMDPs.POMDP, S <: WorldState}
#     # NOTE: we don't use the TRUE location state any where here (except to check if its actual rock)
#     # we only use the GP belief to assign expected rewards since this is called during tree search
#     # graph trial reward is called after we have selected an action during a graph trial and we now want to know
#     # what the actual reward for that action is
#     r = 0.0
#     next_visit = sp.current
#
#     if a.visit_location != nothing
#         # only check for actual rocks
#         if s.location_states[next_visit] == RSBAD || s.location_states[next_visit] == RSGOOD
#
#             # this is equivalent to ΣR(s,a)b(s) by looking at the mean
#             expected_rock_value = s.gp.X == [] ? query_no_data(s.gp)[1][next_visit] : query(s.gp)[1][next_visit]
#
#             # scale expected reward based on expected rock value
#             r += expected_rock_value*(pomdp.env.good_rock_reward - pomdp.env.bad_rock_penalty) + pomdp.env.bad_rock_penalty
#
#             # if expected_rock_value < 0.5 # RSBAD
#             #     r += pomdp.env.bad_rock_penalty
#             # else
#             #     r += pomdp.env.good_rock_reward
#             # end
#         end
#
#     else # sensing action. we don't want to reward for decreasing uncertainty just by visiting bad rocks!
#         if s.gp.X == sp.gp.X
#             r += 0
#             #return marginal_utility(pomdp.env, sp.current, s.visited, s.location_states)
#         else
#             if s.gp.X == []
#                 ν_init = query_no_data(s.gp)[2]
#             else
#                 ν_init = query(s.gp)[2]
#             end
#
#             if sp.gp.X == []
#                 ν_posterior = query_no_data(sp.gp)[2]
#             else
#                 ν_posterior = query(sp.gp)[2]
#             end
#
#             r += 0.7*(sum(ν_init) - sum(ν_posterior))
#             #return marginal_utility(pomdp.env, sp.current, s.visited, s.location_states) + 0.3*(sum(ν_init) - sum(ν_posterior))
#         end
#     end
#     return r
# end

function graph_trial_reward(pomdp::P, s::S, a::MultimodalIPPAction, sp::S) where {P <: POMDPs.POMDP, S <: WorldState}
    return marginal_utility(pomdp.env, sp.current, s.visited, s.location_states)
end

function Base.rand(rng::AbstractRNG, b::BS) where BS <: WorldBeliefState
    S = get_state_of_belstate(eltype(b.gp))
    location_states = S[]

    for lbs in b.gp
        sample = rand(rng, lbs)
        push!(location_states, sample)
    end

    return WorldState(b.current, b.visited, location_states, b.cost_expended)
end


function update_belief(pomdp::P, b::BS, a::MultimodalIPPAction, o::O) where {P <: POMDPs.POMDP, BS <: WorldBeliefState, O <: WorldObservation}

    if a.visit_location != nothing

         new_visited = union(Set{Int}([a.visit_location]), b.visited)
         visit_cost = pomdp.shortest_paths[b.current, a.visit_location]
         new_cost_expended = b.cost_expended + visit_cost

         # Once a good rock is visited, it goes bad
         # sp = ISRSWorldState(a.visit_location, new_visited, new_location_states, new_cost_expended)
         # o = O(a.visit_location, new_visited, new_location_states, new_cost_expended)

         # If visited twice, turn good rock bad # NOTE: I think this was wrong based on whats in the paper
         # if a.visit_location in s.visited && s.location_states[a.visit_location] == RSGOOD
         if pomdp.env.location_states[a.visit_location] == RSGOOD || pomdp.env.location_states[a.visit_location] == RSBAD
             # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
             # for normal dist whereas our GP setup uses σ²_n
             y = 0.0 #we know it becomes bad if it was good and we know it is bad if it was bad
             σ²_n = 1e-9 # don't square this causes singular exception in belief update
             f_posterior = posterior(b.gp, [[CartesianIndices(pomdp.map_size)[a.visit_location].I[1], CartesianIndices(pomdp.map_size)[a.visit_location].I[2]]], [y], [σ²_n])
             bp = ISRSBeliefState(a.visit_location, new_visited, f_posterior, new_cost_expended)
         else
             bp = ISRSBeliefState(a.visit_location, new_visited, b.gp, new_cost_expended)
         end

     else
         new_cost_expended = b.cost_expended + get_energy_cost(a.sensing_action)

         f_posterior = b.gp#deepcopy(s.gp)

         for (i, loc) in enumerate(pomdp.env.location_states)


             # Only bother if true location is a rock
             if loc == RSGOOD || loc == RSBAD

                 dist = norm(pomdp.env.location_metadata[b.current] - pomdp.env.location_metadata[i])
                 prob_correct = 0.5*(1 + 2^(-4*dist/a.sensing_action.efficiency)) # TODO: Check

                 wrong_loc = (loc == RSGOOD) ? RSBAD : RSGOOD

                 if rand(pomdp.rng) < prob_correct
                     y = (loc == RSGOOD) ? 1.0 : 0.0
                 else
                     y = (wrong_loc == RSGOOD) ? 1.0 : 0.0
                 end

                 # correct = ones(Int(floor(1000*(prob_correct))))
                 # incorrect = zeros(Int(floor(1000*(1-prob_correct))))
                 # σ_n = sum((vcat(correct,incorrect) .-  mean(vcat(correct,incorrect))).^2)/length(vcat(correct,incorrect))

                 # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
                 # for normal dist whereas our GP setup uses σ²_n
                 σ²_n = 1-prob_correct
                 f_posterior = posterior(f_posterior, [[CartesianIndices(pomdp.map_size)[i].I[1], CartesianIndices(pomdp.map_size)[i].I[2]]], [y], [σ²_n])
             end
         end

         bp = ISRSBeliefState(b.current, b.visited, f_posterior, new_cost_expended)
     end

     return bp
 end

# function update_belief(pomdp::P, b::BS, a::MultimodalIPPAction, o::O) where {P <: POMDPs.POMDP, BS <: WorldBeliefState, O <: WorldObservation}
#     if a.visit_location != nothing
#         new_location_belief_states = belief_update_location_states_visit(pomdp.env, b.location_belief_states, o.obs_current)
#         bp = BS(o.obs_current, o.obs_visited, new_location_belief_states, o.obs_cost_expended)
#     else
#         new_location_belief_states = belief_update_location_states_sensor(pomdp.env, b.location_belief_states, o.obs_location_states,
#                                                                           o.obs_current, a.sensing_action)
#         bp = BS(o.obs_current, o.obs_visited, new_location_belief_states, o.obs_cost_expended)
#     end
#
#     return bp
# end

struct MultimodalIPPBeliefUpdater{P<:POMDPs.POMDP} <: Updater
    pomdp::P
end

# function BasicPOMCP.extract_belief(::MultimodalIPPBeliefUpdater, node::BeliefNode)
#     return node.b
# end

function BasicPOMCP.extract_belief(::MultimodalIPPBeliefUpdater, node::BeliefNode)
    return node
end

function POMDPs.initialize_belief(updater::MultimodalIPPBeliefUpdater, d)
    return initial_belief_state(updater.pomdp)
end

# TODO: remove @time
function POMDPs.update(updater::MultimodalIPPBeliefUpdater, b::BS, a::MultimodalIPPAction, o::WO) where {BS <: WorldBeliefState, WO <: WorldObservation}
    ub = update_belief(updater.pomdp, b, a, o)
    return ub
end

struct MultimodalIPPGreedyPolicy{P<:POMDPs.POMDP, RNG<:AbstractRNG} <: Policy
    pomdp::P
    lambda::Float64
    rng::RNG
end

function POMDPs.action(p::MultimodalIPPGreedyPolicy, b::BS) where {BS <: WorldBeliefState}
    possible_actions = POMDPs.actions(p.pomdp, b)
    shuffle!(p.rng, possible_actions)
    action_ratios = Vector{Float64}()

    ratio_sum = 0
    for a in possible_actions
        if a.visit_location != nothing
            # Use weight as edge exists
            action_cost = p.pomdp.env.location_graph.weights[b.current, a.visit_location]
            exp_utility = p.lambda*expected_visit_utility(p.pomdp.env, a.visit_location, b)
        else
            action_cost = a.sensing_action.energy_cost
            exp_utility = exp_info_gain(p.pomdp.env, b, a.sensing_action)
        end
        action_ratio = exp(exp_utility/action_cost)
        push!(action_ratios, action_ratio)
        ratio_sum = ratio_sum + action_ratio
    end

    rand_num = rand(p.rng)
    running_sum = 0

    for idx=1:length(action_ratios)
        running_sum += action_ratios[idx]/ratio_sum
        if rand_num < running_sum
            return possible_actions[idx]
        end
    end

    return possible_actions[rand(length(possible_actions))]
end



function POMDPs.action(p::MultimodalIPPGreedyPolicy, s::S) where {S <: WorldState}
    possible_actions = POMDPs.actions(p.pomdp, s)
    action_ratios = Vector{Float64}()

    for a in possible_actions
        if a.visit_location != nothing
            action_cost = p.pomdp.shortest_paths[s.current, a.visit_location]
            exp_utility = POMDPs.reward(p.pomdp, s, a)
            push!(action_ratios, exp_utility/action_cost)
        else
            action_cost = a.sensing_action.energy_cost
            exp_utility = 1
            push!(action_ratios, exp_utility/action_cost)
        end
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

    return possible_actions[rand(length(possible_actions))]
end


## Common functions for trials


# NOTE: Common to both
# function graph_trial(rng::RNG, pomdp::POMDPs.POMDP, policy, isterminal::Function) where {RNG<:AbstractRNG}
#
#     state = initialstate(pomdp)
#     state_hist = [deepcopy(state.current)]
#     location_states_hist = [deepcopy(state.location_states)]
#     gp_hist = [deepcopy(state.gp)]
#     action_hist = []
#     reward_hist = []
#     # belief_state = initial_belief_state(pomdp)
#
#     total_reward = 0.0
#     while true
#
#         a = policy(state)
#
#
#         if isterminal(state)
#             break
#         end
#
#         new_state = transition(pomdp, state, a, rng)
#         # NOTE: this is the TRUE reward it receives based on the TRUE state whereas POMDPs.reward is the belief dependent reward it receives during tree search and rollouts
#         loc_reward = graph_trial_reward(pomdp, state, a, new_state)
#         # obs = generate_o(pomdp, state, a, new_state, rng)
#         # belief_state = update_belief(pomdp, belief_state, a, obs)
#         total_reward += loc_reward
#         state = new_state
#
#         if isterminal(state)
#             break
#         end
#         # println(a)
#         state_hist = vcat(state_hist, deepcopy(state.current))
#         location_states_hist = vcat(location_states_hist, deepcopy(state.location_states))
#         gp_hist = vcat(gp_hist, deepcopy(state.gp))
#         # action_hist = vcat(action_hist, deepcopy(a))
#         reward_hist = vcat(reward_hist, deepcopy(total_reward))
#     end
#     # println(state_hist)
#     # println(location_states_hist)
#     # println(action_hist)
#     # return total_reward, state_hist, action_hist
#     return total_reward, state_hist, location_states_hist, gp_hist, action_hist, reward_hist
# end

function get_pomcp_gcb_policy(env, pomdp, budget, rng,  max_depth=20, queries = 100, lambda=0.00001)
    rollout_policy = MultimodalIPPGreedyPolicy(pomdp, lambda, rng)
    value_estimate = PORollout(rollout_policy, MultimodalIPPBeliefUpdater(pomdp))
    solver = POMCPSolver(rng=rng, estimate_value=value_estimate, max_depth=max_depth, tree_queries = queries)
    pomcp_policy = solve(solver, pomdp)
    return b -> action(pomcp_policy, b)
end

function get_pomcp_basic_policy(env, pomdp, budget, rng, max_depth=20, queries = 100)
    solver = POMCPSolver(rng=rng, max_depth=max_depth, tree_queries = queries)
    pomcp_policy = solve(solver, pomdp)
    return b -> action(pomcp_policy, b)
end


function get_naive_policy(env, ns, budget, num_clusters=5, recursion_depth=2, lambda=0.00025)
    setup(ns, 1, num_clusters, recursion_depth, lambda)
    return b -> plan(ns, b)
end
