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

    for loc_belief_state in curr_bel_state.location_belief_states
        sum_entropy += get_belief_state_mode_density(loc_belief_state)
    end

    return sum_entropy/length(curr_bel_state.location_belief_states)
end


function exp_info_gain(env::E, curr_bel_state::BS, s::SN) where {E <: Environment, BS <: WorldBeliefState, SN <: Sensor}

    # First sample an observed state for each location
    obs_location_states = [rand(lbs) for lbs in curr_bel_state.location_belief_states]

    # Now obtain a new belief state
    new_bel_state_set = belief_update_location_states_sensor(env, curr_bel_state.location_belief_states,
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

function POMDPs.reward(pomdp::P, s::S, a::MultimodalIPPAction, sp::S) where {P <: POMDPs.POMDP, S <: WorldState}
    return marginal_utility(pomdp.env, sp.current, s.visited, s.location_states)
end

function Base.rand(rng::AbstractRNG, b::BS) where BS <: WorldBeliefState
    S = get_state_of_belstate(eltype(b.location_belief_states))
    location_states = S[]

    for lbs in b.location_belief_states
        sample = rand(rng, lbs)
        push!(location_states, sample)
    end

    return WorldState(b.current, b.visited, location_states, b.cost_expended)
end



function update_belief(pomdp::P, b::BS, a::MultimodalIPPAction, o::O) where {P <: POMDPs.POMDP, BS <: WorldBeliefState, O <: WorldObservation}
    if a.visit_location != nothing
        new_location_belief_states = belief_update_location_states_visit(pomdp.env, b.location_belief_states, o.obs_current)
        bp = BS(o.obs_current, o.obs_visited, new_location_belief_states, o.obs_cost_expended)
    else
        new_location_belief_states = belief_update_location_states_sensor(pomdp.env, b.location_belief_states, o.obs_location_states,
                                                                          o.obs_current, a.sensing_action)
        bp = BS(o.obs_current, o.obs_visited, new_location_belief_states, o.obs_cost_expended)
    end

    return bp
end

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
function display_state(pomdp::AreaCoverage2DPOMDP, s::AreaCoverage2DState, a::MultimodalIPPAction)
    current_loc = pomdp.env.location_metadata[s.current].location
    p = scatter([current_loc[1]], [current_loc[2]], fillcolor = :blue,
                    markercolor = :blue, markersize=20, legend=false)

    if a.visit_location == nothing
        sensing_range = a.sensing_action.effective_range
        scatter!(p, [current_loc[1]], [current_loc[2]], markersize=3.14*(sensing_range^2), markercolor = :blue)
    end

    for v=1:length(pomdp.env.location_states)
        v_loc = pomdp.env.location_metadata[v].location
        if v in s.visited
            if pomdp.env.location_states[v].state == GOOD
                scatter!(p, [v_loc[1]], [v_loc[2]], markersize=15, markercolor = :green, legend=false)
            elseif pomdp.env.location_states[v].state == MEDIUM
                scatter!(p, [v_loc[1]], [v_loc[2]], markersize=10, markercolor = :green, legend=false)
            else
                scatter!(p, [v_loc[1]], [v_loc[2]],  markersize=5, markercolor = :green, legend=false)
            end
        else
            if pomdp.env.location_states[v].state == GOOD
                scatter!(p, [v_loc[1]], [v_loc[2]], markersize=15, markercolor = :red, legend=false)
            elseif pomdp.env.location_states[v].state == MEDIUM
                scatter!(p, [v_loc[1]], [v_loc[2]], markersize=10, markercolor = :red, legend=false)
            else
                scatter!(p, [v_loc[1]], [v_loc[2]], markersize=5, markercolor = :red, legend=false)
            end
        end
    end

    for e in pomdp.env.location_edges
        loc1 = pomdp.env.location_metadata[Int(e[1])].location
        loc2 = pomdp.env.location_metadata[Int(e[2])].location
        plot!(p, [loc1[1], loc2[1]], [loc1[2], loc2[2]], linecolor = :black, linealpha = 0.5, linewidth=0.5, legend = false)
    end

    scatter!(p, [current_loc[1]], [current_loc[2]], markercolor = :blue, markersize=20, legend=false)

    display(p)
end

# NOTE: Common to both
function graph_trial(rng::RNG, pomdp::POMDPs.POMDP, policy, isterminal::Function) where {RNG<:AbstractRNG}

    state = initialstate(pomdp)
    belief_state = initial_belief_state(pomdp)

    state_hist = [deepcopy(state.current)]
    location_states_hist = [deepcopy(state.location_states)]
    action_hist = []
    reward_hist = []
    total_planning_time = 0

    total_reward = 0.0
    while true
        a, t = @timed policy(belief_state)

        total_planning_time += t

        if isterminal(state)
            break
        end

        new_state = generate_s(pomdp, state, a, rng)
        loc_reward = reward(pomdp, state, a, new_state)
        obs = generate_o(pomdp, state, a, new_state, rng)
        belief_state = update_belief(pomdp, belief_state, a, obs)
        total_reward += loc_reward
        state = new_state

        if isterminal(state)
            break
        end
        state_hist = vcat(state_hist, deepcopy(state.current))
        location_states_hist = vcat(location_states_hist, deepcopy(state.location_states))
        action_hist = vcat(action_hist, deepcopy(a))
        reward_hist = vcat(reward_hist, deepcopy(total_reward))
    end

    return total_reward, state_hist, location_states_hist, action_hist, reward_hist, total_planning_time, length(reward_hist)
end

function get_pomcp_gcb_policy(env, pomdp, budget, rng,  max_depth=20, queries = 100, lambda=0.00001)
# function get_pomcp_gcb_policy(env, pomdp, budget, rng,  max_depth=20, queries = 100, lambda=0.1)

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

function get_pomcpow_policy(env, pomdp, budget, rng, max_depth=20, queries = 100)
    solver = POMCPOWSolver(rng=rng, max_depth=max_depth, tree_queries = queries, criterion=MaxUCB(20.0))
    pomcpow_policy = solve(solver, pomdp)
    return b -> action(pomcpow_policy, b)
end


function get_naive_policy(env, ns, budget, num_clusters=5, recursion_depth=2, lambda=0.00025)
    setup(ns, 1, num_clusters, recursion_depth, lambda)
    return b -> plan(ns, b)
end
