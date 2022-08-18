mutable struct AreaCoverage2DPOMDP <: POMDPs.POMDP{AreaCoverage2DState, MultimodalIPPAction, AreaCoverage2DObservation}
    env::AreaCoverage2DEnv
    cost_budget::Float64
    actions::Vector{MultimodalIPPAction}
    shortest_paths::Matrix{Real}
end

function AreaCoverage2DPOMDP(env::AreaCoverage2DEnv, cost_budget::Float64)
    action_set = get_action_set(env)
    shortest_paths = Graphs.floyd_warshall_shortest_paths(env.location_graph)
    return AreaCoverage2DPOMDP(env, cost_budget, action_set, shortest_paths.dists)
end

# POMDPs.n_actions(pomdp::AreaCoverage2DPOMDP) = length(pomdp.actions)
# POMDPs.n_states(pomdp::AreaCoverage2DPOMDP) = length(LOCATION_TYPES)^length(pomdp.env.location_metadata)
POMDPs.actions(pomdp::AreaCoverage2DPOMDP) = pomdp.actions
action_index(pomdp::AreaCoverage2DPOMDP, a::MultimodalIPPAction) = a.idx
POMDPs.discount(pomdp::AreaCoverage2DPOMDP) = 1.0

function actions_possible_from_current(pomdp::AreaCoverage2DPOMDP, current::Int,
                                       cost_expended::Float64)
    neighbors = Graphs.neighbors(pomdp.env.location_graph, current)
    possible_actions = Vector{MultimodalIPPAction}()

    for n in neighbors
        pomdp_action = pomdp.actions[n]
        visit_cost = pomdp.shortest_paths[current, pomdp_action.visit_location]
        return_cost = pomdp.shortest_paths[pomdp_action.visit_location, 1]
        if (cost_expended + visit_cost + return_cost) <= pomdp.cost_budget
            push!(possible_actions, pomdp.actions[n])
        end
    end

    num_nodes = length(pomdp.env.location_states)
    for s=1:length(pomdp.env.sensors)
        pomdp_action = pomdp.actions[num_nodes+s]
        sensor_cost = pomdp_action.sensing_action.energy_cost
        return_cost = pomdp.shortest_paths[current, 1]
        if (cost_expended + sensor_cost + return_cost) <= pomdp.cost_budget
            push!(possible_actions, pomdp.actions[num_nodes+s])
        end
    end

    return possible_actions
end

function POMDPs.actions(pomdp::AreaCoverage2DPOMDP, s::AreaCoverage2DState)
    possible_actions = actions_possible_from_current(pomdp, s.current, s.cost_expended)
    return possible_actions
end

function POMDPs.actions(pomdp::AreaCoverage2DPOMDP, o::AreaCoverage2DObservation)
    possible_actions = actions_possible_from_current(pomdp, o.obs_current, o.obs_cost_expended)
    return possible_actions
end

function POMDPs.actions(pomdp::AreaCoverage2DPOMDP, b::AreaCoverage2DBeliefState)
    possible_actions = actions_possible_from_current(pomdp, b.current, b.cost_expended)
    return possible_actions
end

function POMDPs.initialstate(pomdp::AreaCoverage2DPOMDP)
    return AreaCoverage2DState(1, Set{Int}([1]), pomdp.env.location_states, 0.0)
end

function POMDPs.isterminal(pomdp::AreaCoverage2DPOMDP, s::AreaCoverage2DState)
    if s.cost_expended + pomdp.shortest_paths[s.current,1] > pomdp.cost_budget
        return true
    elseif s.current == 1 && length(s.visited) > 1
        neighbors = Graphs.neighbors(pomdp.env.location_graph, 1)
        min_cost_from_start = minimum([pomdp.shortest_paths[1, n] for n in neighbors])
        if (pomdp.cost_budget - s.cost_expended) < 2*min_cost_from_start
            return true
        else
            return false
        end
    else
        return false
    end
end


function generate_s(pomdp::AreaCoverage2DPOMDP, s::AreaCoverage2DState, a::MultimodalIPPAction, rng::RNG) where {RNG <: AbstractRNG}
    if a.visit_location != nothing
        new_visited = union(Set{Int}([a.visit_location]), s.visited)
        visit_cost = pomdp.shortest_paths[s.current, a.visit_location]
        new_cost_expended = s.cost_expended + visit_cost
        sp = AreaCoverage2DState(a.visit_location, new_visited, s.location_states, new_cost_expended)
    else
        new_cost_expended = s.cost_expended + get_energy_cost(a.sensing_action)
        sp = AreaCoverage2DState(s.current, s.visited, s.location_states, new_cost_expended)
    end

    return sp
end

function initial_belief_state(pomdp::AreaCoverage2DPOMDP)
    inital_location_belief_states = Vector{AreaCoverage2DLocationBeliefState}()
    uniform = SVector{3,Float64}([1.0/3.0, 1.0/3.0, 1.0/3.0])

    for i=1:length(pomdp.env.location_states)
        belief_state = AreaCoverage2DLocationBeliefState(SparseCat(LOCATION_TYPES, uniform))
        push!(inital_location_belief_states, belief_state)
    end

    return AreaCoverage2DBeliefState(1, Set{Int}([1]), inital_location_belief_states, 0.0)
end



# NOTE: All functions below work for all environments??
