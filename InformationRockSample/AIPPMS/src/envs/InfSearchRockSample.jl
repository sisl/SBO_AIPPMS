## Create a ISRS type based on RockSample.jl
## From the grid, generate a 4-connected graph
## Each cell is either a good rock (+10), bad rock (-10), or beacon (0)
## Belief about beacons is correct in initial bel-state and never updated
## Sensing from rocks does nothing - exp info gain is 0 too (for Naive)

const ISRSPos = SVector{2, Int64}

@enum ISRS_STATE RSGOOD=1 RSBAD=2 RSBEACON=3 RSNEITHER = 4

const ISRSWorldState = WorldState{ISRS_STATE}
const ISRSLocationBeliefState = SparseCat{SVector{4,ISRS_STATE}, SVector{4,Float64}}
const ISRSBeliefState = WorldBeliefState{ISRSLocationBeliefState}
const ISRSObservation = WorldObservation{ISRS_STATE}

struct ISRSSensor <: Sensor
    energy_cost::Float64
    efficiency::Float64
end
const ISRSNearSensor = ISRSSensor(0.5, 2.5)
const ISRSFarSensor = ISRSSensor(2.0, 10.0)

get_energy_cost(s::ISRSSensor) = s.energy_cost


@with_kw mutable struct ISRSEnv <: Environment

    location_graph::SimpleWeightedGraph{Int,Float64}
    location_neighbors::KDTree
    location_metadata::Vector{ISRSPos}
    location_states::Vector{ISRS_STATE}
    sensors::SVector{2,ISRSSensor}                  = [ISRSNearSensor, ISRSFarSensor]
    bad_rock_penalty::Float64                       = -10.
    good_rock_reward::Float64                       = 10.
end

get_energy_cost(env::ISRSEnv, s::ISRSSensor) = get_energy_cost(s)

function get_action_set(env::ISRSEnv)

    action_set = Vector{MultimodalIPPAction}(undef, 0)
    n_locations = length(env.location_states)
    idx = 1

    for i=1:n_locations
        push!(action_set, MultimodalIPPAction(i, nothing, idx))
        idx += 1
    end

    for s in env.sensors
        push!(action_set, MultimodalIPPAction(nothing, s, idx))
        idx += 1
    end

    return action_set
end


# Define the ISRS POMDP
@with_kw struct ISRSPOMDP <: POMDPs.POMDP{ISRSWorldState, MultimodalIPPAction, ISRSObservation}

    map_size::Tuple{Int64, Int64}       = (5, 5)
    rocks_positions::Vector{ISRSPos}    = [(1,3), (3,5), (4,4)]
    rocks::Vector{ISRS_STATE}           = [RSGOOD, RSGOOD, RSBAD]
    env::ISRSEnv
    actions::Vector{MultimodalIPPAction}
    shortest_paths::Matrix{Real}
    cost_budget::Float64                = 20.0
    init_pos::ISRSPos                   = (1, 1)
end


function are_grid_nbrs(idx1::Tuple{Int64,Int64}, idx2::Tuple{Int64,Int64})
    return (abs(idx1[1] - idx2[1]) + abs(idx1[2] - idx2[2])) == 1
end

function setup_isrs_pomdp(map_size::Tuple{Int64,Int64}, rocks_positions::Vector{ISRSPos}, rocks::Vector{ISRS_STATE},
                    beacon_positions::Vector{ISRSPos}, budget::Float64)

    k = length(rocks_positions)
    (rows, cols) = map_size
    num_nodes = rows*cols

    location_graph = SimpleWeightedGraph(num_nodes)
    location_metadata = Vector{ISRSPos}(undef, num_nodes)
    location_states = Vector{ISRS_STATE}(undef, num_nodes)

    for r = 1:rows
        for c = 1:cols

            # Set metadata and state
            idx = LinearIndices((rows, cols))[r, c]
            location_metadata[idx] = [r, c]

            # See if good or bad rock
            rock_idx = findfirst(isequal([r, c]), rocks_positions)
            beacon_idx = findfirst(isequal([r, c]), beacon_positions)

            if rock_idx == nothing
                if beacon_idx == nothing
                    location_states[idx] = RSNEITHER
                else
                    location_states[idx] = RSBEACON
                end
            else
                location_states[idx] = rocks[rock_idx]
            end
        end
    end

    # Now iterate over nodes and add if nbrs
    for src = 1:num_nodes
        for dst = 1:num_nodes

            src_tup = CartesianIndices((rows, cols))[src].I
            dst_tup = CartesianIndices((rows, cols))[dst].I

            if are_grid_nbrs(src_tup, dst_tup)
                # we know distance is 1
                wt = 1.0
                add_edge!(location_graph, src, dst, wt)
            end
        end
    end

    kd_tree = KDTree(Vector{SVector{2, Int64}}([s for s in location_metadata]))

    env = ISRSEnv(location_graph=location_graph, location_neighbors=kd_tree, location_metadata=location_metadata, location_states=location_states)

    action_set = get_action_set(env)
    fws = Graphs.floyd_warshall_shortest_paths(env.location_graph)

    return ISRSPOMDP(map_size=map_size,
                     rocks_positions=rocks_positions,
                     rocks=rocks,
                     env=env,
                     actions=action_set,
                     shortest_paths=fws.dists,
                     cost_budget = budget)
end

# POMDPs.n_actions(pomdp::ISRSPOMDP) = length(pomdp.actions)
POMDPs.actions(pomdp::ISRSPOMDP) = pomdp.actions
action_index(pomdp::ISRSPOMDP, a::MultimodalIPPAction) = a.idx
POMDPs.discount(pomdp::ISRSPOMDP) = 0.999999#1.0

function actions_possible_from_current(pomdp::ISRSPOMDP, current::Int,
                                       cost_expended::Float64)
    neighbors = Graphs.neighbors(pomdp.env.location_graph, current)
    possible_actions = Vector{MultimodalIPPAction}()
    init_idx = LinearIndices(pomdp.map_size)[pomdp.init_pos[1], pomdp.init_pos[2]]

    for n in neighbors
        pomdp_action = pomdp.actions[n]
        visit_cost = pomdp.shortest_paths[current, pomdp_action.visit_location]
        return_cost = pomdp.shortest_paths[pomdp_action.visit_location, init_idx]
        if (cost_expended + visit_cost + return_cost) <= pomdp.cost_budget
            push!(possible_actions, pomdp.actions[n])
        end
    end

    # ONLY ALLOW SENSING IF AT BEACON
    if pomdp.env.location_states[current] == RSBEACON

        num_nodes = length(pomdp.env.location_states)
        for s=1:length(pomdp.env.sensors)
            pomdp_action = pomdp.actions[num_nodes+s]
            sensor_cost = pomdp_action.sensing_action.energy_cost
            return_cost = pomdp.shortest_paths[current, init_idx]
            if (cost_expended + sensor_cost + return_cost) <= pomdp.cost_budget
                push!(possible_actions, pomdp.actions[num_nodes+s])
            end
        end
    end

    return possible_actions
end

function POMDPs.actions(pomdp::ISRSPOMDP, s::ISRSWorldState)
    possible_actions = actions_possible_from_current(pomdp, s.current, s.cost_expended)
    return possible_actions
end

function POMDPs.actions(pomdp::ISRSPOMDP, o::ISRSObservation)
    possible_actions = actions_possible_from_current(pomdp, o.obs_current, o.obs_cost_expended)
    return possible_actions
end

function POMDPs.actions(pomdp::ISRSPOMDP, b::ISRSBeliefState)
    possible_actions = actions_possible_from_current(pomdp, b.current, b.cost_expended)
    return possible_actions
end

function POMDPs.initialstate(pomdp::ISRSPOMDP)
    curr = LinearIndices(pomdp.map_size)[pomdp.init_pos[1], pomdp.init_pos[2]]
    return ISRSWorldState(curr, Set{Int}([curr]), pomdp.env.location_states, 0.0)
end

function POMDPs.isterminal(pomdp::ISRSPOMDP, s::ISRSWorldState)

    init_idx = LinearIndices(pomdp.map_size)[pomdp.init_pos[1], pomdp.init_pos[2]]

    if s.cost_expended + pomdp.shortest_paths[s.current, init_idx] > pomdp.cost_budget
        return true
    elseif s.current == init_idx && length(s.visited) > 1
        neighbors = Graphs.neighbors(pomdp.env.location_graph, init_idx)
        min_cost_from_start = minimum([pomdp.shortest_paths[init_idx, n] for n in neighbors])
        if (pomdp.cost_budget - s.cost_expended) < 2*min_cost_from_start
            return true
        else
            return false
        end
    else
        return false
    end
end

## Do generate_o according to Area2D as well
## reward according to RockSample
## For generate_o, use the ISRS formula
## Copy over belief update for sensing and visit - use ISRS formula

# function POMDPs.generate_s(pomdp::ISRSPOMDP, s::ISRSWorldState, a::MultimodalIPPAction, rng::RNG) where {RNG <: AbstractRNG}
function generate_s(pomdp::ISRSPOMDP, s::ISRSWorldState, a::MultimodalIPPAction, rng::RNG) where {RNG <: AbstractRNG}

    if a.visit_location != nothing

        new_location_states = deepcopy(s.location_states)

        # If visited twice, turn good rock bad
        # if a.visit_location in s.visited && s.location_states[a.visit_location] == RSGOOD #NOTE: I believe this is inconsistent with the paper
        if s.location_states[a.visit_location] == RSGOOD
            new_location_states[a.visit_location] = RSBAD
        end

        new_visited = union(Set{Int}([a.visit_location]), s.visited)
        visit_cost = pomdp.shortest_paths[s.current, a.visit_location]
        new_cost_expended = s.cost_expended + visit_cost

        # Once a good rock is visited, it goes bad
        sp = ISRSWorldState(a.visit_location, new_visited, new_location_states, new_cost_expended)
    else
        new_cost_expended = s.cost_expended + get_energy_cost(a.sensing_action)
        sp = ISRSWorldState(s.current, s.visited, s.location_states, new_cost_expended)
    end

    return sp
end

function POMDPTools.ModelTools.obs_weight(p, s, a, sp, o)
    return o.obs_weight
end 

function POMDPs.observation(pomdp::P, s::S, a::MultimodalIPPAction, sp::S) where {P <: POMDPs.POMDP, S <: ISRSWorldState}
    O = obstype(P)
    if a.visit_location != nothing
        o = O(sp.current, sp.visited, sp.location_states, sp.cost_expended, 1.0)
    else
        new_obs_location_states, obs_weight = sample_location_states(pomdp.env, s.current, a.sensing_action, pomdp.rng)
        o = O(sp.current, sp.visited, new_obs_location_states, sp.cost_expended, obs_weight)
    end

    return o
end

function POMDPs.gen(pomdp::ISRSPOMDP, s::ISRSWorldState, a::MultimodalIPPAction, rng::RNG) where {RNG <: AbstractRNG}
    sp = generate_s(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp, rng)
    r = reward(pomdp, s, a, sp)

    return (sp=sp, o=o, r=r)
end

get_state_of_belstate(::Type{ISRSLocationBeliefState}) = ISRS_STATE


# No range constraints here - can sense everything
function sample_location_states(env::ISRSEnv, current::Int, sensor::ISRSSensor, rng::RNG) where {RNG <: AbstractRNG}

    new_location_states = Vector{ISRS_STATE}(undef, length(env.location_states))
    obs_weight = 1

    for (i, loc) in enumerate(env.location_states)

        if loc == RSBEACON || loc == RSNEITHER
            new_location_states[i] = loc
        else
            # Now do the sampling of prob correct
            dist = norm(env.location_metadata[current] - env.location_metadata[i])
            prob_correct = 0.5*(1 + 2^(-4*dist/sensor.efficiency)) # TODO: Check
            wrong_loc = (loc == RSGOOD) ? RSBAD : RSGOOD
            obs_weight *= prob_correct

            if rand(rng) < prob_correct
                new_location_states[i] = loc
            else
                new_location_states[i] = wrong_loc
            end
        end
    end

    return new_location_states, obs_weight
end

# Utility of visiting location, given what I've visited already
# Here it is simple - if good rock visited already
function marginal_utility(env::ISRSEnv, next_visit::Int, visited::Set{Int},
                          location_states::Vector{ISRS_STATE})

    # If rock visited already OR visited location is bad, set penalty, else reward
    if location_states[next_visit] == RSBAD
        return env.bad_rock_penalty
    elseif location_states[next_visit] == RSGOOD
        return env.good_rock_reward
    else
        return 0.0
    end
end


# Belief about beacon is correct and does not update
function initial_belief_state(pomdp::ISRSPOMDP)

    initial_loc_belstates = ISRSLocationBeliefState[]

    beacon_dist = @SVector [0., 0., 1.0, 0.0]
    neither_dist = @SVector [0., 0., 0., 1.0]
    rock_dist = @SVector [0.5, 0.5, 0.0, 0.0]
    state_types = @SVector [RSGOOD, RSBAD, RSBEACON, RSNEITHER]

    for ls in pomdp.env.location_states

        if ls == RSBEACON
            belief_state = ISRSLocationBeliefState(state_types, beacon_dist)
        elseif ls == RSNEITHER
            belief_state = ISRSLocationBeliefState(state_types, neither_dist)
        else
            belief_state = ISRSLocationBeliefState(state_types, rock_dist)
        end

        push!(initial_loc_belstates, belief_state)
    end

    curr = LinearIndices(pomdp.map_size)[pomdp.init_pos[1], pomdp.init_pos[2]]

    return ISRSBeliefState(curr, Set{Int}(curr), initial_loc_belstates, 0.0)
end


function belief_update_location_states_visit(env::ISRSEnv, curr_bel_states::Vector{ISRSLocationBeliefState},
                                             current::Int64)

    new_bel_states = deepcopy(curr_bel_states)
    new_dist = zeros(4)

    state_types = @SVector [RSGOOD, RSBAD, RSBEACON, RSNEITHER]

    # Set element to 1.0 based on enum
    # JOSH
    #new_dist[Int64(env.location_states[current])] = 1.0
    new_dist[2] = 1.0 # we know it becomes bad once we visit 

    new_bel_states[current] = ISRSLocationBeliefState(state_types, convert(SVector{4,Float64}, new_dist))

    return new_bel_states
end


function belief_update_location_states_sensor(env::ISRSEnv, curr_bel_states::Vector{ISRSLocationBeliefState},
                                              obs_location_states::Vector{ISRS_STATE}, current::Int, sensor::ISRSSensor)

    new_bel_states = deepcopy(curr_bel_states)
    state_types = @SVector [RSGOOD, RSBAD, RSBEACON, RSNEITHER]

    for (i, loc) in enumerate(env.location_states)

        # Only bother if true location is a rock
        if loc == RSGOOD || loc == RSBAD

            dist = norm(env.location_metadata[current] - env.location_metadata[i])
            prob_correct = 0.5*(1 + 2^(-4*dist/sensor.efficiency)) # TODO: Check

            curr_distribution = curr_bel_states[i]
            new_distribution = convert(Vector{Float64}, curr_distribution.probs)

            for loc_type in [RSGOOD, RSBAD]

                prior = pdf(curr_distribution, loc_type)

                if loc_type == loc
                    likelihood = prob_correct
                else
                    likelihood = 1.0 - prob_correct
                end

                @assert Int64(loc_type) in [1, 2] "loc_type is $(loc_type)!"

                new_distribution[Int64(loc_type)] = likelihood*prior
            end

            # Normalize
            new_dist_static = convert(SVector{4,Float64}, normalize(new_distribution))

            new_bel_states[i] = ISRSLocationBeliefState(state_types, new_dist_static)
        end
    end

    return new_bel_states
end

## NOTE: mode_density is just the maximum
function get_belief_state_mode_density(belief_state::ISRSLocationBeliefState)
    return maximum(belief_state.probs)
end

get_location_graph(env::ISRSEnv) = env.location_graph


function expected_visit_utility(env::ISRSEnv, next_visit::Int, curr_bel_state::ISRSBeliefState)

    exp_val = curr_bel_state.location_belief_states[next_visit].probs[1] * env.good_rock_reward +
                curr_bel_state.location_belief_states[next_visit].probs[2] * env.bad_rock_penalty

    return exp_val
end
