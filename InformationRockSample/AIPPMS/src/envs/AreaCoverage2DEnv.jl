# using RandomNumbers.MersenneTwisters

# LOCATION STATE-UTILITY BLOC
@enum LOCATION_STATE GOOD=1 MEDIUM=2 BAD=3
const STATE_TO_RADIUS = Dict(GOOD=>4, MEDIUM=>2, BAD=>0)

# Every location has some true location state which captures its state
struct AreaCoverage2DLocationState <: LocationState
    state::LOCATION_STATE
end

const LOCATION_TYPES = SVector{3,AreaCoverage2DLocationState}([AreaCoverage2DLocationState(GOOD),
                                                               AreaCoverage2DLocationState(MEDIUM),
                                                               AreaCoverage2DLocationState(BAD)])

function AreaCoverage2DLocationState(rng::AbstractRNG)
    rand_num = Base.rand(rng)
    if rand_num <= 1.0/3.0
        return AreaCoverage2DLocationState(GOOD)
    elseif rand_num <= 2/3.0
        return AreaCoverage2DLocationState(MEDIUM)
    else
        return AreaCoverage2DLocationState(BAD)
    end
end

struct AreaCoverage2DLocationBeliefState <: LocationBeliefState
    belstate::SparseCat{SVector{3,AreaCoverage2DLocationState}, SVector{3,Float64}}
end

get_state_of_belstate(::Type{AreaCoverage2DLocationBeliefState}) = AreaCoverage2DLocationState

function Base.rand(rng::AbstractRNG, b::AreaCoverage2DLocationBeliefState)
    r = rand(rng, b.belstate)
    return r
    # tot = zero(Float64)
    # for (v, p) in dist
    #     tot += p
    #     if r < tot
    #         return v
    #     end
    # end
end

struct AreaCoverage2DLocationMetaData
    location::SVector{2,Float64}
end

function AreaCoverage2DLocationMetaData(rng::AbstractRNG)
    locs = Base.rand(rng, 2)
    location = SVector(locs[1],locs[2])
    return AreaCoverage2DLocationMetaData(location)
end

function get_metadata_distance(loc1::AreaCoverage2DLocationMetaData,
                               loc2::AreaCoverage2DLocationMetaData)
    return get_metadata_distance(loc1.location, loc2.location)
end

function get_metadata_distance(loc1::SVector{2,Float64},loc2::SVector{2,Float64})
    return sqrt((loc1[1] - loc2[1])^2 + (loc1[2] - loc2[2])^2)
end

AreaCoverage2DState = WorldState{AreaCoverage2DLocationState}
AreaCoverage2DBeliefState = WorldBeliefState{AreaCoverage2DLocationBeliefState}
AreaCoverage2DObservation = WorldObservation{AreaCoverage2DLocationState}

struct AreaCoverage2DEnv <: Environment
    location_graph::SimpleWeightedGraph{Int,Float64}
    location_edges::Vector{SVector{2,Float64}}

    location_neighbors::KDTree
    location_coverage::Dict{Tuple{Int,LOCATION_STATE},Set{Int}}

    location_metadata::Vector{AreaCoverage2DLocationMetaData}
    location_states::Vector{AreaCoverage2DLocationState}
    sensors::Vector{Sensor}
end

function get_tiles_covered(center_loc::Int, radius::Int, grid_length::Int)
    center_loc_x = center_loc % grid_length
    center_loc_y = center_loc // grid_length

    tiles_covered = Set{Int}()
    for offset_x=-radius:radius
        for offset_y=-radius:radius
            tile_x = center_loc_x + offset_x
            tile_y = center_loc_y + offset_y

            if (tile_x > 0) && (tile_y > 0) && (tile_x <= grid_length) && (tile_y <= grid_length)
               tile = tile_y*grid_length + tile_x
               push!(tiles_covered, tile)
            end
        end
    end

    return tiles_covered
end

function AreaCoverage2DEnv(num_nodes::Int, connection_radius::Float64, rng::AbstractRNG,
                           sensors::Vector{Sensor}=[FarLidarSensor(),NearLidarSensor(),HDCamSensor()])

    location_metadata = Vector{AreaCoverage2DLocationMetaData}()
    location_states = Vector{AreaCoverage2DLocationState}()
    location_graph = SimpleWeightedGraph(num_nodes)
    location_edges = Vector{SVector{2,Float64}}()

    while true

        for idx=1:num_nodes
            new_metadata::AreaCoverage2DLocationMetaData = AreaCoverage2DLocationMetaData(rng)
            new_state::AreaCoverage2DLocationState = AreaCoverage2DLocationState(rng)
            push!(location_metadata, new_metadata)
            push!(location_states, new_state)
        end

        for src=1:num_nodes
            for dst=1:num_nodes
                if src == dst
                    continue
                end

                dist = get_metadata_distance(location_metadata[src],location_metadata[dst])
                if dist < connection_radius
                    # This is done so coverage distance is of the scale of sensing energy
                    wt = 10*dist
                    add_edge!(location_graph, src, dst, wt)
                    push!(location_edges, SVector{2,Float64}([src, dst]))
                end
            end
        end

        if length(connected_components(location_graph)) == 1
            break
        end

        location_metadata = Vector{AreaCoverage2DLocationMetaData}()
        location_states = Vector{AreaCoverage2DLocationState}()
        location_graph = SimpleWeightedGraph(num_nodes)
        location_edges = Vector{SVector{2,Float64}}()
    end

    kd_tree = KDTree(Vector{SVector{2,Float64}}([s.location for s in location_metadata]))

    location_coverage = Dict{Tuple{Int,LOCATION_STATE},Set{Int}}()
    grid_length = 10*num_nodes

    for node=1:num_nodes
        for t in LOCATION_TYPES
            state_type_radius = STATE_TO_RADIUS[t.state]
            location_coverage[(node,t.state)] = get_tiles_covered(node, state_type_radius, grid_length)
        end
    end

    return AreaCoverage2DEnv(location_graph, location_edges, kd_tree, location_coverage,
                             location_metadata, location_states, sensors)
end

function get_action_set(env::AreaCoverage2DEnv)
    action_set = Vector{MultimodalIPPAction}(undef,0)
    n_locations = length(env.location_states)
    idx = 1

    # Add actions for visiting location
    for i=1:n_locations
        push!(action_set,MultimodalIPPAction(i,nothing,idx))
        idx += 1
    end

    # Add sensors
    for s in env.sensors
        push!(action_set,MultimodalIPPAction(nothing,s,idx))
        idx+=1
    end

    return action_set
end

function get_tiles_covered_by_visited(env::AreaCoverage2DEnv, visited::Set{Int},
                                      location_states::Vector{AreaCoverage2DLocationState})
    all_covered_tiles = Set{Int}()

    for loc in visited
        state_type = location_states[loc].state
        all_covered_tiles = union(all_covered_tiles, env.location_coverage[(loc,state_type)])
    end

    return all_covered_tiles
end

function utility(env::AreaCoverage2DEnv, visited::Set{Int}, location_states::Vector{AreaCoverage2DLocationState})
    return 10*length(get_tiles_covered_by_visited(env, visited, location_states))
end

function marginal_utility(env::AreaCoverage2DEnv, next_visit::Int, visited::Set{Int},
                          location_states::Vector{AreaCoverage2DLocationState})
    new_tiles = setdiff(env.location_coverage[(next_visit,location_states[next_visit].state)],
                        get_tiles_covered_by_visited(env, visited, location_states))
    return 10*length(new_tiles)
end

# Note: can probably speed this up a lot by ignoring an sampled state if it has probability 0
function expected_visit_utility(env::AreaCoverage2DEnv, next_visit::Int, curr_bel_state::AreaCoverage2DBeliefState)
    num_visited = length(curr_bel_state.visited)
    coverages = Vector{Set{Int}}()
    coverage_probabilities = Vector{Float64}()

    for t in LOCATION_TYPES
        belief_state = curr_bel_state.location_belief_states[next_visit].belstate
        probability_of_type = pdf(belief_state, t)

        if probability_of_type == 0
            continue
        end

        push!(coverage_probabilities, probability_of_type)

        coverage = env.location_coverage[(next_visit, t.state)]
        push!(coverages, coverage)
    end

    for visit in curr_bel_state.visited
        length_i_coverages = length(coverages)
        for coverage_idx=1:length_i_coverages
            coverage_prob = popfirst!(coverage_probabilities)
            coverage = popfirst!(coverages)
            for t in LOCATION_TYPES
                belief_state = curr_bel_state.location_belief_states[visit].belstate
                probability_of_type = pdf(belief_state, t)

                if probability_of_type == 0
                    continue
                end

                push!(coverage_probabilities, coverage_prob*probability_of_type)

                new_coverage = env.location_coverage[(visit, t.state)]
                cumulative_coverage = union(coverage, new_coverage)
                push!(coverages, cumulative_coverage)
            end
        end
    end

    cum_expectation = 0

    for idx=1:length(coverages)
        outcome_utility = length(coverages[idx])
        outcome_probability = coverage_probabilities[idx]
        cum_expectation = cum_expectation + outcome_probability*outcome_utility
    end

    return cum_expectation
end

function sample_location(env::AreaCoverage2DEnv, loc::Int, prob_correct::Float64, rng::AbstractRNG)
    true_type = env.location_states[loc].state

    rand_num_1 = Base.rand(rng)
    if rand_num_1 <= prob_correct
        return AreaCoverage2DLocationState(true_type)
    else
        other_possibilities = [state_type.state for state_type in LOCATION_TYPES if state_type.state != true_type]
        rand_num_2 = Base.rand(rng)

        if rand_num_2 <= 0.5
            return AreaCoverage2DLocationState(other_possibilities[1])
        else
            return AreaCoverage2DLocationState(other_possibilities[2])
        end
    end
end

function sample_location_states(env::AreaCoverage2DEnv, current::Int, sensor::Sensor, rng::AbstractRNG)

    location_states = deepcopy(env.location_states)

    sensed_locations = Set{Int}(inrange(env.location_neighbors, env.location_metadata[current].location, sensor.effective_range, false))
    all_locations = Set{Int}(1:length(env.location_states))
    not_sensed = setdiff(all_locations, sensed_locations)

    #Can comment out later if never use non-sensed parts of observation
    for loc in not_sensed
        location_states[loc] = AreaCoverage2DLocationState(rng)
    end

    for loc in sensed_locations
        dist = 10*get_metadata_distance(env.location_metadata[current], env.location_metadata[loc])
        prob_correct = sensor.max_fidelity*(sensor.fidelity_decay_rate^(dist))
        location_states[loc] = sample_location(env, loc, prob_correct, rng)
    end

    return location_states
end

function belief_update_location_states_visit(env::AreaCoverage2DEnv, curr_bel_states::Vector{AreaCoverage2DLocationBeliefState}, current::Int)
    # vector_dist = Vector{Float64}()
    # for i=1:length(LOCATION_TYPES)
    #     loc_type = LOCATION_TYPES[i]
    #     if loc_type.state == env.location_states[current].state
    #         push!(vector_dist,1.0)
    #     else
    #         push!(vector_dist,0.0)
    #     end
    # end
    # new_distribution = SVector{3,Float64}(vector_dist)
    new_distribution = zeros(3)
    new_distribution[Int64(env.location_states[current].state)] = 1.0

    #SparseCat{SVector{3,Float64}, SVector{3,AreaCoverage2DLocationState}}
    new_belief_state = AreaCoverage2DLocationBeliefState(SparseCat(LOCATION_TYPES, convert(SVector{3,Float64}, new_distribution)))

    new_location_belief_states = deepcopy(curr_bel_states)
    new_location_belief_states[current] = new_belief_state
    return new_location_belief_states
end

function update_belief_state(curr_bel_state::AreaCoverage2DLocationBeliefState, obs_state::AreaCoverage2DLocationState, prob_correct::Float64)
    new_distribution = Vector{Float64}()

    for state_type in LOCATION_TYPES
        if state_type.state == obs_state.state
            likelihood = prob_correct
        else
            likelihood = (1.0 - prob_correct)/2.0
        end
        prior = pdf(curr_bel_state.belstate,state_type)
        push!(new_distribution, likelihood*prior)
    end

    marginal = sum(new_distribution)

    for idx in 1:length(new_distribution)
        new_distribution[idx] /= marginal
    end

    return AreaCoverage2DLocationBeliefState(SparseCat(LOCATION_TYPES, SVector{3,Float64}(new_distribution)))
end

function belief_update_location_states_sensor(env::AreaCoverage2DEnv, curr_bel_states::Vector{AreaCoverage2DLocationBeliefState},
                                              obs_location_states::Vector{AreaCoverage2DLocationState}, current::Int, sensor::Sensor)
    new_bel_states = deepcopy(curr_bel_states)
    sensed_locations = inrange(env.location_neighbors, env.location_metadata[current].location, sensor.effective_range, false)

    for loc in sensed_locations
        obs_state = obs_location_states[loc]

        dist = 10*get_metadata_distance(env.location_metadata[current], env.location_metadata[loc])
        prob_correct = sensor.max_fidelity*(sensor.fidelity_decay_rate^(dist))

        new_bel_states[loc] = update_belief_state(curr_bel_states[loc], obs_state, prob_correct)
    end

    return new_bel_states
end

function get_belief_state_mode_density(belief_state::AreaCoverage2DLocationBeliefState)
    return maximum(belief_state.belstate.probs)
end
