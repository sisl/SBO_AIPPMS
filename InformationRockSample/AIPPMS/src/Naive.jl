#=
Implementation of the baseline paper
"Nonmyopic Adaptive Informative Path Planning for Multiple Robots" - Singh et al, IJCAI '09

The paper's NAIVE algorithm needs to be extended to deal with the setting of multimodal
sensing and the ability to sense multiple times at a location.

Notes:
1. DON'T NEED METADATA! Do at graph level
2. ENSURE location graph has a single connected component
=#
mutable struct NaiveSolver{E <: Environment}
    env::E
    cost_budget::Float64
    start_idx::Int
    goal_idx::Int
    terminal::Bool

    # NAIVE parameters
    num_clusters::Int
    graph_clusters::Dict{Int,Set{Int}}
    recursion_depth::Int
    lambda::Float64

    # NAIVE structures
    fws::Graphs.FloydWarshallState{Float64,Int}
end

## Just create. Don't initialize anything that is graph-specific
function NaiveSolver(env::E, budget::Float64) where {E <: Environment}
    return NaiveSolver(env, budget, 0, 0, false, 0, Dict{Int,Set{Int}}(), 0, Inf, Graphs.FloydWarshallState(Array{Float64,2}(undef,0,0),Array{Int,2}(undef,0,0)))
end


# Step 1 of pSPIEL-OR
# ONE TIME?
function get_clusters(ns::NaiveSolver)

    g = get_location_graph(ns.env)

    if length(connected_components(g)) > 1
        println("More than 1 component!")
        throw(RuntimeError())
    end

    # Create Dense from sparse matrix
    # With inf for non-edges
    g_weights = Matrix(g.weights)
    weights_matrix = g_weights + Inf*(g_weights.==0.0)
    for i in vertices(g)
        weights_matrix[i,i] = 0.0
    end

    # @show weights_matrix

    kmeds = kmedoids(weights_matrix, ns.num_clusters)

    # @show kmeds

    clusters = Dict{Int,Set{Int}}()

    for med in kmeds.medoids
        clusters[med] = Set{Int}()
    end

    for (idx,asn) in enumerate(kmeds.assignments)
        push!(clusters[kmeds.medoids[asn]],idx)
    end

    return clusters
end


# Step 2 of pSPIEL-OR
# DONE EACH TIME
function order_clusters(ns::NaiveSolver, curr_bel_state::WBS) where {WBS <: WorldBeliefState}

    # Order based on greedy expected utility on visiting
    # Use edge costs from fws
    # Return a Dict from Int to Vector{Int}

    ordered_clusters = Dict{Int,Vector{Int}}()

    for (idx,members) in ns.graph_clusters

        # Iterate over members and sort based on expected marginal utility
        member_exp_util = Vector{Tuple{Int,Float64}}(undef,0)

        for m in members
            visit_exp_util = expected_visit_utility(ns.env, m, curr_bel_state)
            push!(member_exp_util,(m, visit_exp_util))
        end
        ordered_clusters[idx] = [mu[1] for mu in sort(member_exp_util,by=x->x[2],rev=true)]
    end

    # TODO : Something more sophisticated here??

    return ordered_clusters
end

# Step 3 of pSPIEL-OR
function get_modular_approx_graph(ns::NaiveSolver, ordered_clusters::Dict{Int,Vector{Int}})
    # Create fully connected graph
    # with start, goal, and first indices of each cluster
    mod_vmap = Vector{Int}()
    push!(mod_vmap,ns.start_idx)
    push!(mod_vmap,ns.goal_idx)

    for (idx,asns) in ordered_clusters
        if asns[1] == ns.start_idx || asns[1] == ns.goal_idx
            continue
        end
        push!(mod_vmap,asns[1])
    end

    # @show mod_vmap

    g_prime = SimpleWeightedGraph(length(mod_vmap))

    # Add Edges between all vertices of g' using Floyd-Warshall paths
    for u in vertices(g_prime)
        for v in vertices(g_prime)

            if u != v
                dist = ns.fws.dists[mod_vmap[u], mod_vmap[v]]
                add_edge!(g_prime,u,v,dist)
            end
        end
    end

    return g_prime, mod_vmap
end


# Step 4 of pSPIEL-OR
# Run constrained modular_orienteering
# i - depth of recursion allowed
# Returns a vector of indices in order
# (s,t) is (1,2) to start with
# IMP - Will never generate a path that does not let you return to goal within budget
function modular_orienteering(env::E,
                              curr_bel_state::WBS,
                              mod_approx_graph::SimpleWeightedGraph{Int,Float64},
                              mod_vmap::Vector{Int},
                              B::Float64,
                              i::Int,
                              s::Int,
                              t::Int) where {E <: Environment, WBS <: WorldBeliefState}
    if mod_approx_graph.weights[s,t] > B
        return Vector{Int}()
    end

    P = [s,t]

    if i == 0
        return P
    end

    # Assume additive expected visit utility
    m = 0.0
    for p in P
        m += expected_visit_utility(env, mod_vmap[p], curr_bel_state)
    end

    for v in vertices(mod_approx_graph)
        if v==s || v==t
            continue
        end
        for B1 in 1:sqrt(B):B
            P1 = modular_orienteering(env, curr_bel_state, mod_approx_graph, mod_vmap, B1, i-1, s, v)
            P2 = modular_orienteering(env, curr_bel_state, mod_approx_graph, mod_vmap, B-B1, i-1, v, t)
            Ptemp = vcat(P1,P2)
            mtemp = 0.0

            # Because we are just moving along, no update of other location beliefs
            # And expected utility of visiting locations just decomposes additively
            for p in Ptemp
                mtemp += expected_visit_utility(env, mod_vmap[p], curr_bel_state)
            end

            if mtemp > m
                m = mtemp
                P = copy(Ptemp)
            end
        end
    end

    return P
end

# TODO : Check FWS is correctly used here
function expand_mod_approx_path(ns::NaiveSolver, mod_approx_path::Vector{Int}, mod_vmap::Vector{Int})

    true_path = Vector{Int}()

    for (u,v) in zip(mod_approx_path[1:end-1], mod_approx_path[2:end])

        # Insert u till before v
        u_v_path = enumerate_paths(ns.fws, mod_vmap[u], mod_vmap[v])
        append!(true_path, u_v_path[1:end-1])
    end

    # Last one is guaranteed to be this
    push!(true_path, ns.goal_idx)

    return true_path
end


# Use tour-opt heuristics to smooth
function tour_opt_smooth(ns::NaiveSolver, presmooth_path::Vector{Int})

    g = get_location_graph(ns.env)

    # Get true adj mat with inf weights
    g_weights = Matrix(g.weights)
    g_dense_wts = g_weights + Inf*(g_weights.==0.0)

    smooth_path, cost = two_opt(g_dense_wts, presmooth_path)

    return smooth_path, cost
end


function setup(ns::NaiveSolver, goal::Int, num_clusters::Int, recursion_depth::Int, lambda::Float64)

    # Initialize graph related stuff
    ns.goal_idx = goal
    ns.num_clusters = num_clusters
    ns.recursion_depth = recursion_depth
    ns.lambda = lambda

    # Run Floyd-Warshall
    g = get_location_graph(ns.env)
    ns.fws = floyd_warshall_shortest_paths(g)

    # Step 1 - Get graph clusters
    ns.graph_clusters = get_clusters(ns)
    # @show ns.graph_clusters
end


function update_lambda(ns::NaiveSolver, lambda::Float64)
    ns.lambda = lambda
end



function plan(ns::NaiveSolver, curr_bel_state::WBS) where {WBS <: WorldBeliefState}

    ns.start_idx = curr_bel_state.current

    # Step 1 already done - we have unordered clusters

    # Step 2 - Order the clusters
    ordered_clusters = order_clusters(ns, curr_bel_state)
    # @show ordered_clusters

    # Step 3 - Get mod approx graph
    mod_approx_graph, vmap = get_modular_approx_graph(ns, ordered_clusters)

    # Step 4 - Run modular orienteering with remaining budget
    # We know that for gprime, start = 1 and goal = 2
    #budget = convert(Int64,floor(ns.cost_budget - curr_bel_state.cost_expended))
    budget = ns.cost_budget - curr_bel_state.cost_expended
    mod_approx_path = modular_orienteering(ns.env, curr_bel_state, mod_approx_graph, vmap, budget, ns.recursion_depth, 1, 2)

    # @show mod_approx_path

    if length(mod_approx_path) <= 1
        # Error if not at goal else terminate
        if curr_bel_state.current != ns.goal_idx
            throw(ErrorException())
        else
            ns.terminal = true
            return MultimodalIPPAction(nothing, nothing, 0)
        end
    end

    pruned_mod_approx_path = [1]
    for i = 2:length(mod_approx_path)
        if mod_approx_path[i] != mod_approx_path[i-1]
            push!(pruned_mod_approx_path,mod_approx_path[i])
        end
    end
    # @show pruned_mod_approx_path

    # Step 5 - Expand path according to true graph
    true_path = expand_mod_approx_path(ns, pruned_mod_approx_path, vmap)
    # @show true_path

    # Step 6 - Smooth path using 2-opt heuristic
    smooth_path, cost = tour_opt_smooth(ns, true_path)

    if length(smooth_path) == 1 || smooth_path[2] == curr_bel_state.current
        if curr_bel_state.current != ns.goal_idx
            throw(ErrorException("could not reach goal"))
        else
            ns.terminal = true
            return MultimodalIPPAction(nothing,nothing,0)
        end
    end

    first_idx = smooth_path[2]

    # Now choose between moving or sensing
    # Compute expected utility of visiting nodes along suggested path
    exp_util_move = 0.0
    for p in smooth_path
        exp_util_move += expected_visit_utility(ns.env, p, curr_bel_state)
    end
    # @show exp_util_move

    # Now run through each sensing action and choose the best one
    sensors = get_sensors(ns.env)
    best_exp_util_sense = -Inf
    best_sensor = sensors[1]

    # @show ns.fws.dists[curr_bel_state.current, ns.goal_idx]

    for s in sensors
        #add simulated belief update
        sense_util = exp_info_gain(ns.env, curr_bel_state, s)
        # @show sense_util

        post_sense_budget = ns.cost_budget - (curr_bel_state.cost_expended + get_energy_cost(s))

        # CHECK THAT Taking this action leave you in recoverable state
        if sense_util > best_exp_util_sense && post_sense_budget > ns.fws.dists[curr_bel_state.current, ns.goal_idx]
            best_sensor = s
            best_exp_util_sense = sense_util
        end
    end
    # @show best_exp_util_sense

    # Returns a MultimodalIPPAction but idx does not matter
    # @show exp_util_move
    # @show best_exp_util_sense
    # NOW compare best sense util to best move util using scaling factor and decide
    if ns.lambda*exp_util_move > (1 - ns.lambda)*best_exp_util_sense
        best_action = MultimodalIPPAction(first_idx,nothing,0)
    else
        best_action = MultimodalIPPAction(nothing,best_sensor,0)
    end

    return best_action
end

isterminal_naive(ns::NaiveSolver, state::WorldState) = ns.terminal
