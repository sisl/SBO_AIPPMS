using POMDPModels, POMDPs, POMDPPolicies
using StaticArrays, Parameters, Random, POMDPModelTools, Distributions
using Images, LinearAlgebra, Printf
using Plots
using StatsBase
using BasicPOMCP


export
    RoverPOMDP,
    RoverState,
    RoverPos,
    RoverBelief

const RoverPos = SVector{2,Int}

struct RoverState
    pos::Int
    visited::Set{Int}
    location_states::Matrix{Float64}
    cost_expended::Float64
    drill_samples::Set{Float64}
end

struct ParticleSet
    particles::Matrix{Float64}
    weights::Matrix{Float64}
end

struct RoverBelief
    pos::Int
    visited::Set{Int}
    location_belief::ParticleSet
    cost_expended::Float64
    drill_samples::Set{Float64}
end


@with_kw mutable struct RoverPOMDP <: POMDP{RoverState, Symbol, Float64} # POMDP{State, Action, Observation}
    true_map::Matrix{Float64}
    init_pos::Tuple{Int, Int}              = (1,1)
    map_size::Tuple{Int, Int}              = size(true_map) # size of the map
    query_size::Tuple{Int, Int}            = (101,51)
    goal_pos::Tuple{Int, Int}              = (map_size[1], map_size[2])
    path_to_goal_matrix::Matrix{Float64}   = shortest_path_to_goal_matrix(map_size, goal_pos)

    tprob::Float64                         = 1
    discount::Float64                      = 1.0
    σ_drill::Float64                       = 1e-9
    σ_spec::Float64                        = 0.1
    drill_time::Float64                    = 3.0 # takes 10 time units to drill vs. 1.0 to move to neighbor cell TODO: include uncertain drill time
    cost_budget::Float64                   = 300.0
    step_size::Int64                       = 1 # scales the step the agent takes at each iteration
    new_sample_reward::Float64             = 1 # reward for drilling a unique sample
    repeat_sample_penalty::Float64         = -1 # penalty for drilling a repeat sample
    sample_types::Vector{Float64}          = collect(0:0.1:0.9)
    rng::AbstractRNG
end

function POMDPs.isterminal(pomdp::RoverPOMDP, s::RoverState)
    pos_cartesian = convert_pos_idx_2_pos_coord(pomdp, s.pos)
    if s.cost_expended + shortest_path_to_goal(pomdp, s.pos) >= pomdp.cost_budget
        return true

    elseif pos_cartesian == RoverPos(pomdp.goal_pos[1], pomdp.goal_pos[2])  && length(s.visited) > 1

        neighbor_actions = actions_possible_from_current(pomdp, s.pos, s.cost_expended)#POMDPs.actions(pomdp, s)

        if neighbor_actions == ()
            return true
        else
            min_cost_to_goal = minimum([shortest_path_to_goal(pomdp, pos_cartesian + pomdp.step_size*dir[n]) for n in neighbor_actions])
        end

        # min_cost_from_start = minimum([pomdp.shortest_paths[init_idx, n] for n in neighbors])
        if (pomdp.cost_budget - s.cost_expended) < 2*min_cost_to_goal
            return true
        else
            return false
        end
    else
        return false
    end

end

function POMDPs.isterminal(pomdp::RoverPOMDP, b::RoverBelief)
    return isterminal(pomdp, rand(pomdp.rng, b))
end
# function POMDPs.isterminal(pomdp::RoverPOMDP, b::RoverBelief)
#     if b.cost_expended + shortest_path_to_goal(pomdp, b.pos) > pomdp.cost_budget
#         return true
#     else
#         return false
#     end
# end

function POMDPs.gen(pomdp::RoverPOMDP, s::RoverState, a::Symbol, rng::RNG) where {RNG <: AbstractRNG}
    sp = generate_s(pomdp, s, a, rng)
    o = generate_o(pomdp, s, a, sp, rng)
    r = reward(pomdp, s, a, sp)

    return (sp=sp, o=o, r=r)
end

function inbounds(pomdp::RoverPOMDP, pos::RoverPos)
    if pomdp.map_size[1] >= pos[1] > 0 && pomdp.map_size[2] >= pos[2] > 0
        # i = abs(s[2] - pomdp.map_size[1]) + 1
        # j = s[1]
        return true
    else
        return false
    end
end


isgoal(pomdp::RoverPOMDP, s::RoverState) = (CartesianIndices(pomdp.map_size)[s.pos].I[1] == pomdp.goal_pos[1]) & (CartesianIndices(pomdp.map_size)[s.pos].I[2] == pomdp.goal_pos[2])


# discount
POMDPs.discount(pomdp::RoverPOMDP) = pomdp.discount


include("states.jl")
include("actions.jl")
include("observations.jl")
include("beliefs.jl")
include("transitions.jl")
include("rewards.jl")
