# module MultimodalIPP

# I/O Packages
using JSON
#using CSV
using FileIO
using JLD2

# For sensor location graph
using Graphs, SimpleWeightedGraphs
using MetaGraphs
using TravelingSalesmanHeuristics
using StaticArrays
using NearestNeighbors
using Random
using LinearAlgebra
using Parameters
#using Base.rand

# POMDP dependencies
using POMDPs
using POMDPModels
using POMDPModelTools
using POMDPPolicies
using BasicPOMCP
using POMCPOW
using POMDPTools

# For NAIVE
using Clustering

# Environment and MultimodalIPP stuff
export
    MultimodalIPPAction,
    Environment,
    Sensor,
    LocationState,
    LocationBeliefState,
    Observation,
    MultimodalIPPBeliefUpdater,
    MultimodalIPPGreedyPolicy,
    get_energy_cost,
    get_location_graph,
    utility,
    marginal_utility,
    expected_visit_utility,
    exp_info_gain,
    get_state_of_belstate

# NAIVE
export
    NaiveSolver,
    setup,
    plan

# AreaCoverage 2D Stuff
export
    AreaCoverage2DEnv,
    AreaCoverage2DPOMDP,
    ISRSEnv,
    ISRSPOMDP,
    ISRSPos,
    ISRS_STATE,
    RSGOOD,
    RSBAD,
    RSBEACON,
    RSNEITHER,
    setup_isrs_pomdp


# For trials
export
    graph_trial,
    get_pomcp_basic_policy,
    get_pomcp_gcb_policy,
    get_naive_policy


# The general types needed (see README)
abstract type Environment end # Encapsulates notion of world map
abstract type Sensor end
abstract type LocationState end
abstract type LocationBeliefState end
abstract type Observation end


struct MultimodalIPPAction
    visit_location::Union{Nothing, Int}
    sensing_action::Union{Nothing, Sensor}
    idx::Int
end

struct WorldState{LS}
    current::Int
    visited::Set{Int}
    location_states::Vector{LS}
    cost_expended::Float64
end

# Encoding independence assumption here as well
struct WorldBeliefState{LBS}
    current::Int
    visited::Set{Int}
    location_belief_states::Vector{LBS}
    cost_expended::Float64
end

#=
The observation space is basically all possible states of all possible subsets of locations
The true state belief of the location is updated based on sensor fidelity and distance from the current location.
The observation includes fully observed components as well - since we don't have a separate interface
for mixed observability.
=#
struct WorldObservation{LS}
    obs_current::Int
    obs_visited::Set{Int}
    obs_location_states::Vector{LS}
    obs_cost_expended::Float64
end

# Functions an environment is expected to implement for
# the IPPEnvModel to be able to use it
function get_energy_cost end
"""
    get_energy_cost(env::Environment, s::Sensor)
Return the energy cost of executing a sensing action on a particular sensor
"""

function get_location_graph end
"""
    Just return the SimpleWeightedGraph that encodes the locations. Literally one line
"""

function utility end
"""
    utility(env::Environment, visited::Set{Int})
Compute the utility of visiting the set of locations
"""

function marginal_utility end
"""
    marginal_utility(env::Environment, next_visit::Int, visited::Set{Int})
Compute the marginal utility of visiting some new location, given currently visited locations
"""

function expected_visit_utility end
"""
    function expected_visit_utility(env::E, loc::Int, curr_bel_state::WorldBeliefState{LBS}) where {E <: Environment, LBS <:LocationBeliefState}
Return the expected utility of choosing to visit loc given the current belief state
"""

function exp_info_gain end
"""
    information_gain(env::Environment, curr_bel_state::WorldBeliefState{LBS}) where {LBS <: LocationBeliefState}
A metric for rewarding sensing at a location
"""


function get_state_of_belstate end

include("envs/AreaCoverage2DEnv.jl")
include("envs/AreaCoverage2DSensors.jl")
include("envs/AreaCoverage2DPOMDP.jl")
include("envs/InfSearchRockSample.jl")
include("envs/common.jl")
include("Naive.jl")

# end
