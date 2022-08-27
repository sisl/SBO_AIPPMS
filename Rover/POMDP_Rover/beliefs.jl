struct RoverBeliefUpdater{P<:POMDPs.POMDP} <: Updater
    pomdp::P
end

function Base.rand(rng::AbstractRNG, b::RoverBelief)
    # location belief is Matrix of size N_particles x N_locations
    location_states = zeros(Int(sqrt(size(b.location_belief.particles)[2])), Int(sqrt(size(b.location_belief.particles)[2])))

    for i in size(b.location_belief.particles)[2]
        sample = StatsBase.sample(rng, b.location_belief.particles[:,i], Weights(b.location_belief.weights[:,i]), 1)
        location_states[i] = sample[1]
    end

    return RoverState(b.pos, b.visited, location_states, b.cost_expended, b.drill_samples)
end

# function POMDPs.update(updater::RoverBeliefUpdater, b::RoverBelief, a::Symbol, o::Float64, rng::RNG) where {RNG <: AbstractRNG}
#     ub = update_belief(updater.pomdp, b, a, o)
#     return ub
# end

function POMDPs.update(updater::RoverBeliefUpdater, b::RoverBelief, a::Symbol, o::Float64)
    ub = update_belief(updater.pomdp, b, a, o, updater.pomdp.rng)
    return ub
end


function update_belief(pomdp::P, b::RoverBelief, a::Symbol, o::Float64, rng::RNG) where {P <: POMDPs.POMDP, RNG <: AbstractRNG}
    # check is sp is terminal
    if isterminal(pomdp, b)
        return b
    end

    particles = b.location_belief.particles
    weights = b.location_belief.weights

    if a in [:NE, :NW, :SE, :SW]
        visit_cost = sqrt(2*pomdp.step_size^2)
    elseif a in [:up, :down, :left, :right, :wait]
        visit_cost = 1.0*pomdp.step_size
    elseif a == :drill
        visit_cost = pomdp.drill_time
    end

    if a == :drill
        # if we drill we fully collapse the belief
        idx = first(findall(x-> x== o, pomdp.sample_types))
        weights[:, b.pos] .= 0.0
        weights[idx, b.pos] = 1.0
        # particles[:, b.pos] .= o
        # weights[:, b.pos] .= 1/length(particles[:, b.pos])
        new_cost_expended = b.cost_expended + visit_cost
        new_drill_samples = union(Set{Float64}([o]), b.drill_samples)

        return RoverBelief(b.pos, b.visited, ParticleSet(particles, weights), new_cost_expended, new_drill_samples)

    else
        pos = convert_pos_idx_2_pos_coord(pomdp, b.pos) + pomdp.step_size*dir[a]
        new_pos = convert_pos_coord_2_pos_idx(pomdp, pos)
        new_cost_expended = b.cost_expended + visit_cost
        new_visited = union(Set{Int}([new_pos]), b.visited)

        # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
        # for normal dist whereas our GP setup uses σ²_n
        σ_n = pomdp.σ_spec


        # for j in 1:size(b.location_belief.particles)[2] # rows are particles, columns are different locations
        #     # mu = StatsBase.mean(particles[:,j], Weights(weights[:,j]))
        #     weights[:, j] = pdf(Normal(o, σ_n), particles[:,j])
        # end

        # NOTE: we only resample at the location the observation was received!
        weights[:, new_pos] .*= pdf(Normal(o, σ_n), particles[:,new_pos])
        weights[:, pos] = weights[:, pos] ./ sum(weights[:, pos])
        #particles, weights = resample(pomdp, particles, weights, new_pos, rng)


        return RoverBelief(new_pos, new_visited, ParticleSet(particles, weights), new_cost_expended, b.drill_samples)

    end
end

function resample(pomdp::RoverPOMDP, particles, weights, pos, rng::RNG) where {RNG <: AbstractRNG}
    # NOTE: we only resample at the location the observation was received!
    N = length(particles[:,1]) # number of particles in each location

    weights[:, pos] = weights[:, pos] ./ sum(weights[:, pos])
    particles[:, pos] = StatsBase.sample(rng, particles[:,pos], Weights(weights[:,pos]), N)

    # inject one particle 20% of the time
    if rand(rng) < 0.2
        inject_idx = rand(rng, collect(1:N))
        particles[inject_idx, pos] = rand(rng, pomdp.sample_types)
    end

    weights[:, pos] = (1/N) .* ones(N)

    # for j in 1:size(particles)[2]
    #     weights[:, j] = weights[:, j] ./ sum(weights[:, j])
    #     particles[:, j] = StatsBase.sample(rng, particles[:,j], Weights(weights[:,j]), N) # add some noise to the particles to prevent clumping
    #
    #     # inject one particle 20% of the time
    #     if rand(rng) < 0.2
    #         inject_idx = rand(rng, collect(1:N))
    #         particles[inject_idx, j] = rand(rng, pomdp.sample_types)
    #     end
    #
    #     # try
    #     #     PP.particles[:, j] = StatsBase.sample(rng, PP.particles[:,j], Weights(PP.weights[:,j]), N) + rand(rng, Normal(0, 0.1),N) # add some noise to the particles to prevent clumping
    #     # catch
    #     #     println("caught")
    #     #     # if all of the weights are too small, then we need to resample from the start
    #     #     PP.particles[:, j] = rand(Distributions.Uniform(), N) #TODO: add rng
    #     # end
    #     weights[:, j] = (1/N) .* ones(N)
    # end

    return particles, weights
end

function BasicPOMCP.extract_belief(::RoverBeliefUpdater, node::BeliefNode)
    return node
end


function POMDPs.initialize_belief(updater::RoverBeliefUpdater, d)
    return initial_belief_state(updater.pomdp, updater.pomdp.rng)
end


function POMDPs.initialize_belief(updater::RoverBeliefUpdater, d, rng::RNG) where {RNG <: AbstractRNG}
    return initial_belief_state(updater.pomdp, rng)
end

function initial_belief_state(pomdp::RoverPOMDP, rng::RNG) where {RNG <: AbstractRNG}

    pos = LinearIndices(pomdp.map_size)[pomdp.init_pos[1], pomdp.init_pos[2]]
    visited = Set{Int}(pos)
    location_belief = initialize_particles(pomdp, rng)
    cost_expended = 0.0
    drill_samples = Set{Float64}(Float64[])

    return RoverBelief(pos, visited, location_belief, cost_expended, drill_samples)

end

function initialize_particles(pomdp::RoverPOMDP, rng::RNG) where {RNG <: AbstractRNG}
    N_particles = 10
    # particles is Nx100
    particles = rand(rng, pomdp.sample_types, N_particles, pomdp.map_size[1]*pomdp.map_size[2])#rand(rng, Distributions.Uniform(), N_particles, pomdp.map_size[1]*pomdp.map_size[2])

    column(j) = [j for _ =1:100]
    particles = hcat([column(j) for j=pomdp.sample_types]...)
    particles = Matrix{Float64}(particles')

    weights = (1/N_particles) .* ones((N_particles, pomdp.map_size[1]*pomdp.map_size[2]))
    PS = ParticleSet(particles, weights)
    return PS
end
