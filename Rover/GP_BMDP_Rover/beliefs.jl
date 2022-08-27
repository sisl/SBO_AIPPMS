struct RoverBeliefUpdater{P<:POMDPs.POMDP} <: Updater
    pomdp::P
end

function Base.rand(rng::AbstractRNG, pomdp::RoverPOMDP, b::RoverBelief)

    location_states = rand(rng, b.location_belief, b.location_belief.mXq, b.location_belief.KXqXq)
    location_states = reshape(location_states, pomdp.map_size)

    return RoverState(b.pos, location_states, b.cost_expended, b.drill_samples)
end


function POMDPs.update(updater::RoverBeliefUpdater, b::RoverBelief, a::Symbol, o::Float64)
    ub = update_belief(updater.pomdp, b, a, o, updater.pomdp.rng)
    return ub
end


function update_belief(pomdp::P, b::RoverBelief, a::Symbol, o::Float64, rng::RNG) where {P <: POMDPs.POMDP, RNG <: AbstractRNG}
    if isterminal(pomdp, b)
        return b
    end

    if a in [:NE, :NW, :SE, :SW]
        visit_cost = sqrt(2*pomdp.step_size^2)
    elseif a in [:up, :down, :left, :right, :wait]
        visit_cost = 1.0*pomdp.step_size
    elseif a == :drill
        visit_cost = pomdp.drill_time
    end


    # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
    # for normal dist whereas our GP setup uses σ²_n

    if a == :drill
        # if we drill we fully collapse the belief
        drill_pos = convert_pos_idx_2_pos_coord(pomdp, b.pos)
        σ²_n = pomdp.σ_drill #^2 dont square this causes singular exception in GP update
        f_posterior = posterior(b.location_belief, [[drill_pos[1], drill_pos[2]]], [o], [σ²_n])

        new_cost_expended = b.cost_expended + visit_cost
        new_drill_samples = union(Set{Float64}([o]), b.drill_samples)

        return RoverBelief(b.pos, f_posterior, new_cost_expended, new_drill_samples)

    else
        pos = convert_pos_idx_2_pos_coord(pomdp, b.pos) + pomdp.step_size*dir[a]
        new_pos = convert_pos_coord_2_pos_idx(pomdp, pos)
        new_cost_expended = b.cost_expended + visit_cost

        spec_pos = convert_pos_idx_2_pos_coord(pomdp, new_pos)
        σ²_n = pomdp.σ_spec^2
        f_posterior = posterior(b.location_belief, [[spec_pos[1], spec_pos[2]]], [o], [σ²_n])


        return RoverBelief(new_pos, f_posterior, new_cost_expended, b.drill_samples)

    end
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
    location_belief = pomdp.f_prior
    cost_expended = 0.0
    drill_samples = Set{Float64}(Float64[])

    return RoverBelief(pos, location_belief, cost_expended, drill_samples)

end
