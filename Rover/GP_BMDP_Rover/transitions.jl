function generate_s(pomdp::RoverPOMDP, s::RoverState, a::Symbol, rng::RNG) where {RNG <: AbstractRNG}

    if isterminal(pomdp, s) #|| isgoal(pomdp, s)
        return RoverState(-1, s.location_states, pomdp.cost_budget, s.drill_samples)
    end

    if a in [:NE, :NW, :SE, :SW]
        visit_cost = sqrt(2*pomdp.step_size^2)
    elseif a in [:up, :down, :left, :right, :wait]
        visit_cost = 1.0*pomdp.step_size
    elseif a == :drill
        visit_cost = pomdp.drill_time
    end

    if a == :drill
        new_drill_samples = union(Set{Float64}([s.location_states[s.pos]]), s.drill_samples)
    else
        new_drill_samples = deepcopy(s.drill_samples)
    end

    pos = convert_pos_idx_2_pos_coord(pomdp, s.pos)

    new_pos = pos + pomdp.step_size*dir[a]
    if inbounds(pomdp, new_pos)
        new_pos = convert_pos_coord_2_pos_idx(pomdp, new_pos)
        new_cost_expended = s.cost_expended + visit_cost

        return RoverState(new_pos, s.location_states, new_cost_expended, new_drill_samples)
    else
        return RoverState(s.pos, s.location_states, pomdp.cost_budget*10, new_drill_samples)

    end
end
