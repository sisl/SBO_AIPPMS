# Observations

function generate_o(pomdp::RoverPOMDP, s::RoverState, action::Symbol, sp::RoverState, rng::AbstractRNG)
    # if isterminal(pomdp, sp)
    #     println("HERE")
    #     return -1.0
    # end

    # Remember you make the observation at sp NOT s
    if action == :drill
        o = pomdp.true_map[sp.pos] #sp.location_states[sp.pos]
    else
        o = pomdp.true_map[sp.pos] + round(rand(rng, Normal(0, pomdp.σ_spec)), digits=1)
    end

    return o
end
