# Observations

function generate_o(pomdp::RoverPOMDP, s::RoverState, action::Symbol, sp::RoverState, rng::AbstractRNG)
    if isterminal(pomdp, sp)
        return -1.0
    end

    # Remember you make the observation at sp NOT s
    if action == :drill
        o = pomdp.true_map[sp.pos]#sp.location_states[sp.pos]
    else
        # NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
        # for normal dist whereas our GP setup uses σ²_n

        o = pomdp.true_map[sp.pos] + round(rand(rng, Normal(0, pomdp.σ_spec)), digits=1)
    end

    return o
end
