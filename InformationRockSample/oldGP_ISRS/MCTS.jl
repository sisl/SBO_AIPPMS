struct MonteCarloTreeSearch<: Policy
    rng
    𝒫 # problem
    N # visit counts
    Q # action value estimates
    m # number of simulations
    d # depth
    c # exploration constant
    U # value function estimate
end

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π, s)
    end
    if actions(π.𝒫, s) == ()
        return :wait
    else
        return argmax(a->π.Q[(s,a)], actions(π.𝒫, s))
    end
end

function POMDPs.action(π::MonteCarloTreeSearch, s)
    return π(s)
end

function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return π.U(s)
    end
    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒜, T, γ = actions(𝒫, s), ((s,a)->POMDPs.transition(𝒫, s, a, π.rng)), 𝒫.discount
    R = (s,a,sp)->POMDPs.reward(𝒫, s, a, sp)

    if 𝒜 == MultimodalIPPAction[]
        return 0
    end

    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π, s)
    s′= T(s,a)
    r = R(s,a,s′)
    q = r + γ*simulate!(π, s′, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)
function explore(π::MonteCarloTreeSearch, s)
    𝒜, N, Q, c = actions(π.𝒫, s), π.N, π.Q, π.c
    Ns = sum(N[(s,a)] for a in 𝒜)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end

# TODO: Define rollout policy

function randstep(𝒫, s, a, rng)
    T = ((s,a)->POMDPs.transition(𝒫, s, a, rng))
    R = (s,a,sp)->POMDPs.reward(𝒫, s, a, sp)
    sp = T(s,a)
    return (sp, R(s,a,sp))
end

function RandomRollout(rng, 𝒫, s, d)
    ret = 0.0
    for t in 1:d
        𝒜 = actions(𝒫, s)
        if 𝒜 == MultimodalIPPAction[]
            return 0
        else
            a = rand(rng, 𝒜) #π(s)
        end
        s, r = randstep(𝒫, s, a, rng)
        ret += 𝒫.discount^(t-1) * r
    end
    return ret
end

function SeekTarget(rng, 𝒫, s, target=𝒫.goal_pos)
    if rand(rng) > 0.8
        𝒜 = actions(𝒫, s)
        if 𝒜 == ()
            return Nothing
        end
        a = rand(rng, 𝒜)
        return a
    else
        𝒜 = actions(𝒫, s)
        if 𝒜 == ()
            return Nothing
        end

        if (target[1] > s.pos[1]) & (target[2] > s.pos[2])
            if inbounds(𝒫, s.pos + 𝒫.step_size*dir[:NE])
                return :NE
            else
                return rand(rng, 𝒜)
            end
        elseif (target[1] > s.pos[1])
            if inbounds(𝒫, s.pos + 𝒫.step_size*dir[:right])
                return :right
            else
                return rand(rng, 𝒜)
            end
        elseif target[2] > s.pos[2]
            if inbounds(𝒫, s.pos + 𝒫.step_size*dir[:up])
                return :up
            else
                return rand(rng, 𝒜)
            end
        else
            if inbounds(𝒫, s.pos + 𝒫.step_size*dir[:down])
                return :down
            else
                return rand(rng, 𝒜)
            end
        end
    end
end

function TargetRollout(rng, 𝒫, s, d)
    ret = 0.0
    𝒜 = actions(𝒫, s)
    for t in 1:d
        # if t == d
        #     println("final state: ", s.pos)
        # end
        a = SeekTarget(rng, 𝒫, s) #π(s)
        if a == Nothing
            r = -10000.0
            ret += 𝒫.discount^(t-1) * r
        else
            s, r = randstep(𝒫, s, a, rng)
            ret += 𝒫.discount^(t-1) * r
        end
    end
    return ret
end

# U(s)=RandomRollout(pomdp, s, π.d)
# π_rand=MonteCarloTreeSearch(pomdp, Dict(), Dict(), 30, 20, 1.0, U)

# U(s)=TargetRollout(pomdp, s, π.d)
# π_target=MonteCarloTreeSearch(pomdp, Dict(), Dict(), 30, 20, 1.0, U)


function custom_stepthrough(𝒫, π, s; max_steps=500, rng=MersenneTwister(3))
    s_init = deepcopy(s)
    state_hist = []
    action_hist = []
    for i in 1:max_steps
        println(i)
        state_hist = vcat(state_hist, [[s.pos[1], s.pos[2]]])
        a = π(s)
        println(a)
        action_hist = vcat(action_hist, a)
        s = POMDPs.transition(𝒫, s, a)
    end

#    state_hist2 = []
#    action_hist2 = []
#    s = s_init
#    for i in 1:max_steps
#        println(i)
#        state_hist2 = vcat(state_hist2, [[s.pos[1], s.pos[2]]])
#        a = π(s)
#        println(a)
#        action_hist2 = vcat(action_hist2, a)
#        s = POMDPs.transition(𝒫, s, a).val
#    end

#    return (state_hist, action_hist, state_hist2, action_hist2)
     return (state_hist, action_hist)
end



# include("MCTS.jl")
# π=MonteCarloTreeSearch(pomdp, Dict(), Dict(), 30, 20, 1.0, (x->0.0))
# state_hist, action_hist =stepthrough(pomdp, π, SSTDistribution(GWPos(1,1), 0, f_prior))
# fourth_policy=MonteCarloTreeSearch(pomdp, Dict(), Dict(), 500, 300, 1.0, (x->0.0))
# state_hist, action_hist =stepthrough(pomdp, fourth_policy, SSTDistribution(GWPos(1,1), 0, f_prior))
