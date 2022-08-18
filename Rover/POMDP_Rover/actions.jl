# Actions

function get_neighbor_actions(pomdp::RoverPOMDP, pos::Int)
    actions = [:up, :down, :left, :right, :wait, :NE, :NW, :SE, :SW, :drill]

    pos = convert_pos_idx_2_pos_coord(pomdp, pos)

    if pos[1] == 1
        deleteat!(actions, actions .== :left)
        deleteat!(actions, actions .== :NW)
        deleteat!(actions, actions .== :SW)
    elseif pos[1] == pomdp.map_size[1]
        deleteat!(actions, actions .== :right)
        deleteat!(actions, actions .== :NE)
        deleteat!(actions, actions .== :SE)
    end

    if pos[2] == 1
        deleteat!(actions, actions .== :down)
        deleteat!(actions, actions .== :SW)
        deleteat!(actions, actions .== :SE)
    elseif pos[2] == pomdp.map_size[2]
        deleteat!(actions, actions .== :up)
        deleteat!(actions, actions .== :NE)
        deleteat!(actions, actions .== :NW)
    end

    return actions

end

function shortest_path_to_goal(pomdp::RoverPOMDP, pos::Int)
    return shortest_path_to_goal(pomdp, convert_pos_idx_2_pos_coord(pomdp, pos))
end

function shortest_path_to_goal(pomdp::RoverPOMDP, pos::RoverPos)
    x = pos[1]
    y = pos[2]
    i = abs(y - pomdp.map_size[2]) + 1
    j = x
    if inbounds(pomdp, pos)
        return pomdp.path_to_goal_matrix[i,j]
    else
        return pomdp.cost_budget*10 # high cost so it wont be able to make it to goal
    end
end

function shortest_path_to_goal_matrix(map_size::Tuple{Int, Int}, goal_pos::Tuple{Int, Int})
    path_to_goal_matrix = zeros((map_size[2], map_size[1])) # 51x101
    target = goal_pos

    for x in 1:map_size[1]
        for y in 1:map_size[2]
            position = RoverPos(x,y)
            path_length = 0.0
            while ((position[1] != goal_pos[1]) || (position[2] != goal_pos[2]))
                # println(position)
                if (target[1] > position[1]) & (target[2] > position[2])
                   position += dir[:NE]
                   path_length += sqrt(2)
                elseif (target[1] > position[1])
                    position += dir[:right]
                    path_length += 1
                elseif target[2] > position[2]
                    position += dir[:up]
                    path_length += 1
                else
                    position += dir[:down]
                    path_length += 1
                end
            end
            i = abs(y - map_size[2]) + 1
            j = x
            path_to_goal_matrix[i,j] += path_length
        end
    end
    return path_to_goal_matrix
end

function actions_possible_from_current(pomdp::RoverPOMDP, pos::Int, cost_expended::Float64)
    neighbors_actions = get_neighbor_actions(pomdp, pos)
    possible_actions = []
    pos = convert_pos_idx_2_pos_coord(pomdp, pos)

    for n in neighbors_actions
        if n in [:NE, :NW, :SE, :SW]
            visit_cost = sqrt(2*pomdp.step_size^2)
        elseif n in [:up, :down, :left, :right, :wait]
            visit_cost = 1.0*pomdp.step_size
        elseif n == :drill
            visit_cost = pomdp.drill_time
        end
        return_cost = shortest_path_to_goal(pomdp, pos + pomdp.step_size*dir[n])

        if (cost_expended + visit_cost + return_cost) < pomdp.cost_budget
            if inbounds(pomdp, pos + pomdp.step_size*dir[n])
                push!(possible_actions, n)
            end
        end

    end

    return (possible_actions...,) # return as a tuple
end


function POMDPs.actions(pomdp::RoverPOMDP, s::RoverState)
    if isterminal(pomdp, s)
        return (:up, :down, :left, :right, :wait, :NE, :NW, :SE, :SW, :drill)
    else
        possible_actions = actions_possible_from_current(pomdp, s.pos, s.cost_expended)
    end
    return possible_actions
end

# function POMDPs.action(b::RoverBelief)
#     if isterminal(pomdp, b)
#         return (:up, :down, :left, :right, :wait, :NE, :NW, :SE, :SW, :drill)
#     else
#         possible_actions = actions_possible_from_current(pomdp, b.pos, b.cost_expended)
#     end
#     return possible_actions
# end
#
function POMDPs.action(p::RandomPolicy, b::RoverBelief)
    possible_actions = POMDPs.actions(p.problem, b)
    return rand(p.problem.rng, possible_actions)
    # return possible_actions[rand(collect(1:length(possible_actions)))]
end

function POMDPs.actions(pomdp::RoverPOMDP, s::LeafNodeBelief)
    s = s.sp
    if isterminal(pomdp, s)
        return (:up, :down, :left, :right, :wait, :NE, :NW, :SE, :SW, :drill)
    else
        possible_actions = actions_possible_from_current(pomdp, s.pos, s.cost_expended)
    end
    return possible_actions
end

function POMDPs.actions(pomdp::RoverPOMDP, b::RoverBelief)
    if isterminal(pomdp, b)
        return (:up, :down, :left, :right, :wait, :NE, :NW, :SE, :SW, :drill)
    else
        possible_actions = actions_possible_from_current(pomdp, b.pos, b.cost_expended)
    end
    return possible_actions
end

const dir = Dict(:up=>RoverPos(0,1), :down=>RoverPos(0,-1), :left=>RoverPos(-1,0), :right=>RoverPos(1,0), :wait=>RoverPos(0,0), :NE=>RoverPos(1,1), :NW=>RoverPos(-1,1), :SE=>RoverPos(1,-1), :SW=>RoverPos(-1,-1), :drill=>RoverPos(0,0))
const aind = Dict(:up=>1, :down=>2, :left=>3, :right=>4, :wait=>5, :NE=>6, :NW=>7, :SE=>8, :SW=>9, :drill=>10)

POMDPs.actionindex(POMDP::RoverPOMDP, a::Symbol) = aind[a]
