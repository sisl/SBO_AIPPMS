function convert_pos_idx_2_pos_coord(pomdp::RoverPOMDP, pos::Int)
    if pos == -1
        return RoverPos(-1,-1)
    else
        return RoverPos(CartesianIndices(pomdp.map_size)[pos].I[1], CartesianIndices(pomdp.map_size)[pos].I[2])
    end
end

function convert_pos_coord_2_pos_idx(pomdp::RoverPOMDP, pos::RoverPos)
    if pos == RoverPos(-1,-1)
        return -1
    else
        return LinearIndices(pomdp.map_size)[pos[1], pos[2]]
    end
end

function POMDPs.initialstate(pomdp::RoverPOMDP)
    curr = LinearIndices(pomdp.map_size)[pomdp.init_pos[1], pomdp.init_pos[2]]
    return RoverState(curr, Set{Int}([curr]), pomdp.true_map, 0.0, Set{Float64}(Float64[]))
end
