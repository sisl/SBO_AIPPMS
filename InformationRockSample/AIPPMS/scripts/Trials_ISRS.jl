# using MultimodalIPP
include("/Users/joshuaott/icra2022/AIPPMS/src/MultimodalIPP.jl")
using Graphs
using Random
using BasicPOMCP
using POMDPs
using JSON
using Statistics
using Distributions
using Plots


function plot_trial(state_hist, location_states_hist, action_hist, reward_hist, trial_num, name)


	loc_states = deepcopy(location_states_hist[1])
	plot_scale = 1:0.18:10


    anim = @animate for i = 1:length(state_hist)

		good_rocks = findall(x-> x == RSGOOD, loc_states)
		good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

		bad_rocks = findall(x-> x == RSBAD, loc_states)
		bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

		beacons = findall(x-> x == RSBEACON, loc_states)
		beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


		# Display Total Reward
		if i == 1
			title = "Total Reward: $(reward_hist[1])"
		else
			title = "Total Reward: $(reward_hist[i-1])"
		end

		contourf(collect(plot_scale), collect(plot_scale), ones(length(plot_scale),length(plot_scale)), colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false) # xlims = (1, 10), ylims = (1, 10)

	    # Beacons
		scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect, aspectratio = :equal, xlims = (0.5, 10.5), ylims = (0.5, 10.5))

		# Rocks
		scatter!([good_rocks[i][1] for i in 1:length(good_rocks)], [good_rocks[i][2] for i in 1:length(good_rocks)], legend=false, color=:green)
		scatter!([bad_rocks[i][1] for i in 1:length(bad_rocks)], [bad_rocks[i][2] for i in 1:length(bad_rocks)], legend=false, color=:red)

		# Agent location
		scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title)

		# Update rock visit
		if loc_states[state_hist[i]] == RSGOOD
			# println(loc_states[state_hist[i]])
			loc_states[state_hist[i]] = RSBAD
			# println(loc_states[state_hist[i]])
		end
		# println(location_states_hist)
	end
	Plots.gif(anim, "/Users/joshuaott/icra2022/figures/AIPPMS/$(name)$(trial_num).gif", fps = 1)

	############################################################################
	# Just make the plot
	############################################################################
	loc_states = deepcopy(location_states_hist[1])


	good_rocks = findall(x-> x == RSGOOD, loc_states)
	good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

	bad_rocks = findall(x-> x == RSBAD, loc_states)
	bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

	beacons = findall(x-> x == RSBEACON, loc_states)
	beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


	# Display Total Reward
	title = "Total Reward: $(reward_hist[end-1])"

	contourf(collect(plot_scale), collect(plot_scale), ones(length(plot_scale),length(plot_scale)), colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false) # xlims = (1, 10), ylims = (1, 10)


	# Beacons
	scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect, aspectratio = :equal, xlims = (0.5, 10.5), ylims = (0.5, 10.5))

	# Rocks
	scatter!([good_rocks[i][1] for i in 1:length(good_rocks)], [good_rocks[i][2] for i in 1:length(good_rocks)], legend=false, color=:green)
	scatter!([bad_rocks[i][1] for i in 1:length(bad_rocks)], [bad_rocks[i][2] for i in 1:length(bad_rocks)], legend=false, color=:red)

	# Agent location
	plot!([CartesianIndices((10,10))[state_hist[i]].I[1] for i in 1:length(state_hist)],[CartesianIndices((10,10))[state_hist[i]].I[2] for i in 1:length(state_hist)],legend=false, color=:orchid1, linestyle=:dashdotdot, linewidth=2)
	savefig("/Users/joshuaott/icra2022/figures/AIPPMS/$(name)$(trial_num).pdf")

end

function solver_test_isrs(pref::String;good_prob::Float64=0.5, num_rocks::Int64=10, num_beacons::Int64=25,
                          seed::Int64=1234, num_graph_trials=40)

    isrs_map_size = (10, 10)
    total_budget = 100.

    pos_dist = 1:10

    pomcp_gcb_rewards = Vector{Float64}(undef, 0)
    pomcp_basic_rewards = Vector{Float64}(undef, 0)

	total_planning_time_gcb = 0
	total_plans_gcb = 0

	total_planning_time_basic = 0
	total_plans_basic = 0

    i = 1
    idx = 1
    while idx <= num_graph_trials
        @show i
        # @show idx
        rng = MersenneTwister(seed+i)

        rocks_positions = ISRSPos[]
        rocks = ISRS_STATE[]
        beacon_positions = ISRSPos[]

        n = 1
        while n <= num_beacons
            beac_pos = (rand(rng, pos_dist), rand(rng, pos_dist))
            if findfirst(isequal(beac_pos), rocks_positions) == nothing &&
                findfirst(isequal(beac_pos), beacon_positions) == nothing
                push!(beacon_positions, beac_pos)
                n = n + 1
            end
        end

        n = 1
        while n <= num_rocks
            rock_pos = (rand(rng, pos_dist), rand(rng, pos_dist))

            if rock_pos != (1, 1) && findfirst(isequal(rock_pos), rocks_positions) == nothing

                push!(rocks_positions, rock_pos)

                if rand(rng) < good_prob
                    rock_state = RSGOOD
                else
                    rock_state = RSBAD
                end
                push!(rocks, rock_state)
                n = n + 1
            end
        end


        pomdp = setup_isrs_pomdp(isrs_map_size, rocks_positions, rocks, beacon_positions, total_budget)
        isrs_env = pomdp.env
        # println(rocks_positions)

        ns = NaiveSolver(isrs_env, total_budget)

        pomcp_isterminal(s) = POMDPs.isterminal(pomdp, s)
        naive_isterminal(s) = MultimodalIPP.isterminal_naive(ns, s)

		depth = 30
        pomcp_gcb_policy = get_pomcp_gcb_policy(isrs_env, pomdp, total_budget, rng, depth, 100)
        pomcp_basic_policy = get_pomcp_basic_policy(isrs_env, pomdp, total_budget, rng, depth, 100)

        pomcp_gcb_reward = 0.0
        pomcp_basic_reward = 0.0

		try
			pomcp_gcb_reward, state_hist, location_states_hist, action_hist, reward_hist, planning_time, num_plans = graph_trial(rng, pomdp, pomcp_gcb_policy, pomcp_isterminal)
			total_planning_time_gcb += planning_time
			total_plans_gcb += num_plans
			# plot_trial(state_hist, location_states_hist, action_hist, reward_hist, i, "gcb")
			@show pomcp_gcb_reward


			pomcp_basic_reward, state_hist, location_states_hist, action_hist, reward_hist = graph_trial(rng, pomdp, pomcp_basic_policy, pomcp_isterminal)
			total_planning_time_basic += planning_time
			total_plans_basic += num_plans
			# plot_trial(state_hist, location_states_hist, action_hist, reward_hist, i, "basic")
			@show pomcp_basic_reward
		catch y
			if isa(y, InterruptException)
                throw(InterruptException)
            end
            pomcp_gcb_reward = 0.0
            pomcp_basic_reward = 0.0
            i = i+1
            continue
		end


        # try
        #     pomcp_gcb_reward = graph_trial(rng, pomdp, pomcp_gcb_policy, pomcp_isterminal)
        #     @show pomcp_gcb_reward
		#
        #     pomcp_basic_reward = graph_trial(rng, pomdp, pomcp_basic_policy, pomcp_isterminal)
        #     @show pomcp_basic_reward
        # catch y
        #     if isa(y, InterruptException)
        #         throw(InterruptException)
        #     end
        #     pomcp_gcb_reward = 0.0
        #     pomcp_basic_reward = 0.0
        #     i = i+1
        #     continue
        # end

        i = i+1
        idx = idx+1

        push!(pomcp_gcb_rewards, pomcp_gcb_reward)
        push!(pomcp_basic_rewards, pomcp_basic_reward)
    end

	println("POMCP GCB average planning time: ", total_planning_time_gcb/total_plans_gcb)
	println("POMCP Basic average planning time: ", total_planning_time_basic/total_plans_basic)

    @show mean(pomcp_gcb_rewards)
    @show mean(pomcp_basic_rewards)

    outfile_pomcp_gcb = string("isrs-pomcp-gcb-",pref,".json")
    open(outfile_pomcp_gcb,"w") do f
        JSON.print(f,Dict("rewards"=>pomcp_gcb_rewards),2)
    end

    outfile_pomcp_basic = string("isrs-pomcp-basic-",pref,".json")
    open(outfile_pomcp_basic,"w") do f
        JSON.print(f,Dict("rewards"=>pomcp_basic_rewards),2)
    end
end


solver_test_isrs("test", good_prob=0.5)
