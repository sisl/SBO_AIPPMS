function plot_trial(state_hist, location_states_hist, action_hist, reward_hist, trial_num, name)
	# NOTE reward_hist is actually total_reward_hist

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
	Plots.gif(anim, "/home/jott2/figures/AIPPMS/$(name)$(trial_num).gif", fps = 1)

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
	savefig("/home/jott2/figures/AIPPMS/$(name)$(trial_num).pdf")

end

########################################################################################
# RMSE and Trace 
########################################################################################

function convert_pos_idx_2_pos_coord(pomdp, pos)
    if pos == -1
        return SVector{2,Int}(-1,-1)
    else
        return SVector{2,Int}(CartesianIndices(pomdp.map_size)[pos].I[1], CartesianIndices(pomdp.map_size)[pos].I[2])
    end
end

function convert_pos_coord_2_pos_idx(pomdp, pos)
    if pos == SVector{2,Int}(-1,-1)
        return -1
    else
        return LinearIndices(pomdp.map_size)[pos[1], pos[2]]
    end
end

									
function calculate_rmse_along_traj(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, trial_num)
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
	m(x) = 0.0 # default to 0.5 in the middle of the sample spectrum
	plot_scale = 1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
	f_prior = GaussianProcess(m, μ(X_plot, m), k, [], X_plot, [], [], [], [], KXqXq);
	gp = f_prior

    plot_scale = 1:0.1:10
    RMSE_hist = []

	loc_states = deepcopy(location_states_hist[1])

    for state_idx = 1:length(state_hist)
        if state_idx == 1
            gp = f_prior
            mean_map = query_no_data(gp)[1]
        else
			a = action_hist[state_idx - 1]

			if a.visit_location != nothing
				if loc_states[a.visit_location] == RSGOOD || loc_states[a.visit_location] == RSBAD
					# NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
					# for normal dist whereas our GP setup uses σ²_n
					y = 0.0 #we know it becomes bad if it was good and we know it is bad if it was bad
					σ²_n = 1e-9 # don't square this causes singular exception in belief update
					gp = posterior(gp, [[CartesianIndices(pomdp.map_size)[a.visit_location].I[1], CartesianIndices(pomdp.map_size)[a.visit_location].I[2]]], [y], [σ²_n])
				end

			else
				#for (i, loc) in enumerate(pomdp.env.location_states)
				for (i, loc) in enumerate(obs_hist[state_idx-1].obs_location_states)
					# Only bother if true location is a rock
					if loc == RSGOOD || loc == RSBAD

						dist = norm(pomdp.env.location_metadata[state_hist[state_idx]] - pomdp.env.location_metadata[i])
						prob_correct = 0.5*(1 + 2^(-4*dist/a.sensing_action.efficiency)) # TODO: Check

						y = (loc == RSGOOD) ? 1.0 : 0.0

						σ²_n = 1-prob_correct
						gp = posterior(gp, [[CartesianIndices(pomdp.map_size)[i].I[1], CartesianIndices(pomdp.map_size)[i].I[2]]], [y], [σ²_n])
					end
				end
			end

			if gp.X == []
				mean_map = query_no_data(gp)[1]
			else
				mean_map = query(gp)[1]
			end
		end

		good_rocks = findall(x-> x == RSGOOD, loc_states)
		good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

		bad_rocks = findall(x-> x == RSBAD, loc_states)
		bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

		beacons = findall(x-> x == RSBEACON, loc_states)
		beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]

        RMSE = 0
        for j in 1:length(good_rocks)
            rock_idx = convert_pos_coord_2_pos_idx(pomdp, good_rocks[j])
            RMSE += (1-mean_map[rock_idx])^2
        end
        for j in 1:length(bad_rocks)
            rock_idx = convert_pos_coord_2_pos_idx(pomdp, bad_rocks[j])
            RMSE += (0-mean_map[rock_idx])^2
        end
        num_rocks = length(good_rocks) + length(bad_rocks)
        RMSE = sqrt(RMSE/num_rocks)

        append!(RMSE_hist, RMSE)

		# Update rock visit
		if loc_states[state_hist[state_idx]] == RSGOOD
			# println(loc_states[state_hist[i]])
			loc_states[state_hist[state_idx]] = RSBAD
			# println(loc_states[state_hist[i]])
		end
    end

    return RMSE_hist
end


function calculate_trace_Σ(pomdp, location_states_hist, state_hist, action_hist, obs_hist, total_reward_hist, reward_hist, trial_num)
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
	m(x) = 0.0 # default to 0.5 in the middle of the sample spectrum
	plot_scale = 1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
	f_prior = GaussianProcess(m, μ(X_plot, m), k, [], X_plot, [], [], [], [], KXqXq);
	gp = f_prior

    plot_scale = 1:0.1:10
    trace_hist = []

	loc_states = deepcopy(location_states_hist[1])

    for state_idx = 1:length(state_hist)
        if state_idx == 1
            gp = f_prior
            ν = query_no_data(gp)[2]
			trace_Σ = sum(ν)
        else
			a = action_hist[state_idx - 1]

			if a.visit_location != nothing
				if loc_states[a.visit_location] == RSGOOD || loc_states[a.visit_location] == RSBAD
					# NOTE: σ_n is the stddev whereas σ²_n is the varaiance. Julia uses σ_n
					# for normal dist whereas our GP setup uses σ²_n
					y = 0.0 #we know it becomes bad if it was good and we know it is bad if it was bad
					σ²_n = 1e-9 # don't square this causes singular exception in belief update
					gp = posterior(gp, [[CartesianIndices(pomdp.map_size)[a.visit_location].I[1], CartesianIndices(pomdp.map_size)[a.visit_location].I[2]]], [y], [σ²_n])
				end

			else
				#for (i, loc) in enumerate(pomdp.env.location_states)
				for (i, loc) in enumerate(obs_hist[state_idx-1].obs_location_states)
					# Only bother if true location is a rock
					if loc == RSGOOD || loc == RSBAD

						dist = norm(pomdp.env.location_metadata[state_hist[state_idx]] - pomdp.env.location_metadata[i])
						prob_correct = 0.5*(1 + 2^(-4*dist/a.sensing_action.efficiency)) # TODO: Check

						y = (loc == RSGOOD) ? 1.0 : 0.0

						σ²_n = 1-prob_correct
						gp = posterior(gp, [[CartesianIndices(pomdp.map_size)[i].I[1], CartesianIndices(pomdp.map_size)[i].I[2]]], [y], [σ²_n])
					end
				end
			end
			if gp.X == []
				ν = query_no_data(gp)[2]
			else
				ν = query(gp)[2]
			end
			trace_Σ = sum(ν)
		end

		append!(trace_hist, trace_Σ)

		# Update rock visit
		if loc_states[state_hist[state_idx]] == RSGOOD
			# println(loc_states[state_hist[i]])
			loc_states[state_hist[state_idx]] = RSBAD
			# println(loc_states[state_hist[i]])
		end

	end

	return trace_hist
end
