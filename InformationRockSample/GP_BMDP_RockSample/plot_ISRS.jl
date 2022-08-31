function plot_trial(state_hist, location_states_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num, trial_name, use_ssh_dir)

	# increase GP query resolution for plotting
	# k = with_lengthscale(SqExponentialKernel(), 0.5) + with_lengthscale(MaternKernel(), 0.5)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
	plot_scale = 1:0.1:10
	X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
    #GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

	loc_states = deepcopy(location_states_hist[1])

    anim = @animate for i = 1:length(state_hist)

		# increase GP query resolution for plotting
		if gp_hist[i].X == []
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
		else
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
		end


		good_rocks = findall(x-> x == RSGOOD, loc_states)
		good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

		bad_rocks = findall(x-> x == RSBAD, loc_states)
		bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

		beacons = findall(x-> x == RSBEACON, loc_states)
		beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


		# Display Total Reward
		if i == 1
			title = "Variance \n Total Reward: $(total_reward_hist[1])"
		else
			title = "Variance \n Total Reward: $(total_reward_hist[i-1])"
		end


		# Gaussian Process Variance
		if gp_hist[i].X == []
			# contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) # xlims = (1, 10), ylims = (1, 10)
            contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title=title) # xlims = (1, 10), ylims = (1, 10)
        else
	    	# contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title=title) # xlims = (1, 10), ylims = (1, 10)
        end


	    # Beacons
		scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect)

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
	Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS/variance/GPISRS$(trial_num).gif", fps = 1)
	# Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).gif", fps = 1)


	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
	loc_states = deepcopy(location_states_hist[1])

	gp = GaussianProcess(gp_hist[end].m, μ(X_plot, gp_hist[end].m), k, gp_hist[end].X, X_plot, gp_hist[end].y, gp_hist[end].ν, gp_hist[end].KXX, K(X_plot, gp_hist[end].X, k), KXqXq);


	good_rocks = findall(x-> x == RSGOOD, loc_states)
	good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

	bad_rocks = findall(x-> x == RSBAD, loc_states)
	bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

	beacons = findall(x-> x == RSBEACON, loc_states)
	beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


	# Display Total Reward
	title = "Total Reward: $(reward_hist[end-1])"

	# Gaussian Process Variance
	# contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)
    
	# Beacons
	scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect)

	# Rocks
	scatter!([good_rocks[i][1] for i in 1:length(good_rocks)], [good_rocks[i][2] for i in 1:length(good_rocks)], legend=false, color=:green)
	scatter!([bad_rocks[i][1] for i in 1:length(bad_rocks)], [bad_rocks[i][2] for i in 1:length(bad_rocks)], legend=false, color=:red)

	# Agent location
	plot!([CartesianIndices((10,10))[state_hist[i]].I[1] for i in 1:length(state_hist)],[CartesianIndices((10,10))[state_hist[i]].I[2] for i in 1:length(state_hist)],legend=false, color=:orchid1, linestyle=:dashdotdot, linewidth=2)
	savefig("/Users/joshuaott/icra2022/figures/GPISRS/variance/GPISRS$(trial_num).pdf")
	# savefig("/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).pdf")


end

function plot_trial_with_mean(state_hist, location_states_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num, trial_name, use_ssh_dir)

	# increase GP query resolution for plotting
	# k = with_lengthscale(SqExponentialKernel(), 0.5) + with_lengthscale(MaternKernel(), 0.5)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
	plot_scale = 1:0.1:10
	X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
    #GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

	loc_states = deepcopy(location_states_hist[1])

    anim = @animate for i = 1:length(state_hist)

		# increase GP query resolution for plotting
		if gp_hist[i].X == []
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
		else
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
		end


		good_rocks = findall(x-> x == RSGOOD, loc_states)
		good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

		bad_rocks = findall(x-> x == RSBAD, loc_states)
		bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

		beacons = findall(x-> x == RSBEACON, loc_states)
		beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


		# Display Total Reward
		if i == 1
			title = "Mean \n Total Reward: $(total_reward_hist[1])"
		else
			title = "Mean \n Total Reward: $(total_reward_hist[i-1])"
		end


		# Gaussian Process Variance
		if gp_hist[i].X == []
			# contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) # xlims = (1, 10), ylims = (1, 10)
            contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title=title) # xlims = (1, 10), ylims = (1, 10)
        else
	    	# contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title=title) # xlims = (1, 10), ylims = (1, 10)

        end


	    # Beacons
		scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect)

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
	Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS/mean/mean_GPISRS$(trial_num).gif", fps = 1)
	# Plots.gif(anim, "/Users/joshuaott/icra2022/figures/mean_GPISRS$(trial_num).gif", fps = 1)

	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
	loc_states = deepcopy(location_states_hist[1])

	gp = GaussianProcess(gp_hist[end].m, μ(X_plot, gp_hist[end].m), k, gp_hist[end].X, X_plot, gp_hist[end].y, gp_hist[end].ν, gp_hist[end].KXX, K(X_plot, gp_hist[end].X, k), KXqXq);


	good_rocks = findall(x-> x == RSGOOD, loc_states)
	good_rocks = [CartesianIndices((10,10))[good_rocks[i]].I for i in 1:length(good_rocks)]

	bad_rocks = findall(x-> x == RSBAD, loc_states)
	bad_rocks = [CartesianIndices((10,10))[bad_rocks[i]].I for i in 1:length(bad_rocks)]

	beacons = findall(x-> x == RSBEACON, loc_states)
	beacons = [CartesianIndices((10,10))[beacons[i]].I for i in 1:length(beacons)]


	# Display Total Reward
	title = "Total Reward: $(reward_hist[end-1])"

	# Gaussian Process Variance
    contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Mean") # xlims = (1, 10), ylims = (1, 10)


	# Beacons
	scatter!([beacons[i][1] for i in 1:length(beacons)], [beacons[i][2] for i in 1:length(beacons)], legend=false, color=:grey, markershape=:rect)

	# Rocks
	scatter!([good_rocks[i][1] for i in 1:length(good_rocks)], [good_rocks[i][2] for i in 1:length(good_rocks)], legend=false, color=:green)
	scatter!([bad_rocks[i][1] for i in 1:length(bad_rocks)], [bad_rocks[i][2] for i in 1:length(bad_rocks)], legend=false, color=:red)

	# Agent location
	plot!([CartesianIndices((10,10))[state_hist[i]].I[1] for i in 1:length(state_hist)],[CartesianIndices((10,10))[state_hist[i]].I[2] for i in 1:length(state_hist)],legend=false, color=:orchid1, linestyle=:dashdotdot, linewidth=2)
	savefig("/Users/joshuaott/icra2022/figures/GPISRS/mean/mean_GPISRS$(trial_num).pdf")
	# savefig("/Users/joshuaott/icra2022/figures/mean_GPISRS$(trial_num).pdf")

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

function calculate_rmse_along_traj(pomdp, location_states_hist, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist)
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
	plot_scale = 1:0.1:10
	X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
    #GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

	loc_states = deepcopy(location_states_hist[1])
    RMSE_hist = []

    for i = 1:length(state_hist)
        if gp_hist[i].X == []
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
            mean_map = query_no_data(gp)[1]
		else
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
            mean_map = query(gp)[1]
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
		if loc_states[state_hist[i]] == RSGOOD
			# println(loc_states[state_hist[i]])
			loc_states[state_hist[i]] = RSBAD
			# println(loc_states[state_hist[i]])
		end
    end

    return RMSE_hist
end


function calculate_trace_Σ(pomdp, location_states_hist, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, i)
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
	plot_scale = 1:0.1:10
	X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
    #GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

    trace_hist = []

	for i = 1:length(state_hist)
		if gp_hist[i].X == []
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
			ν = query_no_data(gp)[2]
			trace_Σ = sum(ν)
		else
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
			ν = query(gp)[2]
			trace_Σ = sum(ν)
		end
		append!(trace_hist, trace_Σ)
	end

	return trace_hist
end


#TODO: remove this 
function plot_RMSE_trajectory(pomdp, location_states_hist, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num, trial_name, use_ssh_dir)
	k = with_lengthscale(SqExponentialKernel(), 0.5) # NOTE: check length scale
	plot_scale = 1:0.1:10
	X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
    #GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

    RMSE_hist = calculate_rmse_along_traj(pomdp, location_states_hist, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist)

    anim = @animate for i = 1:length(state_hist)
        plot(collect(1:i), RMSE_hist[1:i], color=RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), legend=false, xlabel="Trajectory Step", ylabel="RMSE", title="RMSE")
    end
	if use_ssh_dir
		Plots.gif(anim, "/home/jott2/figures/GPISRS/RMSE_traj/RMSE_traj$(trial_num).gif", fps = 2)
	else
		Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS/RMSE_traj/RMSE_traj$(trial_num).gif", fps = 2)
	end
end