using Colors

function plot_trial(true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num)

	# increase GP query resolution for plotting
	# k = with_lengthscale(SqExponentialKernel(), 0.5) + with_lengthscale(MaternKernel(), 0.5)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 1.0) # NOTE: check length scale
	plot_scale = 1:0.18:10
	X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
    #GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

	drill_idxs = action_hist .== :drill
	drill_idxs = findall(i-> i==1, drill_idxs)
	good_drill_idx = []
	bad_drill_idx = []
	for i in drill_idxs
		if reward_hist[i] == -1
			append!(bad_drill_idx, i)
		elseif reward_hist[i] == 1
			append!(good_drill_idx, i)
		else
			println("IDX MISALIGNMENT")
		end
	end

    anim = @animate for i = 1:length(state_hist)

		# increase GP query resolution for plotting
		if gp_hist[i].X == []
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
		else
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
		end


		# Display Total Reward
		if i == 1
			title = "Total Reward: $(total_reward_hist[1])"
		else
			title = "Total Reward: $(total_reward_hist[i-1])"
		end


		# Gaussian Process Variance
		if gp_hist[i].X == []
			contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.05:1.2)) # xlims = (1, 10), ylims = (1, 10)
			# contourf(collect(1:0.3:9.7), collect(1:0.3:9.7), reshape(query_no_data(gp_hist[i])[2], (30,30))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
		else
	    	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.05:1.2))
		end

		# Drill Positions
		good_scatter_idx = findlast(good_drill_idx .<= i)
		bad_scatter_idx = findlast(bad_drill_idx .<= i)
		if good_scatter_idx != nothing
			scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1)
		end
		if bad_scatter_idx != nothing
			scatter!([CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[1] for i in collect(1:bad_scatter_idx)],[CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[2] for i in collect(1:bad_scatter_idx)],legend=false, color=:red, markeralpha=1)
		end

		# Agent location
		scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title)

	end
	Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/variance/GP_BMDP_ROVER$(trial_num).gif", fps = 1)
	# Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).gif", fps = 1)


	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
	gp = GaussianProcess(gp_hist[end].m, μ(X_plot, gp_hist[end].m), k, gp_hist[end].X, X_plot, gp_hist[end].y, gp_hist[end].ν, gp_hist[end].KXX, K(X_plot, gp_hist[end].X, k), KXqXq);

	# Display Total Reward
	title = "Total Reward: $(total_reward_hist[end-1])"

	# Gaussian Process Variance
	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.05:1.2))

	# Agent location
	plot!([CartesianIndices((10,10))[state_hist[i]].I[1] for i in 1:length(state_hist)],[CartesianIndices((10,10))[state_hist[i]].I[2] for i in 1:length(state_hist)],legend=false, color=:orchid1, linestyle=:dashdotdot, linewidth=2)
	good_scatter_idx = findlast(good_drill_idx .<= length(action_hist))
	bad_scatter_idx = findlast(bad_drill_idx .<= length(action_hist))
	if good_scatter_idx != nothing
		scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1)
	end
	if bad_scatter_idx != nothing
		scatter!([CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[1] for i in collect(1:bad_scatter_idx)],[CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[2] for i in collect(1:bad_scatter_idx)],legend=false, color=:red, markeralpha=1)
	end

	savefig("/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/variance/GP_BMDP_ROVER$(trial_num).pdf")
	# savefig("/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).pdf")


end

function plot_trial_with_mean(true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num)

	# increase GP query resolution for plotting
	# k = with_lengthscale(SqExponentialKernel(), 0.5) + with_lengthscale(MaternKernel(), 0.5)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 1.0) # NOTE: check length scale
	plot_scale = 1:0.18:10
	X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
    #GP = GaussianProcess(m, μ(X_query, m), k, [], X_query, [], [], [], [], KXqXq);

	drill_idxs = action_hist .== :drill
	drill_idxs = findall(i-> i==1, drill_idxs)
	good_drill_idx = []
	bad_drill_idx = []
	for i in drill_idxs
		if reward_hist[i] == -1
			append!(bad_drill_idx, i)
		elseif reward_hist[i] == 1
			append!(good_drill_idx, i)
		else
			println("IDX MISALIGNMENT")
		end
	end

    anim = @animate for i = 1:length(state_hist)

		# increase GP query resolution for plotting
		if gp_hist[i].X == []
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
		else
			gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
		end


		# Display Total Reward
		if i == 1
			title = "Total Reward: $(total_reward_hist[1])"
		else
			title = "Total Reward: $(total_reward_hist[i-1])"
		end


		# Gaussian Process Variance
		if gp_hist[i].X == []
			contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:viridis, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.05:1.2)) # xlims = (1, 10), ylims = (1, 10)
			# contourf(collect(1:0.3:9.7), collect(1:0.3:9.7), reshape(query_no_data(gp_hist[i])[2], (30,30))', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
		else
	    	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:viridis, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.05:1.2))
		end

		# Drill Positions
		good_scatter_idx = findlast(good_drill_idx .<= i)
		bad_scatter_idx = findlast(bad_drill_idx .<= i)
		if good_scatter_idx != nothing
			scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1)
		end
		if bad_scatter_idx != nothing
			scatter!([CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[1] for i in collect(1:bad_scatter_idx)],[CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[2] for i in collect(1:bad_scatter_idx)],legend=false, color=:red, markeralpha=1)
		end

		# Agent location
		scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title)

	end
	Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/mean/GP_BMDP_ROVER$(trial_num).gif", fps = 1)
	# Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).gif", fps = 1)


	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
	gp = GaussianProcess(gp_hist[end].m, μ(X_plot, gp_hist[end].m), k, gp_hist[end].X, X_plot, gp_hist[end].y, gp_hist[end].ν, gp_hist[end].KXX, K(X_plot, gp_hist[end].X, k), KXqXq);

	# Display Total Reward
	title = "Total Reward: $(total_reward_hist[end-1])"

	# Gaussian Process Variance
	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:viridis, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.05:1.2))
	# println("MIN: ", minimum(query(gp)[1]))
	# println("MAX: ", maximum(query(gp)[1]))

	# Agent location
	plot!([CartesianIndices((10,10))[state_hist[i]].I[1] for i in 1:length(state_hist)],[CartesianIndices((10,10))[state_hist[i]].I[2] for i in 1:length(state_hist)],legend=false, color=:orchid1, linestyle=:dashdotdot, linewidth=2)
	good_scatter_idx = findlast(good_drill_idx .<= length(action_hist))
	bad_scatter_idx = findlast(bad_drill_idx .<= length(action_hist))
	if good_scatter_idx != nothing
		scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1)
	end
	if bad_scatter_idx != nothing
		scatter!([CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[1] for i in collect(1:bad_scatter_idx)],[CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[2] for i in collect(1:bad_scatter_idx)],legend=false, color=:red, markeralpha=1)
	end

	savefig("/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/mean/GP_BMDP_ROVER$(trial_num).pdf")
	# savefig("/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).pdf")


end
