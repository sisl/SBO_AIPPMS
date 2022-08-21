using Colors

theme(:dao)

function plot_trial(true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num, trial_name, use_ssh_dir)

	# increase GP query resolution for plotting
	# k = with_lengthscale(SqExponentialKernel(), 1.0) + with_lengthscale(MaternKernel(), 1.0)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 1.0) # NOTE: check length scale
	# k = with_lengthscale(SqExponentialKernel(), 1.0) + with_lengthscale(MaternKernel(), 1.0)# NOTE: check length scale

	plot_scale = 1:0.1:10
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
			title = "Variance \n Total Reward: $(total_reward_hist[1])"
		else
			title = "Variance \n Total Reward: $(total_reward_hist[i-1])"
		end


		# Gaussian Process Variance
		if gp_hist[i].X == []
			contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)
			# contourf(collect(1:0.3:9.7), collect(1:0.3:9.7), reshape(query_no_data(gp_hist[i])[2], (30,30))', colorbar = true, c =  cgrad(:davos, rev = false), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
		else
			contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)
		end

		# Drill Positions
		good_scatter_idx = findlast(good_drill_idx .<= i)
		bad_scatter_idx = findlast(bad_drill_idx .<= i)
		if good_scatter_idx != nothing
			scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1, markersize=6)
		end
		if bad_scatter_idx != nothing
			scatter!([CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[1] for i in collect(1:bad_scatter_idx)],[CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[2] for i in collect(1:bad_scatter_idx)],legend=false, color=:red, markeralpha=1, markersize=6)
		end

		# Agent location
		scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title, markersize=7)

	end
	if use_ssh_dir
		Plots.gif(anim, "/home/jott2/figures/GP_BMDP_Rover/$(trial_name)/variance/GP_BMDP_ROVER$(trial_num).gif", fps = 2)
	else
		Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/$(trial_name)/variance/GP_BMDP_ROVER$(trial_num).gif", fps = 2)
	end

	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
	gp = GaussianProcess(gp_hist[end].m, μ(X_plot, gp_hist[end].m), k, gp_hist[end].X, X_plot, gp_hist[end].y, gp_hist[end].ν, gp_hist[end].KXX, K(X_plot, gp_hist[end].X, k), KXqXq);

	# Display Total Reward
	title = "Variance \n Total Reward: $(total_reward_hist[end-1])"

	# Gaussian Process Variance
	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[2], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)

	# Agent location
	plot!([CartesianIndices((10,10))[state_hist[i]].I[1] for i in 1:length(state_hist)],[CartesianIndices((10,10))[state_hist[i]].I[2] for i in 1:length(state_hist)],legend=false, color=:orchid1, linestyle=:solid, linewidth=3)
	good_scatter_idx = findlast(good_drill_idx .<= length(action_hist))
	bad_scatter_idx = findlast(bad_drill_idx .<= length(action_hist))
	if good_scatter_idx != nothing
		scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1, markersize=6)
	end
	if bad_scatter_idx != nothing
		scatter!([CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[1] for i in collect(1:bad_scatter_idx)],[CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[2] for i in collect(1:bad_scatter_idx)],legend=false, color=:red, markeralpha=1, markersize=6)
	end

	if use_ssh_dir
		savefig("/home/jott2/figures/GP_BMDP_Rover/$(trial_name)/variance/GP_BMDP_ROVER$(trial_num).pdf")
	else
		savefig("/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/$(trial_name)/variance/GP_BMDP_ROVER$(trial_num).pdf")
	end

end

function plot_trial_with_mean(true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num, trial_name, use_ssh_dir)

	# increase GP query resolution for plotting
	# k = with_lengthscale(SqExponentialKernel(), 1.0) + with_lengthscale(MaternKernel(), 1.0)# NOTE: check length scale
	k = with_lengthscale(SqExponentialKernel(), 1.0) # NOTE: check length scale
	# k = with_lengthscale(SqExponentialKernel(), 1.0) + with_lengthscale(MaternKernel(), 1.0)# NOTE: check length scale

	plot_scale = 1:0.1:10
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
			title = "Mean \n Total Reward: $(total_reward_hist[1])"
		else
			title = "Mean \n Total Reward: $(total_reward_hist[i-1])"
		end


		# Gaussian Process Variance
		if gp_hist[i].X == []
			contourf(collect(plot_scale), collect(plot_scale), reshape(query_no_data(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)
			# contourf(collect(1:0.3:9.7), collect(1:0.3:9.7), reshape(query_no_data(gp_hist[i])[2], (30,30))', colorbar = true, c =  cgrad(:davos, rev = false), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false,  xlabel = "x₁", ylabel = "x₂", aspectratio = :equal) # xlims = (1, 10), ylims = (1, 10)
		else
			contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Variance") # xlims = (1, 10), ylims = (1, 10)

		end

		# Drill Positions
		good_scatter_idx = findlast(good_drill_idx .<= i)
		bad_scatter_idx = findlast(bad_drill_idx .<= i)
		if good_scatter_idx != nothing
			scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1, markersize=6)
		end
		if bad_scatter_idx != nothing
			scatter!([CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[1] for i in collect(1:bad_scatter_idx)],[CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[2] for i in collect(1:bad_scatter_idx)],legend=false, color=:red, markeralpha=1, markersize=6)
		end

		# Agent location
		scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title, markersize=7)

	end
	if use_ssh_dir
		Plots.gif(anim, "/home/jott2/figures/GP_BMDP_Rover/$(trial_name)/mean/GP_BMDP_ROVER$(trial_num).gif", fps = 2)
	else
		Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/$(trial_name)/mean/GP_BMDP_ROVER$(trial_num).gif", fps = 2)
	end

	############################################################################
	# Just make the plot
	############################################################################
	# increase GP query resolution for plotting
	gp = GaussianProcess(gp_hist[end].m, μ(X_plot, gp_hist[end].m), k, gp_hist[end].X, X_plot, gp_hist[end].y, gp_hist[end].ν, gp_hist[end].KXX, K(X_plot, gp_hist[end].X, k), KXqXq);

	# Display Total Reward
	title = "Mean \n Total Reward: $(total_reward_hist[end-1])"

	# Gaussian Process
	contourf(collect(plot_scale), collect(plot_scale), reshape(query(gp)[1], (length(plot_scale),length(plot_scale)))', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Mean") # xlims = (1, 10), ylims = (1, 10)
	# println("MIN: ", minimum(query(gp)[1]))
	# println("MAX: ", maximum(query(gp)[1]))

	# Agent location
	plot!([CartesianIndices((10,10))[state_hist[i]].I[1] for i in 1:length(state_hist)],[CartesianIndices((10,10))[state_hist[i]].I[2] for i in 1:length(state_hist)],legend=false, color=:orchid1, linestyle=:solid, linewidth=3)
	good_scatter_idx = findlast(good_drill_idx .<= length(action_hist))
	bad_scatter_idx = findlast(bad_drill_idx .<= length(action_hist))
	if good_scatter_idx != nothing
		scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1, markersize=6)
	end
	if bad_scatter_idx != nothing
		scatter!([CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[1] for i in collect(1:bad_scatter_idx)],[CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[2] for i in collect(1:bad_scatter_idx)],legend=false, color=:red, markeralpha=1, markersize=6)
	end

	if use_ssh_dir
		savefig("/home/jott2/figures/GP_BMDP_Rover/$(trial_name)/mean/GP_BMDP_ROVER$(trial_num).pdf")
	else
		savefig("/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/$(trial_name)/mean/GP_BMDP_ROVER$(trial_num).pdf")
	end

end


function plot_true_map(true_map, trial_num, trial_name, use_ssh_dir)
	heatmap(collect(1:10), collect(1:10), true_map', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="True Map") # xlims = (1, 10), ylims = (1, 10)

	if use_ssh_dir
		savefig("/home/jott2/figures/GP_BMDP_Rover/$(trial_name)/true_map/true_map$(trial_num).png")
	else
		savefig("/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/$(trial_name)/true_map/true_map$(trial_num).png")
	end
end
#
function plot_error_map(true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num, trial_name, use_ssh_dir)
	k = with_lengthscale(SqExponentialKernel(), 1.0) # NOTE: check length scale
    plot_scale = 1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)

    error_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))
    true_interp_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))
    for i in 1:size(true_interp_map)[1]
        for j in 1:size(true_interp_map)[2]
            idx_i = Int(floor(i/10) + 1)
            idx_j = Int(floor(j/10) + 1)
            true_interp_map[i,j] = true_map[idx_i, idx_j]
        end
    end

    anim = @animate for i = 1:length(state_hist)
        # increase GP query resolution for plotting
        if gp_hist[i].X == []
            gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
            mean_map = query_no_data(gp)[1]

        else
            gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
            mean_map = query(gp)[1]
        end

        error_map = abs.(true_interp_map - reshape(mean_map, size(true_interp_map)))

		heatmap(collect(plot_scale), collect(plot_scale), error_map', colorbar = true, xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, axis=false, ticks=false, title="Difference from True Map")
    end

	if use_ssh_dir
		Plots.gif(anim, "/home/jott2/figures/GP_BMDP_Rover/$(trial_name)/error_map/error_map$(trial_num).gif", fps = 2)
	else
		Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/$(trial_name)/error_map/error_map$(trial_num).gif", fps = 2)
	end
end

function plot_RMSE_trajectory(true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num, trial_name, use_ssh_dir)
	k = with_lengthscale(SqExponentialKernel(), 1.0) # NOTE: check length scale
    plot_scale = 1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)

    error_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))
    true_interp_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))
    RMSE_hist = []

    for i in 1:size(true_interp_map)[1]
        for j in 1:size(true_interp_map)[2]
            idx_i = Int(floor(i/10) + 1)
            idx_j = Int(floor(j/10) + 1)
            true_interp_map[i,j] = true_map[idx_i, idx_j]
        end
    end

    anim = @animate for i = 1:length(state_hist)
        # increase GP query resolution for plotting
        if gp_hist[i].X == []
            gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
            mean_map = query_no_data(gp)[1]

        else
            gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
            mean_map = query(gp)[1]
        end

        error_map = abs.(true_interp_map - reshape(mean_map, size(true_interp_map)))
        RMSE = 0
        for i in 1:length(error_map)
            RMSE += error_map[i]^2
        end
        RMSE = sqrt(RMSE/length(error_map))

        append!(RMSE_hist, RMSE)

        plot(collect(1:i), RMSE_hist, color=RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), legend=false, xlabel="Trajectory Step", ylabel="RMSE", title="RMSE")
    end
	if use_ssh_dir
		Plots.gif(anim, "/home/jott2/figures/GP_BMDP_Rover/$(trial_name)/RMSE_traj/RMSE_traj$(trial_num).gif", fps = 2)
	else
		Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GP_BMDP_Rover/$(trial_name)/RMSE_traj/RMSE_traj$(trial_num).gif", fps = 2)
	end
end

#TODO: in order to do this you would need the history of historys for GP hist, state_hist, and history of true_maps
# but then you can plot the mean RMSE among the 40 different runs and the error bounds
# you can then do the same with the reward along the trajectory

function calculate_rmse_along_traj(true_map, state_hist, gp_hist, action_hist, total_reward_hist, reward_hist, trial_num)
	k = with_lengthscale(SqExponentialKernel(), 1.0) # NOTE: check length scale
	plot_scale = 1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)

    plot_scale = 1:0.1:10
    error_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))
    true_interp_map = zeros((length(collect(plot_scale)),length(collect(plot_scale))))
    RMSE_hist = []

    for i in 1:size(true_interp_map)[1]
        for j in 1:size(true_interp_map)[2]
            idx_i = Int(floor(i/10) + 1)
            idx_j = Int(floor(j/10) + 1)
            true_interp_map[i,j] = true_map[idx_i, idx_j]
        end
    end

    for i = 1:length(state_hist)
        # increase GP query resolution for plotting
        if gp_hist[i].X == []
            gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, [], X_plot, [], [], [], [], KXqXq);
            mean_map = query_no_data(gp)[1]

        else
            gp = GaussianProcess(gp_hist[i].m, μ(X_plot, gp_hist[i].m), k, gp_hist[i].X, X_plot, gp_hist[i].y, gp_hist[i].ν, gp_hist[i].KXX, K(X_plot, gp_hist[i].X, k), KXqXq);
            mean_map = query(gp)[1]
        end

        error_map = abs.(true_interp_map - reshape(mean_map, size(true_interp_map)))
        RMSE = 0
        for i in 1:length(error_map)
            RMSE += error_map[i]^2
        end
        RMSE = sqrt(RMSE/length(error_map))

        append!(RMSE_hist, RMSE)
	end

	return RMSE_hist
end

function plot_RMSE_trajectory_history(rmse_hist, trial_name, use_ssh_dir)
	min_length = minimum([length(rmse_hist[i]) for i in 1:length(rmse_hist)])
	μ = []
	σ = []
	for i in 1:min_length
		mn = []
    	for j in 1:length(rmse_hist)
			append!(mn, rmse_hist[j][i])
		end
		append!(μ, mean(mn))
		append!(σ, sqrt(var(mn)))
	end
	if trial_name == "gp_mcts_dpw"
		plot(collect(1:min_length), μ, ribbon = σ, xlabel="Trajectory Step", ylabel="RMSE",title="RMSE", legend=true, label="GPMCTS-DPW", color = RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073))
	else
		plot(collect(1:min_length), μ, ribbon = σ, xlabel="Trajectory Step", ylabel="RMSE",title="RMSE", legend=true, label="Raster", color = RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073))
	end
	
	if use_ssh_dir
		savefig("/home/jott2/icra2022/figures/RMSE_traj_$(trial_name).pdf")
	else
		savefig("/Users/joshuaott/icra2022/figures/RMSE_traj_$(trial_name).pdf")
	end

end
