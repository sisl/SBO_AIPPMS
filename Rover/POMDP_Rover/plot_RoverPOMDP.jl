using Colors

theme(:dao)

function plot_trial(true_map, state_hist, belief_hist, action_hist, total_reward_hist, reward_hist, trial_num, solver_name, use_ssh_dir)

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

		# Display Total Reward
		if i == 1
			title = "Total Reward: $(total_reward_hist[1])"
		else
			title = "Total Reward: $(total_reward_hist[i-1])"
		end

		particles = belief_hist[i].particles
		var_belief = zeros(size(true_map))
		for i in 1:size(particles)[2] # number of locations
			var_belief[i] = var(particles[:,i])
		end
		contourf(var_belief', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Mean")

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
		Plots.gif(anim, "/home/jott2/figures/POMDP_ROVER/$(solver_name)/variance/POMDP_ROVER$(trial_num).gif", fps = 1)
	else
		Plots.gif(anim, "/Users/joshuaott/icra2022/figures/POMDP_ROVER/$(solver_name)/variance/POMDP_ROVER$(trial_num).gif", fps = 1)
	end

	############################################################################
	# Just make the plot
	############################################################################
	particles = belief_hist[end].particles
	var_belief = zeros(size(true_map))
	for i in 1:size(particles)[2] # number of locations
		var_belief[i] = var(particles[:,i])
	end
	contourf(var_belief', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Mean")

	# Display Total Reward
	title = "Total Reward: $(total_reward_hist[end-1])"

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
		savefig("/home/jott2/figures/POMDP_Rover/$(solver_name)/variance/POMDP_ROVER$(trial_num).pdf")
	else
		savefig("/Users/joshuaott/icra2022/figures/POMDP_Rover/$(solver_name)/variance/POMDP_ROVER$(trial_num).pdf")
	end

end

function plot_trial_with_mean(true_map, state_hist, belief_hist, action_hist, total_reward_hist, reward_hist, trial_num, solver_name, use_ssh_dir)

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

		particles = belief_hist[i].particles
		mean_belief = zeros(size(true_map))
		for i in 1:size(particles)[2] # number of locations
			mean_belief[i] = mean(particles[:,i])
		end
		contourf(mean_belief', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Mean")


		# Display Total Reward
		if i == 1
			title = "Total Reward: $(total_reward_hist[1])"
		else
			title = "Total Reward: $(total_reward_hist[i-1])"
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
		Plots.gif(anim, "/home/jott2/figures/POMDP_Rover/$(solver_name)/mean/POMDP_ROVER$(trial_num).gif", fps = 1)
	else
		Plots.gif(anim, "/Users/joshuaott/icra2022/figures/POMDP_Rover/$(solver_name)/mean/POMDP_ROVER$(trial_num).gif", fps = 1)
	end

	############################################################################
	# Just make the plot
	############################################################################

	particles = belief_hist[end].particles
	mean_belief = zeros(size(true_map))
	for i in 1:size(particles)[2] # number of locations
		mean_belief[i] = mean(particles[:,i])
	end
	contourf(mean_belief', colorbar = true, c = cgrad(:davos, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(-0.2:0.1:1.2), axis=false, ticks=false, title="Mean")


	# Display Total Reward
	title = "Total Reward: $(total_reward_hist[end-1])"

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
		savefig("/home/jott2/figures/POMDP_Rover/$(solver_name)/mean/POMDP_ROVER$(trial_num).pdf")
	else
		savefig("/Users/joshuaott/icra2022/figures/POMDP_Rover/$(solver_name)/mean/POMDP_ROVER$(trial_num).pdf")
	end

end


function calculate_rmse_along_traj(pomdp, true_map, state_hist, belief_hist, action_hist, obs_hist, total_reward_hist, reward_hist, trial_num)
	k = with_lengthscale(SqExponentialKernel(), 1.0) # NOTE: check length scale
	m(x) = 0.0 # default to 0.5 in the middle of the sample spectrum
	plot_scale = 1:0.1:10
    X_plot = [[i,j] for i = plot_scale, j = plot_scale]
    plot_size = size(X_plot)
    X_plot = reshape(X_plot, size(X_plot)[1]*size(X_plot)[2])
    KXqXq = K(X_plot, X_plot, k)
	f_prior = GaussianProcess(m, μ(X_plot, m), k, [], X_plot, [], [], [], [], KXqXq);
	gp = f_prior

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
		b = belief_hist[i]

        if i == 1
            gp = f_prior
            mean_map = query_no_data(gp)[1]
        else
			if action_hist[i-1] == :drill
		        drill_pos = convert_pos_idx_2_pos_coord(pomdp, state_hist[i])
		        σ²_n = 1e-9 #^2 dont square this causes singular exception in GP update
		        gp = posterior(gp, [[drill_pos[1], drill_pos[2]]], [obs_hist[i-1]], [σ²_n])
		    else
		        spec_pos = convert_pos_idx_2_pos_coord(pomdp, state_hist[i])
		        σ²_n = 0.1
		        gp = posterior(gp, [[spec_pos[1], spec_pos[2]]], [obs_hist[i-1]], [σ²_n])
		    end

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
	if trial_name == "gcb"
		plot(collect(1:min_length), μ, ribbon = σ, xlabel="Trajectory Step", ylabel="RMSE",title="RMSE", legend=true, label="POMCP GCB", color = RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073))
	else
		plot(collect(1:min_length), μ, ribbon = σ, xlabel="Trajectory Step", ylabel="RMSE",title="RMSE", legend=true, label="POMCP", color = RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073))
	end

	if use_ssh_dir
		savefig("/home/jott2/figures/RMSE_traj_$(trial_name).pdf")
	else
		savefig("/Users/joshuaott/icra2022/figures/RMSE_traj_$(trial_name).pdf")
	end

end
