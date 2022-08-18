using Colors

function plot_trial(true_map, state_hist, belief_hist, action_hist, total_reward_hist, reward_hist, trial_num, solver_name)

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
		contourf(var_belief', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(0:0.05:1.0)) # xlims = (1, 10), ylims = (1, 10)

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
	Plots.gif(anim, "/Users/joshuaott/icra2022/figures/POMDP_ROVER/$(solver_name)/variance/POMDP_ROVER$(trial_num).gif", fps = 1)
	# Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).gif", fps = 1)


	############################################################################
	# Just make the plot
	############################################################################
	particles = belief_hist[end].particles
	var_belief = zeros(size(true_map))
	for i in 1:size(particles)[2] # number of locations
		var_belief[i] = var(particles[:,i])
	end
	contourf(var_belief', colorbar = true, c = cgrad(:Blues_3, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(0:0.05:1.0)) # xlims = (1, 10), ylims = (1, 10)

	# Display Total Reward
	title = "Total Reward: $(total_reward_hist[end-1])"

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

	savefig("/Users/joshuaott/icra2022/figures/POMDP_Rover/$(solver_name)/variance/POMDP_ROVER$(trial_num).pdf")
	# savefig("/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).pdf")


end

function plot_trial_with_mean(true_map, state_hist, belief_hist, action_hist, total_reward_hist, reward_hist, trial_num, solver_name)

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
		contourf(mean_belief', colorbar = true, c = cgrad(:viridis, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(0:0.05:1.0)) # xlims = (1, 10), ylims = (1, 10)


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
			scatter!([CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[1] for i in collect(1:good_scatter_idx)],[CartesianIndices((10,10))[state_hist[good_drill_idx[i]]].I[2] for i in collect(1:good_scatter_idx)],legend=false, color=:green, markeralpha=1)
		end
		if bad_scatter_idx != nothing
			scatter!([CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[1] for i in collect(1:bad_scatter_idx)],[CartesianIndices((10,10))[state_hist[bad_drill_idx[i]]].I[2] for i in collect(1:bad_scatter_idx)],legend=false, color=:red, markeralpha=1)
		end

		# Agent location
		scatter!([CartesianIndices((10,10))[state_hist[i]].I[1]],[CartesianIndices((10,10))[state_hist[i]].I[2]],legend=false, color=:orchid1, title=title)

	end
	Plots.gif(anim, "/Users/joshuaott/icra2022/figures/POMDP_Rover/$(solver_name)/mean/POMDP_ROVER$(trial_num).gif", fps = 1)
	# Plots.gif(anim, "/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).gif", fps = 1)


	############################################################################
	# Just make the plot
	############################################################################

	particles = belief_hist[end].particles
	mean_belief = zeros(size(true_map))
	for i in 1:size(particles)[2] # number of locations
		mean_belief[i] = mean(particles[:,i])
	end
	contourf(mean_belief', colorbar = true, c = cgrad(:viridis, rev = true), xlims = (0.5, 10.5), ylims = (0.5, 10.5), legend = false, aspectratio = :equal, clim=(0,1), grid=false, levels = collect(0:0.05:1.0)) # xlims = (1, 10), ylims = (1, 10)


	# Display Total Reward
	title = "Total Reward: $(total_reward_hist[end-1])"

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

	savefig("/Users/joshuaott/icra2022/figures/POMDP_Rover/$(solver_name)/mean/POMDP_ROVER$(trial_num).pdf")
	# savefig("/Users/joshuaott/icra2022/figures/GPISRS$(trial_num).pdf")


end
