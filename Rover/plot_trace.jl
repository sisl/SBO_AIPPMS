using Plots
using DelimitedFiles
using StatsBase


theme(:dao)

trace_hist_gcb = readdlm("/Users/joshuaott/icra2022/ssh/Rover/budget_100/trace_hist_gcb.csv", ',')
trace_hist_basic = readdlm("/Users/joshuaott/icra2022/ssh/Rover/budget_100/trace_hist_basic.csv", ',')
trace_hist_raster = readdlm("/Users/joshuaott/icra2022/ssh/Rover/budget_100/trace_hist_raster.csv", ',')
trace_hist_gp_mcts = readdlm("/Users/joshuaott/icra2022/ssh/Rover/budget_100/trace_hist_gp_mcts.csv", ',')

trace_hist_gcb_corrected = []
for i in 1:size(trace_hist_gcb)[1]
	tmp_hist = []
	for j in 1:size(trace_hist_gcb)[2]
		if typeof(trace_hist_gcb[i,j]) == Float64
			append!(tmp_hist, trace_hist_gcb[i,j])
		end
	end
	append!(trace_hist_gcb_corrected, [tmp_hist])
end

trace_hist_basic_corrected = []
for i in 1:size(trace_hist_basic)[1]
	tmp_hist = []
	for j in 1:size(trace_hist_basic)[2]
		if typeof(trace_hist_basic[i,j]) == Float64
			append!(tmp_hist, trace_hist_basic[i,j])
		end
	end
	append!(trace_hist_basic_corrected, [tmp_hist])
end

trace_hist_raster_corrected = []
for i in 1:size(trace_hist_raster)[1]
	tmp_hist = []
	for j in 1:size(trace_hist_raster)[2]
		if typeof(trace_hist_raster[i,j]) == Float64
			append!(tmp_hist, trace_hist_raster[i,j])
		end
	end
	append!(trace_hist_raster_corrected, [tmp_hist])
end

trace_hist_gp_mcts_corrected = []
for i in 1:size(trace_hist_gp_mcts)[1]
	tmp_hist = []
	for j in 1:size(trace_hist_gp_mcts)[2]
		if typeof(trace_hist_gp_mcts[i,j]) == Float64
			append!(tmp_hist, trace_hist_gp_mcts[i,j])
		end
	end
	append!(trace_hist_gp_mcts_corrected, [tmp_hist])
end


# function plot_trace_trajectory_history_together(trace_hist, trial_name, use_ssh_dir, min_length)
function plot_trace_trajectory_history_together(trace_hist, trial_name, use_ssh_dir, min_length, color)
	# min_length = minimum([length(trace_hist[i]) for i in 1:length(trace_hist)])
	μ = []
	σ = []
	for i in 1:min_length
		mn = []
    	for j in 1:length(trace_hist)
			append!(mn, trace_hist[j][i])
		end
		append!(μ, mean(mn))
		append!(σ, sqrt(var(mn)))
	end
	plot!(collect(1:min_length), μ, ribbon = σ, xlabel="Trajectory Step", ylabel="Tr(Σ)",title="Tr(Σ)", legend=false, label=trial_name, color=color,fillalpha=0.2, size=(400,400), xlim=(1,min_length), ylim=(2000,8300))
end




use_ssh_dir = false
# trial_names = ["Raster", "GPMCTS-DPW", "POMCP", "POMCP GCB"]
# trial_names = ["POMCP GCB", "POMCP", "Raster", "GPMCTS-DPW"]
trial_names = ["MCTS-DPW", "POMCP", "POMCP GCB", "Raster"]

hists = [trace_hist_gp_mcts_corrected, trace_hist_basic_corrected, trace_hist_gcb_corrected, trace_hist_raster_corrected]
# hists = [trace_hist_raster_corrected, trace_hist_gp_mcts_corrected, trace_hist_basic_corrected, trace_hist_gcb_corrected]
colors = [RGB{Float64}(0.0,0.6056031611752245,0.9786801175696073), :black, :green, :red]
# alphas = [0.4, 0.3, 0.2, 0.1]

min_length = []
for hist in hists
    append!(min_length, minimum([length(hist[i]) for i in 1:length(hist)]))
end
min_length = minimum(min_length)

plot()
for i in 1:length(hists)
	# plot_trace_trajectory_history_together(hists[i], trial_names[i], use_ssh_dir, min_length, colors[i], alphas[i])
	plot_trace_trajectory_history_together(hists[i], trial_names[i], use_ssh_dir, min_length, colors[i])
end

if use_ssh_dir
	savefig("/home/jott2/icra2022/figures/trace_traj_together.pdf")
else
	savefig("/Users/joshuaott/icra2022/ssh/Rover/budget_100/trace_traj_together.pdf")
end
