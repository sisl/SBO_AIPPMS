import subprocess

NUM_PROCESSES = 30

for i in range(NUM_PROCESSES):
	subprocess.call(['julia', '--project', 'Trials.jl',str(1234*(i+1))])
