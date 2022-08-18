#!/usr/bin/env bash

for i in {1..30}; do
	julia --project Trials.jl $((1234*i)) &
done
