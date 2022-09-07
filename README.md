# Sequential Bayesian Optimization for Adaptive <br /> Informative Path Planning with Multimodal Sensing


![](https://github.com/josh0tt/SBO_AIPPMS/blob/main/img/Figure1.jpg)

<!--
# GPAIPPMS

This repository contains the code for the publication

> Insert paper citation 
```
@inproceedings{fischer2020information,
  title     = {Gaussian Process-based Adaptive Informative Path Planning with Multimodal Sensing},
  author    = {Joshua Ott, Edward Balaban, and Mykel Kochenderfer},
  booktitle = {insert},
  year      = {2023},
  volume    = {insert},
  series    = {insert},
  publisher = {insert},
  address   = {insert},
  month     = {insert}
}
```

 -->

# Description

We compare our formulation of a Gaussian Process Belief MDP using MCTS-DPW with that of the POMDP formulation using POMCP with different rollout policies as first presented in: 
> Choudhury, Shushman, Nate Gruver, and Mykel J. Kochenderfer. "Adaptive informative path planning with multimodal sensing." Proceedings of the International Conference on Automated Planning and Scheduling. Vol. 30. 2020.

We have included their code in this repository for future benchmarking with the permission of the authors. The code uses the [JuliaPOMDP](https://github.com/JuliaPOMDP/POMDPs.jl) framework. 

# Rover Exploration

We introduce a new AIPPMS benchmark problem known as the Rover Exploration problem which is directly inspired by multiple planetary rover exploration missions. The rover begins at a specified starting location and has a set amount of energy available to explore the environment and reach the goal location. The rover is equipped with a spectrometer and a drill. Drilling reveals the true state of the environment at the location the drill sample was taken and as a result, is a more costly action to take from an energy budget perspective. Conversely, the spectrometer provides a noisy observation of the environment and uses less of the energy budget. At each step, the rover can decide whether or not it wants to drill. The rover's goal is to collect as many unique samples as it can while respecting its energy constraints. The rover receives +1 reward for drilling a sample that it has not yet seen and -1 reward for drilling a sample that it has already previously collected. This AIPPMS problem highlights the importance of taking sensing actions to decrease the uncertainty in the belief state before deciding to drill. 

The environment is modeled as an $n \times n$ grid with $\beta$ unique measurement types in the environment. To construct a spatially-correlated environment we first start by sampling each grid cell value from an independent and identically distributed uniform distribution of the $\beta$ unique measurement types. Each cell is then averaged with the value of all of its neighboring cells. This process creates environments with spatial correlation as well as some random noise to simulate a realistic geospatial environment.

For the Rover Exploration problem, we focus on the interplay between the energy budget allotted to the rover and the sensing quality of the spectrometer, where $\sigma_s$ denotes the standard deviation of a Gaussian sensor model. We also include a raster policy that attempts to fully sweep the environment in a raster pattern and evenly distribute its drilling actions along the way. However, the raster policy may not always be able to make it to the goal location in the specified energy budget in which case the rover receives $-\infty$ reward for a mission failure.

<p align="center">
  <img alt="Variance" src="https://github.com/josh0tt/SBO_AIPPMS/blob/main/img/rover.gif" width="100%">
</p>

# Information Rock Sample

We also evaluate our method on the Information Search RockSample (ISRS) problem introduced by He et al. and adapted by Choudry et al. ISRS is a variation of the classic RockSample problem. The agent must move through an environment represented as an $n \times n$ grid. Scattered throughout the environment are $k$ rocks with at most one rock in each grid cell. Only some of the rocks are considered to be `good,' meaning that they have some scientific value. The agent receives $+10$ reward for visiting a good rock and $-10$ reward for visiting a bad rock. Once a good rock is visited it becomes bad. The positions of the agent and rocks are known apriori, but visiting a rock is the only way to reveals its true state. 

There are also $b$ beacons scattered throughout the environment. Upon reaching a beacon location, the agent has the option to take a sensing action where it will receive observations about the state of the nearby rocks. The fidelity of the observation decreases with increasing distance from the beacon location. There are multiple sensing modalities available to the agent with a higher cost for choosing the more accurate sensing modality. Moving between adjacent cells also expends energy cost. The agent's goal is to visit as many good rocks and as few bad rocks as possible while returning to the origin without exceeding its energy budget. 

<p align="center">
  <img alt="Variance" src="https://github.com/josh0tt/SBO_AIPPMS/blob/main/img/isrs.gif" width="100%">
</p>



