#                     Sequential Bayesian Optimization for  <br /> Adaptive Informative Path Planning with Multimodal Sensing


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

Gaussian Process Belief MDP formulation using MCTS-DPW
<p align="center">
  <img alt="Mean" src="https://github.com/josh0tt/SBO_AIPPMS/blob/main/img/mean.gif" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Variance" src="https://github.com/josh0tt/SBO_AIPPMS/blob/main/img/var.gif" width="45%">
</p>

INCLUDE TABLE WITH RESULTS 

POMCP with Random Rollout Policy

POMCP with Generalized Cost-Benefit Rollout Policy

# Information Rock Sample

Gaussian Process Belief MDP formulation using MCTS-DPW

INCLUDE TABLE WITH RESULTS 

POMCP with Random Rollout Policy

POMCP with Generalized Cost-Benefit Rollout Policy


