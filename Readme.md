# Tabular Reinforcement Learning

This repository contains an academic reinforcement learning project focused on implementing and comparing classic tabular, model-free reinforcement learning algorithms in a stochastic gridworld environment. The project evaluates how different learning strategies, exploration methods, and backup depths influence learning performance.

The work is based on the Stochastic Windy Gridworld and follows an experimental methodology consistent with standard reinforcement learning literature.

---

## Project Overview

The objectives of this project are to:

- Implement core tabular reinforcement learning algorithms
- Compare on-policy and off-policy learning methods
- Study the effect of different exploration strategies
- Analyze the impact of backup depth on learning performance
- Empirically evaluate algorithms using repeated experiments

---

## Algorithms Implemented

The following reinforcement learning algorithms are implemented:

- **Q-learning** (off-policy temporal-difference control)
- **SARSA** (on-policy temporal-difference control)
- **n-step Q-learning**
- **Monte Carlo control**

All algorithms use a tabular Q-value representation and support ε-greedy and softmax exploration.

---

## Environment

The environment used in this project is a **Stochastic Windy Gridworld**, characterized by:

- Discrete state and action spaces
- Stochastic wind dynamics
- Terminal goal state
- Episodic task formulation

The environment is shared across all algorithms to ensure fair comparison.

---

## Repository Structure

.
├── Agent.py                 # Base agent with shared action selection and evaluation logic
├── DynamicProgramming.py    # Q-value iteration (model-based DP)
├── Environment.py           # Stochastic Windy Gridworld environment
├── Experiment.py            # SARSA implementation
├── Q_learning.py            # Q-learning implementation
├── Nstep.py                 # n-step Q-learning implementation
├── MonteCarlo.py            # Monte Carlo control implementation
├── Helper.py                # Utilities (softmax, argmax, plotting, smoothing)
├── run_experiments.py       # Main experimental driver and plotting script
└── README.md



---

## Experimental Setup

Experiments are conducted by averaging results over multiple repetitions. The following aspects are evaluated:

- **Exploration strategies**
  - ε-greedy exploration with different ε values
  - Softmax exploration with different temperature parameters

- **On-policy vs off-policy learning**
  - SARSA compared to Q-learning

- **Backup depth**
  - n-step Q-learning with varying n
  - Monte Carlo control as a full-return baseline

Performance is measured using **mean episodic return** over time.

---

## Running the Experiments

To run the full set of experiments and generate learning curves, execute:

python experiment.py


## Requirements:

- Python 3.x

- NumPy

- Matplotlib

- SciPy