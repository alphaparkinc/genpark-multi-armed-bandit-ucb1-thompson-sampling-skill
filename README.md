# genpark-multi-armed-bandit-ucb1-thompson-sampling-skill

[![GitHub stars](https://img.shields.io/github/stars/alphaparkinc/genpark-multi-armed-bandit-ucb1-thompson-sampling-skill?style=social)](https://github.com/alphaparkinc/genpark-multi-armed-bandit-ucb1-thompson-sampling-skill/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-0%20external-brightgreen.svg)](#)
[![Model Context Protocol](https://img.shields.io/badge/MCP-Standard%20Compatible-orange.svg)](#)

> Autonomous Agent Multi-Armed Bandit UCB1 & Thompson Sampling Exploration-Exploitation Optimizer

Part of the **GenPark Autonomous Dynamic Game Theory & Reinforcement Learning Architecture**.

## Architecture Overview

```mermaid
graph TD
    A[K-Arm Multi-Armed Bandit Action Space] --> B{Bandit Strategy: UCB1 / Thompson}
    B -->|UCB1| C[Calculate Upper Confidence Bound Mean + sqrt 2 ln t / Na]
    B -->|Thompson Sampling| D[Sample from Beta Conjugate Posterior Beta alpha, beta]
    C --> E[Select Arm with Maximum Metric]
    D --> E
    E --> F[Pull Selected Arm & Observe Stochastic Reward]
    F --> G[Update Empirical Counts & Reward Registers]
    G --> H[Logarithmic Cumulative Regret Minimization]
```

## Features

- **Pure Python Standard Library**: Zero external dependencies.
- **Production-Grade Design**: Type annotations, robust convergence loops, clean interfaces.
- **MCP Server Ready**: Built-in stdio Model Context Protocol (MCP) server for Claude / Cursor / Agent tool calling.
- **Benchmark Validated**: 100% verified test coverage in isolated sandbox environments.

## Quickstart

```bash
git clone https://github.com/alphaparkinc/genpark-multi-armed-bandit-ucb1-thompson-sampling-skill.git
cd genpark-multi-armed-bandit-ucb1-thompson-sampling-skill
python example_usage.py
```

## Model Context Protocol (MCP) Usage

```bash
python mcp_server.py
```

## License

MIT License. Designed for autonomous agentic workflows.
