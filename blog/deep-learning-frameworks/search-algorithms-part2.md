---
layout: post
title: "Multi-Agent Search & Game Theory"
permalink: /blog/deep-learning-frameworks/search-algorithms-part2.html
---

## Part 2 of 3: AI Search Fundamentals Series

**Author:** Akshay Raj Dhamija  
**Date:** October 4, 2025  
**Tags:** Multi-Agent Search, MiniMax, Alpha-Beta, Game Theory, AI  
**Series:** AI Search Fundamentals (Part 2 of 3)

---

> **📚 Series Navigation:**  
> [← Part 1: Mastering Search Algorithms](search-algorithms-part1.html)  
> **Part 2: Multi-Agent Search & Game Theory** (Current)  
> [Part 3: MDPs to Q-Learning →](search-algorithms-part3.html)

---

Welcome to Part 2 of our AI Search Fundamentals series! In [Part 1](search-algorithms-part1.html), we mastered single-agent search algorithms like DFS, BFS, and A-Star. Now we'll explore the fascinating world of **multi-agent search** where we must consider opponents, allies, and uncertainty.

**What you'll learn in this post:**
- How search changes when adversaries are involved
- MiniMax algorithm for perfect play in games
- Alpha-Beta pruning for computational efficiency
- Handling uncertainty with ExpectiMax
- Applications in game-playing AI (chess, pac-man, tic-tac-toe)

---

## Table of Contents

1. [Search Problems with Adversaries](#1-search-problems-with-adversaries)
2. [Tree Traversal: Single vs Multi-Agent](#2-tree-traversal-in-multi-agent-search-vs-single-agent-search)
3. [MiniMax Search](#3-minimax-search)
4. [Resource Limitations and Depth Trade-off](#4-resource-limitations-and-depth-trade-off)
5. [Alpha-Beta Search](#5-alpha-beta-search)
6. [Action Uncertainty](#6-action-uncertainty)
7. [ExpectiMax](#7-expectimax)
8. [Infinite Search Problems](#8-infinite-search-problems)
9. [Summary & Next Steps](#summary--next-steps)

---

## 1. Search Problems with Adversaries

While the search algorithms we discussed in Part 1 (like uninformed search, informed search and backtracking) were able to find an optimal path with a varied number of search node expansions, they make a very crucial assumption: **the future states and the cost associated with them are only impacted by the action taken by the agent** and not by either other independent agents or environmental factors outside the agent's control.

**Examples of Real-World Multi-Agent Scenarios:**

🎮 **Pac-Man Universe**: A game where the pac-man aims to collect food pellets in presence of ghosts that are trying to eat the pac-man itself

🚗 **Real World Navigation**: While navigating between two cities the path may be impacted by:
- Heavy snow storm (environmental factor outside agent's control)
- Heavy traffic (contributed by independent agents/other drivers)

♟️ **Board Games**: Chess, Go, tic-tac-toe where your opponent is actively trying to defeat you

### Types of Games

**Zero-Sum Games**: The goal of the agents are opposite to each other. The gain of one is the loss of the other, hence leading to the zero sum.
- **Examples**: Pac-man with ghosts, chess, tic-tac-toe
- **Characteristic**: One player's win is another's loss

**General Games**: Each player may have its own goal and while maximizing their own goals two players might end up cooperating, competing or even betraying each other.
- **Example**: Pac-man with multiple ghosts. Each ghost tries to maximize its own score by eating the pac-man but this may end up with two ghosts cooperating with each other to trap the pac-man
- **Characteristic**: Complex dynamics with cooperation and competition

---

## 2. Tree Traversal in Multi-Agent Search vs Single-Agent Search

The important distinction between the search algorithms without adversarial agents versus those with adversarial agents is that when selecting an action, the search algorithms with adversarial agents have to **look at a set of possible outcomes in the future** to minimize its traversal cost or maximize its action's utility.

### Two Key Problems

**1. Computational Expense**

If before taking a single action we have to analyze all possibilities in the future, then effectively we will have to solve the entire search tree for just selecting the first action to take. This is contrary to our aim of minimizing the number of nodes expanded to solve a search problem.

**Solution**: We use a concept similar to heuristics. We replace the overall cost $\(f(n)\)$ with an **evaluation function $\(E(s)\)$** that is representative of the utility of a state $\(s\)$. The evaluation function indicates how close we are to our goal. This utility function can be calculated for each state in the search tree independently. 

The amount of search tree that is expanded (or the amount of steps in the future that the agent can see before taking an action) can be controlled using a **hyperparameter** (depth limit).

**2. Different Goal**

Another important distinction is that:
- **Single agent search**: We aim to find the optimal **path** between the start and the goal state
- **Multi-agent search**: We aim to find the best **action** for a given state

---

## 3. MiniMax Search

MiniMax search can be used to traverse the search tree for selecting the actions taken by the primary agent in order to maximize its utilization while in presence of other adversaries and allies.

### How MiniMax Works

When making such a search tree it is assumed that:
1. Each action taken by the primary agent is followed by an action taken by each of the other agents
2. These agents may contain either **allies** (common goals) or **adversaries** (opposing goals)
3. The **allies** would take an action to **maximize** utilization of the primary agent
4. The **adversarial agents** would take action to **minimize** the utilization of the primary agent

MiniMax search calculates the utility of an action from a state by finding:
- The **minimum** possible utility provided by the adversaries
- The **maximum** possible utility provided by the allies

MiniMax algorithm provides utilities considering:
- **Worst case scenario** put forward by the adversaries
- **Best case scenario** put forward by the allies

---

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/MiniMax_Tree_Exp.png" alt="MiniMax Tree Example" style="max-width: 80%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 1:</strong> MiniMax tree example showing how values propagate from leaf nodes through alternating min and max layers to determine the optimal action.</p>
</div>

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/MiniMax.png" alt="MiniMax Algorithm" style="max-width: 90%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 2:</strong> MiniMax search algorithm pseudocode. The algorithm alternates between maximization (for allies) and minimization (for adversaries) to find the optimal action for the primary agent.</p>
</div>

---

## 4. Resource Limitations and Depth Trade-off

Since the action selection of the primary agent is based on the possible actions the other agents can take in the future, it can be said that **the more the primary agent can see in the future, the better action choices it can make**.

**Depth**: The number of tiers of the search tree or the future steps that an agent can consider are the **depth** of the search tree a primary agent can look into while deciding on which step to take.

### The Trade-off

- **Deeper depth** = Better decisions but more computation
- **Shallow depth** = Faster decisions but potentially suboptimal

**Trade-off**: The deeper the depth an agent can see, the more search nodes it has to expand in order to decide for a single step. Because of both resource constraint and time constraints, our goal is to get the most optimal play while considering the shallowest depth possible.

---

## 5. Alpha-Beta Search

Alpha-Beta search applies a technique of **meta-reasoning** to decide which computation inside the mini-max search is actually worth doing. Both alpha-beta search and mini-max search provide the **exact same result** but alpha-beta search is computationally less expensive.

### Core Idea

Avoid the expansion of unnecessary nodes in the search tree. This unnecessary expansion can be attributed to the repeated nested minimization and maximization operations in the mini-max search.

### Terminology

- **Alpha (α)**: Represents the best possible utility achieved by an ally (maximizer's best)
- **Beta (β)**: Represents the worst possible utility achieved by an adversary (minimizer's best)

### Pruning Logic

**Pruning**: It can be observed that in order to prune the search tree, the order in which actions are considered do play an important role. While they are outside the algorithm's control, there may be ways to smartly select this ordering based on the evaluation function.

**Complexity**: If we consider the best possible ordering of the actions we can get a time complexity of $\(O(b^{m/2})\)$. 

**Key Benefit**: This reduction in the time complexity allows us to run the mini-max search algorithm for **deeper depths** (looking more steps in the future) while having the same amount of resource constraints. This means we can **double the depth** we can search!

---

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/Alpha_Beta.png" alt="Alpha-Beta Pruning" style="max-width: 90%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 3:</strong> Alpha-Beta search explanation. Alpha-Beta search provides the same result as MiniMax but explores fewer nodes by pruning unnecessary branches. Alpha (α) is set by the maximization function and used by the minimization function to decide when to stop traversing the sub-tree.</p>
</div>

---

## 6. Action Uncertainty

Mini-Max search approach always assumed that:
- The adversarial agent would take the outcome that **harms the primary agent the most** (minimum of all outcomes)
- The allies would take the action that **benefits the primary agent the most** (maximum utility)

But this is **not always true** in reality!

### Example: Tic-Tac-Toe

**Playing against an expert**: When playing tic-tac-toe with an experienced adult it may be impossible to win. In the best case we would draw the game.

**Playing against a novice**: In case of a toddler who may make a mistake, it is still possible to win the game.

A mini-max search algorithm would always assume that it is playing against an expert even when it is actually playing against a novice!

### Sources of Uncertainty

Uncertainty in the abilities of the other agent is not the only way to look at action uncertainty. Action uncertainty can also be a result of **environmental factors**.

**Example - Robot Vacuum Cleaner**: When the robot takes the action "left" it is possible that it lands on a different surface (example: a rug) creating slippage in the tire rotation. This can affect its odometry, which means the new state it landed in after taking the left turn might not be 90 degrees from its previous position but rather slightly off.

These kind of environmental factors can be incorporated by considering them as **action uncertainties**.

---

## 7. ExpectiMax

ExpectiMax considers that there is a **probability** associated with each agent picking a specific action. Rather than considering that the agents will either take a minimizing or maximizing approach, expectimax considers the **weighted average value** each agent could provide and then tries to increase the utilization of the primary agent by taking a max of the expected values.

### How ExpectiMax Works

For each agent, rather than calculating the minimum value (as in MiniMax), it tries to calculate the **expected value** based on probabilities.

**Key Difference from MiniMax**:
- **MiniMax**: Assumes worst-case (minimum) from adversary
- **ExpectiMax**: Calculates expected value based on probabilities

### Pruning Limitation

⚠️ While in case of MiniMax we were able to perform tree pruning (since only either the minimum or the maximum node contributed to the actual value), for expectimax it is **not possible to prune the tree** because each node individually contributes towards the value of an action for the primary agent (unless the probability of an action is zero, which is unlikely).

### Generalization: Mixed Trees

The concept of expectimax can also be generalized towards **mixed trees** which contain:
- **Uncertainty nodes** (chance nodes)
- **Minimizer nodes** (adversaries)
- **Maximizer nodes** (allies/primary agent)

**Example**: The search tree for a game of **backgammon** where we have:
- An opponent who is trying to minimize the primary agent
- A dice which is a chance node (uncertainty)

---

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/ExpectiMax.png" alt="ExpectiMax Algorithm" style="max-width: 90%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 4:</strong> ExpectiMax search algorithm. Instead of assuming adversaries will always take the worst action (as in MiniMax), ExpectiMax computes expected values based on probabilities of different actions.</p>
</div>

---

## 8. Infinite Search Problems

In single-agent search problems (Part 1) we focused on traversing **finite search trees** from the start node to the goal node in the most optimal way. This traversal used to provide us a set of actions to be performed in a sequence from the start state that would guarantee us to reach the goal state. This guarantee was because our problem and the environment were **deterministic** and the search tree **finite**.

### The Reality: Most Problems are Infinite

In multi-agent problems, rather than finding the shortest path between two states, we aimed at finding the optimal action given a state. Along with that we took into account some level of **stochasticity** in the actions of other agents when we used ExpectiMax. 

The ExpectiMax solution is capable of being applied to an **infinite search tree** which is the case for most real world search problems.

**Example**: An automated taxi that is aimed at completing as many trips as possible with a pipeline of passengers. The taxi never "reaches a goal" - it continues operating indefinitely.

### The Problem

For such multi-agent search approaches we have to look a few steps in the future to make the decision on the current action, which can be very expensive. This expensive compute can delay the response of an agent especially in a real world problem where the agent needs to run on the edge.

### The Solution: Policies

**Next Evolution**: From this section onward, we move from finding the optimal action in the current state to the problem of finding the optimal action in **every given state**. This map of states to actions is referred to as a **policy π**.

Alongside, now we consider the actions of our primary agent to be **stochastic** (i.e. there is some amount of randomness in the actions of the agent).

💡 **This is where we transition from game-playing to learning!** In [Part 3](search-algorithms-part3.html), we'll explore how to learn these policies using Markov Decision Processes and Reinforcement Learning.

---

## Summary & Next Steps

In this second part of our series, we've mastered multi-agent search and game theory!

### Key Takeaways

✅ **Multi-Agent Search Fundamentals:**
- Distinguished between single-agent and multi-agent problems
- Understood zero-sum vs general games
- Learned evaluation functions for intermediate states

✅ **MiniMax Algorithm:**
- Alternates between maximization (allies) and minimization (adversaries)
- Provides optimal play assuming perfect opponents
- Considers worst-case scenarios

✅ **Alpha-Beta Pruning:**
- Achieves same results as MiniMax with less computation
- Can double the search depth with same resources
- Order of action evaluation matters for pruning effectiveness

✅ **Handling Uncertainty:**
- ExpectiMax for probabilistic opponents
- Mixed trees with chance nodes
- Real-world applications with imperfect information

✅ **Infinite Horizons:**
- Transition from finite to infinite problems
- Introduction to policies (state → action mappings)
- Bridge to reinforcement learning

### What's Next?

You've now mastered search algorithms from simple pathfinding to complex game-playing AI! In **Part 3**, we'll take the final leap into **reinforcement learning**:

🤖 **Markov Decision Processes & Reinforcement Learning**
- Markov Decision Processes (MDPs)
- Bellman Equations for optimal decision making
- Value Iteration and Policy Iteration
- Model-Free Learning with Q-Learning
- Temporal Difference Learning
- Exploration vs Exploitation strategies

**👉 [Continue to Part 3: MDPs to Q-Learning →](search-algorithms-part3.html)**

### Resources

- **Code Examples**: All implementations available in the [pac-man repository](https://github.com/akshay-raj-dhamija/pac-man)
- **Course**: [UC Berkeley CS-188](https://inst.eecs.berkeley.edu/~cs188/su21/)
- **Previous**: [Part 1: Search Algorithms](search-algorithms-part1.html)
- **Next**: [Part 3: Reinforcement Learning](search-algorithms-part3.html)
- **Complete Guide**: [All Three Parts Combined](search-trees-to-reinforcement-learning.html)

---

**[← Part 1: Search Algorithms](search-algorithms-part1.html)** &nbsp;&nbsp;|&nbsp;&nbsp; **[Back to Blog](../index.html)** &nbsp;&nbsp;|&nbsp;&nbsp; **[Part 3: MDPs & Q-Learning →](search-algorithms-part3.html)**

