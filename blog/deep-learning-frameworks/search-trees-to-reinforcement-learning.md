---
layout: post
title: "Search Trees to Reinforcement Learning via MDP"
permalink: /blog/deep-learning-frameworks/search-trees-to-reinforcement-learning.html
---

**Author:** Akshay Raj Dhamija  
**Date:** October 4, 2025  
**Tags:** Reinforcement Learning, MDP, AI, Search Algorithms, Q-Learning

---

This document summarizes the concepts of search trees, MDPs and reinforcement learning learned from the undergraduate course CS-188 provided by University of California, Berkeley ([CS-188](https://inst.eecs.berkeley.edu/~cs188/su21/)). All the code used for experiments in this document is available at [https://github.com/akshay-raj-dhamija/pac-man](https://github.com/akshay-raj-dhamija/pac-man).

## Overview

This comprehensive guide covers:
- How to navigate search trees optimally
- Uninformed and informed search
- Constraint satisfaction problems
- Tree traversal with single agent vs multi-agent
- Infinite search trees
- Markov Decision Processes (MDPs) and how to solve them using value iteration and policy iteration
- Reinforcement learning with temporal difference learning and Q-learning

---

## Table of Contents

1. [Search Problems & Their Terminology](#1-search-problems--their-terminology)
2. [Uninformed Search](#2-uninformed-search)
3. [Informed Search](#3-informed-search)
4. [Constraint Satisfaction Problems](#4-constraint-satisfaction-problems)
5. [Search Problems with Adversaries](#5-search-problems-with-adversaries)
6. [Action Uncertainty](#6-action-uncertainty)
7. [Infinite Search Problems](#7-infinite-search-problems)
8. [Markov Decision Processes](#8-markov-decision-processes)
9. [Bellman Equation](#9-bellman-equation)
10. [Solving the Bellman Equation](#10-solving-the-bellman-equation)
11. [Evaluating Policies](#11-evaluating-policies)
12. [Reinforcement Learning](#12-reinforcement-learning)
13. [Model Based Learning](#13-model-based-learning)
14. [Model Free Learning](#14-model-free-learning)
15. [Passive Reinforcement Learning](#15-passive-reinforcement-learning)
16. [Active Reinforcement Learning](#16-active-reinforcement-learning)
17. [Exploration vs Exploitation](#17-exploration-vs-exploitation)
18. [Conclusion](#18-conclusion)

---

## 1. Search Problems & Their Terminology

Many real world problems can be summarized as search problems in computer science. While many examples can be taken from games like chess, sudoku, Go or pac-man, some real world problems like finding the shortest path in navigation between cities or creating a path for a robot vacuum cleaner between two rooms can also be considered as search problems.

### Understanding Search Problems

One way to understand what search problems are, is by considering the brute force methods to solve a search problem. For example, let's consider the problem of finding the shortest path. This problem may be solved using brute-force i.e. traveling using all the possible paths and then drawing the conclusion of which one is the shortest.

### Key Terminology

**States**: For every search problem, the possible values a problem can take are known as states. In the example of navigation between cities, this could be the cities themselves or in case of a robot vacuum cleaner this may be the coordinates of every position in the environment with respect to the robot's home base.

**Start State and Goal State**: Amongst all states in a search problem, two states are of utmost importance. While search problems would generally only have one start state they may have one or more goal states. For example, if we have to address a navigation problem where we want to visit a set of cities, each combination of these cities irrespective of the order in which they are visited are goal states.

**Actions and Successor Function**: From every state in the search problem an agent can take a set of actions to land in its successor states. These states and the corresponding actions are provided by the successor function. For example, a pac-man agent can have at most four successor states from any given state. These successor states are obtained on taking the action left, right, up or down.

**Cost**: Generally for every action taken from a state there is a cost associated with it. For a navigation problem between cities this cost can be interpreted as the distance between two neighboring cities. For games like pac-man the cost between all adjoining states is constant (1 in the game we considered).

### Search Trees

For search problems, these kind of situations are generally visualized with the help of search trees. Search trees can be considered as a map of every brute force solution possible and more importantly any search problem can be represented as search trees. The aim in solving such search problems is to achieve the solution while doing minimum expansion of the nodes in a search tree.

For example, if the problem was to find the shortest path between two cities A and Z where there were 100 possible paths between A and Z, our goal is to find the shortest path without having to calculate the exact cost of each of the 100 possible paths. The algorithms used to solve such search problems are often referred to as **search algorithms**.

## 2. Uninformed Search

While realistically we may have some prior information about our goal state, for example the direction of city Z with respect to city A, we may not always have this information available to us. One such real life scenario may be with a robot vacuum cleaner that is yet to build a map of its environment. When creating its map for the first time the robot may not have any idea about where the walls in the house are or may not even have any specific goal like cleaning a particular room rather more generalized and broad goal like explore the environment around itself.

The following search algorithms can be used to perform uninformed search in a search tree. These algorithms are also important because more sophisticated algorithms like the ones performing informed search are built using these uninformed search algorithms.

### 2.1 Depth First Search (DFS)

In depth first search we pick one of the various possible successors of a tree node and keep exploring the path leading from it. Whenever a node has multiple possible paths (example: a city A has direct paths leading to B, C and D) we pick one of the cities (example: B) and then continue exploring all of its successors. We do not explore the cities C and D until all possible paths from city B have been explored.

**Limitations**: 
- If the city B was directly connected to our destination city Z we may not ever explore the paths leading from cities C and D even though they may have possibly a shorter path leading to Z. This can be a possible drawback of DFS that though it may provide us a solution, that solution may not be optimal.
- In certain cases of cyclic search trees, DFS may not be able to provide us a solution at all.

**Summary**: DFS is neither complete nor optimal.

---

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/DFS.png" alt="Depth First Search on Pac-Man" style="max-width: 80%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 1:</strong> Depth First Search (DFS) - The brightness of the red color indicates how early during the search that the pac-man visited this position. The pac-man started at the top right state and the food pellet was at the bottom left.</p>
</div>

---

### 2.2 Breadth First Search (BFS)

A counterpart of depth first search can be the breadth first search (BFS). While in DFS we aim at exploring all possible paths leading from a node of the search tree before expanding its siblings (i.e. nodes that were successors of its parent node), in BFS we simultaneously expand each node and its siblings.

**Characteristics**:
- While like DFS, BFS will be unable to provide us the least cost path, it always provides us the path through the least number of nodes.
- If the problem was to find the path from city A to Z while passing through the least number of cities, BFS will be able to provide us the optimal solution, but this solution would not be the shortest path.

**Summary**: BFS is complete but it is optimal only in the special case of constant cost associated with each action.

---

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/BFS.png" alt="Breadth First Search on Pac-Man" style="max-width: 80%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 2:</strong> Breadth First Search (BFS) - BFS simultaneously expands each node and its siblings, resulting in a different exploration pattern compared to DFS.</p>
</div>

---

### 2.3 Performance Comparison

#### Infinite Search Trees

While DFS can be easily used to navigate a search tree, when applied to cyclic search trees it can lead us into an infinite processing loop. For example, if we reconsider the above example of cities, let's say city D had another path leading to city A, in such a case if we ever start expanding the successors of D which includes the city A we will be re-expanding the root of this search tree.

**Solution**: The only possible solution for such a problem is to keep a record of all the visited nodes (i.e. nodes whose successors have been visited). Then before expanding any node in the search tree we simply reference this record of visited nodes and if that node had been expanded earlier we do not revisit it, eventually converting the infinite search tree into a finite search tree. In Python, this book keeping of visited nodes can be done using the `set` data structure.

Unlike DFS, BFS can still navigate an infinite search tree when the goal is to find a path rather than performing backtracking. However, even in such scenarios BFS may highly benefit from the book keeping process.

#### Time Complexity

If the problem was to explore the entire search tree (i.e. a problem like backtracking), both DFS and BFS will have the same time complexity since they will be expanding all the nodes in the search tree (considering that the tree is not infinite or cyclic). This time complexity will be \(b^m\), where \(b\) is the average number of successors for each node and \(m\) is the total number of levels in the tree.

#### Space Complexity

While the time complexity of both DFS and BFS is the same, they have vastly different space complexity:
- **DFS**: \(O(bm)\)
- **BFS**: \(O(b^m)\)

This space complexity difference is a big reason for using DFS over BFS when implemented on edge devices like robot vacuum cleaners.

**DFS Space Complexity Explanation**: At the first search tier, DFS will see \(b\) possible branches, from which it will be expanding one branch, while keeping \((b-1)\) branches in memory. At tier 2 it will have \((b-1)\) branches for the node at tier 2, while also holding \((b-1)\) from the first tier. Assuming \(b_1 = b_2\) we have \(2 \times (b - 1)\), generalizing this further to the \(m\)th level we get \(O(mb)\) as the space complexity.

**BFS Space Complexity Explanation**: At the first search tier, BFS will see \(b\) possible branches. So at the 1st tier BFS will have \(b\) nodes in memory. At tier 2 it expands each of the \(b\) states/nodes into their respective \(b\) successors each. This means at tier 2 it has \(b^2\) nodes in memory. Generalizing this further to the \(m\)th tier we get \(O(b^m)\) as the space complexity.

### 2.4 Uniform Cost Search (UCS)

While both DFS and BFS are able to find a path from a start state to the goal state, they are unable to provide us with the shortest possible path. Mostly because they are unable to take the cost associated with the state transitions into account. This problem is addressed using **Uniform Cost Search (UCS)** which is a cost sensitive search method as compared to DFS and BFS which are insensitive to the cost associated with each action.

**How it works**: UCS may also be thought of as a method that combines both DFS and BFS based on cost. The UCS performs DFS as long as continuing DFS provides it the least cost path; if not it switches to BFS to expand the state where it might have the lowest cost path.

**Optimality**: UCS is able to provide the optimal path from the start to the goal state.

**Complexity**: The space and time complexity of UCS is similar to those of BFS. If \(b\) is the number of branches from a state and let's assume the goal state to have a minimum cost \(C^*\) associated with it, with the minimum cost from any pair of states to be \(\epsilon\), then both the space and time complexity of UCS can be given as \(O(b^{C^*/\epsilon})\).

#### When the transition cost is uniform/constant

If UCS is applied to a problem where the transition cost between two states is constant (e.g. the pac-man game we are considering in this document) it will behave exactly like breadth first search. It may also be said that if the cost of actions is uniform/constant, BFS provides the optimal solution.

---

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/UCS.png" alt="Uniform Cost Search on Pac-Man" style="max-width: 80%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 3:</strong> Uniform Cost Search (UCS) - UCS is cost-sensitive and provides the optimal path. In cases with uniform transition costs, it behaves identically to BFS.</p>
</div>

---

### 2.5 Data Structures Used for Search Trees

While some of the explanation above might sound a bit confusing and the differences between each of the approaches might sound big, when understood from the data structures being used the similarity between these methods become obvious. In fact, for the implementation of these methods we use a single implementation with just a different data structure:

- **DFS**: Uses the **stack** or **LIFO** data structure (in Python: a simple list with the last element being used at every iteration)
- **BFS**: Uses the **queues** or **FIFO** data structure (in Python: a simple list with the first element of the list being used in every iteration. Due to implementation of lists in Python, this operation is generally inefficient and it is recommended to use `deque` or double ended queues)
- **UCS**: Uses **min-heaps** or **priority queues** (in Python: `heapq`)

## 3. Informed Search

In the previous section we did a deep dive into uninformed search, which are the techniques that either do not use any information about the cost associated with an action (like in DFS and BFS) or aim to minimize the actual overall cost associated with all actions (like in UCS).

While such uninformed search approaches have a cost associated with each action from a given state, this cost indicates the cost of transition between two states. When analyzed cumulatively it only tells us the cost to reach the current state from the start state but it does not tell anything about how this cost relates to the actual goal we aim to achieve.

**Terminology**:
- **Backward cost \(g(n)\)**: The cost used by uninformed search approaches, where \(n\) is the action \(a\) taken from a given state \(s\)
- **Forward cost or Heuristic \(h(n)\)**: A cost which is much more representative of the cost that the algorithm might incur from the current state to reach the goal state

### 3.1 Heuristics

Heuristics are very specific to the problem definition being considered and are often the deciding factor of how well an algorithm might work. A heuristic function can be something very simple:
- For a navigation problem: the Euclidean distance between a state and the goal state
- For games like eight puzzle: the amount of tiles out of place in the given state when compared to the goal state

**Example - Pac-Man**: In a game of pac-man we can use either the Euclidean distance or the Manhattan distance between a pac-man and a food pellet. But when we consider the fact that the pac-man cannot move diagonally (only left, right, up or down) we reach the conclusion that Manhattan distance might represent the cost to be incurred in reaching the goal much better than the Euclidean distance.

#### Admissible Heuristics

While one can use any of the many heuristic options available to them, some heuristics should not be used because they may not be representative of the cost needed to reach the goal. Such heuristics are called **inadmissible heuristics**.

**Example of Inadmissible Heuristic**: Let's reconsider the game of pac-man but with a modification that the pac-man is now allowed to move diagonally and is no longer restricted to horizontal or vertical movements. In such a case if we choose the heuristic as Manhattan distance, then when the goal is available one step at the diagonal, the heuristic will indicate it to be available at two pac-man steps (one horizontal and one vertical). This means that the pac-man will think the goal is further away than it actually is, hence leading such a heuristic to be a **pessimistic heuristic**.

**Definition**: An admissible heuristic will always satisfy the property:

$$0 \leq h(n) \leq h^*(n)$$

where \(h^*(n)\) is the true cost to the nearest goal.

This also means that an admissible heuristic will be optimistic of the outcome. However, it should also be noted that over optimism would be fatal. Consider \(h(n) = 0\) which while satisfying the above equation would render the heuristic useless.

### 3.2 Greedy Search

A search performed purely based on a heuristic is called the **greedy search**. In this kind of search technique we are only trying to minimize the forward cost while not considering backward cost or the cost needed to reach a given state.

$$f(n) = g(n) + h(n) \mid g(n) = 0$$
$$f(n) = h(n)$$

---

<div style="display: flex; justify-content: space-around; align-items: flex-start; gap: 20px; margin: 30px 0; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <h4 style="margin-bottom: 10px;">Greedy Search with Euclidean Distance</h4>
    <img src="Images/GD_Euc.png" alt="Greedy Search L2" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  </div>
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <h4 style="margin-bottom: 10px;">Greedy Search with Manhattan Distance</h4>
    <img src="Images/GS_MHD.png" alt="Greedy Search L1" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  </div>
</div>

<p style="text-align: center; font-style: italic; color: #666; margin-top: 10px;"><strong>Figure 4:</strong> Greedy search using different heuristics. While greedy search provides solutions quickly with fewer node expansions, these solutions are not always optimal.</p>

---

### 3.3 A-Star Search

A-Star search, which is also considered to be a generalization of Dijkstra's algorithm, was originally used to solve the navigation problem in Shakey the robot. A-Star search simply combines the idea of greedy search and uniform cost search by considering the total cost to be a summation of the forward \(h(n)\) and backward costs \(g(n)\):

$$f(n) = g(n) + h(n)$$

**Complexity**: The worst case time and space complexity of both A-star search and greedy search is the same as that for UCS, i.e. \(O(b^{C^*/\epsilon})\).

**Optimality**: A-Star search provides the optimal solution provided the heuristic being used is admissible.

**Implementation**: Both A-star search and greedy search can be implemented in the same way as uniform cost search with the min-heap data structure, with the only difference being the cost calculation associated with each state action pair.

---

<div style="display: flex; justify-content: space-around; align-items: flex-start; gap: 20px; margin: 30px 0; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <h4 style="margin-bottom: 10px;">A-Star with Euclidean Distance</h4>
    <img src="Images/Astar_Euc.png" alt="A-Star L2" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  </div>
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <h4 style="margin-bottom: 10px;">A-Star with Manhattan Distance</h4>
    <img src="Images/Astar_MHD.png" alt="A-Star L1" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  </div>
</div>

<p style="text-align: center; font-style: italic; color: #666; margin-top: 10px;"><strong>Figure 5:</strong> A-Star search using different heuristics. A-Star provides optimal solutions while exploring fewer nodes than uninformed search methods. Manhattan distance outperforms Euclidean distance for pac-man due to the grid-based movement.</p>

---

## 4. Constraint Satisfaction Problems

Many real world games like sudoku, rather than being simple tree traversal or graph search problems, are actually **constraint satisfaction problems (CSP)**. Here the goal rather than simply being a special state is actually any state that satisfies a number of predefined constraints.

**Examples**:
- **Sudoku**: Constraints could be:
  - Each cell contains a digit
  - No digits are repeated in each row, each column and each sub-grid of size 3×3
- **Graph coloring problem**
- **N-queen problem**

**Solution Method**: Most constraint satisfaction problems can be solved using **backtracking**, which is simply an uninformed search (usually depth first search) in which at each state a set of conditions is checked and if any of the conditions are violated the sub-tree is not expanded further.

## 5. Search Problems with Adversaries

While the search algorithms discussed above (like uninformed search, informed search and backtracking) were able to find an optimal path with a varied number of search node expansions, they make a very crucial assumption: the future states and the cost associated with them are only impacted by the action taken by the agent and not by either other independent agents or environmental factors outside the agent's control.

**Examples**:
- **Pac-Man Universe**: A game where the pac-man aims to collect food pellets in presence of ghosts that are trying to eat the pac-man itself
- **Real World Navigation**: While navigating between two cities the path may be impacted by a heavy snow storm (environmental factor) or by heavy traffic (other independent agents/drivers)

### Types of Games

**Zero-Sum Games**: The goal of the agents are opposite to each other. The gain of one is the loss of the other, hence leading to the zero sum.
- Examples: Pac-man game with presence of a ghost, chess, tic-tac-toe

**General Games**: Each player may have its own goal and while maximizing their own goals two players might end up cooperating, competing or even betraying each other.
- Example: A game of pac-man with multiple ghosts. Each ghost tries to maximize its own score by eating the pac-man but this may end up with two ghosts cooperating with each other to trap the pac-man.

### 5.1 Tree Traversal in Multi-Agent Search vs Single-Agent Search

The important distinction between the search algorithms without adversarial agents versus those with adversarial agents is that when selecting an action, the search algorithms with adversarial agents have to look at a set of possible outcomes in the future to minimize its traversal cost or maximize its action's utility.

**Two Problems**:

1. **Computational Expense**: If before taking a single action we have to analyze all possibilities in the future, then effectively we will have to solve the entire search tree for just selecting the first action to take. This is contrary to our aim of minimizing the number of nodes expanded to solve a search problem.

   **Solution**: We use a concept similar to heuristics. We replace the overall cost \(f(n)\) with an **evaluation function \(E(s)\)** that is representative of the utility of a state \(s\), i.e. the evaluation function indicates how close we are to our goal. This utility function can be calculated for each state in the search tree independently. The amount of search tree that is expanded (or the amount of steps in the future that the agent can see before taking an action) can be controlled using a hyperparameter.

2. **Different Goal**: Another important distinction is that while in single agent search algorithms we aim to find the optimal path between the start and the goal state, in multi-agent search we aim to find the best action for a given state.

### 5.2 MiniMax Search

MiniMax search can be used to traverse the search tree for selecting the actions taken by the primary agent in order to maximize its utilization while in presence of other adversaries and allies.

**How it works**:
- When making such a search tree it is assumed that each action taken by the primary agent is followed by an action taken by each of the other agents
- These agents may contain either **allies** which may have common goals as the agent or **adversaries** which may have opposing goals
- The allies generally would take an action to **maximize** utilization of the primary agent
- The adversarial agents would take action to **minimize** the utilization of the primary agent

MiniMax search takes this observation into account and calculates the utility of an action from a state by finding the minimum possible utility provided by the adversaries and the maximum possible utility provided by the allies. MiniMax algorithm provides utilities considering the worst case scenario put forward by the adversaries and the best case scenario put forward by the allies.

---

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/MiniMax_Tree_Exp.png" alt="MiniMax Tree Example" style="max-width: 80%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 6a:</strong> MiniMax tree example showing how values propagate from leaf nodes through alternating min and max layers to determine the optimal action.</p>
</div>

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/MiniMax.png" alt="MiniMax Algorithm" style="max-width: 90%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 6b:</strong> MiniMax search algorithm pseudocode. The algorithm alternates between maximization (for allies) and minimization (for adversaries) to find the optimal action for the primary agent.</p>
</div>

---

### 5.3 Resource Limitations and Depth Trade-off

Since the action selection of the primary agent is based on the possible actions the other agents can take in the future, it can be said that the more the primary agent can see in the future, the better action choices it can make.

**Depth**: The number of tiers of the search tree or the future steps that an agent can consider are the depth of the search tree a primary agent can look into while deciding on which step to take.

**Trade-off**: The deeper the depth an agent can see, the more search nodes it has to expand in order to decide for a single step. Because of both resource constraint and time constraints, our goal is to get the most optimal play while considering the shallowest depth possible.

### 5.4 Alpha-Beta Search

Alpha-Beta search applies a technique of meta-reasoning to decide which computation inside the mini-max search is actually worth doing. Both alpha-beta search and mini-max search provide the exact same result but alpha-beta search is computationally less expensive.

**Core Idea**: Avoid the expansion of unnecessary nodes in the search tree. This unnecessary expansion can be attributed to the repeated nested minimization and maximization operations in the mini-max search.

**Terminology**:
- **Alpha (α)**: Represents the best possible utility achieved by an ally
- **Beta (β)**: Represents the worst possible utility achieved by an adversary

**Pruning**: It can be observed that in order to prune the search tree, the order in which actions are considered do play an important role. While they are outside the algorithm's control, there may be ways to smartly select this ordering based on the evaluation function.

**Complexity**: If we consider the best possible ordering of the actions we can get a time complexity of \(O(b^{m/2})\). This reduction in the time complexity allows us to run the mini-max search algorithm for deeper depths (looking more steps in the future) while having the same amount of resource constraints.

---

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/Alpha_Beta.png" alt="Alpha-Beta Pruning" style="max-width: 90%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 7:</strong> Alpha-Beta search explanation. Alpha-Beta search provides the same result as MiniMax but explores fewer nodes by pruning unnecessary branches. Alpha (α) is set by the maximization function and used by the minimization function to decide when to stop traversing the sub-tree.</p>
</div>

---

## 6. Action Uncertainty

Mini-Max search approach always assumed that the adversarial agent would be taking the outcome that harms the primary agent the most (hence it assumes the minimum of all possible outcomes). Similarly for the allies it assumes the action selected would be the one that benefits the primary agent the most. But this is not always true.

**Example**: When playing a game of tic-tac-toe with an experienced adult it may be impossible to win (in the best case we would draw the game), but in case of a toddler who may make a mistake it is still possible to win the game. A mini-max search algorithm would always assume that it is playing against an adult even when it is actually playing against a toddler.

Uncertainty in the abilities of the other agent is not the only way to look at action uncertainty. Action uncertainty can also be a result of **environmental factors**.

**Example - Robot Vacuum Cleaner**: When the robot takes the action "left" it is possible that it lands on a different surface (example: a rug) creating slippage in the tire rotation. This can affect its odometry, which means the new state it landed in after taking the left turn might not be 90 degrees from its previous position but rather slightly off.

These kind of environmental factors can be incorporated by considering them as action uncertainties.

### 6.1 ExpectiMax

ExpectiMax considers that there is a probability associated with each agent picking a specific action. Rather than considering that the agents will either take a minimizing or maximizing approach, expectimax considers the weighted average value each agent could provide and then tries to increase the utilization of the primary agent by taking a max of the expected values.

For each agent, rather than calculating the minimum value, it tries to calculate the expected value. ExpectiMax assumes the probability of each action to be equally likely.

**Pruning Limitation**: While in case of MiniMax we were able to perform tree pruning (since only either the minimum or the maximum node contributed to the actual value), for expectimax it is not possible to prune the tree because each node individually contributes towards the value of an action for the primary agent, unless the probability of an action might be zero (which is unlikely).

**Generalization**: The concept of expectimax can also be generalized towards **mixed trees** which contain uncertainty nodes, minimizer nodes and maximizer nodes because of different agents involved. Example: the search tree for a game of backgammon where we have an opponent who is trying to minimize the primary agent and also a dice which is a chance node.

## 7. Infinite Search Problems

In single-agent search problems we had focused on traversing finite search trees from the start node to the goal node in the most optimal way (i.e. while incurring the shortest cost). This traversal used to provide us a set of actions to be performed in a sequence from the start state that would guarantee us to reach the goal state. This guarantee was because our problem and the environment were deterministic and the search tree finite.

In the multi-agent problems, rather than finding the shortest path between two states, we aimed at finding the optimal action given a state. Along with that we took into account some level of stochasticity in the actions of other agents when we used ExpectiMax. The ExpectiMax solution was also capable of being applied to an infinite search tree which is the case for most real world search problems.

**Example**: An automated taxi that is aimed at completing as many trips as possible with a pipeline of passengers.

**Problem**: For such multi-agent search approaches we have to look a few steps in the future to make the decision on the current action, which can be very expensive. This expensive compute can delay the response of an agent especially in a real world problem where the agent needs to run on the edge.

**Solution**: From this section onward we move from finding the optimal action in the current state to the problem of finding the optimal action in every given state. This map of states to actions is referred to as a **policy π**. Alongside, now we consider the actions of our primary agent to be stochastic (i.e. there is some amount of randomness in the actions of the agent).

## 8. Markov Decision Processes

Markov Decision Processes (MDP's) is one of the solutions to creating such policy maps. The goal here is to create a map of states to actions for a given problem so that during execution such a map can be referred to get the optimal action for a given state.

### MDP Terminology

1. **Set of states \(S\)**
2. **Start state \(s_0\)**
3. **Set of actions \(A\)**
4. **Transitions \(P(s'|s, a)\) or \(T(s, a, s')\)**: Since the actions in an MDP are stochastic, the result of a given action at a given state is not always the same. The probability that a given combination of state \(s\) and action \(a\) will yield state \(s'\) is given by \(P(s'|s, a)\)
5. **Q State**: Q state can be best understood as an abstraction of the above transition probability. An alternate explanation can be seen from the multi-agent search tree. In the search tree when the primary agent has committed to an action \(a\) but the other agents/adversaries haven't, this state is the Q state, where the next state the agent lands in is out of the primary agent's control but rather dependent on some level of stochasticity indicated by the transition function.
6. **Reward \(R(s, a, s')\)**: This is the reward associated with each state
7. **Discounting factor \(\gamma\)**: Discounting factor is important to ensure that the rewards seen earlier are better than the rewards seen much later in an agent's lifetime. The discounting factor is \(0 \leq \gamma < 1\). The discounted reward \(r_t\) at the time \(t\) would be given by \(r_t = \gamma^t \times R_t\) where \(t \geq 0\). Another use of discounting is it can allow us to consider infinite search trees as finite beyond a point because if \(0 < \gamma < 1\), then \(\gamma^{\infty} \approx 0\) making the rewards at the bottom of an infinite search tree immaterial.
8. **Utility \(U\)**: It is the sum of discounted rewards that the agent can earn in the future from a given state. This provides the utility \(U\) of any given future state as:

$$U_s = \sum_{t=0}^{\infty} \gamma^t R_t = \frac{1}{1-\gamma} \sum_{t=0}^{\infty} R_t = \frac{R_{max}}{1-\gamma}$$

9. **Value \(V^*(s)\)**: The maximum utility among all possible actions an agent can take given it is in state \(s\). In other words this can be said to be the utility of a state \(s\) assuming that the agent always takes the optimal action thereafter.

10. **Q Value \(Q^*(s, a)\)**: Expected utility given that the current action taken from state \(s\) is \(a\) and following this action the agent performs optimally. The major difference between Q value and the utility \(U\) defined above is that while the utility depicts the sum of discounted rewards in the future, the Q value takes the transition probabilities associated with each state into account, which is why we are mostly only concerned with the Q value and not the utility \(U\). The difference between V value and Q value can also be interpreted from a multi-agent search tree. While the V-value is associated with the utility of the primary agent, the Q-value is associated with the utility of the other agents whose actions are beyond the primary agent's control.

## 9. Bellman Equation

The Bellman equation is used for optimization problems using dynamic programming. It can be applied to solve for MDP's by simply modeling the assumption that the agent takes the correct action and then onwards continues to be optimal.

For the first assumption of the agent taking the correct action we can define \(V^*(s)\) as:

$$V^*(s) = \max_{a \in A} Q^*(s, a) \tag{1}$$

Now, \(Q^*(s, a)\) indicates the average utility of the agent given action \(a\) at state \(s\). The averaging component is because of the transition function. Here the agent is no longer in control; rather the action is being controlled by a transition function and a certain level of stochasticity. Since we had encountered a similar situation in an ExpectiMax tree, let's consider how we would calculate \(Q^*(s, a)\) for an ExpectiMax tree.

In case of ExpectiMax:

$$Q^*(s, a) = \sum_{s' \in S} T(s, a, s') \times V^*(s') \tag{2}$$

Combining the ExpectiMax idea with the reward seen at the current state and the discounting of the future reward, we get:

$$Q^*(s, a) = \sum_{s' \in S} T(s, a, s') \times [R(s, a, s') + \gamma V^*(s')] \tag{3}$$

Replacing \(Q^*(s, a)\) in equation 1 we get the Bellman equation:

$$V^*(s) = \max_{a \in A} \sum_{s' \in S} T(s, a, s') \times [R(s, a, s') + \gamma V^*(s')] \tag{4}$$

The time complexity of a Bellman equation is \(O(S^2 A)\) where \(S\) is the number of states and \(A\) is the number of possible actions.

## 10. Solving the Bellman Equation

### Solve using ExpectiMax

One easy way to solve the Bellman equation is to consider MDPs as **finite horizon MDPs**. This means that we have a pre-decided depth of the search tree and only consider rewards till that depth in order to select the optimal action. This is very similar to the mini-max search or the expectimax search.

**Problem**: The selected action is only optimal up to the considered depth but might not be optimal to reach the end goal.

### Value Iteration

In order to solve this problem and actually consider the MDPs with their true horizon we solve the Bellman equation using either **value iteration** or **policy iteration**. Because the Bellman equation itself is recursive, if we try and solve it we will only be able to get a solution when we reach the leaf nodes in the search tree and then calculate the utilities bottom up.

Rather, to solve this problem we use the concept of discounting and the fact that for a very deep search tree the discounted value of rewards at the last tier would be so small that it would not impact the overall utility at the topmost node, hence reaching convergence.

$$V_{k+1}(s) = \max_{a \in A} \sum_{s' \in S} T(s, a, s') \times [R(s, a, s') + \gamma V_k(s')]$$

$$V_{k+1} - V_k = \gamma^k \times \max_{i \in k} (R_i)$$

$$\text{if } k \approx \infty, \gamma^k \approx 0, V_{k+1} - V_k \approx 0$$

While in the above equation we only consider the case of \(k \approx \infty\), we do not go to infinite optimization steps. In practice when \(k\) becomes sufficiently large, the numbers become too small to be accommodated in the floating point precision and hence giving the illusion of \(k \approx \infty\).

### Policy Iteration

The solution of an MDP is actually not the values or utilities calculated at each state, rather it is the corresponding action to be taken at those states. When we performed the value iteration, we notice that while reaching the convergence for values may take \(n\) iterations, the only change after \(k\) iterations is in the values corresponding to these iterations and not the optimal action. Since \(k \ll n\), it motivates the concept of **policy iteration** where rather than trying to converge the values we try and converge the policy \(\pi\).

$$\pi_{i+1}(s) = \arg\max_{a \in A} \sum_{s' \in S} T(s, a, s') \times [R(s, a, s') + \gamma V^{\pi_i}(s')]$$

### Mixed Methods

Another advantage of using policy iteration in place of value iteration is that policy iteration uses \(V^{\pi_i}(s')\) instead of \(V_k(s')\). This means that while we have to compute \(V_k(s')\) we have to only fetch \(V^{\pi_i}(s')\) from policy \(\pi_i\). This change can considerably reduce compute time and is also utilized by **mixed methods**, that perform both value iteration and policy iteration. They can use two steps of policy iteration where they only fetch \(V^{\pi_i}(s')\) followed by one step of value iteration to compute the update \(V_k(s')\) or vice versa.

## 11. Evaluating Policies

While we considered how to obtain the policy \(\pi^*\) using the Bellman equation, we did not consider how to evaluate the obtained policy. While evaluating a policy our goal is to find \(V^{\pi}(s)\), the expected total utility starting from state \(s\) and following policy \(\pi\).

Because \(\pi\) already gives us the optimal action we do not need the max operation as we no longer need to choose an action. Evaluation is a **linear operation** versus the iteration is a **non-linear operation** because of the presence of the max operation. Time complexity: \(O(S^2)\).

$$V^{\pi}(s) = \sum_{s' \in S} T(s, \pi(s), s') \times [R(s, \pi(s), s') + \gamma V^{\pi}(s')] \tag{5}$$

## 12. Reinforcement Learning

In MDPs we were provided the rewards and the transitions associated with each state. But for most real world problems finding these transition functions and reward functions can be difficult. For example while theoretically we may consider the transition function of a slot machine, finding it in a real world casino can be fairly challenging. This is where **reinforcement learning** comes into play.

For reinforcement learning we are neither provided the transition function nor the reward associated with each state. This means that the agent is responsible for choosing the next best action without knowing what reward it may or may not get in the future state.

**Key Difference**: MDPs use offline learning because the agent knows the negative or positive reward associated with a future state based on which it may be inclined to either avoid that state or take that state in the future.

There are two possible approaches to reinforcement learning: **model based learning** and **model free learning**.

## 13. Model Based Learning

In model based learning we first try and learn the model of the world, which means we try and approximate the transition function and the reward function. Once we have the transition function and the rewards function we use those to solve the problem similar to an MDP using either value iteration or policy iteration. The only difference being that in MDPs we were solving for the actual world model but now we are solving for an approximation of the world model.

**How it works**: In order to find the transition and the reward functions we take every possible action from every state many times and count the number of times our action from one state results in every possible successor state. Based on the frequency with which we land in these states we can derive an approximation of the transition function \(T(s,a,s')\). We can follow a similar strategy to obtain the reward \(R(s,a,s')\).

**Problem**: This approach is dependent on the **law of large numbers** in probability, i.e. in order to get a more precise transition function we have to collect more samples.

---

<div style="text-align: center; margin: 30px 0;">
  <img src="Images/ExpectiMax.png" alt="ExpectiMax Algorithm" style="max-width: 90%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
  <p style="margin-top: 10px; font-style: italic; color: #666;"><strong>Figure 8:</strong> ExpectiMax search algorithm. Instead of assuming adversaries will always take the worst action (as in MiniMax), ExpectiMax computes expected values based on probabilities of different actions.</p>
</div>

---

## 14. Model Free Learning

While the model based learning approach works, it is dependent on the approximation of the transition and reward functions. In order to get better approximations of these we have to take multiple steps of all possible state-action pairs. This can be expensive especially since we will have to redo these in order to solve the Bellman equation using the approximated equations.

**Solution**: In order to solve this problem we use **model free learning**. If we look closer at the Bellman equation, the transition function is only being used to calculate the weighted average of the V value for a given state. That is, rather than approximating the transition function we could simply replace the transition function with empirical average of the utilities for the following state.

## 15. Passive Reinforcement Learning

In passive reinforcement learning the goal rather than being to learn a new policy is to **evaluate a given policy**. Here we do not choose actions, rather just keep taking actions and learning the values associated with each state for that policy.

### 15.1 Value Learning

In value learning we learn the value of each state by keeping track of the number of times the agent was in this state and the total reward the agent earned for the entire episode during which this state was visited. This enables us to get the average value of each state.

**Problem**: If two states lead to a common state their value might be very different because their values correspond to the reward seen in the corresponding episodes. That is, their value is more representative of the reward seen in the entire episode, which can be biased based on the law of large numbers for probabilities. If the agent was unlucky and saw more negative rewards its value would be lower irrespective that the state they landed in also resulted in positive rewards in other learning episodes.

### 15.2 Temporal Difference Learning

The above problem is solved by **temporal difference learning**, where the update rather than being based on episodes is based on the value of the successor state which in turn is updated using its successors. This is very similar to the Bellman equation used for evaluation:

$$V^{\pi}_{k+1}(s) = \sum_{s'} T(s, \pi(s), s') \times [R(s, \pi(s), s') + \gamma \times V^{\pi}_k(s')]$$

Since we do not have \(T(s, \pi(s), s')\), we replace it by learning from experience. We will follow the policy \(\pi\) multiple times for all states, which will provide us an approximation of the transition functions. So for one sample we calculate the value by:

$$\text{sample} = R(s, \pi(s), s') + \gamma \times V^{\pi}(s')$$

But, if we keep updating the sample each time with the results from the latest values of their successors we would end up biasing the values towards the last run episodes. In order to avoid this problem we update the values by taking a weighted sum of the old and new values.

$$V^{\pi}(s) = (1-\alpha) \times V^{\pi}(s) + \alpha \times \text{sample}$$

Which is the same as:

$$V^{\pi}(s) = V^{\pi}(s) + \alpha \times (\text{sample} - V^{\pi}(s)) \tag{6}$$

For this we use the parameter \(\alpha\) where \(0 < \alpha < 1\):
- If \(\alpha \approx 1\), we give a much higher weight to the recently seen episodes
- If it closer to zero we are giving more weight to values collected by old episodes

As we run for more episodes the value approximations start getting better, so when practically performing temporal difference learning it makes more sense to start with a high alpha and keep reducing it with the number of episodes. One way to reach such an approximation is by using \(\alpha = \frac{1}{n}\), where \(n\) is the number of episodes seen.

## 16. Active Reinforcement Learning

Because the above methods executed a pre-selected policy \(\pi\) they never had a choice of action. They had to execute the action provided by \(\pi\). This meant that though they could approximate the value of each state, they could not decide the optimal action as they had only experienced one action which was provided by the policy. This problem is solved by **active learning** where we do not just evaluate a policy but also derive at the optimal policy.

### 16.1 Q Learning

Q learning is a direct extension of the temporal difference learning. If we look at the temporal difference learning, we were only approximating the V-value of every state. For Q learning we simply calculate the Q-value instead.

Because now we do not have a fixed policy, rather than getting the \(V^{\pi}(s')\) from the policy \(\pi\) we have to calculate the optimal value by taking the max of the Q-values for all possible actions at state \(s'\), i.e. \(\max_{a' \in A} Q(s', a')\).

$$\text{sample} = R(s, a, s') + \gamma \times \max_{a' \in A} Q(s', a')$$

$$Q(s, a) = (1-\alpha) \times Q(s, a) + \alpha \times \text{sample}$$

$$Q(s, a) = (1-\alpha) \times Q(s, a) + \alpha \times [R(s, a, s') + \gamma \times \max_{a' \in A} Q(s', a')]$$

Also written as:

$$Q(s, a) \xleftarrow{\alpha} R(s, a, s') + \gamma \times \max_{a' \in A} Q(s', a') \tag{7}$$

The above simplistic substitutions allow us to now derive the optimal policy rather than simply evaluating a provided policy, i.e. now we are performing active learning rather than passive learning.

**Important Note**: It should be noted that Q learning provides us the optimal policy even when for the repeated experiences we were acting sub-optimally.

**Drawback**: The drawback of Q learning is that now we need to explore every state often to get its corresponding Q value. Eventually the parameter \(\alpha\), also referred to as the **learning rate**, has to be made small enough to avoid high fluctuations in values caused due to randomness. But at the same time \(\alpha\) cannot go down very quickly, so there is careful selection of \(\alpha\) required throughout the learning process to provide an optimal policy.

## 17. Exploration vs Exploitation

During active learning while the agent is building an optimal policy, if it keeps selecting the optimal result from the policy at each step it will not be able to take actions that it might not have taken yet. This brings us to balance between **exploration** vs **exploitation** performed by the agent.

**Definitions**:
- **Exploitation**: Selecting the optimal action from the policy
- **Exploration**: Selecting an action randomly

### ε-Greedy Approach

One possible solution to this problem would be to use a hyperparameter \(\epsilon\) which will act as the probability with which the agent selects an action at random rather than following the policy. With the probability of \((1 - \epsilon)\) the agent exploits the current optimal policy, i.e. selects the action with the highest reward.

**Problem**: Because exploration is dependent on \(\epsilon\) there is always a \((1 - \epsilon)\) probability that all actions from the current state would not be explored.

### Exploration Function

In order to solve this problem we use an exploration function:

$$f(u, n) = u + \frac{k}{n}$$

where:
- \(u\) is the Q value of a state-action pair
- \(n\) is the number of times that state-action pair has been executed
- \(k\) is a constant offset

**Core Idea**: The \(\frac{k}{n}\) term increases the utility of an infrequently visited state more than the utility of the frequently visited states.

$$Q(s, a) \xleftarrow{\alpha} R(s, a, s') + \gamma \times \max_{a' \in A} f(Q(s', a'), N(s', a'))$$

The above is the updated learning equation where \(N(s', a')\) is the number of times action \(a'\) was taken from state \(s'\).

## 18. Conclusion

This comprehensive document covered important concepts in artificial intelligence, from fundamental search algorithms to advanced reinforcement learning techniques.

### Key Takeaways

**Search Algorithms:**
- Explored uninformed search methods (DFS, BFS, UCS) and their trade-offs
- Examined informed search techniques (Greedy, A-Star) using heuristics
- Understood constraint satisfaction problems as specialized tree traversal

**Multi-Agent Systems:**
- Distinguished between single-agent and multi-agent search tree traversal
- Learned MiniMax and Alpha-Beta pruning for adversarial scenarios
- Explored ExpectiMax for handling action uncertainty

**Markov Decision Processes:**
- Mastered the Bellman equation for optimal decision making
- Understood value iteration and policy iteration techniques
- Learned how to evaluate and derive optimal policies

**Reinforcement Learning:**
- Explored model-based vs model-free learning approaches
- Implemented temporal difference learning and Q-learning
- Balanced exploration vs exploitation strategies

### Resources

All code implementations and experiments from this document are available at:
- **GitHub Repository**: [https://github.com/akshay-raj-dhamija/pac-man](https://github.com/akshay-raj-dhamija/pac-man)
- **Course Materials**: [UC Berkeley CS-188](https://inst.eecs.berkeley.edu/~cs188/su21/)

---

**[← Back to Blog](../index.html)**

