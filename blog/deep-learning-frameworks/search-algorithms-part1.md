---
layout: post
title: "Mastering Search Algorithms: From DFS to A-Star"
permalink: /blog/deep-learning-frameworks/search-algorithms-part1.html
---

## Part 1 of 3: AI Search Fundamentals Series

**Author:** Akshay Raj Dhamija  
**Date:** October 4, 2025  
**Tags:** Search Algorithms, AI, DFS, BFS, A-Star, Heuristics  
**Series:** AI Search Fundamentals (Part 1 of 3)

---

> **📚 Series Navigation:**  
> **Part 1: Mastering Search Algorithms** (Current)  
> [Part 2: Multi-Agent Search & Game Theory →](search-algorithms-part2.html)  
> [Part 3: MDPs to Q-Learning →](search-algorithms-part3.html)

---

This is the first part of a three-part series on artificial intelligence search algorithms and reinforcement learning, based on UC Berkeley's CS-188 course. In this post, we'll explore fundamental search algorithms that form the foundation of AI problem-solving.

**What you'll learn in this series:**
- **Part 1 (this post):** Uninformed and informed search algorithms
- **Part 2:** Multi-agent search with adversaries (MiniMax, Alpha-Beta)
- **Part 3:** Markov Decision Processes and Reinforcement Learning

---

## Table of Contents

1. [Search Problems & Their Terminology](#1-search-problems--their-terminology)
2. [Uninformed Search](#2-uninformed-search)
   - [Depth First Search (DFS)](#21-depth-first-search-dfs)
   - [Breadth First Search (BFS)](#22-breadth-first-search-bfs)
   - [Performance Comparison](#23-performance-comparison)
   - [Uniform Cost Search (UCS)](#24-uniform-cost-search-ucs)
   - [Data Structures](#25-data-structures-used-for-search-trees)
3. [Informed Search](#3-informed-search)
   - [Heuristics](#31-heuristics)
   - [Greedy Search](#32-greedy-search)
   - [A-Star Search](#33-a-star-search)
4. [Constraint Satisfaction Problems](#4-constraint-satisfaction-problems)
5. [Summary & Next Steps](#summary--next-steps)

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

---

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

If the problem was to explore the entire search tree (i.e. a problem like backtracking), both DFS and BFS will have the same time complexity since they will be expanding all the nodes in the search tree (considering that the tree is not infinite or cyclic). This time complexity will be $\(b^m\)$, where $\(b\)$ is the average number of successors for each node and $\(m\)$ is the total number of levels in the tree.

#### Space Complexity

While the time complexity of both DFS and BFS is the same, they have vastly different space complexity:
- **DFS**: $\(O(bm)\)$
- **BFS**: $\(O(b^m)\)$

This space complexity difference is a big reason for using DFS over BFS when implemented on edge devices like robot vacuum cleaners.

**DFS Space Complexity Explanation**: At the first search tier, DFS will see $\(b\)$ possible branches, from which it will be expanding one branch, while keeping $\((b-1)\)$ branches in memory. At tier 2 it will have $\((b-1)\)$ branches for the node at tier 2, while also holding $\((b-1)\)$ from the first tier. Assuming $\(b_1 = b_2\)$ we have $\(2 \times (b - 1)\)$, generalizing this further to the $\(m\)$th level we get $\(O(mb)\)$ as the space complexity.

**BFS Space Complexity Explanation**: At the first search tier, BFS will see $\(b\)$ possible branches. So at the 1st tier BFS will have $\(b\)$ nodes in memory. At tier 2 it expands each of the $\(b\)$ states/nodes into their respective $\(b\)$ successors each. This means at tier 2 it has $\(b^2\)$ nodes in memory. Generalizing this further to the $\(m\)$th tier we get $\(O(b^m)\)$ as the space complexity.

### 2.4 Uniform Cost Search (UCS)

While both DFS and BFS are able to find a path from a start state to the goal state, they are unable to provide us with the shortest possible path. Mostly because they are unable to take the cost associated with the state transitions into account. This problem is addressed using **Uniform Cost Search (UCS)** which is a cost sensitive search method as compared to DFS and BFS which are insensitive to the cost associated with each action.

**How it works**: UCS may also be thought of as a method that combines both DFS and BFS based on cost. The UCS performs DFS as long as continuing DFS provides it the least cost path; if not it switches to BFS to expand the state where it might have the lowest cost path.

**Optimality**: UCS is able to provide the optimal path from the start to the goal state.

**Complexity**: The space and time complexity of UCS is similar to those of BFS. If $\(b\)$ is the number of branches from a state and let's assume the goal state to have a minimum cost $\(C^\*\)$ associated with it, with the minimum cost from any pair of states to be $\(\epsilon\)$, then both the space and time complexity of UCS can be given as $\(O(b^{C^\*/\epsilon})\)$.

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

---

## 3. Informed Search

In the previous section we did a deep dive into uninformed search, which are the techniques that either do not use any information about the cost associated with an action (like in DFS and BFS) or aim to minimize the actual overall cost associated with all actions (like in UCS).

While such uninformed search approaches have a cost associated with each action from a given state, this cost indicates the cost of transition between two states. When analyzed cumulatively it only tells us the cost to reach the current state from the start state but it does not tell anything about how this cost relates to the actual goal we aim to achieve.

**Terminology**:
- **Backward cost $\(g(n)\)$**: The cost used by uninformed search approaches, where $\(n\)$ is the action $\(a\)$ taken from a given state $\(s\)$
- **Forward cost or Heuristic $\(h(n)\)$**: A cost which is much more representative of the cost that the algorithm might incur from the current state to reach the goal state

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

where $\(h^\*(n)\)$ is the true cost to the nearest goal.

This also means that an admissible heuristic will be optimistic of the outcome. However, it should also be noted that over optimism would be fatal. Consider $\(h(n) = 0\)$ which while satisfying the above equation would render the heuristic useless.

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

A-Star search, which is also considered to be a generalization of Dijkstra's algorithm, was originally used to solve the navigation problem in Shakey the robot. A-Star search simply combines the idea of greedy search and uniform cost search by considering the total cost to be a summation of the forward $\(h(n)\)$ and backward costs $\(g(n)\)$:

$$f(n) = g(n) + h(n)$$

**Complexity**: The worst case time and space complexity of both A-star search and greedy search is the same as that for UCS, i.e. $\(O(b^{C^\*/\epsilon})\)$.

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

---

## Summary & Next Steps

In this first part of our series, we've covered the fundamental building blocks of AI search algorithms:

### Key Takeaways

✅ **Uninformed Search Methods:**
- Learned DFS, BFS, and UCS with their trade-offs
- Understood time and space complexity differences
- Explored when to use each algorithm

✅ **Informed Search with Heuristics:**
- Mastered the concept of admissible heuristics
- Implemented Greedy search for fast (but not always optimal) solutions
- Understood A-Star as the gold standard for optimal pathfinding

✅ **Practical Applications:**
- Applied algorithms to pac-man navigation
- Compared different heuristics (Euclidean vs Manhattan distance)
- Solved constraint satisfaction problems with backtracking

### What's Next?

Now that you understand single-agent search, you're ready to tackle more complex scenarios! In **Part 2**, we'll explore:

🎮 **Multi-Agent Search & Game Theory**
- How to handle adversaries and allies
- MiniMax algorithm for perfect play
- Alpha-Beta pruning for efficiency
- ExpectiMax for uncertain opponents
- Applications in game-playing AI

**👉 [Continue to Part 2: Multi-Agent Search & Game Theory →](search-algorithms-part2.html)**

### Resources

- **Code Examples**: All implementations available in the [pac-man repository](https://github.com/akshay-raj-dhamija/pac-man)
- **Course**: [UC Berkeley CS-188](https://inst.eecs.berkeley.edu/~cs188/su21/)
- **Next in Series**: [Part 2: Multi-Agent Search](search-algorithms-part2.html)
- **Complete Guide**: [All Three Parts Combined](search-trees-to-reinforcement-learning.html)

---

**[← Back to Blog](../index.html)** &nbsp;&nbsp;|&nbsp;&nbsp; **[Part 2: Multi-Agent Search →](search-algorithms-part2.html)**

