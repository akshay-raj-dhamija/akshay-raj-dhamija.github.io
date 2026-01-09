---
layout: post
title: "From MDPs to Q-Learning: A Reinforcement Learning Journey"
permalink: /blog/deep-learning-frameworks/search-algorithms-part3.html
---

## Part 3 of 3: AI Search Fundamentals Series

**Author:** Akshay Raj Dhamija  
**Date:** October 4, 2025  
**Tags:** Reinforcement Learning, MDP, Q-Learning, Bellman Equation, AI  
**Series:** AI Search Fundamentals (Part 3 of 3)

---

> **📚 Series Navigation:**  
> [← Part 1: Mastering Search Algorithms](search-algorithms-part1.html)  
> [← Part 2: Multi-Agent Search & Game Theory](search-algorithms-part2.html)  
> **Part 3: MDPs to Q-Learning** (Current)

---

Welcome to the final part of our AI Search Fundamentals series! In [Part 1](search-algorithms-part1.html), we learned search algorithms like A-Star. In [Part 2](search-algorithms-part2.html), we handled adversaries with MiniMax. Now we tackle the ultimate challenge: **learning optimal behavior when we don't know the rules of the game**!

**What you'll learn in this post:**
- Markov Decision Processes (MDPs) for sequential decision making
- Bellman Equations for optimal policies
- Value Iteration and Policy Iteration
- Reinforcement Learning fundamentals
- Q-Learning and Temporal Difference Learning
- Balancing exploration vs exploitation

---

## Table of Contents

1. [Markov Decision Processes](#1-markov-decision-processes)
2. [Bellman Equation](#2-bellman-equation)
3. [Solving the Bellman Equation](#3-solving-the-bellman-equation)
4. [Evaluating Policies](#4-evaluating-policies)
5. [Reinforcement Learning](#5-reinforcement-learning)
6. [Model Based Learning](#6-model-based-learning)
7. [Model Free Learning](#7-model-free-learning)
8. [Passive Reinforcement Learning](#8-passive-reinforcement-learning)
9. [Active Reinforcement Learning](#9-active-reinforcement-learning)
10. [Exploration vs Exploitation](#10-exploration-vs-exploitation)
11. [Conclusion](#conclusion)

---

## 1. Markov Decision Processes

As we saw in [Part 2](search-algorithms-part2.html), for multi-agent search we had to look a few steps in the future to make decisions, which can be computationally expensive. **Markov Decision Processes (MDPs)** solve this by creating a **policy map** - a mapping from every state to the optimal action in that state.

The goal here is to create a map of states to actions for a given problem so that during execution such a map can be referred to get the optimal action for a given state instantly!

### MDP Terminology

Let's understand the building blocks of MDPs:

1. **Set of states $\(S\)$**: All possible situations the agent can be in
2. **Start state $\(s_0\)$**: Where the agent begins
3. **Set of actions $\(A\)$**: All possible actions available
4. **Transitions $\(P(s'|s, a)\)$ or $\(T(s, a, s')\)$**: Since the actions in an MDP are **stochastic**, the result of a given action at a given state is not always the same. The probability that a given combination of state $\(s\)$ and action $\(a\)$ will yield state $\(s'\)$ is given by $\(P(s'|s, a)\)$

5. **Q State**: Q state can be best understood as an abstraction of the transition probability. Think of it as: the primary agent has committed to an action $\(a\)$ but the outcome is not yet determined - it depends on the stochastic transition function.

6. **Reward $\(R(s, a, s')\)$**: The reward associated with reaching state $\(s'\)$ from state $\(s\)$ via action $\(a\)$

7. **Discounting factor $\(\gamma\)$**: Ensures that rewards seen earlier are valued more than rewards seen much later. The discounting factor is $\(0 \leq \gamma < 1\)$. The discounted reward $\(r_t\)$ at time $\(t\)$ is: $\(r_t = \gamma^t \times R_t\)$ where $\(t \geq 0\)$
   
   Another use: if $\(0 < \gamma < 1\)$, then $\(\gamma^{\infty} \approx 0\)$, making rewards at the bottom of an infinite search tree negligible.

8. **Utility $\(U\)$**: The sum of discounted rewards that the agent can earn in the future from a given state:

$$U_s = \sum_{t=0}^{\infty} \gamma^t R_t = \frac{R_{max}}{1-\gamma}$$

9. **Value $\(V^\*(s)\)$**: The maximum utility among all possible actions an agent can take given it is in state $\(s\)$. In other words, the utility of a state $\(s\)$ **assuming that the agent always takes the optimal action thereafter**.

10. **Q Value $\(Q^\*(s, a)\)$**: Expected utility given that the current action taken from state $\(s\)$ is $\(a\)$ and following this action the agent performs optimally. 
    
    **Key Difference**: While utility $\(U\)$ depicts the sum of discounted rewards, Q value takes the **transition probabilities** into account. The difference between V value and Q value: V-value is for the primary agent, Q-value is for when outcomes are beyond the agent's control.

---

## 2. Bellman Equation

The **Bellman equation** is used for optimization problems using dynamic programming. It models the assumption that the agent takes the correct action and then onwards continues to be optimal.

### Deriving the Bellman Equation

For the agent taking the correct action, we define $\(V^\*(s)\)$ as:

$$V^*(s) = \max_{a \in A} Q^*(s, a) \tag{1}$$

Now, $\(Q^\*(s, a)\)$ indicates the average utility of the agent given action $\(a\)$ at state $\(s\)$. The averaging component is because of the transition function - the agent is no longer in control.

From ExpectiMax (Part 2), we know:

$$Q^*(s, a) = \sum_{s' \in S} T(s, a, s') \times V^*(s') \tag{2}$$

Combining this with the reward seen at the current state and discounting of future rewards:

$$Q^*(s, a) = \sum_{s' \in S} T(s, a, s') \times [R(s, a, s') + \gamma V^*(s')] \tag{3}$$

Replacing $\(Q^\*(s, a)\)$ in equation 1, we get the **Bellman equation**:

$$V^*(s) = \max_{a \in A} \sum_{s' \in S} T(s, a, s') \times [R(s, a, s') + \gamma V^*(s')] \tag{4}$$

**Time complexity**: $\(O(S^2 A)\)$ where $\(S\)$ is the number of states and $\(A\)$ is the number of possible actions.

---

## 3. Solving the Bellman Equation

### Method 1: Solve using ExpectiMax

One approach is to consider MDPs as **finite horizon MDPs** with a pre-decided depth of the search tree and only consider rewards till that depth. This is similar to the mini-max or expectimax search from Part 2.

**Problem**: The selected action is only optimal up to the considered depth but might not be optimal to reach the end goal.

### Method 2: Value Iteration ⭐

Because the Bellman equation itself is recursive, we can't solve it directly. Instead, we use the concept of **discounting** and the fact that for a very deep search tree, the discounted value of rewards at the last tier would be so small that it would not impact the overall utility, hence reaching convergence.

$$V_{k+1}(s) = \max_{a \in A} \sum_{s' \in S} T(s, a, s') \times [R(s, a, s') + \gamma V_k(s')]$$

$$V_{k+1} - V_k = \gamma^k \times \max_{i \in k} (R_i)$$

$$\text{if } k \approx \infty, \ \gamma^k \approx 0, \ V_{k+1} - V_k \approx 0$$

While we don't go to infinite optimization steps, in practice when $\(k\)$ becomes sufficiently large, the numbers become too small for floating point precision, giving the illusion of $\(k \approx \infty\)$.

### Method 3: Policy Iteration 🎯

The solution of an MDP is actually not the values at each state, but the **corresponding action** to be taken at those states. When we perform value iteration, we notice that while reaching convergence for values may take $\(n\)$ iterations, the optimal action converges after only $\(k\)$ iterations where $\(k \ll n\)$.

This motivates **policy iteration** where rather than trying to converge the values, we try to converge the policy $\(\pi\)$:

$$\pi_{i+1}(s) = \arg\max_{a \in A} \sum_{s' \in S} T(s, a, s') \times [R(s, a, s') + \gamma V^{\pi_i}(s')]$$

### Method 4: Mixed Methods

Another advantage of policy iteration is that it uses $\(V^{\pi_i}(s')\)$ instead of $\(V_k(s')\)$. This means we have to **compute** $\(V_k(s')\)$ but only **fetch** $\(V^{\pi_i}(s')\)$ from policy $\(\pi_i\)$. This reduces compute time.

**Mixed methods** perform both value iteration and policy iteration, e.g.:
- Two steps of policy iteration (fetch $\(V^{\pi_i}(s')\)$)
- One step of value iteration (compute update $\(V_k(s')\)$)
- Or vice versa

---

## 4. Evaluating Policies

While we learned how to obtain the policy $\(\pi^\*\)$ using the Bellman equation, we need to know how to evaluate it. When evaluating a policy, our goal is to find $\(V^{\pi}(s)\)$ - the expected total utility starting from state $\(s\)$ and following policy $\(\pi\)$.

Because $\(\pi\)$ already gives us the optimal action, we **don't need the max operation**. Evaluation is a **linear operation** (versus iteration which is non-linear due to the max operation).

**Time complexity**: $\(O(S^2)\)$

$$V^{\pi}(s) = \sum_{s' \in S} T(s, \pi(s), s') \times [R(s, \pi(s), s') + \gamma V^{\pi}(s')] \tag{5}$$

---

## 5. Reinforcement Learning

In MDPs we were provided the rewards and the transitions associated with each state. But for most real world problems, finding these transition functions and reward functions can be **difficult or impossible**.

**Example**: While theoretically we may consider the transition function of a slot machine, finding it in a real world casino can be fairly challenging!

### The Reinforcement Learning Challenge

For reinforcement learning:
- We are **not provided** the transition function
- We are **not provided** the reward associated with each state
- The agent is responsible for choosing the next best action **without knowing** what reward it may or may not get

**Key Difference from MDPs**: MDPs use **offline learning** because the agent knows the rewards. RL uses **online learning** - the agent learns by trying actions and observing outcomes.

There are two possible approaches:
1. **Model-based learning**: Learn the model (transitions & rewards), then solve
2. **Model-free learning**: Learn optimal behavior directly without learning the model

---

## 6. Model Based Learning

In model based learning we first try to **learn the model of the world** - we approximate the transition function and the reward function. Once we have them, we solve the problem similar to an MDP using value iteration or policy iteration.

### How It Works

To find the transition and reward functions:
1. Take every possible action from every state **many times**
2. Count the number of times our action from one state results in each possible successor state
3. Based on frequency, derive an approximation of $\(T(s,a,s')\)$
4. Follow similar strategy to obtain $\(R(s,a,s')\)$

### Problem

This approach is dependent on the **law of large numbers** in probability. To get more precise transition functions, we have to collect **many samples**. This can be expensive, especially since we'll have to redo these to solve the Bellman equation.

---

## 7. Model Free Learning

While model based learning works, it requires collecting many samples to approximate transitions and rewards, which we then use to solve the Bellman equation.

### Key Insight

If we look closer at the Bellman equation, the transition function is only being used to calculate the **weighted average** of the V value for a given state. Rather than approximating the transition function, we could simply replace it with the **empirical average** of the utilities for the following state!

This is the foundation of **model-free reinforcement learning**.

---

## 8. Passive Reinforcement Learning

In passive reinforcement learning, the goal is to **evaluate a given policy** rather than learn a new one. We don't choose actions - we just keep taking actions according to the policy and learning the values associated with each state.

### 8.1 Value Learning

In value learning we learn the value of each state by:
1. Keeping track of the number of times the agent was in this state
2. Tracking the total reward earned for entire episodes when this state was visited
3. Computing the average value of each state

**Problem**: If two states lead to a common state, their values might be very different because values correspond to rewards seen in the entire episode. If the agent was unlucky and saw more negative rewards, its value would be lower irrespective of positive rewards in other episodes.

### 8.2 Temporal Difference Learning ⭐

The problem above is solved by **temporal difference learning**, where the update is based on the value of the successor state, which in turn is updated using its successors. This is similar to the Bellman equation for evaluation:

$$V^{\pi}_{k+1}(s) = \sum_{s'} T(s, \pi(s), s') \times [R(s, \pi(s), s') + \gamma \times V^{\pi}_k(s')]$$

Since we don't have $\(T(s, \pi(s), s')\)$, we replace it by **learning from experience**. We follow policy $\(\pi\)$ multiple times for all states, providing an approximation of transitions.

For one sample, we calculate:

$$\text{sample} = R(s, \pi(s), s') + \gamma \times V^{\pi}(s')$$

But updating with only the latest sample would bias values towards the last run. Instead, we update with a **weighted sum** of old and new values:

$$V^{\pi}(s) = (1-\alpha) \times V^{\pi}(s) + \alpha \times \text{sample}$$

Which is the same as:

$$V^{\pi}(s) = V^{\pi}(s) + \alpha \times (\text{sample} - V^{\pi}(s)) \tag{6}$$

Where $\(\alpha\)$ is the **learning rate** ($\(0 < \alpha < 1\)$):
- If $\(\alpha \approx 1\)$: Higher weight to recent episodes
- If $\(\alpha \approx 0\)$: More weight to old episodes

**Best practice**: Start with high $\(\alpha\)$ and reduce it over time. One approach: $\(\alpha = \frac{1}{n}\)$ where $\(n\)$ is the number of episodes seen.

---

## 9. Active Reinforcement Learning

In passive RL, we only evaluated a policy - we never had a choice of action. Though we could approximate the value of each state, we couldn't decide the optimal action as we only experienced one action provided by the policy.

**Active learning** solves this: we don't just evaluate a policy but also **derive the optimal policy**.

### 9.1 Q Learning 🌟

Q learning is a direct extension of temporal difference learning. Instead of only approximating V-values, we **calculate Q-values**.

Because we don't have a fixed policy, instead of getting $\(V^{\pi}(s')\)$ from policy $\(\pi\)$, we calculate the optimal value by taking the **max** of Q-values for all possible actions at state $\(s'\)$:

$$\text{sample} = R(s, a, s') + \gamma \times \max_{a' \in A} Q(s', a')$$

$$Q(s, a) = (1-\alpha) \times Q(s, a) + \alpha \times \text{sample}$$

$$Q(s, a) = (1-\alpha) \times Q(s, a) + \alpha \times [R(s, a, s') + \gamma \times \max_{a' \in A} Q(s', a')]$$

Also written as:

$$Q(s, a) \xleftarrow{\alpha} R(s, a, s') + \gamma \times \max_{a' \in A} Q(s', a') \tag{7}$$

### Key Properties of Q-Learning

✅ **Optimal even with sub-optimal exploration**: Q-learning provides the optimal policy **even when during exploration we acted sub-optimally**!

⚠️ **Requires exploration**: We need to explore every state often to get its Q value. The learning rate $\(\alpha\)$ has to be:
- Small enough to avoid high fluctuations due to randomness
- Not decreasing too quickly
- Carefully selected throughout learning

---

## 10. Exploration vs Exploitation

During active learning while building an optimal policy, if the agent always selects the optimal action from the current policy, it will not explore actions it hasn't tried yet. This is the **exploration vs exploitation dilemma**.

**Definitions**:
- **Exploitation**: Selecting the optimal action from the current policy
- **Exploration**: Selecting an action randomly to discover new information

### 10.1 ε-Greedy Approach

Use a hyperparameter $\(\epsilon\)$ as the probability with which the agent selects an action at random rather than following the policy:
- With probability $\((1 - \epsilon)\)$: **exploit** (select action with highest reward)
- With probability $\(\epsilon\)$: **explore** (select random action)

**Problem**: There's always a $\((1 - \epsilon)\)$ probability that not all actions from the current state will be explored.

### 10.2 Exploration Function 🎯

To solve the above problem, we use an exploration function:

$$f(u, n) = u + \frac{k}{n}$$

Where:
- $\(u\)$ is the Q value of a state-action pair
- $\(n\)$ is the number of times that state-action pair has been executed
- $\(k\)$ is a constant offset

**Core Idea**: The $\(\frac{k}{n}\)$ term **increases the utility of infrequently visited states** more than frequently visited states, encouraging exploration of under-explored areas.

$$Q(s, a) \xleftarrow{\alpha} R(s, a, s') + \gamma \times \max_{a' \in A} f(Q(s', a'), N(s', a'))$$

Where $\(N(s', a')\)$ is the number of times action $\(a'\)$ was taken from state $\(s'\)$.

---

## Conclusion

Congratulations! You've completed the AI Search Fundamentals series. Let's recap the incredible journey:

### The Complete Journey

**[Part 1: Search Algorithms](search-algorithms-part1.html)** 🔍
- Mastered uninformed search (DFS, BFS, UCS)
- Learned informed search with heuristics (Greedy, A-Star)
- Found optimal paths in known environments

**[Part 2: Multi-Agent Search](search-algorithms-part2.html)** 🎮
- Handled adversaries with MiniMax and Alpha-Beta pruning
- Dealt with uncertainty using ExpectiMax
- Played perfect games with imperfect information

**Part 3: Reinforcement Learning** 🤖
- Formalized sequential decision making with MDPs
- Derived optimal policies with Bellman equations
- Learned without knowing the rules using Q-learning
- Balanced exploration and exploitation

### From Search to Learning

We've progressed from:
- **Knowing everything** (Part 1: complete map, optimal path)
- **Knowing the rules** (Part 2: game rules, opponent exists)
- **Knowing nothing** (Part 3: learn by trial and error)

This mirrors how AI has evolved: from GPS navigation → chess-playing computers → robots learning to walk!

### Key Reinforcement Learning Takeaways

✅ **MDPs**: Formalize sequential decision-making under uncertainty  
✅ **Bellman Equations**: Provide optimal policies when model is known  
✅ **Value/Policy Iteration**: Solve MDPs efficiently  
✅ **Q-Learning**: Learn optimal behavior without knowing transitions  
✅ **TD-Learning**: Update estimates based on experience  
✅ **Exploration Functions**: Balance discovery vs optimization  

### Real-World Applications

The techniques you've learned power:
- 🎮 **Game AI**: AlphaGo, chess engines, game NPCs
- 🤖 **Robotics**: Autonomous navigation, manipulation
- 🚗 **Self-Driving Cars**: Path planning, decision making
- 💰 **Finance**: Trading algorithms, portfolio optimization
- 🏥 **Healthcare**: Treatment planning, resource allocation
- 🎯 **Recommendations**: Personalized content, advertising

### Continue Learning

Want to dive deeper? Check out:

**Advanced RL Topics**:
- Deep Q-Networks (DQN)
- Policy Gradient Methods
- Actor-Critic Algorithms
- Multi-Agent RL
- Inverse Reinforcement Learning

**Courses & Resources**:
- [UC Berkeley CS-188](https://inst.eecs.berkeley.edu/~cs188/su21/) - Our foundation
- [DeepMind x UCL RL Course](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)
- [Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book.html) - The RL Bible
- [OpenAI Spinning Up](https://spinningup.openai.com/) - Practical deep RL

**Our Code**: All implementations available in the [pac-man repository](https://github.com/akshay-raj-dhamija/pac-man)

### Thank You! 🙏

Thank you for joining me on this journey through AI search and reinforcement learning! I hope these three posts have given you a solid foundation to build upon.

If you found this series helpful, please share it with others who might benefit. Feel free to reach out with questions or feedback!

---

**[← Part 2: Multi-Agent Search](search-algorithms-part2.html)** &nbsp;&nbsp;|&nbsp;&nbsp; **[Back to Blog](../index.html)** &nbsp;&nbsp;|&nbsp;&nbsp; **[← Part 1: Search Algorithms](search-algorithms-part1.html)**

**[📄 Complete Guide (All Parts Combined)](search-trees-to-reinforcement-learning.html)**

