# Overview
The general idea is to build the Quarto agent in 4 steps:
* Find a function that allow to restrict the space of possible game states to reduce the complexity of the next steps.
* Generate a dataset of game states paired with a reward value, computed with some variant of MinMax.
* Use some learning strategy to try and learn from the dataset an extimate heuristic reward function for game states, that hopefully generalize to all the possible game states.
* Build an agent that uses the reward function, possibly paired with other learning strategies, to play Quarto effectively.

# Internal representation of game states
To speed up the processes, game states are represented in a different and more compact way compared to the one used by the Quarto class provided. The agent will then manage the conversion between the two representations.
* Pawns are represented by a number between 0 and 15. The binary representation of them have 4 digits, each one corresponding to one of the features that pawns have.
* The chessboard is a Numpy array of 16 cells, where -1 indicates an empty cell and numbers in [0,15] are pawns.
* States are a tuple containing the chessboard's array and the assigned pawn.

# State reduction
Reducing the space of possible game states can help speeding up the learning process.
The basic idea is to define sets of functionally equivalent states and a collapse function such that:
* Two states S1 and S2 are functionally equivalent iff for each sequence of moves from S1 to a terminal state, there is a sequence of moves of the same size from S2 to a terminal state with equal reward (win, lose, tie), and vice versa.
* The collapse function f(s) is a function such that for every pair of states S1 and S2, f(S1)=f(S2) and f(S1) is functionally equivalent to S1, S2 if and only if S1 is functionally equivalent to S2.

Given this collapse function, every time a state needs to be evaluated, we can instead evaulate its collapsed counterpart.

The main rules used to define the sets of equivalent states are ispired from this article: https://stmorse.github.io/journal/quarto-part1.html

* Rotations or flips of the chessboard are unimportant in Quarto, so states that can be obtained one from another via rotations or flips are equivalent.
* The exact pawns used in the chessboard is not important, only the relations between them and between the ones that can be used next are.
* * for example, we can have two pawns with exactly one feature in common and another one that have one feature in common with the first and zero with the second. This is exactly what we need to evaulate the state, regardless of what specifically the features are.

This is implemented in the *collapse_state.py* script by two function:
* States are collapsed by rotations and flips by trying different rotations and flips and chosing the one that minimise an arbitrary function.
* States are collapsed by pawns by computing the dependency matrix between all pawns and finding another sequences of pawns that minimise another arbitrary function.

# Generating the dataset
The dataset is generated from randomly generated states of the chessboard. Each state is paired with a reward value that encodes how *good* it is to be in that state: a higher value means the player that finds himself in this state can win the game; a negative value means he will lose if the adversary plays good, a neutral means at best he can get a tie.

The basic algorithm to compute the reward value is a variant of the MinMax algorithm with caching and the collapse state function. Compared to the traditional MinMax it does use some pruning, but less strictly compared to traditional alpha-beta pruning, as it always tries to find at least **MIN_NEUT** and **MIN_NEG** child states with a neutral and negative reward value. This prevents the algorithm from generating datasets that are overly imbalanced toward positive states, which would happen with a pure alpha-beta pruning.
Note that here *positive* and *neutral* are aways considered from the point of view of the player that finds himself in the state. So MinMax gives a *positive* value of a state when it can find at least one child state with a *negative* one.

The MinMax function makes use of a cache of states, which is made more useful by the collapse_state function, since collapsed states are more likely to be already visited.

The generation of the dataset is time-consuming but doable in a few hours of computations, but only from states that have at least 8 pawns already placed. States with fewer pawns are closer to the root in the tree that MinMax visits, and computing their reward value is unfeasible. This is a limitation of this first dataset and will be addressed later.

The file containing the logic for the dataset generation is *generate_dataset.py*.

## Reward value bounds:
The reward value is bounded in the range [-100, 140] where:
* The MinMax result if valued 100 points: +100 for winning states, -100 for losing ones, 0 for neutral.
* Another 40 points are given by the ratio between wins and loses in all the branches that unravel from this state: 
* * The reason for these extra point is to favour states that not only are *winnable* by MinMax, but where most of their children leads to victory. This can be useful since the agent will not not play as a perfect MinMax agent and might not always play the best move, so it make sense to favour states where the chances of playing a good move are higher.
* * This ratio is not always computed exactly, since visiting all the children of a node might be too much time consuming. Instead it is extimated on a sub-set of the children.

## Cleaning and pre-processing the dataset
The dataset is then cleaned by the script contained in *pre_process_dataset.py*. 
The *generate_dataset* script generates an unbalanced distribution of states, in favour of states that are positive and have more pawns in it (closer to the leaves of the tree). The pre-process part balances this.
The script also divides the dataset into 2 partitions, used to train and to validate the model.

# Extract rules to extimate the reward value of states
## Training and validating
Training the model make use of a tranining dataset, and the goal is to reduce the average squared differece between the extimated reward and the one provided by the dataset. After the learning phase, the function is then validated by computing the squared error on another partition of the original dataset that was not used during training.
## Training algorithm
This part was done very experimentally, with many different tries.
The general idea is to have a set of binary *features* that can be computed on the state to be evaluated, and a set of weights to multiply to each feature. 
A *feature* can be any arbitrary function computed on the state of the game or on part of it.


The first try involved a GA where both the features and the weights used to compute the reward heuristic were dinamically learned. The number of features was fixed (N) but the algorithm could change which boxes of the chessboard were included in each feature and what function the feature would use. The weights were contained in an NxN matrix, associating a weight to each pair of features that could activate together.
This solution did not work and after many generations the algorithm could not provide better results compared to a pure random guess.

The second version involves a vector of pre defined features, and the learning process would only change the vector of corresponding weights.
The features are:
* For each line in the chessboard:
* * for each possible lengths of lines (0,1,2,3):
* *     One feature if there is a line of this length with at lest 1 feature in common and the line can be blocked with the given pawn.
* *     One feature if there is a line of this length with at lest 1 feature in common and the line can be extended with the given pawn.
* *     One feature if there is a line of this length with at lest 1 feature in common and the line can be blocked with one remaining pawn.
* *     One feature if there is a line of this length with at lest 1 feature in common and the line can be extended with one remaining pawn.
* For each pair of lines that cross in the chessboard, if the cell where they cross is empty and the two lines have at least one feature in common:
* * for each of the three combination: block both lines, extend both lines, block one and extend the other:
* *     One feature regarding the current assigned pawn.
* *     Two features regarding the remaining pawns, one if *at least 1* pawns respect the previous statement, the other if *at least 2* do.

The idea was having a set of features specific enough to be able to extract useful informations from it and to compute the extimated reward only as a linear combination of them (in the first try it was quadratic). 

This showed better results than the first try. 

The third and final version used 15 different vectors of weights, to be used depending on the number of pawns on the chessboard: as the algorithm proved to work best when the dataset had all states with the same number of pawns in it, it made sense to make it learn separately on each number of possible pawns present.

The final algorithm used to learn the weights is a simple **hill climbing**. 
The first idea was to use hill climbing just as an explorative tool for me to check if the set of features was good enough for an algorithm to learn from, and then switch to GA for the real training. But at the end I chosed to keep it, as:
* The main problem with **hill climbing** is that it can get stuck in local minima. But here our main goal is not to improve the results on the training dataset as much as we can, but to avoid over-fitting the model and thus reducing generalizability. GA might help getting a lower square error on the training dataset but does not guarantee in any way to provide a more generalizable result - on the contrary, a square error too low might indicate over-fitting.
* The learning process with GA might be too slow to be doable with my machine, while hill climbing was reasonably fast.

### Training algorithm optimization
The StateReward class evaluate states in two steps: first it computes a vector called *truth_values* that specifies what features are active in the specific state, then it computes the product of this vector with the weights vector.
To speed up training, the *truth_values* of the states in the dataset are pre-computed and cached. This is done in the *Climber* class at creation. The *StateReward* class offers a method called *get_reward_from_truth_value* that computes the reward from the truth_values instead of the state.

### Results
The rewards associated to the dataset are in the range [-100.00, 140.00]; as such, a random guess would give us an average squared error of 14.400. 
After many iterations of learning, the average squared error on the validation dataset was between 900 and 3.000 depending on the number of pawns in the chessboard. 

### Reward for states with fewer pawns
As said before, the first dataset generated contained only states with 8+ pawns already placed, and a good size only for states with 10+, as solving states with fewer pawns with MinMax was unfeasible.
A solution for this was to include the extimated reward function learned so far into the *generate_dataset* script: if we have a reliable extimate for the reward on states with at least 10 pawns, then the MinMax algorithm can stop ad depth 10 and return the extimate. 

To generate this second dataset another variant of MinMax was used.

This was it was possible to generate a more complete dataset and repeat the learning phase.
Of course it can be expected that the learned reward function for states with fewer pawns will be less reliable.


# Agent
## Overview
The basic idea is to build the final agent as a Min-Max-like - or Beam search-like agent that makes use of the reward function. Instead of visiting each state's possible children, resulting in an unbearable number of nodes, this MinMax implementation can make an effective use of the reward extimate by:
* Limit the max depth: instead of waiting for a final state, the algorithm can stop at any depth and return the reward extimate of the given node.
* Limit the width of each node's branch: instead of visiting all the children of a given node, the algorithm can use the reward extimate to visit only the N more promising children.

This brings to the question: given a maximum number of nodes we can visit for each move to make (for time constraints), what is more effective, visiting fewer children nodes each time but going deeper, or going less deep and visit more children of each node?
Another problem to solve is what to do with states with fewer than 5 pawns in it, since the reward function doesn't cover these.

## Fixed rules
States with [0..4] pawns in it cannot be evaluated by the reward function. A way to manage these states is to use fixed rules and let the agent learn which rule to use.
The four fixed rules to be chosen from are:
* Minimize the number of active rows, where a row is considered active if there is at least 1 pawn on it and all the pawns have at least 1 feature in common.
* Maximize the number of active rows
* Minimize the number of pawns in the biggest active line.
* Minimize the maximum number of common features in active lines.

Regardless of which rule is active, the agent always checks if the state is a winning one or if the adversary can win in 1 move from it.

## Using GA to find the params for the agent
The agent have a fixed number of states that can be visited at each move to make: *MAX_NODES*. The more nodes, the more accurate the result will be, but also the more time consuming the function is. This value is fixed at 2000, allowing for a fast execution time
.
Given this value, we can have a *MAX_WIDTH* parameter, which encodes how many children of each node will be visited. The *MAX_DEPTH* will then be computed as the logarithm of *MAX_NODES* with base *MAX_WIDTH*. 
Even better it is to have several *MAX_WIDTH* parameters, to be used for states with a different number of pawns in it. Perhaps for states with fewer pawns it is more useful to have smaller width and more depth, while for states with more pawns the contrary. Or perhaps vice versa. 
The agent have an array of *WIDTH* values, each one to be used for different state's number of pawns.

The other iusse, the one with states with less than 5 pawns, can be solved by using fixed rules in these states. The *FixedRules* class in the *StateReward* file provides 4 of them.

The agent can then have a genome composed by the *WIDTHS* array and one chosen *FixedRule*. From this, a GA can be used to find a good set of values.

Running this learning part was way more problematic than I expected, because of the time consuming reward function: running games takes time, and since there always is a random component in the results of the game, a good reward function should play as many as it can each time.
For this reason I had to use a smaller population than I would have liked and I had to run fewer generations. The lack of differences in the population after a few generations might have compromised the results of this part.
Anyway, the reward function seemed to favour agents that prioritize depth of exploration instead of width, and the first and third fixed rules. In the tests run later it can be seen if these parameters make a real difference compared to random ones.

## States cache and RL layer
The agent algorithm makes use of a cache, like the one used to generate the dataset. This cache is volatile and is used only to avoid repeating the same computations more than once.

But it can be useful to keep in storage another, smaller cache of states, to be used on top of the reward function. The idea is that, since the reward function is an extimate, the agent will indeed make mistakes from time to time. It can be useful to make it learn from them, keeping track of the good and the bad moves it made. 
This works like a RL layer on top of the existing agent.
The implementation is simple: the agent keep a list of all the moves it makes from the start of the game until the end. Whan the game finish, is uses the final result to compute a reward of its previous choices, and stores it. 
To reduce space usage, the states are hashed before being inserted.

During the execution, when the state is present in the cache, the agent uses both this value and the reward function to extimate the reward of the state.

The training of the RL layer has been done making the agent play against itself, against the random player and against limited versions of itself, where some features (like the reward extimate or the cache itself) were deactivated. Making the agent play against less effective adversaries can be useful to find weakness of it: perhaps there are useful moves that the final agent don't see and that less effective agents might stumble upon.

## Wrapper
The agent is split into two classes: *QuartoAgent* and *RealAgent*.
*RealAgent* is the one implementing the algorithm described before, while *QuartoAgent* is a wrapper that converts the state from the format of the Quarto module provided and the format used by my agent.

All the interesting logic is kept in the *RealAgent* class.

## Tests
Here the results of some tests on the final agent are provided.
Testing the agent isn't an easy task since we don't have a deterministic agent like we had with Nim. A possible solution is to test it against versions of itself where some features are deactivated. This way it is possible to check if these features actually provide any advantage.

### Agent VS Random Agent
Most simple test.

### Agent VS blind agent
If we take our agent and change the learned reward function with a flat one, and remove the RL cache too, what remains is a blind agent that uses beam search without any significant heuristic. 