# Overview
The general idea is to build the agent able to play Quarto in 4 steps:
* Find a function that allow to restrict the space of possible game states to reduce the complexity of the next steps.
* Generate a dataset of game states paired with a reward value, computed with some variant of MinMax.
* Use some learning strategy to try to extract some parameters from the dataset and find an extimate reward function that hopefully generalize to all the possible game states.
* Build an agent that uses the reward function, possibly paired with other learning strategies, to play Quarto effectively.

# Internal representation of game states
To speed up the processes, game states are represented in a different and more compact way compared to the one used by the Quarto class provided. The agent will then manage the conversion between the two representations.
* Pawns are represented by a number between 0 and 15. The binary representation of them have 4 digits, each one corresponding to one of the features that pawns have.
* The chessboard is a Numpy array of 256 cells, where -1 indicates an empty cell and numbers in [0,15] are pawns.

# State reduction
Reducing the space of possible game states can help speeding up the learning process.
The basic idea is to define sets of functionally equivalent states and a collapse function such that:
* Two states S1 and S2 are functionally equivalent iff for each sequence of moves from S1 to a terminal state, there is a sequence of moves of the same size from S2 to a terminal state with equal reward (win, lose, tie), and vice versa.
* The collapse function f(s) is a function such that for every pair of states S1 and S2, f(S1)=f(S2) and f(S1) is functionally equivalent to S1, S2 iff S1 is functionally equivalent to S2.

Given this collapse function, every time a state needs to be evaluated, we can instead evaulate its collapsed conterpart.

The main rules used to define the sets of equivalent states are ispired from this article: https://stmorse.github.io/journal/quarto-part1.html

* Rotations or flips of the chessboard are unimportant in Quarto, so states that can be obtained one from another via rotations or flips are equivalent.
* The exact pawns used in the chessboard is not important, but the relations between them and between the ones that can be used next are.
* * for example, we can have two pawns with exactly one feature in common and another one that have one feature in common with the first and zero with the second. This is just what we need to evaulate the state, regardless of what there features are.

This is implemented in the *collapse_state.py* script by two function:
* States are collapsed by rotations and flips by trying different rotations and flips and chosing the one that minimise an arbitrary function.
* States are collapsed by pawns by computing the dependency matrix between all pawns and finding another sequences of pawns that minimise another arbitrary function.

# Generating the dataset
The dataset is generated from randomly generated states of the chessboard. Each state is paired with a reward function that encodes how *good* it is to be in that state. Meaning, a higher value means the player can win starting from the state, and a negative one means he cannot win.

The basic algorithm to compute the reward value is a variant of the MinMax algorithm with caching and the collapse state function. Compared to the traditional MinMax it does use some pruning, but less strictly compared to traditional alpha-beta pruning, as it always tries to find at least **MIN_NEUT** and **MIN_NEG** child states with a neutral and negative reward value. This prevents the algorithm from generating datasets that are overly imbalanced toward positive states, which would happen with a pure alpha-beta pruning.
Note that here *positive* and *neutral* are aways considered from the point of view of the player that finds himself in the state. So MinMax gives a *positive* value of a state when it can find at least one child state with a *negative* one.

The generation of the dataset is time-consuming but doable in a few hours of computations, but only from states that have at least 8 pawns already placed. States with fewer pawns are closer to the root in the tree that MinMax visits, and computing their reward value is unfeasible. This is a limitation of this first dataset and will be addressed later.

The file containing the logic for the dataset generation is *generate_dataset.py*.

## Reward value bounds:
The reward value is bounded in the range [-140, 140] where:
* The MinMax result if valued 100 points: +100 for winning states, -100 for losing ones, 0 for neutral.
* Another 40 points are given by the ratio between wins and loses in all the branches that unravel from this state, regardless of MinMax. 
* * The meaning of these extra point is to favour states that not only are *winnable* from MinMax, but where most of their children leads to victory. This can be useful since the agent will not not play as a perfect MinMax agent and might not always play the best move, so it make sense to favour states where the chances of playing a good move are higher.

## Cleaning and pre-processing the dataset
The dataset is then cleaned by the script contained in *pre_process_dataset.py*. 
The *generate_dataset* script generates an unbalanced distribution of states, in favour of states that are positive and have more pawns in it (closer to the leaves of the tree). The pre-process part balances this.
The script also divides the dataset into 2 partitions, used to train and to validate the model.

# Extract rules to extimate the reward value of states
## Training and validating
Training the model made use of a tranining dataset, and the goal was to reduce the average squared differece between the extimated reward and the one provided by the dataset. After the learning phase, the function is then validated by computing the squared error on another partition of the original dataset that was not used during training.
## Training algorithm
This part was done very experimentally, with many different tries.
The general idea is to have a set of binary *features* that can be computed on the state to be evaluated, and a set of weights to multiply to each feature. A *feature* can be any arbitrary function computed on the state of the game or on part of it.


The first try involved a GA where both the features and the weights used to compute the reward function were dinamically learned. The number of features was fixed (N) but the algorithm could change which boxes of the chessboard were included in each feature and what function the feature would use. The weights were contained in an NxN matrix, associating a weight to any pair of features that could activate together.
This did not work and after many generations the algorithm could get better results compared to a pure random guess.


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
* The learning process with GA might be too slow to be doable with my machine.

### Training algorithm optimization
The StateReward class evaluate states in two steps: first it computes a vector called *truth_values* that specifies what features are active in the specific state, then it computes the product of this vector with the weights vector.
To speed up training, the *truth_values* of the states in the dataset are pre-computed and cached. This is done in the *Climber* class at creation. The *StateReward* class offers a method called *get_reward_from_truth_value* that computes the reward from the truth_values instead of the state.

### Results
The rewards associated to the dataset are in the range [-100.00, 140.00]; as such, a random guess would give us an average squared error of 14.400. 
After many iterations of learning, the average squared error on the validation dataset was of 1.700.

### Reward for states with fewer pawns
As said before, the first dataset generated contained only states with 8+ pawns already placed, and a good size only for states with 10+, as solving states with fewer pawns with MinMax was unfeasible.
A solution for this was to include the extimated reward function learned so far into the *generate_dataset* script: if we have a reliable extimate for the reward on states with at least 10 pawns, then the MinMax algorithm can stop ad depth 10 and return the extimate. 

To generate this second dataset another variant of MinMax was used.

This was it was possible to generate a more complete dataset and repeat the learning phase.
Of course it can be expected that the learned reward function for states with fewer pawns will be less reliable.


