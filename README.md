# Overview
This Quarto agents is implemented as a MinMax with pruning / Beam search-like algorithm, where the heurstic function used to choose which nodes to expand and which to prune is learned with a hill-climbing algorithm over a dataset of states-reward pairs generated by classical MinMax.
On top of this algorithm an unsupervised RL layer has been added.

The steps to come to this solution were:
* Find a function that allow to restrict the space of possible game states to reduce the complexity of the next steps.
* Generate a dataset of game states paired with a reward value, computed with MinMax.
* Use some learning strategy to try and learn from the dataset an extimate reward function for game states, that hopefully generalize to all the possible game states.
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

* Simmetries: Rotations or flips of the chessboard are unimportant in Quarto, so states that can be obtained one from another via rotations or flips are equivalent.
* The exact pawns used in the chessboard is not important, only the relations between them and between the ones that can be used next are.
* * for example, we can have two pawns with exactly one feature in common and another one that have one feature in common with the first and zero with the second. This is exactly what we need to evaulate the state, regardless of what specifically the features are.

This is implemented in the *collapse_state.py* script by two function:
* States are collapsed by rotations and flips by trying different rotations and flips and chosing the one that minimise an arbitrary function.
* States are collapsed by pawns by computing the dependency matrix between all pawns and finding another sequences of pawns that minimise another arbitrary function.

# Generating the dataset
The dataset is generated by randomly generating states of the chessboard. Each state is then solved by a MinMax algorithm, and then stored with its reward value. All the inner states generated by MinMax are stored in the same way.

Since Quarto is a symmetric game, the reward value of states is symmetric too: a positive value (100 in our case) means that it is possible to win starting from this state, while a negative one means it is impossible (given the adversary plays his moves right). 0 is used for ties.

The final reward value of the labelled states is computed as the MinMax calculated reward + (NUM_POSSIBLE_WINS / NUM_POSSIBLE_GAMES), where NUM_POSSIBLE wins is an extimate of the number of possible winning sequences of moves from the state, and NUM_POSSIBLE_GAMES the total number of sequences. The reason behind this is to give a higher reward to states that have more different winning sequences of moves. This might be useful because the final agent won't be able to explore the tree of moves completely, so seeking states where finding one is more likely should improve performances.

About the MinMax algorithm there isn't much to say. It's classic MinMax with caching. It does use some pruning but less strictly compared to traditional alpha-beta pruning, as it always tries to find at least **MIN_NEUT** and **MIN_NEG** child states with a neutral and negative reward value. This prevents the algorithm from generating datasets that are overly imbalanced toward positive states, which would happen with a pure alpha-beta pruning, and allows to extimate the (NUM_POSSIBLE_WINS / NUM_POSSIBLE_GAMES) value.

The generation of the dataset was time-consuming and doable only on states that have at least 8 pawns already placed. States with fewer pawns have too many different paths sprawing from them, and solving them with MinMax is not doable. This iusse will be addressed later.

The file containing the logic for the dataset generation is *quarto_agent_lib/generate_dataset.py*.


## Cleaning and pre-processing the dataset
The dataset is then cleaned by the script contained in *pre_process_dataset.py*. 
The *generate_dataset* script generates an unbalanced distribution of states, in favour of states that are positive and have more pawns in it (closer to the leaves of the tree). The pre-process part balances this.
The script also divides the dataset into 2 partitions, used to train and to validate the model.

# Learning the heuristic reward function

## Training and validating
Training the model make use of a tranining dataset, and the goal is to reduce the average squared differece between the extimated reward and the one provided by the dataset. After the learning phase, the function is then validated by computing the squared error on another partition of the original dataset that was not used during training.

## Training algorithm
This part was done very experimentally, with many different tries.

The general idea is to have a set of binary function that can be computed on the state to be evaluated (called *features*, just to give them a name), and a set of weights to multiply to each feature. The final reward is the vector product of features * weights.

The first try involved a GA, and both the features and the weights were to be learned by it. 
Each feature was characterized by a subset of the chessboard to be avaluated and a function taken from a set of fixed functions. Examples of these fixed functions are "*All selected boxes are empty*" or "*All selected boxes are occupied with pawns with 3 features in common*" or "*At least two boxes are occupied by pawns with 0 features in common*". 
The weights were a matrix of Nx(N+1)/2 (where N is the number of features), and each weight was multiplied to two different features.
This solution did not work, and after many generations the algorithm could not reach better results compared to a pure random guess.

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

The idea is to have a set of fixed features specific enough to be able to extract useful informations from linear combinations of it. This should speed up the learning process as the algorithm doesn't have to find the features (they are fixed) and the number of weights is smaller. 

This showed better results than the first try. 

The third and final version used 15 different vectors of weights, to be used depending on the number of pawns on the chessboard: as the algorithm proved to work best when the dataset had all states with the same number of pawns in it, it made sense to make it learn separately on each possible state's size.

The final algorithm used to learn the weights is a simple **hill climbing**. At first I used it only as an explorative tool, to see if this idea could bring some results, with the intention of then switching to GA. But at the end I chosed to keep it, as:
* The results were good.
* The main problem with **hill climbing** is that it can get stuck in local minima. But here our main goal is not to improve the results on the training dataset as much as we can, but to avoid over-fitting the model and thus reducing generalizability. GA might help getting a lower square error on the training dataset but does not guarantee in any way to provide a more generalizable result - on the contrary, a square error too low might indicate over-fitting.
* The learning process with GA might be too slow to be doable with my machine, while hill climbing was reasonably fast.

To this reward function a few hardcoded were added to determinalistically detect states one move away from winning or defeats.

The scripts used to train the model is in the file *quarto_agent_lib/state_reward.py*

### Training algorithm optimization
The StateReward class evaluate states in two steps: first it computes a vector called *truth_values* that specifies which features are active in the specific state, then it computes the product of this vector with the weights vector.
To speed up training, the *truth_values* of the states in the dataset are pre-computed and cached. This is done in the *Climber* class at creation. The *StateReward* class offers a method called *get_reward_from_truth_value* that computes the reward from the truth_values instead of the state.

### Results
The rewards associated to the dataset are in the range [-100.00, 140.00]; as such, a random guess would give us an average squared error of 14.400. 
After many iterations of learning, the average squared error on the validation dataset was between 900 and 3.000 depending on the number of pawns in the chessboard.

### Reward for states with fewer pawns
As said before, the first dataset generated contained only states with 8+ pawns already placed, and a good size only for states with 10+, as solving states with fewer pawns with MinMax was unfeasible.
A solution for this was to include the extimated reward function learned so far into the *generate_dataset* script: if we have a reliable extimate for the reward on states with at least 10 pawns, then the MinMax algorithm can stop ad depth 10 and return the extimate, allowing to label states with fewer pawns. 

To generate this second dataset another variant of MinMax was used.

Of course it can be expected that the learned reward function for states with fewer pawns will be less reliable.

# Agent implementation
## Overview
The basic idea is to build the final agent as a MinMax with beam-search-like pruning, using the reward functon learned so far as the heuristic to drive its decisions.

In the most basic implementation the algorithm can work as MinMax but:
* Each time it visits a state, instead of recursively visiting all of its children it only visits MAX_NODES most promising ones, using the reward function to extimate how promising they are.
* When MAX_DEPTH is reached, the reward function is used to give a value to the nodes.

The specific parameters of the algorithm are found after some tries-and-errors, and are:
* MAX_DEPTH: 4 to 5 levels; MAX_NODES: 6 to 7.
* Radical pruning: when visiting a node, only the children with extimated reward over a certain threshold are visited, even if they are fewer then MAX_NODES. If the node has 0 promising children, then it is labelled with the reward function and discarted. This is done on the observation that the extimated reward rarely labels a *good* state as *bad*, while most errors are on the other way around, and this allows for early pruning of unpromising branches and saving time for more useful ones.
* If, in a given node, MAX_EVALS children with a good extimated reward are found, the other children are not even labelled by the reward function. This helps speeding up the process as the reward function is slow. The other children are still evaluated to check if they lead to direct winning or losing.
* If there are at least 9 pawns in the board, forget about pruning and work like standard MinMax.
* If there are 3 or fewer pawns on the table, use a fixed *not risky* strategy, placing pawns where they don't help making longer lines.



## States cache and RL layer
The agent algorithm makes use of a cache, like the one used to generate the dataset. This cache is volatile and is used only to avoid repeating the same computations more than once.

But it can be useful to keep in storage another, smaller dictionary of states, to keep track of discovered *good* or *bad states*.
This is done by unsupervised RL training, which in this case is a synonimous of *storing useful results from MinMax explorations*. 
Only states that are determinalistically solved as good or bad are stored, and this dictionary is used each time the reward function is to be computed on a child node. The idea is that if the algorithm finds one node that leads to victory or defeat, it stores it and makes sure that next times it will be spotted earlier.

The training of the RL layer has been done making the agent play against itself, against the random player and against a variant of itself where some extent of randomness was introduced. 

After some tries, the hit-ratio in the dictionary seems to be in the range of 0.005% to 0.02%, which is not too bad. A test will later show if this RL layer makes any difference.


## Wrapper
The agent is split into two classes: *QuartoAgent* and *RealAgent*.
*RealAgent* is the one implementing the algorithm described before, while *QuartoAgent* is a wrapper that converts the state from the format of the Quarto module provided and the format used by my agent.

All the interesting logic is kept in the *RealAgent* class.

## Tests
Here the results of some tests on the final agent are provided.
Testing the agent isn't an easy task since we don't have a deterministic agent like we had with Nim. A possible solution is to test it against versions of itself where some features are deactivated. This way it is possible to check if these features actually provide any advantage.

All tests are balanced, having the agent and the adversary play an even number of games as first and as second player, since the first player seem to have a higher chance of winning. Each test is run 100 times.

### Agent VS Random Agent
This tests shows how good the agent compare to a random strategy.

Results:
* **Win rate**: 100%

Comment:
Poor random agent.

### Agent vs Agent
Since the game has been solved and it showed that the first player, if it plays its moves right, always wins, it makes sense to try to make the agent fight with itself to see how often the one starting the game wins.

Results:
**Player 2 always win**


### Agent VS Semi-Blind Agent
If we take our agent and change the learned reward function weights with random ones, and remove the RL cache too, what remains is a Semi-Blind Agent that uses beam search without any significant heuristic. The agent still have some useful features: the reward function, while initialized with random parameters, can still detect states one move away from a victory/defeat; and it still uses MinMax for states with at least 9 pawns in it.

* **Agent winning rate**: 43%
* **Blind agent winning rate**: 26%
* **Ties**: 31%

### Agent VS No Cache
Results:
**Player 2 always win**


# Iusses and final considerations
I'm happy with the results of this agent, and I'm amazed that a simple hill climbing could find an useful heuristics for the beam search.
There still are some iusses with this project:
* **Slowness of the collapse state function**: this function is incredibly time consuming, and this forced me to abandon it in the final agent. It is a shame because it would have made caching and the RL layer much more efficient.
* **The reward function**: it was meant to be a *guide* for the exploration, giving an advantage over a blind exploration. To some extent it does this, but it is weaker than I'd have liked. There surely are better ways to train a function toward this goal.
* **The dataset used to train the reward function**: it probably needed more states, and most importantly more negative states to be recognized
* **Beam search might be too simple**: the beam search by itself is simple, at each node it chooses N promising children to be visited, and it cannot backtrack and change its decision once taken. This can be a iusse: a child can look promising using the heuristic function, but it is possible that visiting its branches it comes clear that the function misclassified it. A more advanced algorithm could choose to give up earlier visiting some branches and use the extra time available to visit other nodes.

