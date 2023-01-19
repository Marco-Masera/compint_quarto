import random
import quarto
from quarto_agent import QuartoAgent

class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        return random.randint(0, 15)

    def place_piece(self) -> tuple[int, int]:
        return random.randint(0, 3), random.randint(0, 3)


#Simply test against the random agent
def test_against_random_agent():
    print("Testing QuartoAgent against random agent")
    random.seed()
    wins = [0, 0, 0]
    for i in range(30):
        game = quarto.Quarto(no_print=True)
        q = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
        rando = RandomPlayer(game)
        game.set_players((q, rando))
        winner = game.run()
        print(winner)
        wins[winner+1] += 1
    print (wins)


#This class provides tests for the agent
class TestAgent():
    def run_test(iterations):
        random.seed()
        print("Testing QuartoAgent against random agent")
        print(f"Result: {TestAgent.test_against_random_agent(iterations)}")

        print("Testing QuartoAgent against itself with random reward genome and no cache")
        print(f"Result: {TestAgent.test_against_random_reward(iterations, False)}")

        print("Testing QuartoAgent against itself with random reward genome")
        print(f"Result: {TestAgent.test_against_random_reward(iterations)}")

        print("Testing QuartoAgent against itself with cache disabled")
        print(f"Result: {TestAgent.test_against_no_cache(iterations)}")

        print("Testing QuartoAgent against itself with random play parameters")
        print(f"Result: {TestAgent.test_against_no_params(iterations)}")

    
    def test_against_random_agent(iterations):
        wins = 0
        ties = 0
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            q2 = RandomPlayer(game)
            game.set_players((q1, q2))
            winner = game.run()
            if (winner==0): wins += 1
            if (winner==-1): ties += 1
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            q2 = RandomPlayer(game)
            game.set_players((q2, q1))
            winner = game.run()
            if (winner==1): wins += 1
            if (winner==-1): ties += 1
        return {
            "victory": wins / (int(iterations/2)*2),
            "ties": ties / (int(iterations/2)*2)
        }

    #Test against the QuartoAgent with StateReward initialized with random parameters
    def test_against_no_params(iterations):
        wins = 0
        ties = 0
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            q2 = QuartoAgent.get_agent_random_genome(game, use_cache = False, save_states = False, debug_use_random_reward=False)
            game.set_players((q1, q2))
            winner = game.run()
            if (winner==0): wins += 1
            if (winner==-1): ties += 1
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            q2 = QuartoAgent.get_agent_random_genome(game, use_cache = False, save_states = False, debug_use_random_reward=False)
            game.set_players((q2, q1))
            winner = game.run()
            if (winner==1): wins += 1
            if (winner==-1): ties += 1
        return {
            "victory": wins / (int(iterations/2)*2),
            "ties": ties / (int(iterations/2)*2)
        }

    #Test against the QuartoAgent with StateReward initialized with random genome
    def test_against_no_cache(iterations):
        wins = 0
        ties = 0
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            q2 = QuartoAgent.get_agent(game, use_cache = False, save_states = False, debug_use_random_reward=False)
            game.set_players((q1, q2))
            winner = game.run()
            if (winner==0): wins += 1
            if (winner==-1): ties += 1
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            q2 = QuartoAgent.get_agent(game, use_cache = False, save_states = False, debug_use_random_reward=False)
            game.set_players((q2, q1))
            winner = game.run()
            if (winner==1): wins += 1
            if (winner==-1): ties += 1
        return {
            "victory": wins / (int(iterations/2)*2),
            "ties": ties / (int(iterations/2)*2)
        }

    #Test against the QuartoAgent with StateReward initialized with random genome
    def test_against_random_reward(iterations, use_cache = True):
        wins = 0
        ties = 0
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            q_rand = QuartoAgent.get_agent(game, use_cache = use_cache, save_states = False, debug_use_random_reward=True)
            game.set_players((q, q_rand))
            winner = game.run()
            if (winner==0): wins += 1
            if (winner==-1): ties += 1
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            q_rand = QuartoAgent.get_agent(game, use_cache = use_cache, save_states = False, debug_use_random_reward=True)
            game.set_players((q_rand, q))
            winner = game.run()
            if (winner==1): wins += 1
            if (winner==-1): ties += 1
        return {
            "victory": wins / (int(iterations/2)*2),
            "ties": ties / (int(iterations/2)*2)
        }

