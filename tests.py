import random
import quarto
from quarto_agent import QuartoAgent

#This file is meant to test the agent quality


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
        q.new_match()
        rando = RandomPlayer(game)
        game.set_players((q, rando))
        winner = game.run()
        print(winner)
        if (winner != -1):
            q.end_match(winner==0)
        wins[winner+1] += 1
    print (wins)

#Test against the QuartoAgent with StateReward initialized with random genome
def test_against_random_reward():
    random.seed()
    print("Testing QuartoAgent against itself with random reward genome")
    wins = [0, 0, 0]
    for i in range(10):
        game = quarto.Quarto(no_print=True)
        q = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
        q.new_match()
        q_rand = QuartoAgent.get_agent(game, use_cache = True, save_states = False, debug_use_random_reward=True)
        q_rand.new_match()
        game.set_players((q, q_rand))
        winner = game.run()
        if (winner != -1):
            q.end_match(winner==0)
            q_rand.end_match(winner==1)
        print(winner)
        wins[winner+1] += 1
    print (wins)
    print("Now reversed")
    wins = [0, 0, 0]
    for i in range(10):
        game = quarto.Quarto(no_print=True)
        q = QuartoAgent.get_agent(game, use_cache = True, save_states = False, debug_use_random_reward=True)
        q_rand = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
        game.set_players((q_rand, q))
        winner = game.run()
        print(winner)
        wins[winner+1] += 1
    print (wins)


def test_cache_vs_no_cache():
    random.seed()
    print("Testing QuartoAgent cache vs no cache")
    wins = [0, 0, 0]
    for i in range(10):
        game = quarto.Quarto(no_print=True)
        q = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
        q.new_match()
        q2 = QuartoAgent.get_agent(game, use_cache = False, save_states = False)
        game.set_players((q, q2))
        winner = game.run()
        if (winner != -1):
            q.end_match(winner==0)
        print(winner)
        wins[winner+1] += 1
    print (wins)

def test_cache_vs_cache():
    random.seed()
    print("Testing QuartoAgent cache vs itself")
    wins = [0, 0, 0]
    for i in range(10):
        game = quarto.Quarto(no_print=True)
        q = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
        q.new_match()
        q2 = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
        q2.new_match()
        game.set_players((q, q2))
        winner = game.run()
        if (winner != -1):
            q.end_match(winner==0)
        print(winner)
        wins[winner+1] += 1
    print (wins)
    

def test_single_match():
    random.seed()
    game = quarto.Quarto(no_print = True)
    q = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
    q.new_match()
    rando = RandomPlayer(game)
    game.set_players((q, rando))
    winner = game.run()
    if (winner != -1):
        q.end_match(winner==0)
    print(winner)