import logging
import argparse
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
        q = QuartoAgent(game)
        rando = RandomPlayer(game)
        game.set_players((q, rando))
        winner = game.run()
        print(winner)
        wins[winner+1] += 1
    print (wins)

#Test against the QuartoAgent with StateReward initialized with random genome
def test_against_random_reward():
    random.seed()
    print("Testing QuartoAgent against itself with random reward genome")
    wins = [0, 0, 0]
    for i in range(30):
        game = quarto.Quarto(no_print=True)
        q = QuartoAgent(game)
        q_rand = QuartoAgent(game, True)
        game.set_players((q, q_rand))
        winner = game.run()
        print(winner)
        wins[winner+1] += 1
    print (wins)
    

def test_single_match():
    random.seed()
    game = quarto.Quarto(no_print=True)
    q = QuartoAgent(game)
    rando = RandomPlayer(game)
    game.set_players((q, rando))
    winner = game.run()
    print(winner)