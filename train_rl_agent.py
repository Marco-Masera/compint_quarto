import random
import quarto
from quarto_agent import QuartoAgent

class RandomPlayer(quarto.Player):
    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)
    def choose_piece(self) -> int:
        return random.randint(0, 15)
    def place_piece(self) -> tuple[int, int]:
        return random.randint(0, 3), random.randint(0, 3)

#This class provides methods to train the RL part of the agent
class TrainRLAgent:
    def train(iterations):
        random.seed()
        TrainRLAgent.train_against_random(iterations)
        TrainRLAgent.train_against_itself(iterations)
        

    def train_against_itself(iterations):
        print("Starting training against itself")
        for _ in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent.get_agent(game, use_cache = True, save_states = True)
            q1.new_match()
            q2 = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            game.set_players((q1, q2))
            winner = game.run()
            if (winner != -1):
                q1.end_match(winner==0)
        for _ in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent.get_agent(game, use_cache = True, save_states = True)
            q1.new_match()
            q2 = QuartoAgent.get_agent(game, use_cache = True, save_states = False)
            game.set_players((q2, q1))
            winner = game.run()
            if (winner != -1):
                q1.end_match(winner==1)
        print(f"Ended training after {iterations} matches")
            

    def train_against_random(iterations):
        print("Starting training against random agent")
        lost = 0
        for _ in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q = QuartoAgent.get_agent(game, use_cache = True, save_states = True)
            rando = RandomPlayer(game)
            game.set_players((q, rando))
            q.new_match()
            winner = game.run()
            if (winner == 1):
                #Do the training only if agent lost
                q.end_match(False)
                lost += 1
        for _ in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q = QuartoAgent.get_agent(game, use_cache = True, save_states = True)
            rando = RandomPlayer(game)
            game.set_players((rando, q))
            q.new_match()
            winner = game.run()
            if (winner == 0):
                #Do the training only if agent lost
                q.end_match(False)
                lost += 1
        print (f"Over {iterations} matches, player lost {lost} times")
