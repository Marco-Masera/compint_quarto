import random
import quarto
from quarto_agent import QuartoAgent
from quarto_real_agent import QuartoRealAgent
class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        return random.randint(0, 15)

    def place_piece(self) -> tuple[int, int]:
        return random.randint(0, 3), random.randint(0, 3)

#This class provides tests for the agent
class TestAgent():
    def run_test(iterations):
        random.seed()
        print_ = True; save = False
        
        print("Ciao")
        print("Testing QuartoAgent against random agent")
        TestAgent.test_vs_random(iterations, print_=print_, save=save)
        print("Testing QuartoAgent against itself")
        TestAgent.test_vs_itself(iterations, print_=print_, save=save)
        print("Testing QuartoAgent against blind agent")
        TestAgent.test_vs_blind_agent(iterations, print_=print_, save=save)
        print("Testing QuartoAgent against no RL layer")
        TestAgent.test_vs_blind_agent(iterations, print_=print_, save=save, random_reward=False)


    def test_vs_blind_agent(iterations, skip_rl_layer=True, random_reward=True, print_=True, save = False):
        wins = 0; ties = 0
        for i in range(int(iterations/2)):
            game = quarto.Quarto()
            q1 = QuartoAgent(game)
            q2 = QuartoAgent(game, skip_rl_layer=skip_rl_layer, random_reward_function=random_reward)
            game.set_players((q1, q2))
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 0): wins+=1
            if (winner==-1): ties+= 1
            if (save): q2.save_cache()
        
        for i in range(int(iterations/2)):
            game = quarto.Quarto()
            q1 = QuartoAgent(game)
            q2 = QuartoAgent(game, skip_rl_layer=skip_rl_layer, random_reward_function=random_reward)
            game.set_players((q2, q1))
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 1): wins+=1
            if (winner==-1): ties+= 1
            if (save): q1.save_cache()
        print(f"Agent won {wins} over {int(iterations/2)*2}. Ties: {ties}")

    def test_vs_itself(iterations, print_=True, save = False):
        wins = 0; ties = 0
        for i in range(int(iterations/2)):
            game = quarto.Quarto()
            q1 = QuartoAgent(game)
            q2 = QuartoAgent(game)
            game.set_players((q1, q2))
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 0): wins+=1
            if (winner==-1): ties+= 1
            if (save): q2.save_cache()
        
        for i in range(int(iterations/2)):
            game = quarto.Quarto()
            q1 = QuartoAgent(game)
            q2 = QuartoAgent(game)
            game.set_players((q1, q2))
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 0): wins+=1
            if (winner==-1): ties+= 1
            if (save): q1.save_cache()
        print(f"First player won {wins}. Ties: {ties}")

    def test_vs_random(iterations, print_=True, save=False):
        wins = 0; ties = 0
        for i in range(int(iterations/2)):
            game = quarto.Quarto()
            q1 = QuartoAgent(game)
            q2 = RandomPlayer(game)
            game.set_players((q1, q2))
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 0): wins+=1
            if (winner==-1): ties+= 1
            if (save): q1.save_cache()
        for i in range(int(iterations/2)):
            game = quarto.Quarto()
            q1 = QuartoAgent(game)
            q2 = RandomPlayer(game)
            game.set_players((q2, q1))
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 1): wins+=1
            if (winner==-1): ties+= 1
            if (save): q1.save_cache()
        print(wins); print(ties)
