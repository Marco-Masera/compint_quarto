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
        for _ in range(1):
            print("Ciao")

            #print("Testing QuartoAgent against random agent")
            #TestAgent.test_vs_random(iterations, print_=print_, save=save)
            #print("Testing QuartoAgent against itself")
            #TestAgent.test_vs_itself(iterations, print_=print_, save=save)
            #print("Testing QuartoAgent against blind agent")
            #TestAgent.test_vs_blind_agent(iterations, print_=print_, save=save)
            print("Testing QuartoAgent against no RL layer")
            TestAgent.test_vs_blind_agent(iterations, print_=print_, save=save, random_reward=False)

    def test_vs_blind_alt(iterations, skip_rl_layer=True, random_reward=True, print_=True, save = False):
        params = [{"DEPTHS" :         [0]*4 + [3 for _ in range(5)] + [13]*7, #!!!
                "MAX_NODES" :      [0]*4 + [7 for _ in range(5)] + [-1]*7, 
                "MAX_EVALS" :      [0]*4 + [28 for _ in range(5)] + [-1]*7} ,

                {"DEPTHS" :         [0]*4 + [4 for _ in range(5)] + [13]*7, #!!!
                "MAX_NODES" :      [0]*4 + [5 for _ in range(5)] + [-1]*7, 
                "MAX_EVALS" :      [0]*4 + [28 for _ in range(5)] + [-1]*7},
                
                {"DEPTHS" :         [0]*4 + [4 for _ in range(5)] + [13]*7, 
                "MAX_NODES" :      [0]*4 + [4 for _ in range(5)] + [-1]*7, 
                "MAX_EVALS" :      [0]*4 + [150 for _ in range(5)] + [-1]*7}]
       
        results = [(0,0,0) for _ in range(len(params)) ]
        for i in range(len(params)):
            print(f"Iteration {i}")
            ties = 0
            win_1 = 0
            win_2 = 0
            ev_1 = 0; ev_2 = 0;
            for k in range(iterations):
                game = quarto.Quarto(no_print=True)
                q1 = QuartoAgent.get_agent_custom_realagent(game, QuartoRealAgent(params=params[i]))
                q2 = QuartoAgent.get_agent_custom_realagent(game,  QuartoRealAgent(params=params[0], random_reward_function=True))
                game.set_players((q1, q2))
                winner = game.run()
                if (print_): print(f"Iteration {k} - w: {winner}")
                if (winner == 0): win_1+=1
                if (winner==-1): ties+= 1
                ev_1 += q1.realAgent.n_eval; ev_2 += q2.realAgent.n_eval
            for k in range(iterations):
                game = quarto.Quarto(no_print=True)
                q1 = QuartoAgent.get_agent_custom_realagent(game, QuartoRealAgent(params=params[i]))
                q2 = QuartoAgent.get_agent_custom_realagent(game,  QuartoRealAgent(params=params[0], random_reward_function=True))
                game.set_players((q2, q1))
                winner = game.run()
                ev_1 += q1.realAgent.n_eval; ev_2 += q2.realAgent.n_eval
                if (print_): print(f"Iteration {k} - w: {winner}")
                if (winner == 1): win_1+=1
                if (winner==-1): ties+= 1
            for k in range(iterations):
                game = quarto.Quarto(no_print=True)
                q1 = QuartoAgent.get_agent_custom_realagent(game, QuartoRealAgent(params=params[i]))
                q2 = QuartoAgent.get_agent_custom_realagent(game,  QuartoRealAgent(params=params[1], random_reward_function=True))
                game.set_players((q1, q2))
                winner = game.run()
                ev_1 += q1.realAgent.n_eval; ev_2 += q2.realAgent.n_eval
                if (print_): print(f"Iteration {k} - w: {winner}")
                if (winner == 0): win_1+=1
                if (winner==-1): ties+= 1
            for k in range(iterations):
                game = quarto.Quarto(no_print=True)
                q1 = QuartoAgent.get_agent_custom_realagent(game, QuartoRealAgent(params=params[i]))
                q2 = QuartoAgent.get_agent_custom_realagent(game,  QuartoRealAgent(params=params[1], random_reward_function=True))
                game.set_players((q2, q1))
                winner = game.run()
                ev_1 += q1.realAgent.n_eval; ev_2 += q2.realAgent.n_eval
                if (print_): print(f"Iteration {k} - w: {winner}")
                if (winner == 1): win_1+=1
                if (winner==-1): ties+= 1
            for k in range(iterations):
                game = quarto.Quarto(no_print=True)
                q1 = QuartoAgent.get_agent_custom_realagent(game, QuartoRealAgent(params=params[i]))
                q2 = QuartoAgent.get_agent_custom_realagent(game,  QuartoRealAgent(params=params[2], random_reward_function=True))
                game.set_players((q1, q2))
                winner = game.run()
                ev_1 += q1.realAgent.n_eval; ev_2 += q2.realAgent.n_eval
                if (print_): print(f"Iteration {k} - w: {winner}")
                if (winner == 0): win_1+=1
                if (winner==-1): ties+= 1
            for k in range(iterations):
                game = quarto.Quarto(no_print=True)
                q1 = QuartoAgent.get_agent_custom_realagent(game, QuartoRealAgent(params=params[i]))
                q2 = QuartoAgent.get_agent_custom_realagent(game,  QuartoRealAgent(params=params[2], random_reward_function=True))
                game.set_players((q2, q1))
                winner = game.run()
                ev_1 += q1.realAgent.n_eval; ev_2 += q2.realAgent.n_eval
                if (print_): print(f"Iteration {k} - w: {winner}")
                if (winner == 1): win_1+=1
                if (winner==-1): ties+= 1
            results[i] = (results[i][0] + win_1, results[i][1]+ties, 0)
            print(f"End iteration: {win_1} - {ties} - Evaluations {ev_1} {ev_2}")
        for index, result in enumerate(results):
            print(f"Result {index}: {result[0]} - {result[1]}")

    def test_vs_blind_agent(iterations, skip_rl_layer=True, random_reward=True, print_=True, save = False):
        wins = 0; ties = 0
        ev_1 = 0; ev_2 = 0
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent(game)
            q2 = QuartoAgent(game, skip_rl_layer=skip_rl_layer, random_reward_function=random_reward)
            game.set_players((q1, q2))
            winner = game.run()
            ev_1 += q1.realAgent.n_eval; ev_2 += q2.realAgent.n_eval
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 0): wins+=1
            if (winner==-1): ties+= 1
            if (save): q2.save_cache()
        
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent(game)
            q2 = QuartoAgent(game, skip_rl_layer=skip_rl_layer, random_reward_function=random_reward)
            game.set_players((q2, q1))
            ev_1 += q1.realAgent.n_eval; ev_2 += q2.realAgent.n_eval
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 1): wins+=1
            if (winner==-1): ties+= 1
            if (save): q1.save_cache()
        print(f"Agent won {wins} over {int(iterations/2)*2}. Ties: {ties}")
        print(f"The random reward evaluated {ev_2} states, the standard one {ev_1}")

    def test_vs_itself(iterations, print_=True, save = False):
        wins = 0; ties = 0
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent(game)
            q2 = QuartoAgent(game)
            game.set_players((q1, q2))
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 0): wins+=1
            if (winner==-1): ties+= 1
            if (save): q2.save_cache()
        
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
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
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent(game)
            q2 = RandomPlayer(game)
            game.set_players((q1, q2))
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 0): wins+=1
            if (winner==-1): ties+= 1
            if (save): q1.save_cache()
        for i in range(int(iterations/2)):
            game = quarto.Quarto(no_print=True)
            q1 = QuartoAgent(game)
            q2 = RandomPlayer(game)
            game.set_players((q2, q1))
            winner = game.run()
            if (print_): print(f"Iteration {i} - w: {winner}")
            if (winner == 1): wins+=1
            if (winner==-1): ties+= 1
            if (save): q1.save_cache()
        print(wins); print(ties)
