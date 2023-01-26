import quarto
from quarto_real_agent import QuartoRealAgent

class QuartoAgent(quarto.Player):
    def __init__(self, quarto: quarto.Quarto, skip_rl_layer = False, random_reward_function = False) -> None:
        super().__init__(quarto)
        self.chosen_piece = None
        self.realAgent = QuartoRealAgent(skip_rl_layer = skip_rl_layer, random_reward_function = random_reward_function)

    def save_cache(self):
        self.realAgent.save_cache()
    
    def get_agent_custom_realagent(quarto: quarto.Quarto, real_agent):
        q = QuartoAgent(quarto)
        q.realAgent = real_agent
        return q

    def choose_piece(self) -> int:
        if (self.chosen_piece!=None):
            return self.chosen_piece
        #Since it's the relationships between pieces that counts,
        #on the first move it doesn't change anything what piece we choose
        return 0 

    def place_piece(self) -> tuple[int, int]:
        usable_piece = self.get_game().get_selected_piece() #Index
        moves = []
        states = []

        board = self.get_game().get_board_status()
        usable_pieces = set([i for i in range(0,16)]) - set([usable_piece])

        for i in range(0,4):
            for j in range(0,4):
                if (board[i][j] in usable_pieces):
                    usable_pieces.remove(board[i][j])
        if (len(usable_pieces)==0):
            for i in range(0,4):
                for j in range(0,4):
                    if (board[j][i] == -1): 
                        return (i,j)
        for i in range(0,4):
            for j in range(0,4):
                if (board[i][j] != -1): continue
                for piece in usable_pieces:
                    moves.append((i,j,piece))
                    board[i][j] = usable_piece
                    states.append(self.convert_state_format(board, piece)) 
                    board[i][j] = -1
    
        solved = self.realAgent.solve_states(states)
        best = min(solved)
        move = moves[best[1]]
        self.chosen_piece = move[2]
        return (move[1],move[0])

    def convert_state_format(self, board, assigned_pawn):
        new_board = []
        for i in range(0,4):
            for j in range(i, -1, -1):
                new_board.append(board[j][i-j])
        new_board.append((board[3][1]))
        new_board.append((board[2][2]))
        new_board.append((board[1][3]))
        new_board.append((board[3][2]))
        new_board.append((board[2][3]))
        new_board.append((board[3][3]))
        return [new_board, assigned_pawn]