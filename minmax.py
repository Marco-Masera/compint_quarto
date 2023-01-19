import random
import quarto
import copy 
from quarto_agent import QuartoAgent


class MinMaxPlayer(quarto.Player):
    DEPTH = 9
    def __init__(self, quarto: quarto.Quarto) -> None:
        self.chosen_piece = None
        self.used_mm = False
        super().__init__(quarto)
    def choose_piece(self) -> int:
        if (self.chosen_piece!=None):
            return self.chosen_piece
        return 0
    def place_piece(self) -> tuple[int, int]:
        board = self.get_game().get_board_status()
        size = MinMaxPlayer.count_size(board)
        if (size >= MinMaxPlayer.DEPTH):
            self.used_mm = True
            move = self.move(board, self.get_game().get_selected_piece())[1]
            self.chosen_piece = move[2]
            return move[1], move[0]
        self.chosen_piece = random.sample(sorted(MinMaxPlayer.pieces_available(board,self.get_game().get_selected_piece())), k=1)[0]
        return random.randint(0, 3), random.randint(0, 3)
    
    def move(self, board, piece):
        available = list(MinMaxPlayer.pieces_available(board, piece))
        if (MinMaxPlayer.is_win(board)):
            return (-1, None)
        if (len(available)==0):
            return (0, None)
        best_move = None 
        best_p = 1000
        for i in range(4):
            for j in range(4):
                for p in available:
                    if (board[i][j]!=-1):
                        continue
                    copied = copy.copy(board)
                    copied[i][j] = piece 
                    result = self.move(copied, p)[0]
                    if (result < best_p):
                        best_p = result 
                        best_move = (i, j, p)
        return (-best_p, best_move)

    def count_size(board):
        count = 0
        for l in board:
            for b in l:
                if (b!= -1):
                    count += 1
        return count
    def pieces_available(board, p):
        pieces = set([i for i in range(16)])
        for i in range(4):
            for j in range(4):
                if (board[i][j]!= -1):
                    pieces.remove(board[i][j])
        pieces.remove(p)
        return pieces
    def is_win(board):
        g = quarto.Quarto()
        g.setboard(board, 0)
        if (g.check_winner()==0):
            return True 
        return False

def run():
    game = quarto.Quarto(no_print=True)
    q1 = QuartoAgent.get_agent(game, use_cache = True, save_states = True)
    q1.new_match()
    q2 = MinMaxPlayer(game)
    game.set_players((q1, q2))
    winner = game.run()
    if (winner==1):
        q1.end_match(False)
    return (winner, q2.used_mm)

def main():
    wins_1 = 0; wins_2 = 0; mm_uses = 0
    for i in range(40):
        r = run()
        if (r[0]==1):
            wins_1 += 1
        if (r[0]==0):
            wins_2 += 1
        if (r[1]==True):
            mm_uses += 1
        print(r)
    print(wins_1); print(wins_2); print(mm_uses)
main()