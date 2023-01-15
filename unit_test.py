import quarto_agent
import quarto


def print_collapsed(state):
        s = []
        for i in range(len(state)):
            if (state[i] < 0):
                s.append(str(state[i]))
            else:
                s.append(" " + str(state[i]))
        print("    " + s[0])
        print("   " + s[1] + " " + s[2])
        print(" " + s[3] + " " + s[4] + " " + s[5])
        print(s[6] + " " + s[7] + " " + s[8] + " " + s[9])
        print(" " + s[10] + " " + s[11] + " " + s[12])
        print("   " + s[13] + " " + s[14])
        print("    " + s[15])


def run_test(self, final):
        pawns = set([i for i in range(16)])
        board = np.ones(
            shape=(4, 4), dtype=int) * -1
        if (final):
            for i in range(0, 4):
                for j in range(0, 4):
                    board[i][j] = random.sample(pawns, 1)[0]
                    pawns.remove(board[i][j])
        else:
            one_ = False
            for i in range(0, 4):
                for j in range(0, 4):
                    if (random.randint(0, 1) == 0):
                        board[i][j] = random.sample(pawns, 1)[0]
                        pawns.remove(board[i][j])
                    else:
                        one_ = True
            if (not one_):
                pawns.add(board[1][1])
                board[1][1] = -1
        self.get_game().setboard(board)
        win = self.get_game().check_winner() != -1
        for i in pawns:
            converted = self.convert_state_format(board, i)
            collapsed = collapse(converted[0], converted[1])
            full, winning = checkState(collapsed[0])
            print(f"{winning} - {win}")
            assert win == winning
            if (winning != win):
                print(f"{winning} - {win}")
                print("Board 2")
                self.get_game().print()
                print("Converted")
                QuartoAgent.print_collapsed(converted[0])
                print("Collapsed")
                # QuartoAgent.print_collapsed(collapsed[0])
                exit()

    def test(self):

        for i in range(100):
            print(i)
            self.run_test(False)
            self.run_test(True)
        exit()


def test_convert_state_format():
    pass
