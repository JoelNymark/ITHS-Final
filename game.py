# a otello game that can be played by two players or by a player and an AI agent

class othelloGame():

    board = [[0]*8 for _ in range(8)]
    board[3][3], board[4][4] = 1, 1
    board[3][4], board[4][3] = -1, -1

    def print_board(self):
        print('  1 2 3 4 5 6 7 8')
        for i in range(8):
            print(chr(ord('a')+i), end=' ')
            for j in range(8):
                if self.board[i][j] == 1:
                    print('o', end=' ')
                elif self.board[i][j] == -1:
                    print('x', end=' ')
                else:
                    print('.', end=' ')
            print()
        print('  1 2 3 4 5 6 7 8')

    def is_valid_move(self, x, y, player):
        if self.board[x][y] != 0:
            return False
        for dx, dy in (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1):
            i, j = x, y
            while True:
                i += dx
                j += dy
                if not (0 <= i < 8 and 0 <= j < 8):
                    break
                if self.board[i][j] == 0:
                    break
                if self.board[i][j] == player:
                    if i != x + dx or j != y + dy:
                        return True
                    break
        return False
    
    def current_player(self):
        return 1 if sum(map(sum, self.board)) == 0 else -1

    def player_make_move(self, x, y, player):
        self.board[x][y] = player
        for dx, dy in (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1):
            i, j = x, y
            while True:
                i += dx
                j += dy
                if not (0 <= i < 8 and 0 <= j < 8):
                    break
                if self.board[i][j] == 0:
                    break
                if self.board[i][j] == player:
                    if i != x + dx or j != y + dy:
                        while True:
                            i -= dx
                            j -= dy
                            if i == x and j == y:
                                break
                            self.board[i][j] = player
                    break
        
    def ai_make_move(self, action, player):
        # action is a tuple (x, y)
        # return the new state of the board and the reward of the move (the number of pieces flipped) and whether the game is over and the score of the game

        x, y = action
        self.board[x][y] = player
        reward = 0
        for dx, dy in (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1):
            i, j = x, y
            while True:
                i += dx
                j += dy
                if not (0 <= i < 8 and 0 <= j < 8):
                    break
                if self.board[i][j] == 0:
                    break
                if self.board[i][j] == player:
                    if i != x + dx or j != y + dy:
                        while True:
                            i -= dx
                            j -= dy
                            if i == x and j == y:
                                break
                            self.board[i][j] = player
                            reward += 1
                    break
        return reward, len(self.get_valid_moves(1)) == 0 and len(self.get_valid_moves(-1)) == 0, self.get_score(player)
    
        
    def get_valid_moves(self, player):
        valid_moves = []
        for i in range(8):
            for j in range(8):
                if self.is_valid_move(i, j, player):
                    valid_moves.append((i, j))
        return valid_moves
    
    def is_game_over(self):
        return len(self.get_valid_moves(1)) == 0 and len(self.get_valid_moves(-1)) == 0
    
    def get_winner(self):
        score = 0
        for i in range(8):
            for j in range(8):
                score += self.board[i][j]
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0
        
    def get_reward(self, player):
        score = 0
        for i in range(8):
            for j in range(8):
                score += self.board[i][j]
        if player == 1:
            return score
        else:
            return -score
        
    def get_score(self, player):
        # return the score of the game for the player

        player_1_score = 0
        player_2_score = 0
        for i in range(8):
            for j in range(8):
                if player == 1:
                    if self.board[i][j] == 1:
                        player_1_score += 1
                elif player == -1:
                    if self.board[i][j] == -1:
                        player_2_score += 1
                
        if player == 1:
            return player_1_score
        elif player == -1:
            return player_2_score
        else:
            return 0

    def get_state(self):
        return self.board
    

class human_Play():
    def __init__(self):
        pass

    def play(self):
        # play the game as 2 human players print the board after each move and print the move of each player can do as a number for the columns and a letter for the rows

        game = othelloGame()
        player = -1
        valid_moves = game.get_valid_moves(player)
            # print(f"Valid moves for player {player}: {valid_moves}")
        while True:
            print(f'score : x', game.get_score(-1))
            print(f'score : o', game.get_score(1))
            game.print_board()
            valid_moves = game.get_valid_moves(player)
            if not valid_moves:
                print('No valid moves for player', player)
                player = -player
                valid_moves = game.get_valid_moves(player)
                if not valid_moves:
                    break
            print('Valid moves for player', player, ':')
            for move in valid_moves:
                row_letter = chr(ord('a') + move[0])
                col_num = move[1] + 1
                print(row_letter + str(col_num), end=' ')
            print()
            while True:
                move = input('Your move? ')
                if move == 'pass':
                    break
                if len(move) != 2 or move[0] not in 'abcdefgh' or move[1] not in '12345678':
                    print('Invalid move')
                    continue
                x = ord(move[0]) - ord('a')
                y = int(move[1]) - 1
                if (x, y) not in valid_moves:
                    print('Invalid move')
                    continue
                break
            if move == 'pass':
                player = -player
                continue
            game.player_make_move(x, y, player)
            player = -player
        game.print_board()
        winner = game.get_winner()
        if winner == 1:
            print('Player 1 wins!')
        elif winner == -1:
            print('Player 2 wins!')
        else:
            print('Tie!')
