piece_value = {
    "K": 0, "Q": 9, "R": 5, "B": 3, "N": 3, "p": 1
}

win_reward = 1.0
draw_reward = 0.0
loss_reward = -1.0

class Reward():
    def __init__ (self, gs, move): # đối tượng ChessEngine.GameState, ChessEngine.Move
        self.gs = gs
        self.move = move
            
    def get_reward(self):
        # vì đã thực hiện makeMove nên đảo chiều người chơi
        # đảo ngược lại để tính điểm
        # tính điểm cho bên đen
        ######################
        if self.gs.checkMate:
            if self.gs.whiteToMove: 
                return win_reward
            else:
                return loss_reward
        elif self.gs.staleMate:
            return draw_reward

        if self.move.pieceCaptured != '--':
            val = piece_value[self.move.pieceCaptured[1]]
            if self.gs.whiteToMove:
                return float(val/39)
            else:
                return -float(val/39)
        return 0.0