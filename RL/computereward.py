piece_value = {
    "K": 0, "Q": 9, "R": 5, "B": 3, "N": 3, "p": 1
}

win_reward = 1.0
draw_reward = 0.0
loss_reward = -1.0

class Reward():
    def __init__ (self, gs, move, captured=None): # đối tượng ChessEngine.GameState, ChessEngine.Move
        self.gs = gs
        self.move = move
        self.captured = captured
            
    def get_reward(self):
        # tính điểm cho bên đen
        ######################
        # nước tiếp theo chiếu hết
        if self.gs.checkMate:
            if not self.gs.whiteToMove: # đang lượt đen
                print(f'win {win_reward}')
                return win_reward
            else: # đang lượt trắng
                print(f'loss {loss_reward}')
                return loss_reward
        elif self.gs.staleMate:
            print(f'draw {draw_reward}')
            return draw_reward

        # nước tiếp theo ăn quân
        if self.move.pieceCaptured != '--':
            val = piece_value[self.move.pieceCaptured[1]]
            if not self.gs.whiteToMove: # đang lượt đen
                print(f'take {float(val/39)}')
                return float(val/39)
        
        # nước tiếp theo trắng ăn quân
        if self.captured != None and self.gs.whiteToMove:
            val = piece_value[self.captured]
            return -float(val/39)
        return 0.0