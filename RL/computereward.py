import copy
piece_value = {
    "K": 0, "Q": 9, "R": 5, "B": 3, "N": 3, "p": 1
}

class Reward():
    def __init__ (self, gs, move, white_capture=None): # đối tượng ChessEngine.GameState, ChessEngine.Move
        self.gs = gs
        self.move = move
        self.white_capture = white_capture
        next_gs = copy.deepcopy(gs)
        next_gs.makeMove(move)
        self.next_gs = next_gs
        
    def get_reward(self):
        # tính điểm cho bên đen
        #####################
        # lượt của đen
        white_moved = 0.0
        next_black_move = 0.0
        if self.white_capture != None:
            val = piece_value[self.white_capture]
            white_moved = -float(val/39)

        # nước tiếp theo ăn quân
        if self.move.pieceCaptured != '--':
            val = piece_value[self.move.pieceCaptured[1]]
            next_black_move = float(val/39)
        
        if self.white_capture != None and self.move.pieceCaptured != '--':
            print(f'total sum reward {white_moved + next_black_move}')
        return white_moved + next_black_move