import os, sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
import importlib
import ChessEngine
import computereward
import replaybuffer
import torch
import copy 

importlib.reload(ChessEngine)
importlib.reload(computereward)
importlib.reload(replaybuffer)

# xây dựng trạng thái cờ vua
class Env():
    def __init__(self):
        self.gs = ChessEngine.GameState()
        self.__initial__gs = copy.deepcopy(self.gs)
        self.action_size = len(self.all_moves())
        self.moveid_to_index = {mid: idx for idx, mid in enumerate(self.all_moves())}
        self.index_to_moveid = {idx: mid for idx, mid in enumerate(self.all_moves())}
        
    def reset(self):
        self.gs = copy.deepcopy(self.__initial__gs)
    
    def all_moves(self):
        return sorted(
            sr * 1000 + sc * 100 + er * 10 + ec # startrow, startcol, endrow, endcol
            for sr in range(8)
            for sc in range(8)
            for er in range(8)
            for ec in range(8)
            if not (sr == er and sc == ec)
        )
    
    def legal_moves_mask(self):
        legal_mask = torch.zeros(self.action_size, dtype=torch.bool)
        for m in self.gs.getValidMoves():
            mid = m.moveID
            if mid in self.moveid_to_index:
                legal_mask[self.moveid_to_index[mid]] = True
        return legal_mask

    def step(self, index, captured=None): # đầu vào là chỉ số nước đi và nước trắng đi có ăn quân hay ko
        legal_mask = self.legal_moves_mask()
        if legal_mask[index]:
            moveid = self.index_to_moveid[index]
            s = str(moveid).zfill(4)
            sr, sc, er, ec = (int(c) for c in s)
            action = ChessEngine.Move((sr, sc), (er, ec), self.gs.board)
            r = computereward.Reward(self.gs, action, captured)
            self.gs.makeMove(action)

            done = self.gs.checkMate or self.gs.staleMate
            return self.state_to_tensor(), r.get_reward(), done, legal_mask
        else:
            return self.state_to_tensor(), -10, True, legal_mask

    def state_to_tensor(self): # từ bàn cờ chuyển thành đầu vào cho DQN
        mapping = ["wp","wN","wB","wR","wQ","wK","bp","bN","bB","bR","bQ","bK"] 
        planes = torch.zeros((12, 8, 8), dtype=torch.float32)
        
        for r in range(8):
            for c in range(8):
                sq = self.gs.board[r][c]
                if sq in mapping:
                    planes[(mapping.index(sq), r, c)] = 1.0

        # side to move
        stm = torch.full((1, 8, 8), float(self.gs.whiteToMove), dtype=torch.float32)
        return torch.cat([planes, stm], dim=0) # (13,8,8)