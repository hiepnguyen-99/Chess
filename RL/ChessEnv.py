import importlib
import ChessEngine
import RL.computereward as computereward
import RL.replaybuffer as replaybuffer
import torch
import copy 
from Minimax import SmartMoveFinder
from Minimax import Evaluate

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
        # print (f'self.gs.getValidMoves() {len(self.gs.getValidMoves())}')
        for m in self.gs.getValidMoves():
            mid = m.moveID
            # print(mid)
            if mid in self.moveid_to_index:
                legal_mask[self.moveid_to_index[mid]] = True

                # print(f'legal_mask[self.moveid_to_index[mid]] {legal_mask[self.moveid_to_index[mid]]}')

        # print (legal_mask[75:85])
        return legal_mask

    def step(self, index, white_capture=None, lenlegalmove=None): # đầu vào là chỉ số nước đi và nước trắng đi có ăn quân hay ko
        before_legal_mask = self.legal_moves_mask()
        state = self.state_to_tensor()
        index = index.item() if torch.is_tensor(index) else index 
        # true_indices = torch.nonzero(before_legal_mask, as_tuple=False).squeeze().tolist()
        # print(f'Các chỉ số có giá trị True trong before_legal_mask: {true_indices}')

        # for idx in true_indices:
        #     moveid = self.index_to_moveid[idx]
        #     print(f'Index: {idx} -> MoveID: {moveid}')

        # hết nước đi hợp lệ 
        # loss
        if lenlegalmove == 0 and self.gs.checkMate:
            print(f'hết nước đi hợp lệ, self.gs.checkMate, -1' )
            return state, -1, True, before_legal_mask, before_legal_mask
        # draw
        if lenlegalmove == 0 and self.gs.staleMate:
            print(f'hết nước đi hợp lệ, self.gs.staleMate, 0' )
            return state, 0, True, before_legal_mask, before_legal_mask

        # print(f'legal_mask[{index}] {legal_mask[index]}')
        if before_legal_mask[index]:
            moveid = self.index_to_moveid[index]
            s = str(moveid).zfill(4)
            sr, sc, er, ec = (int(c) for c in s)
            action = ChessEngine.Move((sr, sc), (er, ec), self.gs.board)
            r = computereward.Reward(self.gs, action, white_capture).get_reward()
            self.gs.makeMove(action)
            done = self.gs.checkMate or self.gs.staleMate

            # win
            if done and self.gs.checkMate:
                print(f'done = self.gs.checkMate or self.gs.staleMate {done}, reward 1')
                return state, 1, True, before_legal_mask, before_legal_mask
            # draw
            elif done and self.gs.staleMate:
                print(f'done = self.gs.checkMate or self.gs.staleMate {done}, reward 0')
                return state, 0, True, before_legal_mask, before_legal_mask
            else:
                # trắng đi
                if Evaluate.check_mid_game(self.gs):
                    SmartMoveFinder.DEPTH = 4
                else:
                    SmartMoveFinder.DEPTH = 3
                MinimaxMove = SmartMoveFinder.findBestMinimaxMove(self.gs, self.gs.getValidMoves())
                if MinimaxMove is None:
                    MinimaxMove = SmartMoveFinder.findRandomMove(self.gs.getValidMoves())
                self.gs.makeMove(MinimaxMove)

                # next_state của đen
                next_state = self.state_to_tensor()
                next_legal_mask = self.legal_moves_mask()

                # trả lại nước đi cho trắng
                self.gs.undoMove()

                return next_state, r, False, before_legal_mask, next_legal_mask
        
        else:
            # print(f'inlegal move -10')
            return state, -10, True, before_legal_mask, before_legal_mask

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