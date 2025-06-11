import importlib
from . import ChessEnv
import ChessEngine
from . import network
import torch
import os

importlib.reload(ChessEnv)
importlib.reload(ChessEngine)
importlib.reload(network)

# khởi tạo môi trường và mô hình
env = ChessEnv.Env()

device = torch.device('cpu')
# device = torch.device('cpu')

print(f'device {device}')

print(f'env.action_size {env.action_size}')
q_net = network.DQN(env.action_size).to(device)
model_path = os.path.join(os.path.dirname(__file__), 'DQN.pth')
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    q_net.load_state_dict(checkpoint)
    print("Đã load model")
else:
    print("Không tìm thấy model")

def state_to_tensor(state, whiteToMove): # từ bàn cờ chuyển thành đầu vào cho DQN
    mapping = ["wp","wN","wB","wR","wQ","wK","bp","bN","bB","bR","bQ","bK"] 
    planes = torch.zeros((12, 8, 8), dtype=torch.float32)
    
    for r in range(8):
        for c in range(8):
            sq = state[r][c]
            if sq in mapping:
                planes[(mapping.index(sq), r, c)] = 1.0

    # side to move
    stm = torch.full((1, 8, 8), float(whiteToMove), dtype=torch.float32)
    return torch.cat([planes, stm], dim=0) # (13,8,8)

def legal_moves_mask(env, validmoves):
    legal_mask = torch.zeros(env.action_size, dtype=torch.bool)
    for m in validmoves:
        mid = m.moveID
        if mid in env.moveid_to_index:
            legal_mask[env.moveid_to_index[mid]] = True
    return legal_mask

def BestRLMove(gs):
    state = gs.board
    whiteToMove = gs.whiteToMove
    validmoves = gs.getValidMoves()

    # input
    state_tensor = state_to_tensor(state, whiteToMove)   
    legal_mask = legal_moves_mask(env, validmoves)

    with torch.no_grad():
        q_values = q_net(state_tensor.unsqueeze(0).to(device)) # action_size
        q_values[~legal_mask.unsqueeze(0)] = -torch.inf
        best_action_index = q_values.argmax()
    
    best_action_mid = env.index_to_moveid[best_action_index.item()]
    s = str(best_action_mid).zfill(4)
    sr, sc, er, ec = map(int, s)
    return ChessEngine.Move((sr, sc), (er, ec), gs.board)
