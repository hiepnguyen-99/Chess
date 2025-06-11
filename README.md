# Chess
**Members**: 
- Nguyễn Xuân Hiệp - 22022591
- Phó Viết Tiến Anh - 22022568
- Nguyễn Phương Đông - 22022593

University of Engineering and Technology (UET-VNU)

## Introduction
This project builts a chess game integrating Minimax algorithm and Deep Q-Network of Reinforcement Learning model

## Features
1. **User Interface**:
    - Simple 2D graphics use Pygame platform.
    - Displays moved piece and board state.

2. **Minimax Alpha-Beta Pruning**:
    - Configurable search depth.
    - Evaluation function based on material count, piece positioning, and king safety.

3. **Deep Q-Network (DQN)**:
    - Custom chess environment.
    - Input: one-hot encoded board state matrix.
    - Multi-layer perceptron (MLP) estimate Q-values for each legal move.
    - Replay buffer, epsilon-greedy strategy, and periodic target network updates.

4. **Training mode**:
    - RL is played against Minimax.
    - Periodic model checkpoint saving.
    - Tracking `q_value max`, `q_target max`, `rewards`

## Project Structure
```
Chess/
├── images/                 # Images of pieces
├── Minimax/                # Minimax Alpha-Beta Prunning
│   ├── Evaluate.py
│   ├── SmartMoveFinder.py
├── RL/                     # Reinforcement Learning
│   ├── ChessEnv.py         # Chess Environment
│   ├── computereward.py    # Reward
│   ├── DQN.pth             # Parameter
│   ├── network.py          # Deep Q-Network
│   ├── replaybuffer.py     # Moved history
│   ├── Move.py             
│   └── train.py            # Train DQN model
├── ChessEngine.py          # Logic game
├── ChessMain.py            # Main file to run project
├── README.md
└── requirements.txt
```

## Usage
1. Clone repository to your local

2. Install the required libraries:
```bash
pip install -r requirements.txt
```

3. Run project and play Chess
```
python ChessMain.py
```
