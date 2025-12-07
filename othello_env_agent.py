# othello_env_agent.py
# (경로: /Users/liam/Desktop/Othello/othello_env_agent.py)

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.mps

# ============================================================
# Q-Table 초기화 함수
# ============================================================
def q_table_initializer(action_dim):
    return np.zeros(action_dim)

# ============================================================
# 1️⃣ 환경 정의 (OthelloEnv 클래스)
# ============================================================
class OthelloEnv(gym.Env):
    def __init__(self, size=8):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(self.size * self.size)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=int)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.size, self.size), dtype=int)
        mid = self.size // 2
        # 초기 배치: 백(2)은 흑(1)과 대각선으로 배치
        self.board[mid-1][mid-1], self.board[mid][mid] = 2, 2
        self.board[mid-1][mid], self.board[mid][mid-1] = 1, 1
        self.current_player = 1 # 흑(1)부터 시작
        self.game_over = False
        return self.board.copy(), {}

    def get_opponent(self):
        return 3 - self.current_player

    def is_on_board(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    # 돌을 뒤집을 수 있는지 확인하고 뒤집을 돌 목록을 반환 (내부용)
    def _check_direction(self, x, y, x_dir, y_dir, player):
        opponent = 3 - player
        flips = []
        
        # 첫 번째 칸은 상대 돌이어야 함
        if not self.is_on_board(x + x_dir, y + y_dir) or self.board[x + x_dir][y + y_dir] != opponent:
            return []
        
        cur_x, cur_y = x + 2 * x_dir, y + 2 * y_dir
        
        while self.is_on_board(cur_x, cur_y):
            cell = self.board[cur_x][cur_y]
            if cell == 0:
                return []
            if cell == player:
                # 플레이어 돌을 찾았으면, 그 사이에 있는 모든 상대 돌을 뒤집을 수 있음
                px, py = x + x_dir, y + y_dir
                while (px, py) != (cur_x, cur_y):
                    flips.append((px, py))
                    px += x_dir
                    py += y_dir
                return flips
            
            # 계속 진행
            cur_x += x_dir
            cur_y += y_dir

        return [] 

    # 유효한 이동 위치 계산
    def valid_moves(self):
        moves = set()
        
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x][y] == 0: # 빈 칸인 경우
                    is_valid = False
                    for x_dir in [-1, 0, 1]:
                        for y_dir in [-1, 0, 1]:
                            if x_dir == 0 and y_dir == 0:
                                continue
                                
                            flips = self._check_direction(x, y, x_dir, y_dir, self.current_player)
                            if flips:
                                moves.add(x * self.size + y) # 행동은 0~63의 인덱스
                                is_valid = True
                                break
                        if is_valid:
                            break
                            
        return moves

    # 실제로 돌을 뒤집는 함수
    def flip_discs(self, x, y):
        total_flips = 0
        for x_dir in [-1, 0, 1]:
            for y_dir in [-1, 0, 1]:
                if x_dir == 0 and y_dir == 0:
                    continue
                
                flips = self._check_direction(x, y, x_dir, y_dir, self.current_player)
                for fx, fy in flips:
                    self.board[fx][fy] = self.current_player
                    total_flips += 1
        return total_flips

    # 환경의 상태를 업데이트하는 함수
    def step(self, action):
        x, y = divmod(action, self.size)
        
        # 1. 유효성 검사
        if self.board[x][y] != 0 or action not in self.valid_moves():
            reward = -10.0  # 잘못된 수에 큰 패널티
            return self.board.copy(), reward, self.game_over, True, {} 
        
        # 2. 돌 놓기 및 뒤집기
        self.board[x][y] = self.current_player
        flips_count = self.flip_discs(x, y)
        
        # 3. 보상 계산
        reward = flips_count * 1.0 
        if (x, y) in [(0, 0), (0, 7), (7, 0), (7, 7)]: # 코너 보상
            reward += 10.0 
        
        # 4. 플레이어 전환
        self.current_player = self.get_opponent()
        
        # 5. 게임 종료 및 패스 체크
        is_truncated = False
        if not self.valid_moves():
            # 상대방에게 턴을 넘김
            self.current_player = self.get_opponent() 
            
            if not self.valid_moves():
                # 양쪽 모두 둘 곳이 없으면 게임 종료
                self.game_over = True
                is_truncated = True

        if self.game_over:
            # 최종 보상
            black_score = np.sum(self.board == 1)
            white_score = np.sum(self.board == 2)
            
            # 최종 승패 결정 (현재 플레이어는 턴을 받은 상태이므로, 그 이전 플레이어의 승리/패배를 보상)
            if black_score > white_score and self.current_player == 2: 
                reward += 100.0
            elif white_score > black_score and self.current_player == 1:
                 reward += 100.0
            elif black_score == white_score:
                reward += 0.0
            else:
                reward -= 50.0 

        return self.board.copy(), reward, self.game_over, is_truncated, {} 

    def render(self, mode='human'):
        if mode == 'human':
            print("  0 1 2 3 4 5 6 7")
            for i in range(self.size):
                row = str(i) + " "
                for j in range(self.size):
                    cell = self.board[i][j]
                    if cell == 1:
                        row += "⚫" # 흑돌
                    elif cell == 2:
                        row += "⚪" # 백돌
                    else:
                        row += " " # 빈칸
                    row += " "
                print(row)

# ============================================================
# 2️⃣ Q-learning Agent (QLearningAgent 클래스)
# ============================================================
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.01):
        self.env = env
        # ⭐ Q-Table 정의 (defaultdict 사용) ⭐
        self.q_table = defaultdict(lambda: q_table_initializer(env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=1)

    def state_to_key(self, state):
        return tuple(int(x) for x in state.flatten())

    def act(self, state, valid_moves):
        if not valid_moves:
            return None
        
        if np.random.rand() < self.epsilon:
            return random.choice(list(valid_moves))
        else:
            state_key = self.state_to_key(state)
            q_values = self.q_table[state_key]
            
            q_filtered = [q_values[a] if a in valid_moves else -np.inf
                          for a in range(self.env.action_space.n)]
            
            return int(np.argmax(q_filtered))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if not self.memory:
            return 
            
        s, a, r, s2, done = self.memory.popleft()
        
        s_key = self.state_to_key(s)
        s2_key = self.state_to_key(s2)
        
        current_q = self.q_table[s_key][a]
        
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[s2_key])
            
        # Q-Learning 업데이트 공식
        new_q = current_q + self.alpha * (r + self.gamma * max_future_q - current_q)
        self.q_table[s_key][a] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target(self):
        pass

# ============================================================
# 3️⃣ CNN 기반 DQN Network 및 Agent (M1/LR=1e-4)
# ============================================================
class DQNetwork(nn.Module):
    def __init__(self, input_channels=1, output_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

class DQNAgent:
    # LR=1e-4, batch_size=128 적용
    def __init__(self, env, gamma=0.95, lr=1e-4, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1, buffer_size=50000, batch_size=128):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # M1/MPS 디바이스 설정
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🧠 PyTorch M1 (MPS) 사용.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🧠 PyTorch CUDA 사용.")
        else:
            self.device = torch.device("cpu")
            print("🧠 PyTorch CPU 사용.")

        self.memory = deque(maxlen=buffer_size)
        self.q_net = DQNetwork().to(self.device)
        self.target_net = DQNetwork().to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr) 
        self.loss_fn = nn.MSELoss()
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def _state_to_tensor(self, state):
        return torch.FloatTensor(np.array(state)).unsqueeze(0).unsqueeze(0).to(self.device)

    def act(self, state, valid_moves):
        if not valid_moves:
            return None
        if np.random.rand() < self.epsilon:
            return random.choice(list(valid_moves))

        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_net(state_tensor).cpu().numpy()[0]

        q_filtered = [q_values[a] if a in valid_moves else -np.inf
                      for a in range(self.env.action_space.n)]

        return int(np.argmax(q_filtered))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, done = zip(*batch)

        s = torch.FloatTensor(np.array(s)).unsqueeze(1).to(self.device)
        s2 = torch.FloatTensor(np.array(s2)).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        done = torch.FloatTensor(done).to(self.device) 

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        next_q = self.target_net(s2).max(1)[0]
        target = r + (1 - done) * self.gamma * next_q

        loss = self.loss_fn(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)