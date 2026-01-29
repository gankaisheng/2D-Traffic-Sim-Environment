import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import traffic_sim_env  # 引用環境
import pygame           # 用來偵測按鍵
import os

# --- 超參數設定 ---
BATCH_SIZE = 64
LR = 0.001
GAMMA = 0.99
# 這裡設定的是「如果是新遊戲」的數值
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995 
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    def forward(self, x): return self.net(x)

def train():
    env = traffic_sim_env.TrafficSim()
    
    # Lv3 輸入維度
    n_states = 50  
    n_actions = 5

    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    
    epsilon = EPSILON_START
    loaded_save = False

    # --- [關鍵修改] 智慧讀檔系統 ---
    if os.path.exists("traffic_dqn.pth"):
        try:
            saved_state = torch.load("traffic_dqn.pth")
            policy_net.load_state_dict(saved_state)
            target_net.load_state_dict(saved_state)
            loaded_save = True
            print("✅ 成功讀取存檔！")
            
            # [重點] 如果是讀檔，不要從頭變笨 (1.0)，而是從 0.3 開始
            # 這樣它會保留大部分實力，同時還能繼續學習
            epsilon = 0.3 
            print(f"👉 繼承訓練模式：Epsilon 起始值設為 {epsilon}")
            
        except Exception as e:
            print(f"⚠️ 存檔格式不符或損壞，將重新訓練: {e}")
            # 如果讀檔失敗，建議刪除舊檔以免下次又報錯
            # os.remove("traffic_dqn.pth")
    else:
        print("🆕 找不到存檔，開始全新的訓練...")
        target_net.load_state_dict(policy_net.state_dict())

    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    loss_func = nn.MSELoss()
    
    memory = deque(maxlen=MEMORY_SIZE)
    
    # --- 控制開關 ---
    render_mode = False  
    
    print("---------------------------------------------")
    print("🚀 Lv3 訓練開始！")
    print("👉 按下 [Z] 鍵：切換「看畫面」或「背景極速訓練」")
    print("👉 按下 [S] 鍵：存檔並離開 (Pause)")
    print("---------------------------------------------")

    episodes = 0
    steps_done = 0

    try:
        while True:
            state = env.reset()
            state = torch.FloatTensor(state).unsqueeze(0)
            total_reward = 0
            done = False
            
            while not done:
                steps_done += 1

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("使用者關閉視窗，停止訓練。")
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_z:
                            render_mode = not render_mode
                            mode_text = "📺 觀看模式" if render_mode else "🚀 極速模式"
                            print(f"切換模式: {mode_text}")
                        if event.key == pygame.K_s:
                            print("\n💾 存檔中...")
                            torch.save(policy_net.state_dict(), "traffic_dqn.pth")
                            print("✅ 存檔完成！您現在可以安全退出了。")
                            # 這裡您可以選擇要不要直接退出，或者繼續
                            # return 

                # 決定動作
                if random.random() < epsilon:
                    action = random.randint(0, n_actions - 1)
                else:
                    with torch.no_grad():
                        action = policy_net(state).argmax().item()

                # 執行動作
                next_state_np, reward, done = env.step(action)
                next_state = torch.FloatTensor(next_state_np).unsqueeze(0)
                
                # 記憶
                memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                # 繪圖
                if render_mode:
                    info = [f"Ep: {episodes}", f"Eps: {epsilon:.2f}", f"Rwd: {total_reward:.1f}"]
                    env.draw(extra_info=info)
                    env.clock.tick(60)
                else:
                    if steps_done % 1000 == 0:
                         env.draw(extra_info=["TURBO MODE", f"Ep: {episodes}"])
                         print(f"\rEpisode: {episodes}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}", end="")

                # 學習 (每5步)
                if len(memory) > BATCH_SIZE and steps_done % 5 == 0:
                    batch = random.sample(memory, BATCH_SIZE)
                    b_state = torch.cat([x[0] for x in batch])
                    b_action = torch.LongTensor([x[1] for x in batch]).unsqueeze(1)
                    b_reward = torch.FloatTensor([x[2] for x in batch]).unsqueeze(1)
                    b_next = torch.cat([x[3] for x in batch])
                    b_done = torch.FloatTensor([float(x[4]) for x in batch]).unsqueeze(1)

                    curr_q = policy_net(b_state).gather(1, b_action)
                    next_q = target_net(b_next).max(1)[0].unsqueeze(1)
                    expected_q = b_reward + (GAMMA * next_q * (1 - b_done))

                    loss = loss_func(curr_q, expected_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            episodes += 1
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            
            if episodes % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                if not render_mode:
                    print(f"\nEpisode {episodes}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}")

    except KeyboardInterrupt:
        print("\n🛑 強制中斷，存檔中...")
        torch.save(policy_net.state_dict(), "traffic_dqn.pth")
        env.pygame.quit()

if __name__ == "__main__":
    train()