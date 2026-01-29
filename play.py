import torch
import torch.nn as nn
import numpy as np
import traffic_sim_env  # 引用環境
import pygame
import sys

# --- 模型結構保持一致 ---
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

def draw_ui_button(screen, rect, text, bg_color, text_color):
    """繪製按鈕 (在側邊欄)"""
    pygame.draw.rect(screen, bg_color, rect, border_radius=8)
    pygame.draw.rect(screen, (255, 255, 255), rect, 2, border_radius=8) 
    
    font = pygame.font.SysFont('Arial', 24, bold=True)
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

def play():
    env = traffic_sim_env.TrafficSim()
    
    # [Lv3 修改] 輸入層變大
    n_states = 50 
    n_actions = 5
    model = DQN(n_states, n_actions)
    
    try:
        print("📂 正在讀取 traffic_dqn.pth ...")
        model.load_state_dict(torch.load("traffic_dqn.pth"))
        model.eval()
        print("✅ 讀取成功！")
    except FileNotFoundError:
        print("❌ 找不到 traffic_dqn.pth！請確認是否已完成 Lv3 訓練。")
        return
    except RuntimeError:
        print("❌ 模型形狀不符！請刪除舊的 traffic_dqn.pth 並重新訓練。")
        return

    clock = pygame.time.Clock()
    
    # 狀態變數
    game_active = False 
    crashed = False     
    
    # 按鈕位置 (Dashboard)
    btn_width, btn_height = 180, 50
    btn_x = traffic_sim_env.GAME_WIDTH + 10 
    btn_y = traffic_sim_env.HEIGHT - 80     
    
    button_rect = pygame.Rect(btn_x, btn_y, btn_width, btn_height)

    # 初始重置
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    total_reward = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if not game_active:
                    if button_rect.collidepoint(event.pos):
                        print("▶️ 開始遊戲...")
                        game_active = True
                        crashed = False
                        state = env.reset()
                        state = torch.FloatTensor(state).unsqueeze(0)
                        total_reward = 0

        if game_active:
            with torch.no_grad():
                q_values = model(state)
                action = q_values.argmax().item()

            next_state_np, reward, done = env.step(action)
            state = torch.FloatTensor(next_state_np).unsqueeze(0)
            total_reward += reward

            if done:
                game_active = False
                crashed = True
                print(f"💀 撞車！Score: {total_reward:.1f}")

        # --- 繪圖 ---
        status_msg = "PLAYING"
        if not game_active:
            status_msg = "CRASHED" if crashed else "READY"

        info = [
            "MODE: Lv3 DEMO",
            f"Score: {total_reward:.1f}",
            f"State: {status_msg}"
        ]
        
        env.draw(extra_info=info, do_flip=False)

        if not game_active:
            if crashed:
                draw_ui_button(env.screen, button_rect, "RESTART", (220, 50, 50), WHITE)
            else:
                draw_ui_button(env.screen, button_rect, "START", (50, 200, 50), WHITE)
        else:
            draw_ui_button(env.screen, button_rect, "RUNNING...", (100, 100, 100), (200, 200, 200))

        pygame.display.flip()
        clock.tick(60)

WHITE = (255, 255, 255)

if __name__ == "__main__":
    play()