import pygame
import math
import random

# --- 常數設定 ---
# 視窗配置
GAME_WIDTH = 500  # 遊戲區寬度
UI_WIDTH = 200    # 儀表板寬度
WIDTH, HEIGHT = GAME_WIDTH + UI_WIDTH, 600

CAR_SIZE = (40, 80)
FPS = 60

# 道路設定
LANE_WIDTH = 80
NUM_LANES = 3
ROAD_WIDTH = LANE_WIDTH * NUM_LANES  # 3 * 80 = 240
ROAD_X = (GAME_WIDTH - ROAD_WIDTH) // 2 # 道路置中

# 顏色定義
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
UI_BG_COLOR = (50, 50, 60)

class Car:
    def __init__(self, x, y):
        self.original_image = pygame.Surface(CAR_SIZE, pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, GREEN, (0, 0, CAR_SIZE[0], CAR_SIZE[1]), border_radius=5)
        # 車頭燈 (黃色)
        pygame.draw.rect(self.original_image, YELLOW, (5, 0, 10, 5))
        pygame.draw.rect(self.original_image, YELLOW, (CAR_SIZE[0]-15, 0, 10, 5))
        # 車尾燈 (紅色，倒車時視覺輔助)
        pygame.draw.rect(self.original_image, (150, 0, 0), (5, CAR_SIZE[1]-5, 10, 5))
        pygame.draw.rect(self.original_image, (150, 0, 0), (CAR_SIZE[0]-15, CAR_SIZE[1]-5, 10, 5))
        
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        
        self.position = pygame.math.Vector2(x, y)
        self.angle = 0 
        self.speed = 0 
        self.max_speed = 15 
        self.min_speed = -5 # 啟用倒車功能
        self.radars = [] 
        self.alive = True
        self.distance_traveled = 0

    def handle_input(self):
        # 僅在手動模式下呼叫此函式
        if not self.alive:
            return

        keys = pygame.key.get_pressed()
        
        # 左右平移
        if keys[pygame.K_LEFT]:
            self.position.x -= 6
            self.angle = 5
        elif keys[pygame.K_RIGHT]:
            self.position.x += 6
            self.angle = -5
        else:
            self.angle = 0

        # 油門與倒車控制
        if keys[pygame.K_UP]:
            self.speed = min(self.speed + 0.2, self.max_speed)
        elif keys[pygame.K_DOWN]:
            self.speed = max(self.speed - 0.5, self.min_speed) 
        else:
            # 自然減速 (摩擦力)
            if self.speed > 0:
                self.speed = max(self.speed - 0.1, 0)
            elif self.speed < 0:
                self.speed = min(self.speed + 0.1, 0)

    def update(self, walls, npcs):
        # 限制移動範圍
        self.position.x = max(ROAD_X + 20, min(self.position.x, ROAD_X + ROAD_WIDTH - 20))
        
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.position)
        
        self.check_collision(walls, npcs)
        
        # 360度雷達更新
        self.radars.clear()
        # 定義 8 個方向: 0(前), 45(左前), 90(左), 135(左後), 180(後), -135(右後), -90(右), -45(右前)
        sensor_angles = [0, 45, 90, 135, 180, -135, -90, -45]
        
        for degree in sensor_angles: 
            self.cast_ray(degree, walls, npcs)
            
        if self.alive:
            self.distance_traveled += self.speed

    def check_collision(self, walls, npcs):
        if not self.alive:
            return

        for wall in walls:
            if self.rect.colliderect(wall):
                self.alive = False
                break
        for npc in npcs:
            if self.rect.colliderect(npc.rect):
                self.alive = False
                break

    def cast_ray(self, angle_offset, walls, npcs):
        length = 0
        max_length = 200 # 雷達長度
        
        rad = math.radians(90 + angle_offset)
        dx = math.cos(rad)
        dy = -math.sin(rad) 
        
        # 雷達發射點：車身中心
        x, y = self.position.x, self.position.y
        
        while length < max_length:
            x += dx * 3
            y += dy * 3 
            length += 3
            
            point_rect = pygame.Rect(x, y, 4, 4)
            hit = False
            
            # 檢測牆壁
            for wall in walls:
                if wall.colliderect(point_rect):
                    hit = True
                    break
            
            # 檢測 NPC 車輛
            for npc in npcs:
                if npc.rect.colliderect(point_rect):
                    hit = True
                    break

            if hit:
                break
                
        dist = int(math.sqrt((x - self.position.x)**2 + (y - self.position.y)**2))
        self.radars.append(((x, y), dist))

    def draw(self, screen, show_radar=True):
        if not self.alive:
            filter_surf = pygame.Surface(self.image.get_size(), pygame.SRCALPHA)
            filter_surf.fill((255, 0, 0, 100))
            screen.blit(self.image, self.rect)
            screen.blit(filter_surf, self.rect)
        else:
            screen.blit(self.image, self.rect)
        
        if show_radar:
            for radar in self.radars:
                 end_pos, dist = radar
                 color = GREEN
                 if dist < 60: color = RED
                 elif dist < 120: color = YELLOW
                 
                 start_pos = (self.position.x, self.position.y)
                 pygame.draw.line(screen, color, start_pos, end_pos, 1)
                 pygame.draw.circle(screen, color, end_pos, 3)

class NPC_Car:
    def __init__(self, lane_index, start_y, speed):
        lane_x = ROAD_X + (lane_index * LANE_WIDTH) + (LANE_WIDTH // 2)
        self.rect = pygame.Rect(0, 0, CAR_SIZE[0], CAR_SIZE[1])
        self.rect.center = (lane_x, start_y) 
        
        self.color = (random.randint(50, 255), random.randint(50, 255), 255)
        self.speed = speed
        self.max_speed = speed 

    def update(self, player_car, npcs):
        closest_dist = 9999
        obstacle_speed = 0
        found_obstacle = False
        
        targets = [player_car] + [npc for npc in npcs if npc != self]

        for target in targets:
            if abs(target.rect.centerx - self.rect.centerx) < 30:
                if target.rect.y < self.rect.y:
                    dist = self.rect.top - target.rect.bottom
                    if dist < closest_dist:
                        closest_dist = dist
                        obstacle_speed = target.speed
                        found_obstacle = True

        if found_obstacle and closest_dist < 400:
            brake_force = 0.2
            if closest_dist < 150: brake_force = 0.8 
            elif closest_dist < 250: brake_force = 0.4 

            target_speed_limit = max(0, obstacle_speed - 1)
            if self.speed > target_speed_limit:
                self.speed -= brake_force
        else:
            if self.speed < self.max_speed:
                self.speed += 0.1

        relative_speed = player_car.speed - self.speed
        self.rect.y += relative_speed

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=5)

class TrafficSim:
    """
    TrafficSim 封裝了所有的遊戲邏輯。
    這樣設計可以方便未來被強化學習 (RL) 環境調用，
    也可以被 main() 函式調用來進行手動遊玩。
    """
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Traffic Sim - Refactored")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.lidar_font = pygame.font.SysFont('Arial', 14)
        
        # 路邊牆壁
        self.walls = [
            pygame.Rect(0, 0, ROAD_X, HEIGHT),
            pygame.Rect(ROAD_X + ROAD_WIDTH, 0, GAME_WIDTH - (ROAD_X + ROAD_WIDTH), HEIGHT)
        ]
        
        # 遊戲狀態變數
        self.reset()
        
        # UI 狀態
        self.show_radar = True
        self.btn_rect = pygame.Rect(GAME_WIDTH + 20, HEIGHT - 60, 160, 40)
        
    def reset(self):
        """重置遊戲狀態"""
        self.player_car = Car(GAME_WIDTH // 2, HEIGHT - 150)
        self.npcs = []
        self.lane_offset = 0
        self.spawn_timer = 0
        self.crash_timer = 0
        self.traffic_start_dist = 5000

    def step(self):
        """執行一步遊戲邏輯 (Update Loop)"""
        # 如果是手動模式，這裡處理輸入。
        # 未來如果是 AI 模式，這裡會接收 action 參數並傳給 player_car.control(action)
        self.player_car.handle_input()
        
        if self.player_car.alive:
            # 1. 背景捲動邏輯
            self.lane_offset += self.player_car.speed 
            if self.lane_offset >= 40: self.lane_offset = 0
            if self.lane_offset < 0: self.lane_offset = 40 
            
            # 2. 生成 NPC 邏輯
            if self.player_car.distance_traveled > self.traffic_start_dist:
                self.spawn_timer += 1
                if self.spawn_timer > 40 and random.random() < 0.05:
                    lane = random.randint(0, NUM_LANES - 1)
                    npc_speed = random.uniform(5, 12)
                    
                    spawn_y = -100 
                    if npc_speed > self.player_car.speed + 5: 
                         spawn_y = HEIGHT + 100
                    else:
                         spawn_y = -100

                    safe = True
                    for npc in self.npcs:
                        # 簡單的重疊檢查
                        lane_center = ROAD_X + (lane * LANE_WIDTH) + (LANE_WIDTH // 2)
                        if abs(npc.rect.centerx - lane_center) < 10:
                            if abs(npc.rect.y - spawn_y) < 200:
                                safe = False
                                break
                    
                    if safe:
                        self.npcs.append(NPC_Car(lane, spawn_y, npc_speed))
                        self.spawn_timer = 0
            
            # 3. 更新 NPC
            for npc in self.npcs[:]:
                npc.update(self.player_car, self.npcs)
                if npc.rect.y > HEIGHT + 2000 or npc.rect.y < -300:
                    self.npcs.remove(npc)
        else:
            # 4. 碰撞後重置邏輯
            self.crash_timer += 1
            if self.crash_timer > 30: 
                self.reset()

        # 5. 更新玩家車輛 (包含雷達與碰撞檢測)
        self.player_car.update(self.walls, self.npcs)

    def draw(self):
        """繪製遊戲畫面 (Render Loop)"""
        self.screen.fill(BLACK) 
        
        # 畫草地
        pygame.draw.rect(self.screen, (34, 139, 34), (0, 0, ROAD_X, HEIGHT))
        pygame.draw.rect(self.screen, (34, 139, 34), (ROAD_X + ROAD_WIDTH, 0, GAME_WIDTH - (ROAD_X + ROAD_WIDTH), HEIGHT))
        
        # 畫路邊線
        pygame.draw.line(self.screen, WHITE, (ROAD_X, 0), (ROAD_X, HEIGHT), 5)
        pygame.draw.line(self.screen, WHITE, (ROAD_X + ROAD_WIDTH, 0), (ROAD_X + ROAD_WIDTH, HEIGHT), 5)
        
        # 畫車道分隔線
        for i in range(1, NUM_LANES):
            x = ROAD_X + i * LANE_WIDTH
            for y in range(-50, HEIGHT, 40):
                draw_y = y + self.lane_offset
                if draw_y > HEIGHT: draw_y -= (HEIGHT + 50)
                pygame.draw.line(self.screen, WHITE, (x, draw_y), (x, draw_y + 20), 2)

        # 畫實體
        for npc in self.npcs:
            npc.draw(self.screen)
            
        self.player_car.draw(self.screen, self.show_radar)
        
        # 畫 UI
        self._draw_sidebar()

        # 撞擊提示
        if not self.player_car.alive:
            font = pygame.font.SysFont('Arial', 50, bold=True)
            text = font.render('CRASHED!', True, RED)
            text_rect = text.get_rect(center=(GAME_WIDTH//2, HEIGHT//2))
            
            bg_rect = text_rect.inflate(20, 20)
            s = pygame.Surface((bg_rect.width, bg_rect.height))
            s.set_alpha(200)
            s.fill(BLACK)
            self.screen.blit(s, bg_rect)
            self.screen.blit(text, text_rect)

        pygame.display.flip()

    def _draw_sidebar(self):
        """內部方法：繪製側邊儀表板"""
        pygame.draw.rect(self.screen, UI_BG_COLOR, (GAME_WIDTH, 0, UI_WIDTH, HEIGHT))
        pygame.draw.line(self.screen, WHITE, (GAME_WIDTH, 0), (GAME_WIDTH, HEIGHT), 2)
        
        self.screen.blit(self.title_font.render("Dashboard", True, YELLOW), (GAME_WIDTH + 20, 30))
        
        y = 70
        gap = 35
        
        # 數據顯示
        self.screen.blit(self.font.render(f"Speed: {self.player_car.speed:.1f} km/h", True, WHITE), (GAME_WIDTH + 20, y))
        y += gap
        self.screen.blit(self.font.render(f"Dist: {int(self.player_car.distance_traveled)} m", True, WHITE), (GAME_WIDTH + 20, y))
        y += gap
        
        status_text = "ALIVE" if self.player_car.alive else "CRASHED"
        status_color = GREEN if self.player_car.alive else RED
        self.screen.blit(self.title_font.render(status_text, True, status_color), (GAME_WIDTH + 20, y))

        y += gap + 10
        self.screen.blit(self.font.render(f"360 Sensors:", True, GRAY), (GAME_WIDTH + 20, y))
        y += 25
        
        labels = ["Front", "F-Left", "Left", "R-Left", "Rear", "R-Right", "Right", "F-Right"]
        text_color = WHITE if self.show_radar else GRAY
        
        for i, radar in enumerate(self.player_car.radars):
            dist = radar[1]
            label = labels[i]
            col = i % 2 
            row = i // 2
            x_pos = GAME_WIDTH + 10 + col * 90
            y_pos = y + row * 20
            
            text = f"{label}: {dist}"
            data_color = RED if dist < 50 and self.show_radar else text_color
            self.screen.blit(self.lidar_font.render(text, True, data_color), (x_pos, y_pos))

        # 按鈕繪製
        btn_color = (0, 150, 0) if self.show_radar else (150, 0, 0)
        pygame.draw.rect(self.screen, btn_color, self.btn_rect, border_radius=5)
        btn_text = self.font.render(f"Lidar: {'ON' if self.show_radar else 'OFF'} (L)", True, WHITE)
        self.screen.blit(btn_text, btn_text.get_rect(center=self.btn_rect.center))

    def handle_events(self):
        """處理 Pygame 事件 (Quit, Keys, Mouse)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: 
                    self.reset()
                if event.key == pygame.K_l: 
                    self.show_radar = not self.show_radar
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    if self.btn_rect.collidepoint(event.pos):
                        self.show_radar = not self.show_radar
        return True

def main():
    """
    主程式進入點：
    現在它只是一個『驅動器』，負責建立 TrafficSim 物件並呼叫 step/draw。
    這跟未來 RL 訓練程式呼叫 env.step() 的結構是一模一樣的。
    """
    game = TrafficSim()
    running = True
    
    while running:
        # 1. 處理系統事件
        running = game.handle_events()
        
        # 2. 執行遊戲邏輯 (Step)
        game.step()
        
        # 3. 渲染畫面 (Render)
        game.draw()
        
        # 4. 控制幀率
        game.clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()