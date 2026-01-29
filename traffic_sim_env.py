import pygame
import math
import random
import numpy as np

# --- 常數設定 ---
GAME_WIDTH = 500
UI_WIDTH = 200
WIDTH, HEIGHT = GAME_WIDTH + UI_WIDTH, 600

CAR_SIZE = (40, 80)
FPS = 60

# 道路設定
LANE_WIDTH = 80
NUM_LANES = 3
ROAD_WIDTH = LANE_WIDTH * NUM_LANES
ROAD_X = (GAME_WIDTH - ROAD_WIDTH) // 2

# 顏色定義
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
UI_BG_COLOR = (50, 50, 60)
PURPLE = (255, 0, 255) # EES 觸發時的顏色

class Car:
    def __init__(self, x, y, allow_reverse=False):
        self.original_image = pygame.Surface(CAR_SIZE, pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, GREEN, (0, 0, CAR_SIZE[0], CAR_SIZE[1]), border_radius=5)
        # 車頭燈
        pygame.draw.rect(self.original_image, YELLOW, (5, 0, 10, 5))
        pygame.draw.rect(self.original_image, YELLOW, (CAR_SIZE[0]-15, 0, 10, 5))
        # 車尾燈
        pygame.draw.rect(self.original_image, (150, 0, 0), (5, CAR_SIZE[1]-5, 10, 5))
        pygame.draw.rect(self.original_image, (150, 0, 0), (CAR_SIZE[0]-15, CAR_SIZE[1]-5, 10, 5))
        
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(x, y))
        
        self.position = pygame.math.Vector2(x, y)
        self.angle = 0 
        self.speed = 0 
        self.max_speed = 15 
        self.min_speed = -5 
        self.radars = [] 
        self.alive = True
        self.distance_traveled = 0
        self.allow_reverse = allow_reverse
        
        # 狀態標記
        self.is_emergency_braking = False 
        self.is_ees_active = False # [新增] 緊急閃避狀態

    def handle_input(self, action=None):
        if not self.alive: return

        turn = 0      
        throttle = 0  

        if action is not None:
            if action == 1: turn = -1
            elif action == 2: turn = 1
            if action == 3: throttle = 1
            elif action == 4: throttle = -1
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: turn = -1
            elif keys[pygame.K_RIGHT]: turn = 1
            if keys[pygame.K_UP]: throttle = 1
            elif keys[pygame.K_DOWN]: throttle = -1

        # ==========================================
        # 安全輔助系統 (Safety Systems)
        # ==========================================
        self.is_emergency_braking = False
        self.is_ees_active = False
        
        if action is not None: 
            # 1. ADAS 盲點偵測 (防側撞)
            allow_turn = True
            if turn != 0 and len(self.radars) == 24:
                for r in self.radars:
                    # r: ((x,y), dist, rel_speed, angle)
                    angle = r[3]
                    dist = r[1]
                    if turn == -1: # 左轉
                        if -110 <= angle <= -70 and dist < 60:
                            allow_turn = False; break
                    elif turn == 1: # 右轉
                        if 70 <= angle <= 110 and dist < 60:
                            allow_turn = False; break
            if not allow_turn: turn = 0

            # 2. EES 後方緊急閃避 (Emergency Evasion System)
            # 邏輯：如果屁股後面有車衝過來，不管前方怎樣，先踩油門再說
            if len(self.radars) == 24:
                rear_danger = False
                for r in self.radars:
                    angle = r[3]
                    dist = r[1]
                    rel_speed = r[2] # 對方速度 - 我方速度
                    
                    # 檢查正後方 (-165 ~ 180 ~ 165)
                    if angle <= -165 or angle >= 165:
                        # 條件：距離近 (<80) 且 對方比我快很多 (rel_speed > 2)
                        # 正值代表對方正在接近我 (追撞風險)
                        # 注意：這裡的 rel_speed 計算方式取決於 cast_ray 的寫法
                        # 我們在 cast_ray 寫的是 target_speed - self.speed
                        # 如果對方快(10)，我慢(5)，結果是 5 (正值危險)
                        if dist < 80 and rel_speed > 2:
                            rear_danger = True
                            break
                
                if rear_danger:
                    throttle = 1 # 強制補油門逃跑！
                    self.is_ees_active = True

            # 3. AEB 自動緊急煞車 (防追撞前車)
            # AEB 的優先級最高 (Priority High)，因為撞前車比被後車撞更痛
            if len(self.radars) == 24:
                min_front_dist = 200
                for r in self.radars:
                    if -15 <= r[3] <= 15:
                        if r[1] < min_front_dist:
                            min_front_dist = r[1]
                
                if min_front_dist < 80 and self.speed > 2:
                    throttle = -1
                    turn = 0
                    self.is_emergency_braking = True
                    self.is_ees_active = False # 如果前方有牆，取消閃避，優先煞車

        # ==========================================
        # 物理執行
        # ==========================================
        if turn == -1:
            self.position.x -= 4 
            self.angle = 5
        elif turn == 1:
            self.position.x += 4
            self.angle = -5
        else:
            self.angle = 0

        speed_limit = self.min_speed
        if action is not None and not self.allow_reverse:
            speed_limit = 0 

        if throttle == 1:
            # EES 觸發時，給予額外的瞬間加速力 (彈射起步)
            accel = 0.5
            if self.is_ees_active: accel = 0.8 
            self.speed = min(self.speed + accel, self.max_speed)
        elif throttle == -1:
            self.speed = max(self.speed - 0.5, speed_limit)
        else:
            if self.speed > 0: self.speed = max(self.speed - 0.1, 0)
            elif self.speed < 0: self.speed = min(self.speed + 0.1, 0)

    def update(self, walls, npcs):
        self.position.x = max(ROAD_X + 20, min(self.position.x, ROAD_X + ROAD_WIDTH - 20))
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.position)
        self.check_collision(walls, npcs)
        
        self.radars.clear()
        
        for i, degree in enumerate(range(-180, 180, 15)):
            self.cast_ray(degree, walls, npcs)
            
        if self.alive:
            self.distance_traveled += self.speed

    def check_collision(self, walls, npcs):
        if not self.alive: return
        for wall in walls:
            if self.rect.colliderect(wall):
                self.alive = False; break
        for npc in npcs:
            if self.rect.colliderect(npc.rect):
                self.alive = False; break

    def cast_ray(self, angle_offset, walls, npcs):
        length = 0
        max_length = 200 
        
        rad = math.radians(90 + angle_offset)
        dx = math.cos(rad)
        dy = -math.sin(rad) 
        x, y = self.position.x, self.position.y
        
        target_speed = 0 
        
        while length < max_length:
            x += dx * 3
            y += dy * 3 
            length += 3
            point_rect = pygame.Rect(x, y, 4, 4)
            hit = False
            
            for wall in walls:
                if wall.colliderect(point_rect): 
                    hit = True; target_speed = 0; break
            
            if not hit:
                for npc in npcs:
                    if npc.rect.colliderect(point_rect): 
                        hit = True; target_speed = npc.speed; break
            
            if hit: break
            
        dist = int(math.sqrt((x - self.position.x)**2 + (y - self.position.y)**2))
        # 計算相對速度：(目標 - 我)
        # 如果追撞：目標快(10) - 我慢(5) = 5 (正值危險)
        rel_speed = target_speed - self.speed
        
        self.radars.append(((x, y), dist, rel_speed, angle_offset))

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
                 end_pos, dist, rel_speed, angle = radar
                 color = GREEN
                 if dist < 60: color = RED
                 elif dist < 120: color = YELLOW
                 
                 # 後方追撞警告 (紫色)
                 if (angle <= -160 or angle >= 160) and dist < 100 and rel_speed > 1:
                     color = PURPLE

                 start_pos = (self.position.x, self.position.y)
                 pygame.draw.line(screen, color, start_pos, end_pos, 1)
                 pygame.draw.circle(screen, color, end_pos, 3)

# --- NPC 類別 (維持原樣，不做修改，因為您說不要修改A) ---
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
            if self.speed > target_speed_limit: self.speed -= brake_force
        else:
            if self.speed < self.max_speed: self.speed += 0.1
        self.rect.y += (player_car.speed - self.speed)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=5)

class TrafficSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AI Traffic Sim (EES Active)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.lidar_font = pygame.font.SysFont('Arial', 14)
        self.walls = [
            pygame.Rect(0, 0, ROAD_X, HEIGHT),
            pygame.Rect(ROAD_X + ROAD_WIDTH, 0, GAME_WIDTH - (ROAD_X + ROAD_WIDTH), HEIGHT)
        ]
        self.show_radar = True
        self.score = 0
        self.reset()
        
    def reset(self):
        self.player_car = Car(GAME_WIDTH // 2, HEIGHT - 150, allow_reverse=False)
        self.player_car.speed = 10 
        
        self.npcs = []
        self.lane_offset = 0
        self.spawn_timer = 0
        self.crash_timer = 0
        self.traffic_start_dist = 500
        self.score = 0
        return self.get_state()

    def get_state(self):
        radar_data = []
        for r in self.player_car.radars:
            dist_norm = r[1] / 200.0
            speed_rel_norm = max(-1, min(1, r[2] / 20.0))
            radar_data.extend([dist_norm, speed_rel_norm])
        
        while len(radar_data) < 48: 
            radar_data.extend([1.0, 0.0])
            
        norm_speed = self.player_car.speed / self.player_car.max_speed
        norm_x = (self.player_car.position.x - ROAD_X) / ROAD_WIDTH
        
        return np.array(radar_data + [norm_speed, norm_x], dtype=np.float32)

    def step(self, action=None):
        self.player_car.handle_input(action)
        
        if self.player_car.alive:
            self.lane_offset += self.player_car.speed 
            if self.lane_offset >= 40: self.lane_offset = 0
            if self.lane_offset < 0: self.lane_offset = 40 
            
            if self.player_car.distance_traveled > self.traffic_start_dist:
                self.spawn_timer += 1
                if self.spawn_timer > 40 and random.random() < 0.05:
                    lane = random.randint(0, NUM_LANES - 1)
                    npc_speed = random.uniform(5, 12)
                    spawn_y = -100 
                    if npc_speed > self.player_car.speed + 5: spawn_y = HEIGHT + 100
                    else: spawn_y = -100
                    
                    safe = True
                    for npc in self.npcs:
                        lane_center = ROAD_X + (lane * LANE_WIDTH) + (LANE_WIDTH // 2)
                        if abs(npc.rect.centerx - lane_center) < 10:
                            if abs(npc.rect.y - spawn_y) < 200: safe = False; break
                    if safe:
                        self.npcs.append(NPC_Car(lane, spawn_y, npc_speed))
                        self.spawn_timer = 0
            
            for npc in self.npcs[:]:
                npc.update(self.player_car, self.npcs)
                if npc.rect.y > HEIGHT + 2000 or npc.rect.y < -300:
                    self.npcs.remove(npc)
        else:
            self.crash_timer += 1
            if self.crash_timer > 30: 
                if action is None: self.reset()

        self.player_car.update(self.walls, self.npcs)

        reward = 0
        done = False
        
        if not self.player_car.alive:
            reward = -100
            done = True
        else:
            reward = (self.player_car.speed / 10.0)
            if self.player_car.is_emergency_braking: reward -= 5.0 
            if self.player_car.speed < 5: reward -= 1.0
            self.score += reward

        return self.get_state(), reward, done

    def draw(self, extra_info=[], plot_surface=None, do_flip=True):
        self.screen.fill(BLACK) 
        
        pygame.draw.rect(self.screen, (34, 139, 34), (0, 0, ROAD_X, HEIGHT))
        pygame.draw.rect(self.screen, (34, 139, 34), (ROAD_X + ROAD_WIDTH, 0, GAME_WIDTH - (ROAD_X + ROAD_WIDTH), HEIGHT))
        pygame.draw.line(self.screen, WHITE, (ROAD_X, 0), (ROAD_X, HEIGHT), 5)
        pygame.draw.line(self.screen, WHITE, (ROAD_X + ROAD_WIDTH, 0), (ROAD_X + ROAD_WIDTH, HEIGHT), 5)
        
        for i in range(1, NUM_LANES):
            x = ROAD_X + i * LANE_WIDTH
            for y in range(-50, HEIGHT, 40):
                draw_y = y + self.lane_offset
                if draw_y > HEIGHT: draw_y -= (HEIGHT + 50)
                pygame.draw.line(self.screen, WHITE, (x, draw_y), (x, draw_y + 20), 2)

        for npc in self.npcs: npc.draw(self.screen)
        self.player_car.draw(self.screen, self.show_radar)
        self._draw_sidebar()

        if extra_info:
            info_font = pygame.font.SysFont('Arial', 14)
            for i, line in enumerate(extra_info):
                self.screen.blit(info_font.render(line, True, YELLOW), (5, 5 + i*15))

        if not self.player_car.alive:
            font = pygame.font.SysFont('Arial', 50, bold=True)
            text = font.render('CRASHED!', True, RED)
            text_rect = text.get_rect(center=(GAME_WIDTH//2, HEIGHT//2))
            self.screen.blit(text, text_rect)

        if do_flip:
            pygame.display.flip()

    def _draw_sidebar(self):
        pygame.draw.rect(self.screen, UI_BG_COLOR, (GAME_WIDTH, 0, UI_WIDTH, HEIGHT))
        pygame.draw.line(self.screen, WHITE, (GAME_WIDTH, 0), (GAME_WIDTH, HEIGHT), 2)
        
        self.screen.blit(self.title_font.render("Dashboard", True, YELLOW), (GAME_WIDTH + 20, 30))
        y = 70; gap = 35
        self.screen.blit(self.font.render(f"Speed: {self.player_car.speed:.1f} km/h", True, WHITE), (GAME_WIDTH + 20, y))
        y += gap
        self.screen.blit(self.font.render(f"Dist: {int(self.player_car.distance_traveled)} m", True, WHITE), (GAME_WIDTH + 20, y))
        y += gap
        status_text = "ALIVE" if self.player_car.alive else "CRASHED"
        status_color = GREEN if self.player_car.alive else RED
        self.screen.blit(self.title_font.render(status_text, True, status_color), (GAME_WIDTH + 20, y))
        
        y += gap + 20
        self.screen.blit(self.font.render(f"Lv3 Sensors", True, (0, 255, 255)), (GAME_WIDTH + 20, y))
        
        y += gap
        if self.player_car.is_emergency_braking:
             self.screen.blit(self.font.render(f"⚠️ AEB BRAKE", True, RED), (GAME_WIDTH + 20, y))
        elif self.player_car.is_ees_active:
             self.screen.blit(self.font.render(f"⚡️ EES ESCAPE!", True, PURPLE), (GAME_WIDTH + 20, y)) # [新增] EES 狀態顯示

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

def main():
    game = TrafficSim()
    running = True
    while running:
        running = game.handle_events()
        game.step(action=None) 
        game.draw()
        game.clock.tick(FPS)
    pygame.quit()

if __name__ == "__main__":
    main()