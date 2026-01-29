# 2D Traffic Simulation Environment (Phase I)

這是一個基於 Python 和 Pygame 開發的 2D 交通模擬環境。
此專案旨在作為 **強化學習 (Reinforcement Learning)** 的訓練環境基礎，具備物理碰撞檢測、NPC 車流邏輯以及 360 度光達 (Lidar) 感測模擬。

## 🚀 功能特點 (Features)

* **玩家控制 (Player Control)**：支援加速、自然減速（摩擦力）、剎車與倒車功能。
* **360 度環境感知 (360° Perception)**：
    * 配備 8 個方位的雷達傳感器（前、後、左、右及四個對角）。
    * 即時回傳與障礙物（牆壁、其他車輛）的距離數據。
    * 視覺化雷達線：根據距離變色（綠 -> 黃 -> 紅）。
* **NPC 智能車流**：
    * NPC 車輛會隨機生成於不同車道。
    * 具備基本的避障 AI：當前方有車時會自動減速以避免追撞。
* **即時儀表板 (Live Dashboard)**：
    * 顯示車速、行駛距離。
    * 即時顯示 8 個方位的傳感器數值。
    * 車輛狀態監測（存活/碰撞）。

## 🛠️ 安裝需求 (Requirements)

請確保你的電腦已安裝 Python 3.x，並安裝 `pygame` 套件：

```bash
pip install pygame
#train模型補充
pip install torch

#已修改的部分
traffic_sim_env +了：
0.輸入層>26
1.增加雷達傳感器，避免死角造成的訓練的問題。
2.ADAS 盲點偵測 ：想轉彎時，先檢查旁邊有沒有車。有車就鎖住方向盤。
3.AEB 自動緊急煞車 ：當前方雷達偵測到距離過近 (< 80px)，且左右無路可走時，強制覆蓋 AI 的油門指令，直接煞車到底，避免3車包圍還要超車的情況。
以上是到lv2自動駕駛，下面是嘗試lv3：
0.輸入層>50
1.讓雷達能讀取速度
2.避免AI為了獎勵特意頻繁啟動AEB

train.py：訓練用py
play.py：展示用py

