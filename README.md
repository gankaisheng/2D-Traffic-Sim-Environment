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
