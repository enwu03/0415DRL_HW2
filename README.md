# 0415DRL_HW2 - Cliff Walking: Q-learning vs SARSA

**[Live Demo](https://enwu03.github.io/0415DRL_HW2/)**

![Cliff Walking Demo](demo.webp)

This project implements and compares two classic reinforcement learning algorithms — **Q-learning** (Off-policy) and **SARSA** (On-policy) — on the **Cliff Walking** gridworld environment. Both algorithms are trained under identical parameters for fair comparison.

## Environment Description

- **Grid**: Configurable rectangular grid (default 4 × 12)
- **Start (S)**: Bottom-left corner
- **Goal (G)**: Bottom-right corner
- **Cliff (☠)**: Bottom row between Start and Goal
- **Rewards**: Step = −1, Cliff = −100 (reset to Start), Goal = episode ends

## Algorithm Implementation

### Q-learning (Off-policy)
```
Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') − Q(s, a)]
```
Uses `max Q(s', a')` to update — learns the **optimal policy** regardless of exploration behavior.

### SARSA (On-policy)
```
Q(s, a) ← Q(s, a) + α [r + γ · Q(s', a') − Q(s, a)]
```
Uses the **actual next action** `Q(s', a')` to update — learns a policy that accounts for exploration risk.

## Parameters (User-Tunable)

| Parameter | Default | Range |
|---|---|---|
| Learning Rate (α) | 0.1 | 0.01 - 1.0 |
| Discount Factor (γ) | 0.9 | 0 - 1.0 |
| Exploration Rate (ε) | 0.1 | 0 - 0.5 |
| Episodes | 500 | 100 - 5000 |
| Grid Rows | 4 | 3 - 10 |
| Grid Columns | 12 | 4 - 20 |

## Results Analysis

The app auto-generates a comprehensive analysis with three sections:

### 一、學習表現 (Learning Performance)
- Cumulative reward curve per episode (smoothed)
- Convergence speed comparison (50-episode moving average)

### 二、策略行為 (Policy Behavior)
- Learned path visualization with directional arrows
- Q-learning → **Risky** shortest path along cliff edge
- SARSA → **Safe** longer path away from cliff

### 三、穩定性分析 (Stability Analysis)
- Reward standard deviation (volatility)
- Cliff fall count during training
- Exploration (ε) impact on each algorithm with probability calculations

---

## Setup and Execution

This is a **pure frontend** application (HTML/CSS/JS). No backend server is required.

### Option 1: GitHub Pages
Simply visit the **[Live Demo](https://enwu03.github.io/0415DRL_HW2/)**.

### Option 2: Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/enwu03/0415DRL_HW2.git
   cd 0415DRL_HW2
   ```

2. Open `index.html` in any modern browser, or serve via a local HTTP server:
   ```bash
   python -m http.server 8080
   ```
   Then navigate to **[http://localhost:8080](http://localhost:8080)**.

3. Adjust parameters, click **「開始訓練」** to train, and review the results!

## Tech Stack
- **HTML5** — Semantic structure
- **CSS3** — Dark glassmorphism theme with responsive layout
- **JavaScript (ES6+)** — RL algorithms, Canvas chart rendering, DOM visualization

## 四、理論比較

**Q-learning（離策略，Off-policy）**
- **更新規則**：`Q(s, a) ← Q(s, a) + α[r + γ * max_a' Q(s', a') - Q(s, a)]`
- **理論探討**：Q-learning 在計算下一個狀態 `s'` 的價值時，直接取未來可獲得的最大 Q 值（`max_a'`），這代表它是基於**完美策略（最佳策略）**進行更新的，完全無視於當前探索機制（ε-greedy）帶來的隨機性風險。
- **行為結果**：Q-learning 會學到理論上最短、收益最高的「最佳路徑」（緊貼著懸崖邊走）。但這也意味著在訓練過程或仍帶有 ε 探索率的執行環境中，它因為太過靠近邊緣而有極大的機率掉入懸崖。

**SARSA（同策略，On-policy）**
- **更新規則**：`Q(s, a) ← Q(s, a) + α[r + γ * Q(s', a') - Q(s, a)]`
- **理論探討**：SARSA 的更新依賴於實際被選中的下一個動作 `a'`，也就是說，包含 ε-greedy 策略帶來的隨機探索動作也會被反映在該路徑的估值中。
- **行為結果**：因為探索掉入懸崖的高額懲罰被真實反映，SARSA 會傾向於遠離懸崖，學到一條「保守而安全」的較長路徑。這樣雖然路徑長度增加（通常是多退後一格再前進），但在有隨機性的狀況下能夠極大程度降低掉入懸崖的風險。

---

## 五、結論

1. **安全性與風險控制：** 
   SARSA 的核心優勢在於將未知的隨機風險（探索行為）納入其價值考量中。在 Cliff Walking 這個環境下，SARSA 將會選擇遠離危險的保守路徑。在實務上（例如自動駕駛、醫療器材等），如果系統發生錯誤的代價無法承受，SARSA 這類 On-policy 的方法會是更安全的選項。
   
2. **收斂性與最優解：** 
   Q-learning 承擔了高風險，但能保證在足夠的迭代後收斂出全局最佳策略（最短路徑）。如果訓練成本非常低，並且我們預期部署後的系統沒有隨機探索（ε = 0），那麼 Q-learning 能提供更高上限的效率與回報。

3. **穩定度比較：** 
   根據我們的實驗結果與穩定性分析（標準差與懸崖掉落次數），SARSA 在訓練過程中的表現遠比 Q-learning 穩定。Q-learning 因策略過於冒險，很容易在最後幾步因 10% 的隨機性跌下懸崖（帶來 -100 懲罰），造成整體平均獎勵大幅波動。

> 因此，在這兩個演算法的選擇上，必須衡量「訓練期間失敗的代價」以及「系統上線後是否保留隨機性」。這正是 Cliff Walking 經典實驗想傳達的核心觀念。

---

## Development Workflow
This project is developed and orchestrated using **[OpenSpec](https://github.com/Fission-AI/OpenSpec)** (`@fission-ai/openspec`). 

Using the Spec-driven development (SDD) methodology, this workflow leverages AI-assisted commands, dynamic `tasks.md` timelines, and automatic `handover.md` synchronization (via `npm run dev:ending`) to ensure precise alignment between the architectural specs and the final implementation.
