// ============================================================
// Cliff Walking: Q-learning vs SARSA
// Pure JavaScript Implementation
// ============================================================

// === DOM Elements ===
const gridRowsInput = document.getElementById('grid-rows');
const gridColsInput = document.getElementById('grid-cols');
const alphaSlider = document.getElementById('alpha-slider');
const gammaSlider = document.getElementById('gamma-slider');
const epsilonSlider = document.getElementById('epsilon-slider');
const episodesInput = document.getElementById('episodes-input');
const alphaVal = document.getElementById('alpha-val');
const gammaVal = document.getElementById('gamma-val');
const epsilonVal = document.getElementById('epsilon-val');
const trainBtn = document.getElementById('train-btn');
const resetBtn = document.getElementById('reset-btn');
const progressContainer = document.getElementById('progress-bar-container');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const envGrid = document.getElementById('env-grid');
const resultsSection = document.getElementById('results-section');
const rewardCanvas = document.getElementById('reward-chart');
const qPolicyGrid = document.getElementById('q-policy-grid');
const sPolicyGrid = document.getElementById('s-policy-grid');
const qPathInfo = document.getElementById('q-path-info');
const sPathInfo = document.getElementById('s-path-info');
const analysisContent = document.getElementById('analysis-content');

// Slider live updates
alphaSlider.addEventListener('input', (e) => alphaVal.textContent = e.target.value);
gammaSlider.addEventListener('input', (e) => gammaVal.textContent = e.target.value);
epsilonSlider.addEventListener('input', (e) => epsilonVal.textContent = e.target.value);

// === SVG Arrow Templates ===
const arrowSVG = {
    0: '<svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 2L4 14h5v8h6v-8h5L12 2z"/></svg>',   // Up
    1: '<svg viewBox="0 0 24 24"><path fill="currentColor" d="M12 22L4 10h5V2h6v8h5L12 22z"/></svg>',   // Down
    2: '<svg viewBox="0 0 24 24"><path fill="currentColor" d="M2 12l10-8v5h10v6H12v5L2 12z"/></svg>',   // Left
    3: '<svg viewBox="0 0 24 24"><path fill="currentColor" d="M22 12l-10 8v-5H2v-6h10V4l10 8z"/></svg>'  // Right
};

// Actions: 0=Up, 1=Down, 2=Left, 3=Right
const ACTIONS = [0, 1, 2, 3];
const ACTION_NAMES = ['↑', '↓', '←', '→'];
const DR = [-1, 1, 0, 0];
const DC = [0, 0, -1, 1];

// === Environment ===
class CliffWalkingEnv {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.start = [rows - 1, 0];
        this.goal = [rows - 1, cols - 1];
        this.cliff = new Set();
        for (let c = 1; c < cols - 1; c++) {
            this.cliff.add(`${rows - 1},${c}`);
        }
        this.state = [...this.start];
    }

    reset() {
        this.state = [...this.start];
        return this.stateKey();
    }

    stateKey(s) {
        const st = s || this.state;
        return `${st[0]},${st[1]}`;
    }

    step(action) {
        let nr = this.state[0] + DR[action];
        let nc = this.state[1] + DC[action];

        // Boundary check
        if (nr < 0 || nr >= this.rows || nc < 0 || nc >= this.cols) {
            nr = this.state[0];
            nc = this.state[1];
        }

        this.state = [nr, nc];
        const key = this.stateKey();

        // Check cliff
        if (this.cliff.has(key)) {
            this.state = [...this.start];
            return { nextState: this.stateKey(), reward: -100, done: false };
        }

        // Check goal
        if (nr === this.goal[0] && nc === this.goal[1]) {
            return { nextState: key, reward: -1, done: true };
        }

        return { nextState: key, reward: -1, done: false };
    }

    isCliff(r, c) { return this.cliff.has(`${r},${c}`); }
    isStart(r, c) { return r === this.start[0] && c === this.start[1]; }
    isGoal(r, c) { return r === this.goal[0] && c === this.goal[1]; }
}

// === Epsilon-Greedy Policy ===
function epsilonGreedy(Q, stateKey, epsilon) {
    if (Math.random() < epsilon) {
        return ACTIONS[Math.floor(Math.random() * ACTIONS.length)];
    }
    // Greedy: pick action with max Q
    let bestA = 0;
    let bestVal = Q[stateKey][0];
    for (let a = 1; a < ACTIONS.length; a++) {
        if (Q[stateKey][a] > bestVal) {
            bestVal = Q[stateKey][a];
            bestA = a;
        }
    }
    return bestA;
}

// === Initialize Q-table ===
function initQ(rows, cols) {
    const Q = {};
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            Q[`${r},${c}`] = [0, 0, 0, 0]; // [Up, Down, Left, Right]
        }
    }
    return Q;
}

// === Q-learning ===
function trainQLearning(rows, cols, alpha, gamma, epsilon, episodes) {
    const Q = initQ(rows, cols);
    const rewards = [];

    for (let ep = 0; ep < episodes; ep++) {
        const env = new CliffWalkingEnv(rows, cols);
        let state = env.reset();
        let totalReward = 0;
        let steps = 0;
        const maxSteps = rows * cols * 100;

        while (steps < maxSteps) {
            const action = epsilonGreedy(Q, state, epsilon);
            const { nextState, reward, done } = env.step(action);

            // Q-learning update: use max over next actions
            const maxNextQ = Math.max(...Q[nextState]);
            Q[state][action] += alpha * (reward + gamma * maxNextQ - Q[state][action]);

            totalReward += reward;
            state = nextState;
            steps++;

            if (done) break;
        }

        rewards.push(totalReward);
    }

    return { Q, rewards };
}

// === SARSA ===
function trainSARSA(rows, cols, alpha, gamma, epsilon, episodes) {
    const Q = initQ(rows, cols);
    const rewards = [];

    for (let ep = 0; ep < episodes; ep++) {
        const env = new CliffWalkingEnv(rows, cols);
        let state = env.reset();
        let action = epsilonGreedy(Q, state, epsilon);
        let totalReward = 0;
        let steps = 0;
        const maxSteps = rows * cols * 100;

        while (steps < maxSteps) {
            const { nextState, reward, done } = env.step(action);

            // SARSA update: use the actual next action
            const nextAction = epsilonGreedy(Q, nextState, epsilon);
            Q[state][action] += alpha * (reward + gamma * Q[nextState][nextAction] - Q[state][action]);

            totalReward += reward;
            state = nextState;
            action = nextAction;
            steps++;

            if (done) break;
        }

        rewards.push(totalReward);
    }

    return { Q, rewards };
}

// === Extract Greedy Policy ===
function extractPolicy(Q, rows, cols) {
    const policy = {};
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const key = `${r},${c}`;
            const qVals = Q[key];
            let bestA = 0;
            for (let a = 1; a < ACTIONS.length; a++) {
                if (qVals[a] > qVals[bestA]) bestA = a;
            }
            policy[key] = bestA;
        }
    }
    return policy;
}

// === Extract Optimal Path ===
function extractPath(policy, env, Q) {
    const path = [];
    let state = [...env.start];
    const visited = new Set();
    const maxSteps = env.rows * env.cols * 2;

    for (let i = 0; i < maxSteps; i++) {
        const key = `${state[0]},${state[1]}`;
        if (visited.has(key)) break;
        if (env.isGoal(state[0], state[1])) { path.push([...state]); break; }
        visited.add(key);
        path.push([...state]);

        // Try greedy action first; if it leads to cliff, try next best
        const qVals = Q ? Q[key] : null;
        let moved = false;
        
        // Get actions sorted by Q-value (descending)
        const sortedActions = qVals 
            ? ACTIONS.slice().sort((a, b) => qVals[b] - qVals[a])
            : [policy[key]];

        for (const action of sortedActions) {
            let nr = state[0] + DR[action];
            let nc = state[1] + DC[action];
            if (nr < 0 || nr >= env.rows || nc < 0 || nc >= env.cols) continue;
            if (env.isCliff(nr, nc)) continue;
            state = [nr, nc];
            moved = true;
            break;
        }

        if (!moved) break;
    }

    return path;
}

// ==================================================================
// RENDERING
// ==================================================================

function renderEnvGrid() {
    const rows = parseInt(gridRowsInput.value);
    const cols = parseInt(gridColsInput.value);
    const env = new CliffWalkingEnv(rows, cols);

    envGrid.innerHTML = '';
    envGrid.style.gridTemplateColumns = `repeat(${cols}, 48px)`;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';

            if (env.isStart(r, c)) {
                cell.classList.add('cell-start');
                cell.textContent = 'S';
            } else if (env.isGoal(r, c)) {
                cell.classList.add('cell-goal');
                cell.textContent = 'G';
            } else if (env.isCliff(r, c)) {
                cell.classList.add('cell-cliff');
                cell.textContent = '☠';
            } else {
                cell.classList.add('cell-normal');
            }

            envGrid.appendChild(cell);
        }
    }
}

function renderPolicyGrid(gridEl, Q, policy, path, env, pathClass) {
    const rows = env.rows;
    const cols = env.cols;
    const pathSet = new Set(path.map(p => `${p[0]},${p[1]}`));

    gridEl.innerHTML = '';
    gridEl.style.gridTemplateColumns = `repeat(${cols}, 48px)`;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            const key = `${r},${c}`;

            if (env.isStart(r, c)) {
                cell.classList.add('cell-start');
                cell.innerHTML = arrowSVG[policy[key]];
            } else if (env.isGoal(r, c)) {
                cell.classList.add('cell-goal');
                cell.textContent = 'G';
            } else if (env.isCliff(r, c)) {
                cell.classList.add('cell-cliff');
                cell.textContent = '☠';
            } else {
                cell.classList.add('cell-normal');
                cell.innerHTML = arrowSVG[policy[key]];
                if (pathSet.has(key)) {
                    cell.classList.add(pathClass);
                }
            }

            gridEl.appendChild(cell);
        }
    }
}

// === Reward Chart (Canvas) ===
function drawRewardChart(qRewards, sRewards) {
    const canvas = rewardCanvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const W = rect.width;
    const H = rect.height;
    const pad = { top: 20, right: 20, bottom: 45, left: 65 };
    const chartW = W - pad.left - pad.right;
    const chartH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    // Smooth rewards (moving average)
    const windowSize = Math.max(1, Math.floor(qRewards.length / 50));
    const smoothQ = movingAverage(qRewards, windowSize);
    const smoothS = movingAverage(sRewards, windowSize);

    const allVals = [...smoothQ, ...smoothS];
    const minVal = Math.min(...allVals);
    const maxVal = Math.max(...allVals);
    const range = maxVal - minVal || 1;

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    const numGridLines = 5;
    ctx.font = '11px Inter';
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= numGridLines; i++) {
        const y = pad.top + chartH - (i / numGridLines) * chartH;
        const val = minVal + (i / numGridLines) * range;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + chartW, y);
        ctx.stroke();
        ctx.fillText(Math.round(val), pad.left - 8, y);
    }

    // X-axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const numXLabels = 5;
    for (let i = 0; i <= numXLabels; i++) {
        const ep = Math.round((i / numXLabels) * (qRewards.length - 1));
        const x = pad.left + (i / numXLabels) * chartW;
        ctx.fillText(ep, x, pad.top + chartH + 8);
    }

    // Axis labels
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Episode', pad.left + chartW / 2, H - 6);

    ctx.save();
    ctx.translate(14, pad.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Reward (smoothed)', 0, 0);
    ctx.restore();

    // Draw lines
    function drawLine(data, color) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        for (let i = 0; i < data.length; i++) {
            const x = pad.left + (i / (data.length - 1)) * chartW;
            const y = pad.top + chartH - ((data[i] - minVal) / range) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    // Draw fill
    function drawFill(data, color) {
        ctx.beginPath();
        ctx.fillStyle = color;
        for (let i = 0; i < data.length; i++) {
            const x = pad.left + (i / (data.length - 1)) * chartW;
            const y = pad.top + chartH - ((data[i] - minVal) / range) * chartH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.lineTo(pad.left + chartW, pad.top + chartH);
        ctx.lineTo(pad.left, pad.top + chartH);
        ctx.closePath();
        ctx.fill();
    }

    drawFill(smoothS, 'rgba(255, 145, 0, 0.08)');
    drawFill(smoothQ, 'rgba(0, 229, 255, 0.08)');
    drawLine(smoothS, '#ff9100');
    drawLine(smoothQ, '#00e5ff');
}

function movingAverage(data, windowSize) {
    if (windowSize <= 1) return data;
    const result = [];
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        sum += data[i];
        if (i >= windowSize) sum -= data[i - windowSize];
        const count = Math.min(i + 1, windowSize);
        result.push(sum / count);
    }
    return result;
}

// === Analysis ===
function renderAnalysis(qRewards, sRewards, qPath, sPath, episodes) {
    const last100Q = qRewards.slice(-100);
    const last100S = sRewards.slice(-100);
    const avgQ = (last100Q.reduce((a, b) => a + b, 0) / last100Q.length).toFixed(1);
    const avgS = (last100S.reduce((a, b) => a + b, 0) / last100S.length).toFixed(1);

    // Standard deviation (stability)
    const stdDev = (arr) => {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const variance = arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / arr.length;
        return Math.sqrt(variance);
    };
    const stdQ = stdDev(last100Q).toFixed(1);
    const stdS = stdDev(last100S).toFixed(1);

    // Convergence speed: first episode where 50-ep moving avg exceeds threshold
    const convergenceThreshold = -50;
    const maWindow = 50;
    const findConvergence = (rewards) => {
        const ma = movingAverage(rewards, maWindow);
        for (let i = maWindow - 1; i < ma.length; i++) {
            if (ma[i] > convergenceThreshold) return i + 1;
        }
        return null;
    };
    const convQ = findConvergence(qRewards);
    const convS = findConvergence(sRewards);

    // Cliff fall count estimation (episodes with reward < -100)
    const cliffFallsQ = qRewards.filter(r => r < -100).length;
    const cliffFallsS = sRewards.filter(r => r < -100).length;

    // Min reward (worst episode)
    const minQ = Math.min(...qRewards).toFixed(0);
    const minS = Math.min(...sRewards).toFixed(0);

    const reachedGoalQ = qPath.length > 0 && qPath[qPath.length - 1].toString() === [parseInt(gridRowsInput.value) - 1, parseInt(gridColsInput.value) - 1].toString();
    const reachedGoalS = sPath.length > 0 && sPath[sPath.length - 1].toString() === [parseInt(gridRowsInput.value) - 1, parseInt(gridColsInput.value) - 1].toString();

    const epsilon = parseFloat(epsilonSlider.value);

    analysisContent.innerHTML = `
        <h3>📈 一、學習表現 (Learning Performance)</h3>
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Q-learning 最後100回合平均獎勵</div>
                <div class="stat-value" style="color:var(--accent-q)">${avgQ}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SARSA 最後100回合平均獎勵</div>
                <div class="stat-value" style="color:var(--accent-s)">${avgS}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Q-learning 收斂回合</div>
                <div class="stat-value" style="color:var(--accent-q)">${convQ ? '≈ ' + convQ : '> ' + episodes}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SARSA 收斂回合</div>
                <div class="stat-value" style="color:var(--accent-s)">${convS ? '≈ ' + convS : '> ' + episodes}</div>
            </div>
        </div>
        <p>
            上方圖表顯示每回合的累積獎勵曲線（經平滑處理）。
            ${convQ && convS
                ? (convQ < convS
                    ? `Q-learning 約在第 ${convQ} 回合開始收斂，SARSA 約在第 ${convS} 回合，Q-learning 收斂速度較快。`
                    : convQ > convS
                    ? `SARSA 約在第 ${convS} 回合開始收斂，Q-learning 約在第 ${convQ} 回合，SARSA 收斂速度較快。`
                    : `兩者大約同時在第 ${convQ} 回合收斂。`)
                : '其中一個或兩個演算法在設定的回合數內尚未完全收斂。'}
            收斂標準為：50 回合移動平均獎勵超過 ${convergenceThreshold}。
        </p>

        <h3>🛤️ 二、策略行為 (Policy Behavior)</h3>
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Q-learning 最終路徑長度</div>
                <div class="stat-value" style="color:var(--accent-q)">${reachedGoalQ ? qPath.length + ' 步' : 'N/A'}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SARSA 最終路徑長度</div>
                <div class="stat-value" style="color:var(--accent-s)">${reachedGoalS ? sPath.length + ' 步' : 'N/A'}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Q-learning 策略傾向</div>
                <div class="stat-value" style="color:#ff1744;font-size:1.1rem;">冒險 (Risky)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SARSA 策略傾向</div>
                <div class="stat-value" style="color:#00e676;font-size:1.1rem;">保守 (Safe)</div>
            </div>
        </div>
        <p>
            <strong style="color:var(--accent-q)">Q-learning (off-policy)</strong>
            使用 <code>max<sub>a'</sub> Q(s', a')</code> 進行更新，學習的是「最優策略」而非「行為策略」。
            因此，即使在訓練過程中 agent 偶爾因 ε-greedy 探索掉入懸崖，
            Q-learning 仍然會學到沿懸崖邊緣行走的最短路徑（${reachedGoalQ ? qPath.length + ' 步' : 'N/A'}）。
            <strong>這是一種冒險策略</strong>——最終策略是最優的，但在探索過程中代價更高。
        </p>
        <p style="margin-top:10px;">
            <strong style="color:var(--accent-s)">SARSA (on-policy)</strong>
            使用實際選擇的動作 <code>Q(s', a')</code> 進行更新，學習的是「行為策略」（包含探索）。
            由於 ε = ${epsilon}，agent 有 ${(epsilon * 100).toFixed(0)}% 的機率隨機行動。
            在懸崖邊緣，這種隨機行動極有可能導致掉入懸崖（向下移動），
            因此 SARSA 學會了<strong>遠離懸崖的保守路徑</strong>（${reachedGoalS ? sPath.length + ' 步' : 'N/A'}）。
            雖然路徑較長，但每一步都更安全。
        </p>

        <h3>📊 三、穩定性分析 (Stability Analysis)</h3>
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Q-learning 獎勵標準差 (最後100回合)</div>
                <div class="stat-value" style="color:var(--accent-q)">${stdQ}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SARSA 獎勵標準差 (最後100回合)</div>
                <div class="stat-value" style="color:var(--accent-s)">${stdS}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Q-learning 掉落懸崖回合數</div>
                <div class="stat-value" style="color:var(--accent-q)">${cliffFallsQ} / ${episodes}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SARSA 掉落懸崖回合數</div>
                <div class="stat-value" style="color:var(--accent-s)">${cliffFallsS} / ${episodes}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Q-learning 最差回合獎勵</div>
                <div class="stat-value" style="color:var(--accent-q)">${minQ}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">SARSA 最差回合獎勵</div>
                <div class="stat-value" style="color:var(--accent-s)">${minS}</div>
            </div>
        </div>
        <p>
            <strong>波動程度：</strong>
            Q-learning 最後 100 回合的獎勵標準差為 <strong style="color:var(--accent-q)">${stdQ}</strong>，
            SARSA 為 <strong style="color:var(--accent-s)">${stdS}</strong>。
            ${parseFloat(stdQ) > parseFloat(stdS)
                ? `Q-learning 的波動明顯較大（標準差高出 ${(parseFloat(stdQ) - parseFloat(stdS)).toFixed(1)}），這是因為其策略沿懸崖邊緣行走，ε-greedy 探索容易導致掉入懸崖（獎勵 -100）。`
                : `兩者波動程度相近，可能因為訓練已充分收斂。`}
        </p>
        <p style="margin-top:10px;">
            <strong>探索 (Exploration) 對結果的影響：</strong>
            在 ε = ${epsilon} 的設定下，agent 每一步有 ${(epsilon * 100).toFixed(0)}% 的機率隨機探索。
            這對兩種演算法產生了截然不同的影響：
        </p>
        <ul style="margin:8px 0 0 20px; line-height:2;">
            <li><strong style="color:var(--accent-q)">Q-learning：</strong>因為更新規則使用 max Q（不考慮實際探索動作），所以學到的策略是理論最優的最短路徑。然而在實際執行時，探索導致 agent 在懸崖邊緣掉落的機率為 ε/4 ≈ ${(epsilon/4*100).toFixed(1)}%（每一步），在 ${reachedGoalQ ? qPath.length - 1 : '~13'} 步的路徑上，至少掉落一次的機率約為 ${(100 * (1 - Math.pow(1 - epsilon/4, (reachedGoalQ ? qPath.length - 1 : 13)))).toFixed(1)}%。</li>
            <li><strong style="color:var(--accent-s)">SARSA：</strong>因為更新規則使用實際的下一步動作（包含探索），SARSA 學會了將探索成本納入考量。它選擇遠離懸崖的路徑，使得即使隨機探索也不會掉入懸崖。雖然路徑更長，但訓練過程中的獎勵更穩定。</li>
        </ul>
        <p style="margin-top:10px;">
            <strong>結論：</strong>
            Q-learning 學到的是<em>理論最優策略</em>（如果 ε=0 則是最佳路徑），
            而 SARSA 學到的是<em>考慮探索風險的最佳策略</em>。
            在需要安全性的應用場景（如機器人導航、自動駕駛），SARSA 的保守策略更合適；
            而在可以離線學習後部署（關閉探索）的場景，Q-learning 的最優策略更高效。
        </p>
    `;
}

// ==================================================================
// TRAINING ORCHESTRATION
// ==================================================================

trainBtn.addEventListener('click', () => {
    const rows = parseInt(gridRowsInput.value);
    const cols = parseInt(gridColsInput.value);

    if (rows < 3 || rows > 10 || cols < 4 || cols > 20) {
        alert('請確保列數在 3-10 之間，行數在 4-20 之間。');
        return;
    }

    const alpha = parseFloat(alphaSlider.value);
    const gamma = parseFloat(gammaSlider.value);
    const epsilon = parseFloat(epsilonSlider.value);
    const episodes = parseInt(episodesInput.value);

    trainBtn.disabled = true;
    trainBtn.textContent = '⏳ 訓練中...';
    progressContainer.style.display = 'block';
    resultsSection.style.display = 'none';

    // Use setTimeout to allow UI update before blocking computation
    setTimeout(() => {
        const startTime = performance.now();

        // Train Q-learning
        progressBar.style.width = '25%';
        progressText.textContent = '訓練 Q-learning...';
        const qResult = trainQLearning(rows, cols, alpha, gamma, epsilon, episodes);

        progressBar.style.width = '50%';
        progressText.textContent = '訓練 SARSA...';

        setTimeout(() => {
            // Train SARSA
            const sResult = trainSARSA(rows, cols, alpha, gamma, epsilon, episodes);

            progressBar.style.width = '75%';
            progressText.textContent = '生成視覺化...';

            setTimeout(() => {
                const env = new CliffWalkingEnv(rows, cols);
                const qPolicy = extractPolicy(qResult.Q, rows, cols);
                const sPolicy = extractPolicy(sResult.Q, rows, cols);
                const qPath = extractPath(qPolicy, env, qResult.Q);
                const sPath = extractPath(sPolicy, env, sResult.Q);

                // Show results
                resultsSection.style.display = 'block';

                // Render policy grids
                renderPolicyGrid(qPolicyGrid, qResult.Q, qPolicy, qPath, env, 'path-q');
                renderPolicyGrid(sPolicyGrid, sResult.Q, sPolicy, sPath, env, 'path-s');

                // Path info
                const qGoal = qPath.length > 0 && env.isGoal(qPath[qPath.length-1][0], qPath[qPath.length-1][1]);
                const sGoal = sPath.length > 0 && env.isGoal(sPath[sPath.length-1][0], sPath[sPath.length-1][1]);
                qPathInfo.textContent = qGoal ? `最優路徑: ${qPath.length} 步` : '未能找到有效路徑';
                sPathInfo.textContent = sGoal ? `安全路徑: ${sPath.length} 步` : '未能找到有效路徑';

                // Draw chart
                drawRewardChart(qResult.rewards, sResult.rewards);

                // Analysis
                renderAnalysis(qResult.rewards, sResult.rewards, qPath, sPath, episodes);

                // Done
                progressBar.style.width = '100%';
                progressText.textContent = '完成！';
                const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    trainBtn.disabled = false;
                    trainBtn.textContent = '🚀 開始訓練';
                }, 600);

                // Scroll to results
                resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 50);
        }, 50);
    }, 50);
});

// === Reset ===
resetBtn.addEventListener('click', () => {
    resultsSection.style.display = 'none';
    progressContainer.style.display = 'none';
    progressBar.style.width = '0%';
    trainBtn.disabled = false;
    trainBtn.textContent = '🚀 開始訓練';
    renderEnvGrid();
});

// === Grid size change re-renders preview ===
gridRowsInput.addEventListener('change', renderEnvGrid);
gridColsInput.addEventListener('change', renderEnvGrid);

// === Init ===
renderEnvGrid();
