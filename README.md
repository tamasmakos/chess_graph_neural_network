# Spatio-Temporal Heterogeneous Graph Attention Network (ST-HGAT) for Chess

## 1. Abstract
This repository implements a **Value-Based Chess Tutor** using a Spatio-Temporal Heterogeneous Graph Attention Network (ST-HGAT). Instead of predicting moves directly (Policy Network), this model learns a **Value Function** $V(s)$ that estimates the winning probability of board states. By utilizing a "Spatio-Temporal" architecture, it processes the full sequence of recent moves to understand game dynamics, capturing patterns that single-frame evaluations miss. The system recommends moves by simulating all legal future states and ranking them by their predicted value.

## 2. The Logic: "Evaluate & Rank"
Comparing to traditional engines or policy networks:

1.  **See the Future**: For any given position, the Tutor generates all legal moves.
2.  **Simulation**: It applies each move to create resulting board states.
3.  **Temporal Evaluation**: Each resulting state (and its history) is fed into the ST-HGAT.
4.  **Ranking**: The model predicts a "Win Probability" for each state. The Tutor recommends the moves that lead to the best outcome for the current player.

**Why this matters**: This eliminates the "Easy Negative" trap. Instead of learning to distinguish "valid moves" from "random blunders" (which is trivial), the model is trained purely on game outcomes, forcing it to learn the true strategic value of positions.

## 3. Architecture

The model is defined in `chessgnn/model.py`.

### 3.1 3-Layer Weighted ST-HGAT
We stack **3 Layers** of Weighted Heterogeneous Graph Convolutions to perform spatial reasoning.
*   **Layer 1**: Direct physical attacks/defenses.
*   **Layer 2**: Secondary support and control.
*   **Layer 3**: Complex tactical chains (pins, batteries).

### 3.2 True Temporal Processing
Unlike standard GNNs that see only a snapshot, this model effectively uses **Temporal Memory (GRU)**.
*   **Input**: A sequence of the last 8 board states (sliding window).
*   **Process**: The GNN encoder runs on each frame to extract spatial embeddings. These are fed sequentially into a GRU (Gated Recurrent Unit).
*   **Output**: The final GRU state, enriched with historical context, drives the prediction.

### 3.3 Bounded Value Head
The model uses `nn.Softsign` activation in the final layer.
*   **Bounded Output**: Strictly outputs values in $(-1, 1)$, matching the game result targets (-1: Black Win, 0: Draw, 1: White Win).
*   **Stability**: Prevents the "infinite score" problem where unbounded regression models explode gradients by trying to be "too correct" (e.g., scoring +10.0 for a checkmate).

## 4. Training Methodology: Outcome Regression

We train using **Mean Squared Error (MSE)** directly on the game result.

$$ \text{Loss} = \text{MSE}(\text{Predicted\_Value}, \text{Game\_Result}) $$

*   **No Negative Sampling**: We do not use random negatives. The model learns strictly from the ground truth outcome of Grandmaster games.
*   **Value Function**: By regressing to the final result (who actually won), the model implicitly learns that "good positions" are those statistically correlated with winning, without needing explicit "bad move" examples.

## 5. Usage

### Training with Deep Inspection
Run the training script to see the Tutor learn in real-time. Features **Parallel Data Loading** to maximize GPU utilization.

```bash
python3 train.py
```

**Deep Inspection Logs**:
Every 100 steps, the trainer pauses to perform a full analysis. It generates valid temporal sequences for all legal moves to see how the model evaluates potential futures:

```text
Evaluating 34 legal moves with history context...
--- Top 5 Recommended Moves ---
#1: e4e5 | WinProb: 55.2% | Raw: 0.15
#2: g1f3 | WinProb: 54.1% | Raw: 0.12
...
```

### Inference (The Tutor)
Use `tutor.py` to get recommendations.

```python
from tutor import CaseTutor
# ... load model ...
tutor = CaseTutor(model, device)
best_move, win_prob, analysis = tutor.recommend_move("rnbqkbnr/...")
print(f"Recommended: {best_move} ({win_prob:.1f}%)")
# Note: tutor.py currently implements a basic 1-ply simulation.
```

## 6. Project Structure

```
├── chessgnn/
│   ├── graph_builder.py  # FEN -> HeteroData (Nodes, Edges, Weights)
│   ├── dataset.py        # PGN -> Sharded Parallel Dataset + Short Game Filter
│   ├── model.py          # 3-Layer Weighted HGT + Temporal GRU + Softsign
│   └── ...
├── train.py              # Main Loop + Deep Inspection + Parallel Workers
├── tutor.py              # Inference Logic (Evaluate & Rank)
└── input/                # PGN datasets
```
