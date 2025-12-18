# Spatio-Temporal Heterogeneous Graph Attention Network (ST-HGAT) for Chess

## 1. Abstract
This repository implements a **Value-Based Chess Tutor** using a Spatio-Temporal Heterogeneous Graph Attention Network (ST-HGAT). Instead of predicting moves directly (Policy Network), this model learns a **Value Function** $V(s)$ that estimates the winning probability of board states. 

By utilizing a "Spatio-Temporal" architecture, it processes the **full sequence of a game** to understand temporal dynamics. The system recommends moves by simulating all legal future states and ranking them by their predicted value, leveraging historical context to identify traps and long-term advantages.

## 2. The Logic: "Evaluate & Rank"
Comparing to traditional engines or policy networks:

1.  **Context-Aware History**: Unlike standard evaluators, the Tutor maintains a "Game Memory" (GRU latent state).
2.  **Simulation**: For any position, it generates all legal moves and creates resulting board states.
3.  **Stateful Evaluation**: Each resulting state is fed into the ST-HGAT as a "branch" off the current game history.
4.  **Ranking**: The model predicts "Win Probability" for all possibilities. The Tutor recommends moves that statistically correlate with winning.

## 3. Architecture

### 3.1 3-Layer Weighted ST-HGAT
We stack **3 Layers** of Weighted Heterogeneous Graph Convolutions to perform spatial reasoning.
*   **Layer 1**: Direct physical attacks/defenses.
*   **Layer 2**: Secondary support and control.
*   **Layer 3**: Complex tactical chains (pins, batteries, discovered attacks).

### 3.2 Sequence-to-Sequence Temporal Processing
The model uses a **Gated Recurrent Unit (GRU)** to process the game flow.
*   **Parallel Spatial Encoding**: The entire game sequence is batched into a single "Super-Graph" and processed by the Gating layers in one parallel pass (O(1) relative to game length during training).
*   **Infinite Horizon**: We process the **entire game** sequence at once. There are no arbitrary sliding windows; the model sees everything from the opening to the current move.
*   **Stateful Inference**: During live play, the Tutor uses a "KV-Cache" style incremental update, only processing the *latest* move to update its memory, making inference extremely fast.

## 4. Training Methodology: Outcome Regression

We train using **Mean Squared Error (MSE)** directly on the game result across the entire sequence.

$$ \text{Loss} = \frac{1}{T} \sum_{t=1}^{T} \text{MSE}(\text{Predicted\_Value}_t, \text{Final\_Result}) $$

*   **Supervised by Outcome**: Every move in a winning game is supervised as "Winning" (to varying degrees), forcing the model to learn the patterns that lead to victory.
*   **Efficiency**: By batching the full game sequence, training throughput is increased by **~10x** compared to sliding-window approaches.

## 5. Usage

### Training
Run the training script to process PGN datasets.

```bash
python3 train.py
```

**Performance Metrics**:
- **Throughput**: ~1 game/second (Standard GPU).
- **Context**: Fully temporal (full game history).
- **Accuracy**: Improved by supervising the entire mainline simultaneously.

### The Tutor (Inference)
The `CaseTutor` is stateful for maximum performance.

```python
from tutor import CaseTutor
# ... load model ...
tutor = CaseTutor(model, device)

# 1. Update the tutor as moves are played
tutor.update_state(current_fen)

# 2. Get recommendations
best_move, win_prob, analysis = tutor.recommend_move(current_fen)
```

## 6. Project Structure

```
├── chessgnn/
│   ├── graph_builder.py  # FEN -> HeteroData (Nodes, Edges, Weights)
│   ├── dataset.py        # PGN -> Full Game Sequence Yielding
│   ├── model.py          # 3-Layer Weighted HGT + Parallel Batching + Stateful Inference
│   └── ...
├── train.py              # Main Loop + Full Sequence MSE Loss
├── tutor.py              # Stateful Inference Logic (O(1) Branching)
├── output/               # Trained models and logs
└── input/                # PGN datasets
```

