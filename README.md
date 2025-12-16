# Spatio-Temporal Heterogeneous Graph Attention Network (ST-HGAT) for Chess

## 1. Abstract
This repository implements a **Value-Based Chess Tutor** using a Spatio-Temporal Heterogeneous Graph Attention Network (ST-HGAT). Instead of predicting moves directly (Policy Network), this model learns to **evaluate board states** (Value Network) and ranks legal moves by simulating their outcomes. By constructing a heterogeneous graph of physical pieces, squares, and "Ray" edges (for pins/forks), the model learns an inductive bias for tactical patterns. Training utilizes **Contrastive Ranking Loss** to explicitly differentiate between "played" moves (ground truth) and random legal moves, forcing the model to learn relative move quality.

## 2. The Logic: "Evaluate & Rank"
Comparing to traditional engines or policy networks:

1.  **See the Future**: For any given position, the Tutor generates all legal moves.
2.  **Simulation**: It applies each move to create $N$ resulting board states.
3.  **Graph Evaluation**: Each resulting state is converted into a graph and fed into the GNN.
4.  **Ranking**: The GNN predicts a "Win Probability" for each state. The Tutor recommends the moves that lead to the highest win probability for the current player.

**Why this matters**: This mimics human calculation ("If I go here, the position looks good. If I go there, it looks bad") rather than just pattern matching ("Masters play e4 here").

## 3. Architecture

The model is defined in `chessgnn/model.py`.

### 3.1 3-Layer Weighted ST-HGAT
We stack **3 Layers** of Weighted Heterogeneous Graph Convolutions.
*   **Layer 1 (Direct Interaction)**: "My Knight attacks your Pawn."
*   **Layer 2 (Defense/Support)**: "Your Pawn is defended by your Queen."
*   **Layer 3 (Deep Tactics)**: "The Queen is pinned to the King by my Rook."
This depth allows the model to see complex tactical chains that a single-layer GNN would miss.

### 3.2 Unbounded Linear Value Head
Instead of using `Tanh` (which saturates predictions at -1/1) or `Sigmoid` (0-1), we use a linear identity output. This prevents gradient vanishing during training and allows the model to express strong confidence (e.g., scores > 1.0 for "Winning" or < -1.0 for "Losing"). For display, these scores are soft-clipped to a 0-100% Win Probability.

## 4. Training Methodology: Contrastive Ranking

We do not train on "Answer Key" labels (Cross-Entropy). We train on **comparison**:

$$ \text{Loss} = \text{MSE}(\text{State}) + \lambda \times \max(0, \text{Score}_{\text{Bad}} - \text{Score}_{\text{Good}} + \text{Margin}) $$

*   **Positive Sample**: The move actually played by the Grandmaster in the game.
*   **Negative Sample**: A random legal move *not* played in that position.
*   **Goal**: The model must assign a higher score to the GM's position than the random position. This explicitly teaches the model "Move A is better than Move B."

## 5. Usage

### Training with Deep Inspection
Run the training script to see the Tutor learn in real-time.

```bash
python3 train.py
```

**Deep Inspection Logs**:
Every 100 steps, the trainer pauses to perform a full analysis of the current board. It evaluates all legal moves and prints a ranked list:

```text
Evaluating 34 legal moves...
--- Top 5 Recommended Moves ---
#1: e4e5 | WinProb: 55.2% | Raw: 0.15
#2: g1f3 | WinProb: 54.1% | Raw: 0.12
...
--- Worst 3 Blunders ---
X : g2g4 | WinProb: 32.5% | Raw: -0.41
```

### Inference (The Tutor)
Use `tutor.py` to get recommendations for any FEN string.

```python
from tutor import CaseTutor
# ... load model ...
tutor = CaseTutor(model, device)
best_move, win_prob, analysis = tutor.recommend_move("rnbqkbnr/...")
print(f"Recommended: {best_move} ({win_prob:.1f}%)")
```

## 6. Project Structure

```
├── chessgnn/
│   ├── graph_builder.py  # FEN -> HeteroData (Nodes, Edges, Weights)
│   ├── dataset.py        # PGN -> Graphs + Played Move Extraction
│   ├── model.py          # 3-Layer ST-HGAT + Value Head
│   └── ...
├── train.py              # Main Loop + Deep Inspection + Ranking Loss
├── tutor.py              # Inference Logic (Evaluate & Rank)
└── input/                # PGN datasets
```
