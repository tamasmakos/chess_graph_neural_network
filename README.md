# Spatio-Temporal Heterogeneous Graph Attention Network (ST-HGAT) for Chess

## 1. Abstract
This repository implements a **Spatio-Temporal Heterogeneous Graph Attention Network (ST-HGAT)** designed to reason about chess positions as trajectories on a high-dimensional graph manifold, rather than sequences of static image tensors. By constructing a heterogeneous graph of physical pieces and spatial squares, explicitly modeling tactical relationships (pins, forks) via geometric "Ray" edges, and employing a Pointer Network for move selection, this architecture provides a strong inductive bias for tactical pattern recognition in chess.

## 2. How it Works (In Plain English)
For those not familiar with Graph Neural Networks, here is the intuition behind our approach:

1.  **The Board as a Web**: Instead of seeing the chessboard as a simplr 8x8 image (like a photo), we treat it as a **web of connections**. Every piece is a "node" in this web.
2.  **Encoding Tension**: We draw lines (edges) between pieces that are attacking or defending each other. The *strength* of these lines depends on the value of the pieces. A Queen attacking a King creates a very "heavy" line, while a Pawn attacking a Pawn is lighter. This helps the AI instantly recognize danger without having to "calculate" it from scratch every time.
3.  **X-Ray Vision**: We also draw special lines for Pins and Skewers. If a Bishop is looking at a King but a Pawn is in the way, we draw a "Ray" that connects them. This allows the AI to "see through" pieces and understand that moving the Pawn is illegal or dangerous (a Pin).
4.  **Selecting the Move**: Traditional engines check thousands of future moves. Our model looks at the current web of tension and assigns a score to every possible legal move. It asks: "Given the pressure on the board, which move releases tension or increases my advantage?" and selects the one with the highest score.

## 3. Theoretical Formulation: The Chess Manifold
We define the game of chess $\Gamma$ not as a sequence of grid matrices, but as a trajectory through a dynamic heterogeneous graph manifold $\mathcal{G} = \{G_0, G_1, \dots, G_T\}$.

### 2.1 Heterogeneous Node Set $\mathcal{V}_t$
The graph $G_t$ consists of two distinct node types:
*   **Piece Nodes ($\mathcal{V}_t^P$)**: Represent physical pieces ($N \le 32$). Feature vector $\mathbf{x}_i \in \mathbb{R}^{d_P}$ encodes `[type, color, value, pos_x, pos_y]`.
*   **Square Nodes ($\mathcal{V}_t^S$)**: Represent the spatial board distinct from occupancy ($N=64$). Feature vector $\mathbf{s}_j \in \mathbb{R}^{d_S}$ encodes `[coord_x, coord_y, occupancy_flag]`.

### 2.2 Multi-Relational Weighted Edge Tensor $\mathcal{E}_t$
Edges are constructed to interpret the "force field" of the board:
*   **Interaction Edges** ($\mathcal{E}_{int}$): Connect attacking/defending pieces. Weights are derived from the sigmoid of the material value difference $\sigma(\alpha \Delta V)$, explicitly encoding the "quality" of a potential trade (e.g., trading a Queen for a Pawn is heavily penalized/weighted).
*   **Ray-Alignment Edges** ($\mathcal{E}_{ray}$): Connect pieces along the same rank, file, or diagonal specifically to model non-local "X-ray" tactics like Pins and Skewers.
*   **Spatial Adjacency** ($\mathcal{E}_{grid}$): Connects square nodes based on Chebyshev distance ($L_\infty=1$), providing the underlying lattice geometry.
*   **Positional Edges** ($\mathcal{E}_{on}$): Bipartite edges linking Piece nodes to the Square nodes they occupy.

## 4. Architecture Specification

The model is implemented in `chessgnn/model.py` and processing logic in `chessgnn/graph_builder.py`.

### 4.1 Module 1: Weighted Heterogeneous Graph Transformer (WeightedHGT)
We extend the standard HGT operator to incorporate scalar edge weights $w_{st}$ directly into the multi-head attention mechanism. For a source node $s$ and target $t$ with relationship $\phi(e)$:

$$
\text{Attn}(s, t) = \text{Softmax}\left( \frac{K(s) W_{\phi}^Q Q(t)}{\sqrt{d}} \cdot (1 + \lambda w_{st}) \right)
$$

This modulation forces the network to attend more heavily to high-stakes interactions (e.g., threats to the King or Queen) defined by the graph builder.

### 4.2 Module 2: Ray-Alignment Block
To detect pinned pieces (which cannot move), we utilize features aggregated along $\mathcal{E}_{ray}$. The implementation creates direct edges between aligned pieces, carrying attributes like `blocking_count` to allow the network to infer visibility and obstruction.

### 4.3 Module 3: Temporal Evolution
The input to the network is a sliding window of graphs $[G_{t-L}, \dots, G_t]$. We employ a Recurrent Neural Network (GRU) over the sequence of graph embeddings to capture the momentum and history of the game (e.g., "Has this piece been dormant?").

### 4.4 Module 4: Pointer Network Policy Head
Unlike AlphaZero which predicts a policy over a fixed output space (4672 moves), we use a **Pointer Network**.
1.  **Candidate Selection**: We generate a subgraph of legal moves $M_t = \{(u, v) \mid \text{move } u \to v \text{ is legal}\}$.
2.  **Scoring**: For each candidate move $k$, we concatenate the learned embeddings of the source piece $\mathbf{h}_u$ and the target (piece or square) $\mathbf{h}_v$:
    $$ u_k = \text{MLP}([\mathbf{h}_u || \mathbf{h}_v]) $$
3.  **Policy**: $P(m_k) = \text{Softmax}(u_k)$.

This allows the network to handle the variable action space of chess naturally and leverages the rich node representations learned by the GNN.

## 5. Implementation Details

### Data Pipeline
*   **Source**: PGN files parsed via `python-chess`.
*   **Graph Construction**: `chessgnn.graph_builder.ChessGraphBuilder` builds `PyG` (PyTorch Geometric) `HeteroData` objects on the fly.
*   **Batching**: Custom collation handles variable-sized graphs and disjoint unions.

### Project Structure
```
├── chessgnn/
│   ├── graph_builder.py  # FEN -> HeteroData logic (Nodes, Edges, Weights)
│   ├── dataset.py        # PGN -> Sliding Window Sequences
│   ├── model.py          # ST-HGAT + Pointer Network implementation
│   └── ...
├── train.py              # Training loop with Curriculum support
└── input/                # PGN datasets
```

## 6. Usage

### Prerequisites
*   Python 3.10+
*   PyTorch >= 2.0
*   PyTorch Geometric (PyG)
*   python-chess

### Training
To start training on the provided sample dataset (Lichess subset):

```bash
python3 train.py
```

Configuration parameters (Batch size, Learning Rate, Window Size) are defined at the top of `train.py`.

## 7. Future Work
*   **ListMLE Loss**: Currently trained with Cross-Entropy on the single best move. Implementing ListMLE would allow ranking *all* legal moves based on Stockfish evaluations.
*   **Dynamic Centrality**: Integrating calculating Betweenness Centrality on the fly as a dynamic node feature.
*   **Transmissibility Gate**: Refine the Ray-Block using a learned gating mechanism based on obstruction density.
