"""
Single-position visualization using the analysis from `position_to_graph.py`.

Usage (example):
    python -m chessgnn.position_visualize \
        --fen "rnbqkbnr/pppppppp/8/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 1 2" \
        --output output/position_test.png \
        --centrality pagerank_centrality
"""

from __future__ import annotations

import os
import argparse
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from .position_to_graph import analyze_position, PositionAnalysis, chess as chess


def _draw_board_background(ax) -> None:
    for i in range(8):
        for j in range(8):
            color = '#d4b896' if (i + j) % 2 == 0 else '#8b6f47'
            ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=color, alpha=0.7, zorder=0))


def _draw_defense_edges(ax, analysis: PositionAnalysis) -> None:
    for src_square, dst_square, _ in analysis.defense_edges:
        src_rank = chess.square_rank(src_square)
        src_file = chess.square_file(src_square)
        dst_rank = chess.square_rank(dst_square)
        dst_file = chess.square_file(dst_square)
        ax.plot([src_file, dst_file], [7 - src_rank, 7 - dst_rank], color='limegreen', alpha=0.9, linewidth=3, zorder=2)


def _draw_attack_edges(ax, analysis: PositionAnalysis) -> None:
    # Normalize attack weights for coloring. Range approx [-8, +8].
    norm = mcolors.Normalize(vmin=-8, vmax=8)
    cmap = plt.cm.get_cmap('hot_r')
    for src_square, dst_square, weight in analysis.attack_edges:
        src_rank = chess.square_rank(src_square)
        src_file = chess.square_file(src_square)
        dst_rank = chess.square_rank(dst_square)
        dst_file = chess.square_file(dst_square)
        edge_color = cmap(norm(weight))
        line_width = 2.5 + max(0.0, weight) * 0.5
        ax.annotate('', xy=(dst_file, 7 - dst_rank), xytext=(src_file, 7 - src_rank),
                    arrowprops=dict(arrowstyle='->', color=edge_color, lw=line_width, alpha=0.9, shrinkA=5, shrinkB=5),
                    zorder=4)


def _draw_pieces(ax, analysis: PositionAnalysis, centrality_scores: Dict[int, float]) -> None:
    # Prepare size scaling
    min_score, max_score, score_range = 0.0, 1.0, 1.0
    if centrality_scores:
        vals = list(centrality_scores.values())
        if vals:
            min_score = min(vals)
            max_score = max(vals)
            score_range = max(max_score - min_score, 1e-9)

    for piece in analysis.pieces:
        rank = chess.square_rank(piece.square)
        file = chess.square_file(piece.square)
        score = centrality_scores.get(piece.square, min_score)
        normalized = (score - min_score) / score_range
        scale_factor = 0.5 + normalized * 1.0  # [0.5, 1.5]

        # For simplicity and portability, draw text pieces with background
        color = 'white' if piece.is_white else 'black'
        piece_bg_color = 'lightblue' if piece.is_white else 'lightcoral'
        piece_edge_color = 'darkblue' if piece.is_white else 'darkred'
        fontsize = 18 * scale_factor
        ax.text(file, 7 - rank, piece.symbol, fontsize=fontsize, ha='center', va='center',
                color=color, weight='bold', zorder=3,
                bbox=dict(boxstyle="circle,pad=0.15", facecolor=piece_bg_color,
                          edgecolor=piece_edge_color, linewidth=2, alpha=0.9))


def render_position_frame(fen: str, centrality: str = "pagerank_centrality", title: str = None) -> np.ndarray:
    analysis = analyze_position(fen)
    centrality_scores = analysis.centralities.get(centrality, {})

    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111)

    _draw_board_background(ax)
    _draw_defense_edges(ax, analysis)
    _draw_attack_edges(ax, analysis)
    _draw_pieces(ax, analysis, centrality_scores)

    ax.set_xlim(-1.0, 8.0)
    ax.set_ylim(-1.0, 8.0)
    ax.set_aspect('equal')
    ax.set_xticks(range(8))
    ax.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    ax.set_yticks(range(8))
    ax.set_yticklabels(['8', '7', '6', '5', '4', '3', '2', '1'])
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title, fontsize=14)

    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


def _save_frame(array_rgb: np.ndarray, output_path: str) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(array_rgb).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single chess position frame from FEN.")
    parser.add_argument('--fen', type=str, required=False, default=None, help='FEN string of the position')
    parser.add_argument('--output', type=str, required=True, help='Output image path (e.g., output/position.png)')
    parser.add_argument('--centrality', type=str, default='pagerank_centrality',
                        choices=['degree_centrality','betweenness_centrality','closeness_centrality','eigenvector_centrality','pagerank_centrality'],
                        help='Centrality measure for piece sizing')
    parser.add_argument('--title', type=str, default=None, help='Optional figure title')
    args = parser.parse_args()

    fen = args.fen or chess.STARTING_FEN
    frame = render_position_frame(fen, args.centrality, args.title)
    _save_frame(frame, args.output)
    print(f"Saved frame to {args.output}")


if __name__ == "__main__":
    main()


