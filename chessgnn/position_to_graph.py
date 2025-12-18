"""
Single-position chess graph builder.

This module provides a focused API to convert one chess position (FEN) into a
weighted bipartite graph and compute centrality measures over that graph.

Extracted with surgical precision from `chess_to_graphv2.py`:
- Edge construction rules (attacks/defenses with weights)
- NetworkX graph assembly suitable for centrality analysis
- Computation of degree, betweenness, closeness, eigenvector, and PageRank

Public API:
- analyze_position(fen: str) -> PositionAnalysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Ensure we import the system-installed python-chess, not this package
import sys  # noqa: E402
import os  # noqa: E402
import site  # noqa: E402
import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402

# Load third-party python-chess explicitly from site-packages to avoid name clash
def _load_third_party_chess():  # noqa: E402
    candidate_paths = []
    try:
        candidate_paths.extend(site.getsitepackages())  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        candidate_paths.append(site.getusersitepackages())
    except Exception:
        pass
    # Fallback: common prefixes
    candidate_paths.extend([
        "/usr/local/lib/python3.10/dist-packages",
        "/usr/lib/python3/dist-packages",
    ])

    for base in candidate_paths:
        try:
            spec = importlib.machinery.PathFinder.find_spec('chess', [base])
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
                return mod
        except Exception:
            continue
    raise ImportError("python-chess not found in site-packages; please install with `pip install python-chess`. ")

chess = _load_third_party_chess()  # noqa: E402
import networkx as nx
from networkx.exception import PowerIterationFailedConvergence


# --- Data structures ---------------------------------------------------------------------------

@dataclass
class PieceInfo:
    square: int
    square_name: str
    piece_type: int
    is_white: bool
    symbol: str
    value: int


@dataclass
class PositionGraph:
    fen: str
    pieces: List[PieceInfo]
    attack_edges: List[Tuple[int, int, float]]
    defense_edges: List[Tuple[int, int, float]]

@dataclass
class GameState:
    """Represents a single game state with its graph (legacy support for V3)."""
    move_number: int
    is_white_turn: bool
    fen: str
    pieces: List[PieceInfo]
    attack_edges: List[Tuple[int, int, float]]  # (from_square, to_square, weight)
    defense_edges: List[Tuple[int, int, float]]  # (from_square, to_square, weight)
    timestamp: float = 0.0  # Time in seconds from game start


@dataclass
class PositionAnalysis:
    fen: str
    pieces: List[PieceInfo]
    attack_edges: List[Tuple[int, int, float]]
    defense_edges: List[Tuple[int, int, float]]
    centralities: Dict[str, Dict[int, float]]
    node_to_is_white: Dict[int, bool]


# --- Core logic ---------------------------------------------------------------------------------

_PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 20,
}


def _get_piece_value(piece: chess.Piece) -> int:
    return _PIECE_VALUES.get(piece.piece_type, 0)


def _create_bipartite_chess_graph(fen: str) -> PositionGraph:
    board = chess.Board(fen)

    # Collect piece info
    pieces: List[PieceInfo] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        pieces.append(
            PieceInfo(
                square=square,
                square_name=chess.square_name(square),
                piece_type=piece.piece_type,
                is_white=(piece.color == chess.WHITE),
                symbol=piece.symbol(),
                value=_get_piece_value(piece),
            )
        )

    attack_edges: List[Tuple[int, int, float]] = []
    defense_edges: List[Tuple[int, int, float]] = []

    # Attack edges: src can capture dst. Weight encodes trade quality.
    for src_square in chess.SQUARES:
        src_piece = board.piece_at(src_square)
        if src_piece is None:
            continue

        attacks = board.attacks(src_square)
        if not hasattr(attacks, "__iter__"):
            continue

        for dst_square in attacks:
            dst_piece = board.piece_at(dst_square)
            if dst_piece is not None and dst_piece.color != src_piece.color:
                src_value = _get_piece_value(src_piece)
                dst_value = _get_piece_value(dst_piece)

                # If destination is undefended, weight is value of captured piece;
                # else weight reflects trade (dst - src).
                defenders = board.attackers(dst_piece.color, dst_square)
                if not defenders:
                    weight = float(dst_value)
                else:
                    weight = float(dst_value - src_value)

                attack_edges.append((src_square, dst_square, weight))

    # Defense edges: same-color defenders of an occupied square.
    for defended_square in chess.SQUARES:
        defended_piece = board.piece_at(defended_square)
        if defended_piece is None:
            continue

        defenders = board.attackers(defended_piece.color, defended_square)
        for defender_square in defenders:
            if defender_square == defended_square:
                continue
            defender_piece = board.piece_at(defender_square)
            if defender_piece is None:
                continue

            defended_value = _get_piece_value(defended_piece)
            defender_value = _get_piece_value(defender_piece)
            if defender_value > 0:
                weight = 1.0 + (defended_value / float(defender_value))
            else:
                weight = 1.0
            defense_edges.append((defender_square, defended_square, float(weight)))

    return PositionGraph(
        fen=fen,
        pieces=pieces,
        attack_edges=attack_edges,
        defense_edges=defense_edges,
    )


def _build_networkx_graph(pg: PositionGraph) -> Tuple[nx.Graph, Dict[int, bool]]:
    node_to_is_white: Dict[int, bool] = {p.square: p.is_white for p in pg.pieces}

    G = nx.Graph()
    G.add_nodes_from(node_to_is_white.keys())

    # Attacks: ensure nonnegative weights for PageRank; use epsilon for non-positive
    for edge in pg.attack_edges:
        u, v, weight = edge
        if u in node_to_is_white and v in node_to_is_white:
            w = max(0.0, float(weight))
            if w == 0.0:
                w = 1e-6
            G.add_edge(u, v, weight=w)

    # Defenses: add as-is (already >= 1.0)
    for u, v, weight in pg.defense_edges:
        if u in node_to_is_white and v in node_to_is_white:
            G.add_edge(u, v, weight=float(weight))

    return G, node_to_is_white


def _compute_all_centralities(G: nx.Graph) -> Dict[str, Dict[int, float]]:
    n = G.number_of_nodes()
    if n == 0:
        return {
            "degree_centrality": {},
            "betweenness_centrality": {},
            "closeness_centrality": {},
            "eigenvector_centrality": {},
            "pagerank_centrality": {},
        }

    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, normalized=True)
    cls = nx.closeness_centrality(G)

    if G.number_of_edges() == 0:
        eig = {node: 0.0 for node in G.nodes}
        pr = {node: 0.0 for node in G.nodes}
    else:
        try:
            eig = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-6)
        except PowerIterationFailedConvergence:
            try:
                eig = nx.eigenvector_centrality_numpy(G)
            except Exception:
                eig = {node: 0.0 for node in G.nodes}

        try:
            pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6, weight="weight")
        except Exception:
            try:
                pr = nx.pagerank_numpy(G, alpha=0.85, weight="weight")
            except Exception:
                pr = {node: 0.0 for node in G.nodes}

    return {
        "degree_centrality": deg,
        "betweenness_centrality": btw,
        "closeness_centrality": cls,
        "eigenvector_centrality": eig,
        "pagerank_centrality": pr,
    }


# --- Public API --------------------------------------------------------------------------------

def analyze_position(fen: str) -> PositionAnalysis:
    """
    Convert a single chess position (FEN) into a weighted bipartite graph and
    compute centrality measures over the resulting undirected, weighted graph.

    Returns a PositionAnalysis containing:
      - pieces and weighted edges (attacks/defenses)
      - node_to_is_white mapping
      - centralities: degree, betweenness, closeness, eigenvector, pagerank
    """
    pg = _create_bipartite_chess_graph(fen)
    G, node_to_is_white = _build_networkx_graph(pg)
    centralities = _compute_all_centralities(G)
    return PositionAnalysis(
        fen=fen,
        pieces=pg.pieces,
        attack_edges=pg.attack_edges,
        defense_edges=pg.defense_edges,
        centralities=centralities,
        node_to_is_white=node_to_is_white,
    )


__all__ = [
    "PieceInfo",
    "PositionGraph",
    "PositionAnalysis",
    "analyze_position",
]


