import matplotlib.pyplot as plt
import numpy as np
import chess
from typing import Dict, Tuple, List, Any
import os
from PIL import Image
import networkx as nx

# Try importing leidenalg, if not available, handle gracefully
try:
    import leidenalg as la
    import igraph as ig
    _HAS_LEIDEN = True
except ImportError:
    _HAS_LEIDEN = False

from .position_to_graph import GameState, PieceInfo, analyze_position, PositionAnalysis

class ChessVisualizer:
    def __init__(self):
        self.piece_images = self._load_piece_images()
        self.community_colors_cache = {}

    def _load_piece_images(self) -> Dict[str, Image.Image]:
        """Load and cache custom PNG piece images, pre-resized for performance."""
        piece_images = {}
        
        # Define piece symbol to filename mapping
        piece_mapping = {
            'p': 'p', 'k': 'k', 'n': 'n', 'b': 'b', 'r': 'r', 'q': 'q',  # black pieces
            'P': 'P', 'K': 'K', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q'   # white pieces
        }
        
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        for symbol, filename in piece_mapping.items():
            try:
                # Determine if it's a white or black piece
                folder = 'white' if symbol.isupper() else 'black'
                file_path = os.path.join(script_dir, folder, f"{filename}.png")
                
                if os.path.exists(file_path):
                    # Load and immediately resize to a reasonable size for performance
                    img = Image.open(file_path)
                    # Resize to a reasonable size (e.g., 200x200) to avoid memory issues
                    img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                    # Keep RGBA mode to preserve transparency - matplotlib handles it correctly
                    piece_images[symbol] = img
                else:
                    # Fallback or silent warning could go here, but for now we just skip
                    pass
                    
            except Exception as e:
                print(f"Error loading piece image for {symbol}: {e}")
        
        return piece_images

    def _get_resized_piece_image(self, symbol: str, square_size: float = 1.0) -> Image.Image:
        """Get a resized piece image that fits within a chess square."""
        if symbol not in self.piece_images:
            return None
        return self.piece_images[symbol].copy()

    def compute_leiden_communities(self, G: nx.Graph) -> Dict[int, int]:
        """
        Compute community structure using the Leiden algorithm.
        Returns a mapping from node (square int) to community ID.
        """
        if G.number_of_nodes() == 0:
            return {}

        if not _HAS_LEIDEN:
            return {}

        try:
            # Build igraph from networkx
            nodes = list(G.nodes())
            node_index = {n: i for i, n in enumerate(nodes)}
            edges_idx = [(node_index[u], node_index[v]) for u, v in G.edges()]

            G_ig = ig.Graph(n=len(nodes), edges=edges_idx, directed=False)
            G_ig.vs["square"] = nodes

            weights = [G.get_edge_data(u, v).get('weight', 1.0) for u, v in G.edges()]

            # Simple resolution for now, can be made adaptive if needed
            partition = la.find_partition(G_ig, la.CPMVertexPartition, resolution_parameter=0.3, weights=weights)
            
            membership = partition.membership
            communities = {nodes[i]: membership[i] for i in range(len(nodes))}
            
            return communities
        except Exception as e:
            print(f"Warning: Leiden community detection failed: {e}")
            return {}

    def create_stable_community_colors(self, per_move_communities: List[Dict[int, int]]) -> Tuple[List[Dict[int, int]], List[Any]]:
        """
        Analyzes community data across all moves to assign stable colors.
        """
        stable_color_map: Dict[Tuple[int, ...], int] = {}
        color_palette = plt.get_cmap('tab20').colors
        next_color_idx = 0
        
        # Pass 1: Discover unique communities
        for move_communities in per_move_communities:
            if not move_communities:
                continue
            
            community_groups = {}
            for square, comm_id in move_communities.items():
                if comm_id not in community_groups:
                    community_groups[comm_id] = set()
                community_groups[comm_id].add(square)
            
            for comm_id, squares_set in community_groups.items():
                stable_id = tuple(sorted(list(squares_set)))
                if stable_id not in stable_color_map:
                    stable_color_map[stable_id] = next_color_idx
                    next_color_idx += 1

        # Pass 2: Remap
        remapped_per_move_communities = []
        for move_communities in per_move_communities:
            if not move_communities:
                remapped_per_move_communities.append({})
                continue

            community_groups = {}
            for square, comm_id in move_communities.items():
                if comm_id not in community_groups:
                    community_groups[comm_id] = set()
                community_groups[comm_id].add(square)
                
            new_mapping_for_move = {}
            for comm_id, squares_set in community_groups.items():
                stable_id = tuple(sorted(list(squares_set)))
                stable_color_index = stable_color_map.get(stable_id)
                
                if stable_color_index is not None:
                    for square in squares_set:
                        new_mapping_for_move[square] = stable_color_index
            
            remapped_per_move_communities.append(new_mapping_for_move)
            
        return remapped_per_move_communities, list(color_palette)

    def visualize_game_state(self, game_state: GameState, ax, title: str = None, 
                           centrality_scores: Dict[int, float] = None,
                           community_mapping: Dict[int, int] = None,
                           highlight_move: Tuple[int, int] = None):
        """Visualize a chess position with graph overlay."""
        
        # Draw chess board background
        for i in range(8):
            for j in range(8):
                color = '#d4b896' if (i + j) % 2 == 0 else '#8b6f47'
                ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, color=color, alpha=0.7, zorder=0))

        # Community overlays
        if community_mapping:
            community_colors = plt.get_cmap('tab20').colors
            num_colors = len(community_colors)
            for square_index, community_id in community_mapping.items():
                file = chessgnn.square_file(square_index)
                rank = chessgnn.square_rank(square_index)
                comm_color = community_colors[community_id % num_colors]
                ax.add_patch(plt.Rectangle(
                    (file - 0.5, (7 - rank) - 0.5), 1, 1,
                    color=comm_color,
                    alpha=0.6,
                    zorder=1
                ))

        # Highlight move
        if highlight_move:
            try:
                from_sq, to_sq = highlight_move
                
                if from_sq is not None:
                    fx = chessgnn.square_file(from_sq)
                    fy = 7 - chessgnn.square_rank(from_sq)
                    cross_size = 0.45
                    ax.plot([fx - cross_size, fx + cross_size], [fy - cross_size, fy + cross_size],
                            color='red', linewidth=3, alpha=0.9, zorder=5)
                    ax.plot([fx - cross_size, fx + cross_size], [fy + cross_size, fy - cross_size],
                            color='red', linewidth=3, alpha=0.9, zorder=5)
                
                if to_sq is not None:
                    tx = chessgnn.square_file(to_sq)
                    ty = 7 - chessgnn.square_rank(to_sq)
                    circle = plt.Circle(
                        (tx, ty), 0.45,
                        color='red', fill=False, linewidth=3, alpha=0.9, zorder=5
                    )
                    ax.add_patch(circle)
            except Exception:
                pass
        
        # Piece sizing based on centrality
        min_score, max_score, score_range = 0, 1, 1
        if centrality_scores:
            all_scores = list(centrality_scores.values())
            if all_scores:
                min_score = min(all_scores)
                max_score = max(all_scores)
                score_range = max_score - min_score
                if score_range < 1e-9:
                    score_range = 1
        
        # Draw attack edges (Optional - can be added back if needed, present in V3 but sometimes cluttered)
        # For this refactor, I will focus on pieces and communities as per V3's primary draw method
        
        # Draw pieces
        for piece in game_state.pieces:
            rank = chessgnn.square_rank(piece.square)
            file = chessgnn.square_file(piece.square)
            
            scale_factor = 1.0
            if centrality_scores:
                piece_score = centrality_scores.get(piece.square, min_score)
                normalized_score = (piece_score - min_score) / score_range
                scale_factor = 0.5 + normalized_score * 1.0
            
            piece_img = self._get_resized_piece_image(piece.symbol)
            
            if piece_img:
                piece_array = np.array(piece_img)
                x_pos = file
                y_pos = 7 - rank
                
                base_img_size = 0.85
                img_width = base_img_size * scale_factor
                img_height = base_img_size * scale_factor
                
                extent = [x_pos - img_width/2, x_pos + img_width/2, 
                          y_pos - img_height/2, y_pos + img_height/2]
                
                ax.imshow(piece_array, extent=extent, zorder=3, aspect='equal')
            else:
                # Text fallback
                color = 'white' if piece.color else 'black'
                piece_bg_color = 'lightblue' if piece.color else 'lightcoral'
                piece_edge_color = 'darkblue' if piece.color else 'darkred'
                ax.text(file, 7 - rank, piece.symbol, fontsize=18*scale_factor, ha='center', va='center',
                       color=color, weight='bold', zorder=3,
                       bbox=dict(boxstyle="circle,pad=0.15", facecolor=piece_bg_color,
                                 edgecolor=piece_edge_color, linewidth=2, alpha=0.9))

        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)
        ax.set_aspect('equal')
        ax.axis('off')

