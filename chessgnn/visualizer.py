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
        
        # Use absolute workspace path for images
        # The user specified to use images from the images folder in the workspace root
        base_images_dir = "/workspaces/chessgnn/images"
        
        for symbol, filename in piece_mapping.items():
            try:
                # Determine if it's a white or black piece
                folder = 'white_pieces' if symbol.isupper() else 'black_pieces'
                file_path = os.path.join(base_images_dir, folder, f"{filename}.png")
                
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
                file = chess.square_file(square_index)
                rank = chess.square_rank(square_index)
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
                    fx = chess.square_file(from_sq)
                    fy = 7 - chess.square_rank(from_sq)
                    cross_size = 0.45
                    ax.plot([fx - cross_size, fx + cross_size], [fy - cross_size, fy + cross_size],
                            color='red', linewidth=3, alpha=0.9, zorder=5)
                    ax.plot([fx - cross_size, fx + cross_size], [fy + cross_size, fy - cross_size],
                            color='red', linewidth=3, alpha=0.9, zorder=5)
                
                if to_sq is not None:
                    tx = chess.square_file(to_sq)
                    ty = 7 - chess.square_rank(to_sq)
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
            rank = chess.square_rank(piece.square)
            file = chess.square_file(piece.square)
            
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


class GameVideoGenerator:
    def __init__(self, visualizer: ChessVisualizer):
        self.visualizer = visualizer

    def generate_video(self, 
                       game_states: List[GameState], 
                       win_probs: List[float], 
                       stockfish_evals: List[float],
                       output_path: str, 
                       fps: int = 1):
        """
        Generates a video of the game with a win probability plot and eval bar.
        
        Args:
            game_states: List of GameState objects.
            win_probs: List of model win probabilities (0-100).
            stockfish_evals: List of Stockfish evaluations (centipawns, e.g. +150, -50).
            output_path: Path to save the video.
            fps: Frames per second.
        """
        try:
            import cv2
        except ImportError:
            print("Error: cv2 (opencv-python) is required for video generation.")
            return

        print(f"Generating video to {output_path} with {len(game_states)} frames...")
        
        frames = []
        
        for i, state in enumerate(game_states):
            current_probs = win_probs[:i+1]
            # Get current stockfish eval if available, else 0
            current_sf = stockfish_evals[i] if i < len(stockfish_evals) else 0.0
            
            frame = self._render_frame(state, current_probs, current_sf, total_moves=len(game_states))
            frames.append(frame)
            
        if not frames:
            return

        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)

        video.release()
        print(f"Video saved successfully: {output_path}")

    def _render_frame(self, 
                      state: GameState, 
                      win_probs: List[float], 
                      sf_eval: float,
                      total_moves: int) -> np.ndarray:
        """Renders a single frame with Eval Bar, Board, and Win Prob Plot."""
        
        # Figure setup: 3 rows (Eval Bar, Board, Plot)
        # Ratios: Bar=0.5, Board=5, Plot=2
        fig = plt.figure(figsize=(6, 9), dpi=100)
        gs = fig.add_gridspec(3, 1, height_ratios=[0.4, 4, 1.5], hspace=0.3)
        
        # 1. Stockfish Eval Bar (Top)
        ax_eval = fig.add_subplot(gs[0])
        self._draw_eval_bar(ax_eval, sf_eval)
        
        # 2. Board View (Middle)
        ax_board = fig.add_subplot(gs[1])
        self.visualizer.visualize_game_state(state, ax_board)
        
        turn_text = "White" if state.is_white_turn else "Black"
        ax_board.set_title(f"Move {state.move_number} - {turn_text} to Move", fontsize=12)

        # 3. Win Probability Plot View (Bottom)
        ax_plot = fig.add_subplot(gs[2])
        
        x_vals = range(len(win_probs))
        ax_plot.plot(x_vals, win_probs, color='blue', linewidth=2, label='White Win %')
        ax_plot.fill_between(x_vals, win_probs, 50, where=(np.array(win_probs) >= 50), facecolor='green', alpha=0.1, interpolate=True)
        ax_plot.fill_between(x_vals, win_probs, 50, where=(np.array(win_probs) < 50), facecolor='red', alpha=0.1, interpolate=True)

        if win_probs:
            ax_plot.plot(len(win_probs)-1, win_probs[-1], 'o', color='red', markersize=6)
            ax_plot.text(len(win_probs)-1, win_probs[-1] + 8, f"{win_probs[-1]:.1f}%", fontsize=10, ha='center')

        ax_plot.set_xlim(0, max(total_moves, 1))
        ax_plot.set_ylim(0, 100)
        ax_plot.set_ylabel("Win Prob (%)", fontsize=10)
        ax_plot.set_yticks([0, 50, 100])
        ax_plot.grid(True, linestyle='--', alpha=0.5)
        ax_plot.axhline(50, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        # Draw canvas
        fig.canvas.draw()
        
        # Convert to numpy array
        try:
            # Old Matplotlib
            buf = fig.canvas.tostring_rgb()
            buf = np.frombuffer(buf, dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except AttributeError:
            # Modern Matplotlib (3.8+)
            buf = np.asarray(fig.canvas.buffer_rgba())
            if buf.ndim == 3 and buf.shape[2] == 4:
                buf = buf[:, :, :3]
        
        plt.close(fig)
        return buf

    def _draw_eval_bar(self, ax, score: float):
        """Draws an evaluation bar. Score in centipawns."""
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Sigmoid-like normalization for visualization
        # +1000 cp -> almost full white
        # -1000 cp -> almost full black
        # score is typical cp. max range +/- 400 (4 pawns) is usually good visual range
        
        # Clamp visual range
        vis_score = max(-400, min(400, score))
        norm_score = (vis_score + 400) / 800.0 # 0 to 1
        
        # Draw background (Black)
        ax.add_patch(plt.Rectangle((-1, 0), 2, 1, color='black'))
        
        # Draw white advantage (Left to Right or just split?)
        # Let's do a split bar. Center is 0.
        # Actually chess eval bars are usually Full Bar with White % vs Black %.
        # norm_score = 0.5 is equal. 1.0 is full white.
        
        ax.add_patch(plt.Rectangle((-1, 0), 2 * norm_score, 1, color='white'))
        
        # Label
        eval_text = f"{score/100:.2f}"
        if score > 0: eval_text = "+" + eval_text
        
        # Text color depends on who has advantage? Or just simple overlay
        # Center text
        ax.text(0, 0.5, f"Est. Stockfish: {eval_text}", ha='center', va='center', 
                fontsize=10, color='gray', weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))


