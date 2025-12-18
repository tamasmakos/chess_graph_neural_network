
import torch
from torch_geometric.data import HeteroData
import chess
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional

# Constants
SQUARES = chess.SQUARES
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0 # King value handled separately or 0 for trade logic
}

class ChessGraphBuilder:
    def __init__(self):
        self.node_type_map = {'piece': 0, 'square': 1}
        
    def fen_to_graph(self, fen: str, history_emb: Optional[torch.Tensor] = None) -> HeteroData:
        """
        Converts a FEN string into a HeteroData object.
        Nodes:
            - pieces: [type, color, value, pos_emb]
            - squares: [coord, occupancy]
        Edges:
            - interacts (piece-piece): [weight, type] (Attack/Defense)
            - on (piece-square): [1] (Location)
            - adjacent (square-square): [1] (Grid)
            - ray (piece-piece): [distance, blocking_count]
        """
        board = chess.Board(fen)
        data = HeteroData()

        # 1. Nodes Construction
        pieces = []
        piece_indices = {} # map square -> piece_idx
        
        squares = []
        square_indices = {sq: i for i, sq in enumerate(SQUARES)}
        
        # Squares Nodes
        # Feature: [file, rank, is_occupied]
        for sq in SQUARES:
            file, rank = chess.square_file(sq), chess.square_rank(sq)
            piece = board.piece_at(sq)
            is_occupied = 1.0 if piece else 0.0
            squares.append([file/7.0, rank/7.0, is_occupied])
        
        data['square'].x = torch.tensor(squares, dtype=torch.float)

        # Pieces Nodes
        # Feature: [type(onehot 6), color(1), value(1), file(1), rank(1)]
        current_piece_idx = 0
        for sq in SQUARES:
            piece = board.piece_at(sq)
            if piece:
                # One-hot type
                type_vec = [0]*6
                type_vec[PIECE_TYPES.index(piece.piece_type)] = 1
                
                # Color (White=1, Black=-1)
                color = 1.0 if piece.color == chess.WHITE else -1.0
                
                # Value
                val = PIECE_VALUES[piece.piece_type] / 10.0
                
                # Pos
                file, rank = chess.square_file(sq), chess.square_rank(sq)
                
                feat = type_vec + [color, val, file/7.0, rank/7.0]
                pieces.append(feat)
                piece_indices[sq] = current_piece_idx
                current_piece_idx += 1

        if pieces:
            data['piece'].x = torch.tensor(pieces, dtype=torch.float)
        else:
            data['piece'].x = torch.empty((0, 10), dtype=torch.float)

        # 2. Edges Construction
        
        # Piece-Square (Location)
        # Edge type: ('piece', 'on', 'square')
        edge_index_on = [[], []]
        for sq, p_idx in piece_indices.items():
            edge_index_on[0].append(p_idx)
            edge_index_on[1].append(square_indices[sq])
        
        data['piece', 'on', 'square'].edge_index = torch.tensor(edge_index_on, dtype=torch.long)

        # Reverse Edge: Square -> Piece (Occupied By)
        # Allows squares to inform pieces about their location properties (e.g. center control)
        data['square', 'occupied_by', 'piece'].edge_index = torch.tensor([edge_index_on[1], edge_index_on[0]], dtype=torch.long)

        
        # Square-Square (Adjacency)
        # Edge type: ('square', 'adjacent', 'square')
        # King moves (Chebyshev distance = 1)
        edge_index_adj = [[], []]
        for sq1 in SQUARES:
            for sq2 in SQUARES:
                if sq1 == sq2: continue
                f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
                f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
                if max(abs(f1-f2), abs(r1-r2)) == 1:
                    edge_index_adj[0].append(square_indices[sq1])
                    edge_index_adj[1].append(square_indices[sq2])

        data['square', 'adjacent', 'square'].edge_index = torch.tensor(edge_index_adj, dtype=torch.long)
        
        # Piece-Piece (Interaction: Attack/Defense)
        # Edge type: ('piece', 'interacts', 'piece')
        # Weight computed based on value difference
        edge_index_int = [[], []]
        edge_attr_int = [] 

        # We iterate over all pieces and check attacks
        for sq_src, p_idx_src in piece_indices.items():
            piece_src = board.piece_at(sq_src)
            attacks = board.attacks(sq_src)
            
            for sq_dst in attacks:
                if sq_dst in piece_indices:
                    p_idx_dst = piece_indices[sq_dst]
                    piece_dst = board.piece_at(sq_dst)
                    
                    # Attack or Defense?
                    if piece_src.color != piece_dst.color:
                        # Attack
                        # Weight = sigmoid(Val(Target) - Val(Attacker))
                        # Favors capturing high value with low value
                        diff = PIECE_VALUES[piece_dst.piece_type] - PIECE_VALUES[piece_src.piece_type]
                        weight = self.sigmoid(diff)
                        edge_type = 1.0 # Attack
                    else:
                        # Defense
                        # Weight = sigmoid(Val(Defender)) ?? Or just constant?
                        # Plan says: function of value difference.
                        # Using 1.0 as standard strong connection
                        weight = 1.0 # Strong support
                        edge_type = -1.0 # Defense
                    
                    edge_index_int[0].append(p_idx_src)
                    edge_index_int[1].append(p_idx_dst)
                    edge_attr_int.append([weight, edge_type])
        
        if edge_index_int[0]:
            data['piece', 'interacts', 'piece'].edge_index = torch.tensor(edge_index_int, dtype=torch.long)
            data['piece', 'interacts', 'piece'].edge_attr = torch.tensor(edge_attr_int, dtype=torch.float)
        else:
             data['piece', 'interacts', 'piece'].edge_index = torch.empty((2, 0), dtype=torch.long)
             data['piece', 'interacts', 'piece'].edge_attr = torch.empty((0, 2), dtype=torch.float)

        # Piece-Piece (Ray Edges) - Simplified for now
        # We need to detect pins/skewers. Any piece on the same rank/file/diagonal.
        # This is O(N^2) but N <= 32.
        
        edge_index_ray = [[], []]
        edge_attr_ray = [] # [distance, blocking_count]
        
        pieces_locs = list(piece_indices.items())
        for i in range(len(pieces_locs)):
            for j in range(len(pieces_locs)):
                if i == j: continue
                
                sq1, p1_idx = pieces_locs[i]
                sq2, p2_idx = pieces_locs[j]
                
                # Check alignment
                if self.is_aligned(sq1, sq2):
                    dist = chess.square_distance(sq1, sq2)
                    blocking = self.count_blocking(board, sq1, sq2)
                    
                    edge_index_ray[0].append(p1_idx)
                    edge_index_ray[1].append(p2_idx)
                    edge_attr_ray.append([float(dist)/7.0, float(blocking)])

        if edge_index_ray[0]:
            data['piece', 'ray', 'piece'].edge_index = torch.tensor(edge_index_ray, dtype=torch.long)
            data['piece', 'ray', 'piece'].edge_attr = torch.tensor(edge_attr_ray, dtype=torch.float)
        else:
            data['piece', 'ray', 'piece'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['piece', 'ray', 'piece'].edge_attr = torch.empty((0, 2), dtype=torch.float)

        return data

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def is_aligned(self, sq1, sq2):
        f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
        f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
        return (f1 == f2) or (r1 == r2) or (abs(f1-f2) == abs(r1-r2))

    def count_blocking(self, board, sq1, sq2):
        # Ray cast
        # We use chess.Ray logic or manual steps
        # Manual step iteration
        f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
        f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
        
        df = 0 if f1 == f2 else (1 if f2 > f1 else -1)
        dr = 0 if r1 == r2 else (1 if r2 > r1 else -1)
        
        curr_f, curr_r = f1 + df, r1 + dr
        blocking = 0
        
        while (curr_f != f2) or (curr_r != r2):
            sq = chess.square(curr_f, curr_r)
            if board.piece_at(sq):
                blocking += 1
            curr_f += df
            curr_r += dr
            
        return blocking
