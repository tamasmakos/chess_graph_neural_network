
import os
import chess
import chess.pgn
import torch
from torch_geometric.data import Dataset, HeteroData
from typing import List, Tuple
from tqdm import tqdm
from .graph_builder import ChessGraphBuilder

import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import HeteroData
from .graph_builder import ChessGraphBuilder

class ChessGraphIterableDataset(IterableDataset):
    def __init__(self, pgn_file: str, num_games: int = 10, offset: int = 0, window_size: int = 8):
        self.pgn_file = pgn_file
        self.num_games = num_games
        self.offset = offset
        self.window_size = window_size
        self.builder = ChessGraphBuilder()
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            # Multi-worker split: This is naive (all workers read same file) unless we shard.
            # For PGN, random access is hard. We suggest num_workers=0 or 1.
            # Or simplified: only worker 0 yields data.
            if worker_info.id != 0:
                return
        
        with open(self.pgn_file) as f:
            # Skip offset
            for _ in range(self.offset):
                if chess.pgn.read_game(f) is None:
                    return

            games_processed = 0
            while games_processed < self.num_games:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                    
                # Process this game's moves
                # We yield samples game by game
                yield from self._process_game_stream(game, game_id=games_processed + self.offset)
                
                games_processed += 1

    def _process_game_stream(self, game: chess.pgn.Game, game_id: int):
        # 1. Extract Result (1-0, 0-1, 1/2-1/2)
        result = game.headers.get("Result", "*")
        if result == "1-0":
            game_value = 1.0
        elif result == "0-1":
            game_value = -1.0
        else:
            game_value = 0.0 # Draw or unknown

        board = game.board()
        
        # Buffer for sliding window: List of (fen, graph)
        window_buffer = []  
        
        # Initial State
        initial_fen = board.fen()
        initial_graph = self.builder.fen_to_graph(initial_fen)
        window_buffer.append(initial_graph)
        
        # Move Sequence
        game_moves = list(game.mainline_moves())
        
        for i, move in enumerate(game_moves):
            # 1. Construct Sample from current state (before push)
            
            # Slice buffer
            start_index = max(0, len(window_buffer) - self.window_size)
            sequence_graphs = window_buffer[start_index:]
            
            current_legal_moves = list(board.legal_moves)

            # KEY CHANGE: The target is now the Game Value
            # If turn is WHITE: target is game_value
            # If turn is BLACK: target is -game_value (or handle relative to 'player to move')
            
            current_turn_value = game_value if board.turn == chess.WHITE else -game_value
            
            yield {
                'sequence': sequence_graphs, 
                'legal_moves': current_legal_moves, # Keep for visualization/inference
                'target_value': current_turn_value, # <--- NEW TARGET
                'fen': board.fen(),
                'game_id': game_id
            }
            
            # 2. Advance State
            board.push(move)
            new_fen = board.fen()
            new_graph = self.builder.fen_to_graph(new_fen)
            window_buffer.append(new_graph)
            
            # Optimize buffer size
            if len(window_buffer) > self.window_size:
                window_buffer.pop(0)

# Dataset alias for backward compatibility if needed, but we should update train.py
ChessGraphDataset = ChessGraphIterableDataset

def custom_collate(batch):
    return batch
