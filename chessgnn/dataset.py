
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
        board = game.board()
        
        # Buffer for sliding window: List of (fen, graph)
        # We only need to store the last 'window_size' graphs to avoid re-computing.
        window_buffer = []  
        
        # Initial State
        initial_fen = board.fen()
        initial_graph = self.builder.fen_to_graph(initial_fen)
        window_buffer.append(initial_graph)
        
        # Move Sequence
        game_moves = list(game.mainline_moves())
        # We iterate through moves to generate (Input, Target) pairs
        
        # Input at step t: window of graphs ending at state t
        # Target at step t: move chosen at state t (which leads to t+1)
        
        # current board is at initial state (step 0)
        # We have 1 graph in buffer.
        
        for i, move in enumerate(game_moves):
            # i is step index. move is the action taken at step i.
            
            # 1. Construct Sample from current state (before push)
            # We need a sequence of length up to window_size ending at current state.
            
            # Slice buffer
            start_index = max(0, len(window_buffer) - self.window_size)
            sequence_graphs = window_buffer[start_index:]
            
            # Target Index
            current_legal_moves = list(board.legal_moves)
            try:
                target_idx = current_legal_moves.index(move)
            except ValueError:
                target_idx = 0 # Fallback
                
            yield {
                'sequence': sequence_graphs, # List of HeteroData
                'legal_moves': current_legal_moves,
                'target_index': target_idx,
                'fen': board.fen(),
                'game_id': game_id
            }
            
            # 2. Advance State
            board.push(move)
            new_fen = board.fen()
            new_graph = self.builder.fen_to_graph(new_fen)
            window_buffer.append(new_graph)
            
            # Optimize buffer size?
            if len(window_buffer) > self.window_size:
                window_buffer.pop(0)

# Dataset alias for backward compatibility if needed, but we should update train.py
ChessGraphDataset = ChessGraphIterableDataset

def custom_collate(batch):
    return batch
