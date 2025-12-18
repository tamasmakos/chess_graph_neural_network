
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
        worker_id = 0
        num_workers = 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
             worker_id = worker_info.id
             num_workers = worker_info.num_workers
        
        with open(self.pgn_file) as f:
            # Skip offset
            for _ in range(self.offset):
                if chess.pgn.read_game(f) is None:
                    return

            games_processed = 0
            while games_processed < self.num_games:
                # We interpret "games_processed" as the index relative to offset
                current_idx = games_processed
                
                # Check ownership BEFORE reading? 
                # No, we must read sequentially to advance file pointer.
                # PGN is not random access. We must parse every game header/content 
                # to advance the stream, even if we discard it.
                # This is IO bound but avoids Parsing Moves/Building Graphs (CPU bound).
                
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                    
                # Sharding: Process only if (index % num_workers) == worker_id
                if current_idx % num_workers == worker_id:
                    yield from self._process_game_stream(game, game_id=current_idx + self.offset)
                
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
        
        # Filter: Skip short games (< 5 moves = 10 plies)
        if len(game_moves) < 5:
            return
        
        if len(game_moves) < 5:
            return
        
        # Pre-calculate all graphs for the whole game
        all_graphs = []
        all_graphs.append(initial_graph)

        for i, move in enumerate(game_moves):
            board.push(move)
            # We want the state *after* the move? 
            # Original code was: 
            # 1. Slice buffer (which has initial) -> predict
            # 2. Push move -> add to buffer
            #
            # The model predicts the *evaluation of the current position*.
            # So a sequence of length N means N positions.
            # Position 0: Start
            # Position 1: After Move 1
            # ...
            # Position N: After Move N
            
            new_fen = board.fen()
            new_graph = self.builder.fen_to_graph(new_fen)
            all_graphs.append(new_graph)
            
        # Yield the FULL game
        # We process the entire game history.
        # target_value is 1.0 (White Win), -1.0 (Black Win), 0.0 (Draw)
        # We can pass this single scalar, and the loss function will expand it.
        
        yield {
            'sequence': all_graphs,      # List[HeteroData], length N+1
            'target_value': game_value,  # Scalar
            'game_id': game_id,
            'result': result,
            'fen': board.fen() # Final position FEN
        }

# Dataset alias for backward compatibility if needed, but we should update train.py
ChessGraphDataset = ChessGraphIterableDataset

def custom_collate(batch):
    return batch
