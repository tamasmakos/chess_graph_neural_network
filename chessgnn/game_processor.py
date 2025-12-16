import chess
import chessgnn.pgn
import chessgnn.engine
import os
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import sys

# Try to handle chessgnn.engine availability
try:
    _HAS_CHESS_ENGINE = True
except ImportError:
    chessgnn.engine = None
    _HAS_CHESS_ENGINE = False

from .position_to_graph import GameState, analyze_position, PositionAnalysis, _create_bipartite_chess_graph

class ChessGameProcessor:
    def __init__(self, stockfish_path: str = None):
        self.stockfish_path = stockfish_path

    def get_last_n_days_games(self, pgn_file_path: str, n_days: int = 3) -> List[chessgnn.pgn.Game]:
        """Extract games from the last N days."""
        print(f"Extracting games from the last {n_days} days...")
        
        all_games_with_dates = []
        
        with open(pgn_file_path, 'r', encoding='utf-8') as pgn:
            while True:
                game = chessgnn.pgn.read_game(pgn)
                if game is None:
                    break
                
                date_str = game.headers.get('Date', '')
                if date_str:
                    try:
                        game_date = datetime.strptime(date_str, '%Y.%m.%d')
                        all_games_with_dates.append((game, game_date))
                    except ValueError:
                        continue
        
        if not all_games_with_dates:
            return []
        
        all_games_with_dates.sort(key=lambda x: x[1], reverse=True)
        latest_date = all_games_with_dates[0][1]
        cutoff_date = latest_date - timedelta(days=n_days-1)
        
        filtered_games = [g for g, d in all_games_with_dates if d >= cutoff_date]
        print(f"Found {len(filtered_games)} games from the last {n_days} days")
        return filtered_games

    def _estimate_time_per_move(self, headers: Dict[str, str]) -> float:
        """Estimate time per move based on game type."""
        time_control = headers.get('TimeControl', '')
        if time_control and time_control != '?':
            try:
                if '+' in time_control:
                    base, _ = time_control.split('+')
                    return max(5.0, int(base) / 40.0)
                else:
                    return max(5.0, int(time_control) / 40.0)
            except ValueError:
                pass
        
        event = headers.get('Event', '').lower()
        if 'blitz' in event: return 15.0
        if 'rapid' in event: return 30.0
        if 'classical' in event: return 60.0
        return 30.0

    def process_game(self, game: chessgnn.pgn.Game) -> Tuple[List[GameState], List[str]]:
        """Process a full game into GameStates and FENs."""
        board = game.board()
        # Initialize with starting position
        fens = [chessgnn.STARTING_FEN]
        # We need to create the initial GameState manually or via helper
        # Using position_to_graph's analyze_position equivalent logic but returning GameState
        
        # Actually, let's stick to what V3 did: create GameState objects
        game_states = []
        
        # Initial state
        # Helper to bridge the gap between position_to_graph (PositionGraph) and V3 (GameState)
        # We will reuse the logic from visualizer or here. 
        # Ideally position_to_graph should export a way to get GameState-like object or we just build it here.
        # But V3 had `create_bipartite_chess_graph` returning `GameState`.
        # `position_to_graph` has `_create_bipartite_chess_graph` returning `PositionGraph`.
        # They are very similar. Let's adapt here.
        
        time_per_move = self._estimate_time_per_move(game.headers)
        current_time = 0.0
        
        # Add start state
        game_states.append(self._create_game_state_from_fen(chessgnn.STARTING_FEN, 0, current_time))
        
        for i, move in enumerate(game.mainline_moves()):
            board.push(move)
            current_time += time_per_move
            fen = board.fen()
            fens.append(fen)
            game_states.append(self._create_game_state_from_fen(fen, i+1, current_time))
            
        return game_states, fens

    def _create_game_state_from_fen(self, fen: str, move_number: int, timestamp: float) -> GameState:
        """Bridge function to create GameState from FEN using position_to_graph logic."""
        # We can use the internal function from position_to_graph or reimplement
        # Ideally we import `_create_bipartite_chess_graph` or `analyze_position`
        pos_graph = _create_bipartite_chess_graph(fen)
        
        # Convert PositionGraph to GameState (which V3 expected)
        # V3 GameState: move_number, is_white_turn, fen, pieces, attack_edges, defense_edges, timestamp
        
        return GameState(
            move_number=move_number,
            is_white_turn=chessgnn.Board(fen).turn == chessgnn.WHITE,
            fen=fen,
            pieces=pos_graph.pieces,
            attack_edges=pos_graph.attack_edges,
            defense_edges=pos_graph.defense_edges,
            timestamp=timestamp
        )

    def get_stockfish_evaluations(self, fens: List[str]) -> List[float]:
        """Get evaluations for a list of FENs."""
        if not self.stockfish_path or not os.path.exists(self.stockfish_path):
            return [0.0] * len(fens)
            
        if not _HAS_CHESS_ENGINE:
            return [0.0] * len(fens)

        evals = []
        engine = None
        try:
            engine = chessgnn.engine.SimpleEngine.popen_uci(self.stockfish_path)
            for i, fen in enumerate(fens):
                if i == 0:
                    evals.append(0.0)
                    continue
                    
                board = chessgnn.Board(fen)
                try:
                    info = engine.analyse(board, chessgnn.engine.Limit(time=0.1)) # Faster than V3 1.0s
                    score = info["score"].white()
                    if score.is_mate():
                        val = 15000 * (1 if score.mate() > 0 else -1)
                    else:
                        val = score.score()
                    evals.append(val)
                except Exception:
                    evals.append(0.0)
        except Exception as e:
            print(f"Engine error: {e}")
            return [0.0] * len(fens)
        finally:
            if engine:
                engine.quit()
        
        return evals

    def infer_move(self, prev_fen: str, curr_fen: str) -> Tuple[int, int]:
        """Infer move from two FENs."""
        # Simple implementation
        try:
            board = chessgnn.Board(prev_fen)
            for move in board.legal_moves:
                board.push(move)
                if board.fen() == curr_fen:
                    return move.from_square, move.to_square
                board.pop()
        except:
            pass
        return None, None
