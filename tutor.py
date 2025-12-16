import torch
import chess
from chessgnn.graph_builder import ChessGraphBuilder

class CaseTutor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.builder = ChessGraphBuilder()
        
    def recommend_move(self, fen):
        """
        Evaluates all legal moves from the given FEN and recommends the best one.
        Returns:
            best_move (chess.Move): The best move found.
            best_prob (float): Win probability of the best move (0-100%).
            ranking (list): List of (move, prob) tuples sorted by quality.
        """
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None, 0.0, []
            
        move_scores = []
        
        # Determine optimization direction
        # If it's White's turn, higher score (+1) is better.
        # If it's Black's turn, lower score (-1) is better.
        is_white_turn = board.turn == chess.WHITE
        
        # Batching logic could be added here, but for now we loop
        # (30 moves is fast enough for 1-ply)
        
        for move in legal_moves:
            board.push(move)
            next_fen = board.fen()
            
            # Convert to Graph
            graph = self.builder.fen_to_graph(next_fen)
            graph = graph.to(self.device)
            
            # Predict
            with torch.no_grad():
                # Model expects sequence list
                raw_score = self.model([graph]).item()
                
            # Convert -1..1 score to White Win Probability %
            # +1 -> 100%, -1 -> 0%
            white_win_prob = (raw_score + 1) / 2 * 100
            
            move_scores.append((move, white_win_prob))
            
            board.pop()
            
        # Sort
        # If White to move, pick Highest White Win Prob
        # If Black to move, pick Lowest White Win Prob 
        # (Wait: Black wants to MINIMIZE White's win prob, i.e. MAXIMIZE Black's win prob)
        
        if is_white_turn:
            move_scores.sort(key=lambda x: x[1], reverse=True) # Descending
            best_move = move_scores[0][0]
            best_prob = move_scores[0][1] # Probability White wins
        else:
            move_scores.sort(key=lambda x: x[1], reverse=False) # Ascending
            best_move = move_scores[0][0]
            # Convert "White Win Prob" to "Black Win Prob" for display comfort?
            # Or keep it consistent? 
            # Let's return the probability relative to the Player Moving.
            best_prob = 100.0 - move_scores[0][1] # Probability Black wins
            
            # Update the list to show "My Win Prob" instead of "White Win Prob"?
            # Let's keep the raw white_win_prob in the list for consistency, but return relative best_prob.
            
        return best_move, best_prob, move_scores
