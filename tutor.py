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
        self.current_hidden = None
        
    def reset(self):
        """Resets the internal hidden state (New Game)."""
        self.current_hidden = None
        
    def update_state(self, fen):
        """
        Advances the internal hidden state with the played moves.
        Call this AFTER a move is committed to the board.
        """
        graph = self.builder.fen_to_graph(fen).to(self.device)
        with torch.no_grad():
             # Advance hidden state without caring about value
             _, self.current_hidden = self.model.forward_step(graph, self.current_hidden)

    def recommend_move(self, fen):
        """
        Evaluates all legal moves using the CURRENT hidden state context.
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
        is_white_turn = board.turn == chess.WHITE
        
        for move in legal_moves:
            board.push(move)
            next_fen = board.fen()
            
            # Convert to Graph
            graph = self.builder.fen_to_graph(next_fen)
            graph = graph.to(self.device)
            
            # Predict
            with torch.no_grad():
                # EFFICIENT INFERENCE:
                # Use the CACHED hidden state from previous moves.
                # Do NOT overwrite self.current_hidden (this is just exploring a branch)
                raw_score, _ = self.model.forward_step(graph, self.current_hidden)
                raw_score = raw_score.item()
                
            # Convert -1..1 score to White Win Probability %
            white_win_prob = (raw_score + 1) / 2 * 100
            
            move_scores.append((move, white_win_prob))
            
            board.pop()
            
        # Sort logic remains the same
        if is_white_turn:
            move_scores.sort(key=lambda x: x[1], reverse=True) # Descending (Max White Prob)
            best_move = move_scores[0][0]
            best_prob = move_scores[0][1] # Probability White wins
        else:
            move_scores.sort(key=lambda x: x[1], reverse=False) # Ascending (Min White Prob)
            best_move = move_scores[0][0]
            best_prob = 100.0 - move_scores[0][1] # Probability Black wins
            
        return best_move, best_prob, move_scores
