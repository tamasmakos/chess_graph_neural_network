import torch
import unittest
import chess
from chessgnn.model import STHGATLikeModel
from chessgnn.graph_builder import ChessGraphBuilder
from torch_geometric.data import HeteroData
from tutor import CaseTutor

class TestStatefulInference(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mock metadata
        self.metadata = (
            ['piece', 'square'],
            [('piece', 'interacts', 'piece'), 
             ('piece', 'ray', 'piece'),
             ('piece', 'on', 'square')] # Add if needed by model, though model only uses piece-interacts-piece explicitly in conv if configured
        )
        
        # Initialize Board
        self.board = chess.Board()
        self.builder = ChessGraphBuilder()
        
        # Get one graph to extract metadata from it for model init
        g = self.builder.fen_to_graph(self.board.fen())
        self.real_metadata = g.metadata()
        
        self.model = STHGATLikeModel(self.real_metadata, hidden_channels=32, num_layers=2).to(self.device)
        self.model.eval()

    def test_forward_step_equivalence(self):
        """
        Verify that running forward_step iteratively produces the same result 
        as running forward() on the full sequence.
        """
        # Generate a sequence of 3 moves
        moves = list(self.board.legal_moves)[:3]
        sequence_graphs = []
        
        # 1. Collect graphs
        g0 = self.builder.fen_to_graph(self.board.fen()).to(self.device)
        sequence_graphs.append(g0)
        
        # Play 3 valid moves sequentially
        for _ in range(3):
            legal = list(self.board.legal_moves)
            if not legal: break
            move = legal[0] # Just pick first valid move
            self.board.push(move)
            
            g = self.builder.fen_to_graph(self.board.fen()).to(self.device)
            sequence_graphs.append(g)
            
        print(f"Sequence length: {len(sequence_graphs)}")
        
        # 2. Run Full Sequence Forward
        with torch.no_grad():
            full_out_seq = self.model(sequence_graphs) # [1, T, 1]
            last_val_full = full_out_seq[0, -1, 0].item()
            
        # 3. Run Iterative Forward Step
        h_curr = None
        last_val_step = 0.0
        
        with torch.no_grad():
            for i, graph in enumerate(sequence_graphs):
                val, h_curr = self.model.forward_step(graph, h_curr)
                last_val_step = val.item()
                
                # Verify intermediate steps
                full_step_val = full_out_seq[0, i, 0].item()
                diff = abs(last_val_step - full_step_val)
                print(f"Step {i}: Full={full_step_val:.5f}, Step={last_val_step:.5f}, Diff={diff:.5f}")
                self.assertTrue(diff < 1e-5, f"Mismatch at step {i}")
                
        print("Equivalence Test Passed!")

    def test_tutor_statefulness(self):
        """
        Verify CaseTutor updates its state correctly.
        """
        tutor = CaseTutor(self.model, self.device)
        tutor.reset()
        
        self.assertIsNone(tutor.current_hidden)
        
        # Update State
        fen = chess.Board().fen()
        tutor.update_state(fen)
        
        self.assertIsNotNone(tutor.current_hidden)
        self.assertEqual(tutor.current_hidden.shape, (1, 1, 32)) # Hidden dim 32
        
        h_prev = tutor.current_hidden.clone()
        
        # Recommend should NOT change state
        tutor.recommend_move(fen)
        self.assertTrue(torch.equal(tutor.current_hidden, h_prev))
        
        print("Tutor State Test Passed!")

if __name__ == '__main__':
    unittest.main()
