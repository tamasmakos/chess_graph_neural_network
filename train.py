
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import HeteroData
from torch.utils.data import DataLoader
from tqdm import tqdm
import chess
from chessgnn.dataset import ChessGraphDataset, custom_collate
from chessgnn.model import STHGATLikeModel

# Configuration
PGN_FILE = "/workspaces/chessgnn/input/lichess_db_standard_rated_2013-01.pgn"
BATCH_SIZE = 10 # We use batch size 1 (sequence of 1 game) for simplicity
HIDDEN_DIM = 128
LR = 0.0005
EPOCHS = 2
TRAIN_GAMES = 100
TEST_GAMES = 50

# Setup Logging
import logging
import os
os.makedirs("output", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("output/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def tutor_spotlight(model, test_loader, device):
    """
    Selects a random position from the test set and has the model 'commentate' on it.
    """
    model.eval()
    logger.info("\n" + "="*50)
    logger.info("  ♟️  TUTOR SPOTLIGHT ANALYST  ♟️")
    logger.info("="*50)
    
    # Grab one batch (game)
    batch = next(iter(test_loader))
    sample_dict = batch[0]
    
    # Pick a random step in the game sequence
    seq_len = len(sample_dict['sequence'])
    if seq_len == 0: return # Safety
    
    # Use the last frame (latest position)
    graph = sample_dict['sequence'][-1].to(device)
    fen = sample_dict['fen']
    target_val = sample_dict['target_value']
    
    # Run Model
    with torch.no_grad():
        # Model expects a list of graphs (sequence)
        score = model([graph]).item() # -1 to 1
    
    # Visual Interpretation
    win_prob = (score + 1) / 2 * 100 # 0% to 100%
    
    # Color/Emoji logic
    if win_prob > 55:
        judgement = "White is winning"
        emoji = "⚪"
    elif win_prob < 45:
        judgement = "Black is winning"
        emoji = "⚫"
    else:
        judgement = "Position is Equal"
        emoji = "⚖️"
        
    actual_res = "Draw"
    if target_val > 0.5: actual_res = "White Won"
    elif target_val < -0.5: actual_res = "Black Won"
    
    board = chess.Board(fen)
    logger.info(f"Position: {fen}")
    logger.info(f"Analysis: {emoji} {judgement} ({win_prob:.1f}% Win Prob)")
    logger.info(f"Actual Game Result: {actual_res}")
    
    # Simple ASCII Board
    logger.info("\n" + str(board) + "\n")
    logger.info("="*50 + "\n")


# ... existing code ...
from chessgnn.graph_builder import ChessGraphBuilder

def deep_inspection(model, fen, sequence, device, builder):
    """
    Evaluates ALL legal moves in the position and logs a ranked list.
    Now temporally aware: uses the provided history sequence.
    """
    model.eval()
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return

    logger.info(f"\n🔍 DEEP INSPECTION (Step Analysis) 🔍")
    logger.info(f"Position: {fen}")
    logger.info(f"Evaluating {len(legal_moves)} legal moves with history context...")
    
    move_analysis = []
    
    # Pre-process history: ensure it's on device
    history_window = [g.to(device) for g in sequence]
    # If history is full (window size 8), we'll slice [1:] when appending new move
    
    for move in legal_moves:
        board.push(move)
        # Convert resultant position to graph
        try:
            graph = builder.fen_to_graph(board.fen())
            graph = graph.to(device)
            
            # Construct Temporal Sequence for this candidate
            # Pivot: Slide window if needed found in dataset.py logic?
            # Model GRU handles variable length, but for consistency with training (window=8),
            # we should maintain similar context length.
            # If len(history_window) >= 8, drop first.
            
            if len(history_window) >= 8:
                 candidate_seq = history_window[1:] + [graph]
            else:
                 candidate_seq = history_window + [graph]
            
            # Model evaluation
            # Pass as list of graphs (model expects sequence_graphs)
            score = model(candidate_seq).item()
                
            # Softsign is already in model, outputs -1..1
            display_score = score 
            # Convert to Win Prob %
            win_prob = (display_score + 1) / 2 * 100
            
            move_analysis.append((move.uci(), win_prob, score))
        except Exception as e:
            pass 
        finally:
            board.pop()
            
    # Sort Analysis
    # If White Turn: Descending (Higher is better for White)
    # If Black Turn: Ascending (Lower is better for Black, i.e. higher Black Win Prob)
    
    # Actually, simpler interpretation:
    # The model outputs "White Advantage".
    # So we sort by raw Score.
    
    # But for "Best Move" display, we want the best move for the CURRENT player.
    if board.turn == chess.WHITE:
        move_analysis.sort(key=lambda x: x[2], reverse=True) # Maximize White Score
        best_label = "White (Max Score)"
    else:
        move_analysis.sort(key=lambda x: x[2], reverse=False) # Minimize White Score (Maximize Black)
        best_label = "Black (Min Score)"

    logger.info(f"--- Top 5 Recommended Moves for {best_label} ---")
    for i, (uci, prob, score) in enumerate(move_analysis[:5]):
        logger.info(f"#{i+1}: {uci} | WinProb: {prob:.1f}% | Raw: {score:.4f}")
        
    logger.info(f"--- Worst 3 Blunders (Avoid) ---")
    for i, (uci, prob, score) in enumerate(move_analysis[-3:]):
        logger.info(f"X : {uci} | WinProb: {prob:.1f}% | Raw: {score:.4f}")
        
    logger.info("------------------------------------------------\n")
    model.train() # Switch back to train mode

# ... imports ...
import random

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize Builder
    graph_builder = ChessGraphBuilder()
    
    # Load Data (Iterable)
    logger.info(f"Initializing Datasets with PGN: {PGN_FILE}")
    train_dataset = ChessGraphDataset(PGN_FILE, num_games=TRAIN_GAMES, offset=0)
    test_dataset = ChessGraphDataset(PGN_FILE, num_games=TEST_GAMES, offset=TRAIN_GAMES)
    
    # Initialize Model
    try:
        sample_iter = iter(ChessGraphDataset(PGN_FILE, num_games=1, offset=0))
        sample = next(sample_iter)
        sample_graph = sample['sequence'][-1]
        metadata = sample_graph.metadata()
    except StopIteration:
        logger.error("Dataset is empty/failed to load first sample")
        raise ValueError("Dataset is empty. Check PGN file path.")
        
    model = STHGATLikeModel(metadata, hidden_channels=HIDDEN_DIM, num_layers=3).to(device)
    logger.info(f"Model Initialized: STHGATLikeModel with hidden_dim={HIDDEN_DIM}, layers=3")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Loss Functions
    mse_criterion = nn.MSELoss() 
    # Ranking Margin removed (Outcome Regression Only)
    
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate, num_workers=4)
    
    logger.info("Starting training loop...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_steps = 0
        
        # Per-Game Stats placeholders
        current_game_id = None
        game_loss_accum = 0.0
        game_steps = 0
        
        for batch in pbar:
            sample_dict = batch[0]
            
            # Game Boundary Check
            sample_game_id = sample_dict['game_id']
            if current_game_id is None: current_game_id = sample_game_id
            # Logspam reduction: With parallel workers, IDs flip constantly. 
            # We disable "Game Finished" logging to clean up the output.
            if sample_game_id != current_game_id:
                # avg_game_loss = game_loss_accum / game_steps if game_steps > 0 else 0
                # logger.info(f"==> Game {current_game_id} Finished | Avg Loss: {avg_game_loss:.4f}")
                current_game_id = sample_game_id
                game_loss_accum = 0.0
                game_steps = 0
            
            # 1. Positive Sample (Played Move)
            sequence = sample_dict['sequence'] 
            # Move to device
            sequence = [g.to(device) for g in sequence]
            
            target_val = sample_dict['target_value'] # -1, 0, 1 (Game Result)
            # MSE Target for the STATE
            target = torch.tensor([target_val], device=device).float()
            
            optimizer.zero_grad()
            
            # Forward Pass (Positive - The real state that occurred)
            score_pos = model(sequence) # Scalar output for State_t
            
            mse_loss = mse_criterion(score_pos, target)
            
            # --- MODIFIED: Removed "Easy Negative" Ranking Loss ---
            # We now rely solely on Outcome Regression (MSE) to learn the Value Function V(s).
            # This avoids the trap of learning "Safe > Blunder" without learning true strategy.
            # To recommend moves, we simply evaluate V(s') for all legal next states s'.

            # Loss
            loss = mse_loss

            
            loss.backward()
            
            # Clip Gradients to prevent explosion driving Tanh to saturation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            total_samples += 1
            epoch_steps += 1
            
            game_loss_accum += current_loss
            game_steps += 1
            
            # Logging detailed step info every 50 steps
            if epoch_steps % 50 == 0:
                 # Convert score to Win Probability %
                 # Since we removed Tanh from model, we apply it here for display interpretation
                 # score 0 -> 50%, score 1 -> ~88%, score -1 -> ~12%
                 # Actually, let's just stick to the -1..1 semantics for 'WinProb'
                 raw_score = score_pos.item()
                 # Apply soft clipping for display
                 # Softsign maps to -1..1
                 display_score = raw_score 
                 win_prob = (display_score + 1) / 2 * 100
                 
                 logger.info(f"Step {epoch_steps} | Game {current_game_id} | Loss: {current_loss:.4f} (MSE:{mse_loss.item():.4f}) | WinProb: {win_prob:.1f}%")

            # --- NEW: DEEP INSPECTION EVERY 100 STEPS ---
            if epoch_steps % 100 == 0:
                 try:
                    deep_inspection(model, sample_dict['fen'], sample_dict['sequence'], device, graph_builder)
                 except Exception as e:
                    logger.error(f"Deep inspection failed: {e}")




    # Save Model
    save_path = os.path.join("output", "st_hgat_model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
