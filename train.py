
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

def deep_inspection(model, fen, device, builder):
    """
    Evaluates ALL legal moves in the position and logs a ranked list.
    This provides the 'Why' behind the model's thinking.
    """
    model.eval()
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return

    logger.info(f"\n🔍 DEEP INSPECTION (Step Analysis) 🔍")
    logger.info(f"Position: {fen}")
    logger.info(f"Evaluating {len(legal_moves)} legal moves to find the best continuation...")
    
    move_analysis = []
    
    # Batch evaluation would be faster, but loop is safer for memory/implementation simplicity right now
    for move in legal_moves:
        board.push(move)
        # Convert resultant position to graph
        try:
            graph = builder.fen_to_graph(board.fen())
            graph = graph.to(device)
            
            # Model evaluation
            score = model([graph]).item()
                
            # Apply Tanh for display (map unbounded score to -1..1)
            display_score = math.tanh(score)
            win_prob = (display_score + 1) / 2 * 100
            
            move_analysis.append((move.uci(), win_prob, score))
        except Exception as e:
            pass # Skip invalid graph generations if any
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
    # Ranking Margin (how much better the played move should be than the random move)
    ranking_margin = 0.1 
    
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate)
    
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
            if sample_game_id != current_game_id:
                avg_game_loss = game_loss_accum / game_steps if game_steps > 0 else 0
                logger.info(f"==> Game {current_game_id} Finished | Avg Loss: {avg_game_loss:.4f}")
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
            
            # 2. Ranking Loss (Constraint: Played Move > Random Move)
            # We are at State_t. The played move (Move_P) leads to State_{t+1}.
            # The Random Move (Move_R) leads to State_{t+1}'
            
            # User wants: V(Move_P) > V(Move_R)
            # But we are forwarding State_t. Does the model see the move?
            # NO. The model sees the BOARD. 
            # So V(State_t) is the evaluation of the position BEFORE the move.
            
            # Wait. If we want to recommend a move, we perform input: Graph(State_After_Move).
            # So to train this, we must feed:
            # Pos: Graph(State_After_Move_P)  (Which is actually State_{t+1})
            # Neg: Graph(State_After_Move_R)
            
            # Challenge: Our Dataset yields State_t.
            # State_{t+1} is in the NEXT sample?
            # Yes, `sequence` is shifting window/buffer.
            # But for *Ranking*, we need to generate these graphs ON THE FLY from State_t.
            
            ranking_loss = torch.tensor(0.0, device=device)
            
            played_move_uci = sample_dict.get('played_move_uci')
            fen = sample_dict['fen']
            
            if played_move_uci:
                board = chess.Board(fen)
                legal_moves = list(board.legal_moves)
                
                # Check if we have enough moves to rank
                if len(legal_moves) > 1:
                    played_move = chess.Move.from_uci(played_move_uci)
                    
                    # 2a. Generate Graph for Positive (Result of Played Move)
                    board.push(played_move)
                    try:
                        g_pos = graph_builder.fen_to_graph(board.fen()).to(device)
                        score_after_pos = model([g_pos])
                    except:
                        score_after_pos = None
                    board.pop()
                    
                    # 2b. Generate Graph for Negative (Result of Random Move)
                    # Pick random move != played_move
                    candidates = [m for m in legal_moves if m != played_move]
                    if candidates and score_after_pos is not None:
                        neg_move = random.choice(candidates)
                        board.push(neg_move)
                        try:
                            g_neg = graph_builder.fen_to_graph(board.fen()).to(device)
                            score_after_neg = model([g_neg])
                            
                            # 2c. Compute Ranking Loss
                            # If White to move: Want Score_Pos > Score_Neg
                            # If Black to move: Want Score_Pos < Score_Neg (Score is "White Advantage")
                            
                            margin = ranking_margin
                            
                            if board.turn == chess.BLACK: # Note: We popped, turn is back to original player
                                # White just moved. White wants Score to be higher.
                                # Loss = max(0, Score_Neg - Score_Pos + Margin)
                                r_loss = torch.relu(score_after_neg - score_after_pos + margin)
                            else:
                                # Black just moved. Black wants Score to be lower.
                                # Target is e.g. -1. 
                                # We want Score_Pos < Score_Neg
                                # Loss = max(0, Score_Pos - Score_Neg + margin)
                                r_loss = torch.relu(score_after_pos - score_after_neg + margin)
                                
                            ranking_loss = r_loss
                            
                        except:
                            pass
                        board.pop()

            # Loss
            loss = mse_loss + ranking_loss
            
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
                 display_score = math.tanh(raw_score)
                 win_prob = (display_score + 1) / 2 * 100
                 
                 r_loss_val = ranking_loss.item() if isinstance(ranking_loss, torch.Tensor) else 0
                 logger.info(f"Step {epoch_steps} | Game {current_game_id} | Loss: {current_loss:.4f} (MSE:{mse_loss.item():.4f} Rnk:{r_loss_val:.4f}) | WinProb: {win_prob:.1f}%")

            # --- NEW: DEEP INSPECTION EVERY 100 STEPS ---
            if epoch_steps % 100 == 0:
                 try:
                    deep_inspection(model, sample_dict['fen'], device, graph_builder)
                 except Exception as e:
                    logger.error(f"Deep inspection failed: {e}")




    # Save Model
    save_path = os.path.join("output", "st_hgat_model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
