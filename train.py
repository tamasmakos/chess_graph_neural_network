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
BATCH_SIZE = 1 # We use batch size 1 (sequence of 1 game) for simplicity
HIDDEN_DIM = 64
LR = 0.001
EPOCHS = 10
TRAIN_GAMES = 300
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

def _map_moves_to_indices(legal_moves, last_graph, fen):
    """
    Maps chess.Move objects to (src_node_idx, dst_node_idx, is_dst_piece) tuple.
    """
    board = chess.Board(fen)
    
    # Reconstruct Piece Map (Square -> Piece Node Index)
    # Logic matches Reader/GraphBuilder: pieces added in 0..63 order of squares.
    piece_map = {} 
    curr_idx = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            piece_map[sq] = curr_idx
            curr_idx += 1
            
    # Square Node Index is just the square integer (0..63)
    square_map = {sq: i for i, sq in enumerate(chess.SQUARES)} 
    
    legal_indices = []
    for move in legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        
        if from_sq not in piece_map:
            continue
            
        src_idx = piece_map[from_sq]
        dst_piece = board.piece_at(to_sq)
        
        if dst_piece:
            # Capturing a piece
            if to_sq in piece_map:
                dst_idx = piece_map[to_sq]
                legal_indices.append((src_idx, dst_idx, True))
            else:
                pass
        else:
            # Moving to a square
            dst_idx = square_map[to_sq]
            legal_indices.append((src_idx, dst_idx, False))
            
    return legal_indices

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
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
        
    model = STHGATLikeModel(metadata, hidden_channels=HIDDEN_DIM).to(device)
    logger.info(f"Model Initialized: STHGATLikeModel with hidden_dim={HIDDEN_DIM}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate)
    
    logger.info("Starting training loop...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_samples = 0
        
        # Use simple counter since len() is not available
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        epoch_steps = 0
        
        # Per-Game Stats placeholders
        current_game_id = None
        game_correct = 0
        game_total = 0
        game_loss_accum = 0.0
        
        for batch in pbar:
            sample_dict = batch[0]
            
            # Game Boundary Check
            # DataLoader collates into batch, so we access item 0
            # sample_dict['game_id'] might be a Tensor if collated default, 
            # but custom_collate returns list of dicts? 
            # No, custom_collate returns batch (list of dicts). 
            # sample_dict is the dict itself.
            # 'game_id' is int because we didn't convert to tensor in dataset.
            
            sample_game_id = sample_dict['game_id']
            
            if current_game_id is None:
                current_game_id = sample_game_id
                
            if sample_game_id != current_game_id:
                # Finish previous game
                acc = game_correct / game_total if game_total > 0 else 0
                avg_game_loss = game_loss_accum / game_total if game_total > 0 else 0
                logger.info(f"==> Game {current_game_id} Finished | Accuracy: {acc*100:.1f}% ({game_correct}/{game_total}) | Avg Loss: {avg_game_loss:.4f}")
                
                # Reset for new game
                current_game_id = sample_game_id
                game_correct = 0
                game_total = 0
                game_loss_accum = 0.0
            
            sequence = sample_dict['sequence'] 
            # Move to device
            sequence = [g.to(device) for g in sequence]
            
            legal_moves = sample_dict['legal_moves']
            target_idx = torch.tensor([sample_dict['target_index']], device=device)
            
            # Map Moves
            last_graph = sequence[-1]
            legal_indices = _map_moves_to_indices(legal_moves, last_graph, sample_dict['fen'])
            
            # Forward
            optimizer.zero_grad()
            scores = model(sequence, legal_indices) 
            
            # Loss
            loss = criterion(scores.unsqueeze(0), target_idx)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            total_samples += 1
            epoch_steps += 1
            
            # Update Game Stats
            game_loss_accum += current_loss
            pred_idx = torch.argmax(scores).item()
            if pred_idx == target_idx.item():
                game_correct += 1
            game_total += 1
            
            # Logging detailed step info every 10 steps
            if epoch_steps % 10 == 0:
                 pred_idx = torch.argmax(scores).item()
                 # Get UCI moves for logging
                 try:
                     pred_move = legal_moves[pred_idx]
                     true_move = legal_moves[target_idx.item()]
                     logger.info(f"Epoch {epoch+1} | Step {epoch_steps} | Game {current_game_id} | Loss: {current_loss:.4f} | Pred: {pred_move} | Target: {true_move}")
                 except:
                     pass # Fallback if indices are off
            
            pbar.set_postfix({'Loss': total_loss / max(1, total_samples)})
            
        # Report last game of epoch
        if game_total > 0:
            acc = game_correct / game_total if game_total > 0 else 0
            avg_game_loss = game_loss_accum / game_total if game_total > 0 else 0
            logger.info(f"==> Game {current_game_id} Finished | Accuracy: {acc*100:.1f}% ({game_correct}/{game_total}) | Avg Loss: {avg_game_loss:.4f}")

            
        avg_loss = total_loss / max(1, total_samples)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} Completed | Avg Loss: {avg_loss:.4f}")
        
    # Evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    logger.info("Evaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            sample_dict = batch[0]
            sequence = [g.to(device) for g in sample_dict['sequence']]
            legal_moves = sample_dict['legal_moves']
            target_idx = sample_dict['target_index']
            
            legal_indices = _map_moves_to_indices(legal_moves, sequence[-1], sample_dict['fen'])
            scores = model(sequence, legal_indices)
            
            pred_idx = torch.argmax(scores).item()
            if pred_idx == target_idx:
                test_correct += 1
            test_total += 1
            
    acc = test_correct/test_total if test_total > 0 else 0
    logger.info(f"Test Evaluation Completed | Accuracy: {acc:.4f} ({test_correct}/{test_total})")

    # Save Model
    save_path = os.path.join("output", "st_hgat_model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
