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
LR = 0.002
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
    criterion = nn.MSELoss() 
    
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
        game_loss_accum = 0.0
        game_steps = 0
        
        for batch in pbar:
            sample_dict = batch[0]
            
            # Game Boundary Check
            sample_game_id = sample_dict['game_id']
            
            if current_game_id is None:
                current_game_id = sample_game_id
                
            if sample_game_id != current_game_id:
                # Finish previous game
                avg_game_loss = game_loss_accum / game_steps if game_steps > 0 else 0
                logger.info(f"==> Game {current_game_id} Finished | Avg Loss: {avg_game_loss:.4f}")
                
                # Reset for new game
                current_game_id = sample_game_id
                game_loss_accum = 0.0
                game_steps = 0
            
            sequence = sample_dict['sequence'] 
            # Move to device
            sequence = [g.to(device) for g in sequence]
            
            # Target
            target = torch.tensor([sample_dict['target_value']], device=device).float()
            
            # Forward
            optimizer.zero_grad()
            scores = model(sequence) # Returns scalar [-1, 1]
            
            # Loss
            loss = criterion(scores, target)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            total_loss += current_loss
            total_samples += 1
            epoch_steps += 1
            
            # Update Game Stats
            game_loss_accum += current_loss
            game_steps += 1
            
            # Logging detailed step info every 50 steps
            if epoch_steps % 50 == 0:
                 # Convert score to Win Probability %
                 win_prob = (scores.item() + 1) / 2 * 100
                 logger.info(f"Epoch {epoch+1} | Step {epoch_steps} | Game {current_game_id} | Loss: {current_loss:.4f} | WinProb: {win_prob:.1f}% | Target: {target.item():.2f}")
            
            pbar.set_postfix({'Loss': total_loss / max(1, total_samples)})
            
        # Report last game of epoch
        if game_steps > 0:
            avg_game_loss = game_loss_accum / game_steps if game_steps > 0 else 0
            logger.info(f"==> Game {current_game_id} Finished | Avg Loss: {avg_game_loss:.4f}")

        avg_loss = total_loss / max(1, total_samples)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} Completed | Avg Loss: {avg_loss:.4f}")
        
    # Evaluation
    model.eval()
    test_loss = 0
    test_total = 0
    
    logger.info("Evaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            sample_dict = batch[0]
            sequence = [g.to(device) for g in sample_dict['sequence']]
            target = torch.tensor([sample_dict['target_value']], device=device).float()
            
            scores = model(sequence)
            loss = criterion(scores, target)
            
            test_loss += loss.item()
            test_total += 1
            
    avg_test_loss = test_loss/test_total if test_total > 0 else 0
    logger.info(f"Test Evaluation Completed | Avg MSE Loss: {avg_test_loss:.4f}")

    # Run Tutor Spotlight
    tutor_spotlight(model, test_loader, device)

    # Save Model
    save_path = os.path.join("output", "st_hgat_model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
