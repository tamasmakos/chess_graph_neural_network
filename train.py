
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
BATCH_SIZE = 1 
HIDDEN_DIM = 256
LR = 0.005 
EPOCHS = 2
TRAIN_GAMES = 100
TEST_GAMES = 5

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


from chessgnn.game_processor import ChessGameProcessor
from chessgnn.visualizer import ChessVisualizer, GameVideoGenerator
from chessgnn.graph_builder import ChessGraphBuilder

def visualize_test_games(model, pgn_file, num_games, offset, device, epoch):
    """
    Visualizes test games by generating videos with dynamic win probability plots.
    """
    model.eval()
    logger.info("\n" + "="*50)
    logger.info(f"  🎥  VISUALIZING {num_games} TEST GAMES (Epoch {epoch})  🎥")
    logger.info("="*50)
    
    vis = ChessVisualizer()
    video_gen = GameVideoGenerator(vis)
    processor = ChessGameProcessor(stockfish_path="/workspaces/chessgnn/stockfish/src/stockfish")
    builder = ChessGraphBuilder()
    
    # Skip to test games
    with open(pgn_file) as f:
        for _ in range(offset):
            if chess.pgn.read_game(f) is None:
                return

        for i in range(num_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
                
            game_states, fens = processor.process_game(game)
            
            # Get Stockfish Evaluations
            stockfish_evals = processor.get_stockfish_evaluations(fens)

            # Predict Win Probs
            # We need to feed the sequence to the model
            # Model expects [Batch, Time, Features] or List[Graph] if handled by collate/forward
            # Our model.forward takes a list of HeteroData (history)
            
            win_probs = []
            history_graphs = []
            
            # ... (inference loop) ...
            for fen in fens:
                graph = builder.fen_to_graph(fen).to(device)
                history_graphs.append(graph)
                
                # Keep history window reasonable (e.g. 8) OR pass full history if model supports it
                seq_window = history_graphs[-16:]
                
                with torch.no_grad():
                    # model(seq) -> [1, T, 1]
                    # We only care about the last step
                    scores = model(seq_window)
                    last_score = scores[0, -1, 0].item()
                    
                prob = (last_score + 1) / 2 * 100
                win_probs.append(prob)
                
            # Generate Video
            output_path = f"output/videos/epoch_{epoch}_game_{i+1}.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            video_gen.generate_video(game_states, win_probs, stockfish_evals, output_path, fps=2)
            logger.info(f"Generated video: {output_path}")

    logger.info("="*50 + "\n")
    model.train()


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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate, num_workers=4)
    
    logger.info(f"Starting training loop... Batch Size={BATCH_SIZE}, LR={LR}")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_steps = 0
        
        for batch_list in pbar:
            # batch_list: List[Dict]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            total_target_val = 0.0
            count = 0
            
            last_sample = None
            
            # Gradient Accumulation Implementation
            for sample_dict in batch_list:
                
                sequence = sample_dict['sequence'] 
                # sequence is already a list of HeteroData.
                # Ensure they are on device.
                sequence = [g.to(device) for g in sequence]
                
                target_val = sample_dict['target_value'] 
                
                # Model Forward
                # Output: [1, T, 1] (Batch=1)
                score_seq = model(sequence) 
                
                # Target Construction
                # We want to supervise EVERY step with the final game result.
                # Target: [1, T, 1]
                T = score_seq.shape[1]
                target = torch.full((1, T, 1), target_val, device=device).float()
                
                loss = mse_criterion(score_seq, target)
                
                # Normalize by batch size (usually 1 here)
                loss_scaled = loss / len(batch_list)
                loss_scaled.backward()
                
                if not math.isnan(loss.item()):
                    batch_loss += loss.item()
                count += 1
                
                total_target_val += target_val
                last_sample = sample_dict
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            avg_loss = batch_loss / count if count > 0 else 0
            
            total_loss += avg_loss
            total_samples += 1
            epoch_steps += 1
            
            # LOGGING FREQUENCY CHANGED: EACH STEP
            if epoch_steps % 1 == 0:
                 # Take mean of the last game's predictions or just the last step?
                 # Let's take the Last Step of the last game processed
                 last_score = score_seq[0, -1, 0].item()
                 win_prob = (last_score + 1) / 2 * 100
                 actual_win = (total_target_val / count + 1) / 2 * 100 if count > 0 else 50.0
                 logger.info(f"Step {epoch_steps} | Loss: {avg_loss:.4f} | WinProb: {win_prob:.1f}% | ActualWin: {actual_win:.1f}%")


    # Save Model
    save_path = os.path.join("output", "st_hgat_model.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    # Final Visualization
    try:
        visualize_test_games(model, PGN_FILE, TEST_GAMES, TRAIN_GAMES, device, "final")
    except Exception as e:
        logger.error(f"Final visualization failed: {e}")

if __name__ == "__main__":
    train()
