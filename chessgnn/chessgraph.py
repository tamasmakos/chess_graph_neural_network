#!/usr/bin/env python3
"""
Chess Game to Bipartite Graph Converter
Main CLI Entry point.
"""

import sys
import os
import argparse
import json
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

# Local imports
# Local imports
try:
    # Try relative imports first (when running as package)
    from .position_to_graph import analyze_position
    from .visualizer import ChessVisualizer
    from .game_processor import ChessGameProcessor
except ImportError:
    # Handle running as script
    # We need to add the parent directory to path to import 'chess' package
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # We must correspond to the directory name 'chess'
    from chessgnn.position_to_graph import analyze_position
    from chessgnn.visualizer import ChessVisualizer
    from chessgnn.game_processor import ChessGameProcessor

def main():
    parser = argparse.ArgumentParser(description="Convert chess games to bipartite graphs and visualizations.")
    parser.add_argument("pgn_file", help="Path to input PGN file")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--stockfish-path", "-s", help="Path to Stockfish executable")
    parser.add_argument("--create-animation", "-a", action="store_true", help="Create video animation")
    parser.add_argument("--last-days", type=int, default=3, help="Process games from last N days")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize implementation modules
    processor = ChessGameProcessor(stockfish_path=args.stockfish_path)
    visualizer = ChessVisualizer()
    
    # 1. Get Games
    games = processor.get_last_n_days_games(args.pgn_file, n_days=args.last_days)
    if not games:
        print("No games found.")
        return

    print(f"Processing {len(games)} games...")
    
    for game_idx, game in enumerate(games):
        headers = game.headers
        white = headers.get("White", "Unknown")
        black = headers.get("Black", "Unknown")
        date = headers.get("Date", "Unknown")
        game_name = f"{date}_{white}_vs_{black}".replace(" ", "_").replace("/", "-")
        print(f"\n[{game_idx+1}/{len(games)}] Processing {game_name}...")
        
        # 2. Process Game to GameStates
        game_states, fens = processor.process_game(game)
        
        # 3. Get Evaluations
        print("  Evaluating positions...")
        evaluations = processor.get_stockfish_evaluations(fens)
        
        # 4. Analyze Graphs & Communities
        print("  analyzing graphs...")
        per_move_communities = []
        centrality_data = [] # List of dicts
        
        for gs in tqdm(game_states, desc="Graph Analysis"):
            # We recreate the graph to get networkx object for analysis
            analysis = analyze_position(gs.fen)
            # Use pagerank from analysis
            centrality_data.append(analysis.centralities.get('pagerank_centrality', {}))
            
            # Compute communities (we need to rebuild the graph or use analysis?)
            # creating nx graph from analysis is not directly exposed but we can infer or use helper
            # Actually visualizer has compute_leiden_communities which takes nx graph
            # Let's add a helper in main or modify visualizer api to take analysis?
            # For now, let's just rebuild quickly or ideally analyze_position should return community?
            # It doesn't. 
            # We can rebuild simplified graph for community detection
            
            # Quick rebuild from edges for community detection
            G = nx.Graph()
            G.add_nodes_from([p.square for p in gs.pieces])
            for u, v, w in gs.attack_edges: G.add_edge(u, v, weight=w)
            for u, v, w in gs.defense_edges: G.add_edge(u, v, weight=w)
            
            comms = visualizer.compute_leiden_communities(G)
            per_move_communities.append(comms)
            
        # 5. Stabilize Colors
        print("  Stabilizing community colors...")
        stable_communities, palette = visualizer.create_stable_community_colors(per_move_communities)
        
        # 6. Save Graph Data
        graph_data = {
            "game_info": {"white": white, "black": black, "date": date},
            "moves": []
        }
        for i, gs in enumerate(game_states):
            graph_data["moves"].append({
                "move_number": gs.move_number,
                "fen": gs.fen,
                "evaluation": evaluations[i],
                # serialize other data if needed
            })
            
        json_path = os.path.join(args.output_dir, f"{game_name}_data.json")
        with open(json_path, 'w') as f:
            json.dump(graph_data, f, indent=2)
            
        # 7. Visualization / Animation
        if args.create_animation:
            print("  Generating animation...")
            video_path = os.path.join(args.output_dir, f"{game_name}.avi")
            
            # Setup video writer
            # Using matplotlib animation or cv2? V3 used cv2 but relied on drawing logic
            # Let's use matplotlib animation for cleaner code or manual frame rendering
            
            # We will generate frames and write with cv2 for speed/control as in V3
            temp_dir = os.path.join(args.output_dir, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Render frames
            frame_paths = []
            for i, gs in enumerate(tqdm(game_states, desc="Rendering Frames")):
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Determine move highlight
                highlight = None
                if i > 0:
                    highlight = processor.infer_move(game_states[i-1].fen, gs.fen)
                
                visualizer.visualize_game_state(
                    gs, ax, 
                    title=f"{white} vs {black}\nMove {gs.move_number} - Eval: {evaluations[i]/100:.2f}",
                    centrality_scores=centrality_data[i],
                    community_mapping=stable_communities[i],
                    highlight_move=highlight
                )
                
                # Save frame
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                plt.savefig(frame_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                frame_paths.append(frame_path)
            
            # Compile video
            if frame_paths:
                first_frame = cv2.imread(frame_paths[0])
                height, width, layers = first_frame.shape
                video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 2, (width, height))
                
                for path in frame_paths:
                    video.write(cv2.imread(path))
                    os.remove(path) # cleanup
                
                cv2.destroyAllWindows()
                video.release()
                os.rmdir(temp_dir)
                print(f"  Video saved to {video_path}")

if __name__ == "__main__":
    main()