
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Linear
from torch.nn import ParameterDict
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones

class WeightedHGTConv(MessagePassing):
    """
    Heterogeneous Graph Transformer with Edge Weights.
    """
    def __init__(self, in_channels, out_channels, metadata, heads=1, **kwargs):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.d_k = out_channels // heads
        self.metadata = metadata

        self.k_lin = nn.ModuleDict()
        self.q_lin = nn.ModuleDict()
        self.v_lin = nn.ModuleDict()
        self.a_lin = nn.ModuleDict()
        
        self.skip = ParameterDict()
        
        node_types, edge_types = metadata
        
        for nt in node_types:
            self.k_lin[nt] = Linear(in_channels, out_channels)
            self.q_lin[nt] = Linear(in_channels, out_channels)
            self.v_lin[nt] = Linear(in_channels, out_channels)
            self.a_lin[nt] = Linear(out_channels, out_channels)
            
            if in_channels != out_channels:
                self.skip[nt] = Linear(in_channels, out_channels)
            else:
                self.skip[nt] = nn.Identity()
            
        self.relation_att = nn.ParameterDict()
        self.relation_msg = nn.ParameterDict()
        self.relation_pri = nn.ParameterDict()
        
        for et in edge_types:
            et_str = '__'.join(et)
            self.relation_att[et_str] = nn.Parameter(torch.Tensor(heads, self.d_k, self.d_k))
            self.relation_msg[et_str] = nn.Parameter(torch.Tensor(heads, self.d_k, self.d_k))
            self.relation_pri[et_str] = nn.Parameter(torch.ones(1))
            
            glorot(self.relation_att[et_str])
            glorot(self.relation_msg[et_str])

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        out_dict = {}
        
        # Prepare Q, K, V
        k_dict, q_dict, v_dict = {}, {}, {}
        for nt, x in x_dict.items():
            k_dict[nt] = self.k_lin[nt](x).view(-1, self.heads, self.d_k)
            q_dict[nt] = self.q_lin[nt](x).view(-1, self.heads, self.d_k)
            v_dict[nt] = self.v_lin[nt](x).view(-1, self.heads, self.d_k)

        # Propagate
        for et, edge_index in edge_index_dict.items():
            src_type, _, dst_type = et
            et_str = '__'.join(et)
            
            # Edge Weights
            # If edge_weight_dict is provided and has this edge type
            edge_weight = None
            if edge_weight_dict and et in edge_weight_dict:
                edge_weight = edge_weight_dict[et] # Expecting [E, 1] or [E]
            
            out = self.propagate(edge_index, 
                                 k=k_dict[src_type], 
                                 q=q_dict[dst_type], 
                                 v=v_dict[src_type], 
                                 rel_att=self.relation_att[et_str],
                                 rel_msg=self.relation_msg[et_str],
                                 rel_pri=self.relation_pri[et_str],
                                 edge_weight=edge_weight)
                                 
            if dst_type not in out_dict:
                out_dict[dst_type] = out
            else:
                out_dict[dst_type] += out
                
        # Skip connection & Update
        for nt in out_dict:
            out_dict[nt] = self.a_lin[nt](out_dict[nt].view(-1, self.out_channels))
            out_dict[nt] += self.skip[nt](x_dict[nt])
            out_dict[nt] = F.gelu(out_dict[nt])
            
        return out_dict

    def message(self, k_j, q_i, v_j, rel_att, rel_msg, rel_pri, index, edge_weight):
        # k_j: [E, Heads, D]
        
        # Attention Score
        # (K * W_att) * Q
        k_att = torch.einsum('ehd, hdk -> ehk', k_j, rel_att)
        alpha = (k_att * q_i).sum(dim=-1) * rel_pri / math.sqrt(self.d_k) # [E, Heads]
        
        # Integrate Edge Weight into Attention
        if edge_weight is not None:
            # edge_weight: [E] or [E, 1]
            if edge_weight.dim() == 1: edge_weight = edge_weight.unsqueeze(-1)
            # alpha = alpha * (1 + lambda * w)
            # Simplified: alpha = alpha + log(w)? Or multiplicative?
            # Specification: "AttnHead ... * (1 + lambda w_st)" -> This modulates the ATTENTION VALUE (pre-softmax) or PROBABILITY?
            # Usually pre-softmax logits.
            # Let's add it to logits to bias attention.
            # alpha += edge_weight
            alpha = alpha * (1.0 + edge_weight)

        alpha = softmax(alpha, index)
        
        # Message
        # V * W_msg
        v_msg = torch.einsum('ehd, hdk -> ehk', v_j, rel_msg)
        return v_msg * alpha.unsqueeze(-1)

class RayAlignmentBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.transmissibility_net = nn.Linear(in_channels, 1)
        self.ray_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x_dict, edge_index_ray, edge_attr_ray):
        # x_dict['piece']: [N_p, D]
        # edge_index_ray: [2, E_ray]
        # edge_attr_ray: [E_ray, 2] (dist, blocking)
        
        x = x_dict['piece']
        
        # 1. Compute Transmissibility per node
        # T_v = sigmoid(W z_v + b)
        # Represents likelihood to PASS influence.
        t_val = torch.sigmoid(self.transmissibility_net(x)) # [N, 1]
        
        # 2. Propagate
        # Influence = Gate(blocking_count) * Gate(Transmissibility of blockers)?
        # For MVP: We assume direct edge exists if aligned.
        # We modulate the message by 1 / (1 + blocking_count).
        # This is a soft "inverse distance" weighting based on obstacles.
        
        # E_ray: [dist, blocking]
        blocking_count = edge_attr_ray[:, 1].unsqueeze(-1) # [E, 1]
        dist = edge_attr_ray[:, 0].unsqueeze(-1) # [E, 1]
        
        # Weight = 1 / (1 + blocking + 0.1 * dist) 
        # Favor closer and unblocked pieces.
        ray_weight = 1.0 / (1.0 + blocking_count + 0.1 * dist)

        # Apply Source Transmissibility
        # If the source piece is NOT a slider (e.g. Knight), it should not be sending rays.
        # ray_weight *= t_val[source]
        row, col = edge_index_ray
        source_transmissibility = t_val[row] 
        ray_weight = ray_weight * source_transmissibility
        
        # Message passing manually or via sparse MM?
        # Simple GAT-like: Target = Sum( Weight * Source )
        # Using sparse matrix multiplication logic or loop.
        
        # Project source
        x_proj = self.ray_proj(x)
        
        # Scatter add (Message Passing)
        row, col = edge_index_ray
        
        # Message = x_proj[source] * weight
        msg = x_proj[row] * ray_weight
        
        # Aggregate to target
        # Using a simple scatter add implementation if torch_scatter is avail, else loop (slow) or use index_add?
        # Standard GNN approach:
        out = torch.zeros_like(x)
        out.index_add_(0, col, msg)
        
        # Update piece features
        # Residual connection
        x_new = x + out
        
        # Update dict
        x_dict['piece'] = x_new
        return x_dict



class STHGATLikeModel(nn.Module):
    def __init__(self, metadata, hidden_channels=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        self.square_proj = Linear(3, 10) 
        self.ray_block = RayAlignmentBlock(hidden_channels) 
        
        # Stack of GNN Layers
        self.convs = nn.ModuleList()
        # First layer: input dim 10 -> hidden
        self.convs.append(WeightedHGTConv(10, hidden_channels, metadata, heads=4))
        
        # Subsequent layers: hidden -> hidden
        for _ in range(num_layers - 1):
             self.convs.append(WeightedHGTConv(hidden_channels, hidden_channels, metadata, heads=4))

        # Temporal RNN (Simplified: GRU over global graph state)
        # Input to GRU is now 2 * hidden (Piece + Square pool)
        self.global_gru = nn.GRU(hidden_channels * 2, hidden_channels, batch_first=True)
        
        # Add Value Head
        self.value_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Softsign() # Outputs bounded scalar (-1 to 1)
        )
        
        # Initialize final layer to 0 to start with 0.0 prediction
        # Initialize final layer to 0 to start with 0.0 prediction
        nn.init.constant_(self.value_head[2].weight, 0.0)
        nn.init.constant_(self.value_head[2].bias, 0.0)
        # glorot(self.value_head[2].weight)
        # self.value_head[2].bias.data.fill_(0)

    def forward(self, sequence_graphs):
        # sequence_graphs: List[HeteroData] (Length T) representing the full game
        
        # 1. Batch the entire sequence into one Super-Graph
        # This allows parallel computation of GNN over all time steps
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(sequence_graphs)
        
        # 2. Spatial Encode (Parallel)
        x_dict = batch.x_dict.copy()
        x_dict['square'] = self.square_proj(x_dict['square'])
        
        # Edge Weights
        ew_dict = {}
        if ('piece', 'interacts', 'piece') in batch.edge_attr_dict:
                ew_dict[('piece', 'interacts', 'piece')] = batch['piece', 'interacts', 'piece'].edge_attr[:, 0]

        # Multi-Layer GNN Pass
        for i, conv in enumerate(self.convs):
                x_dict = conv(x_dict, batch.edge_index_dict, ew_dict)
        
        z_dict = x_dict 

        # Ray Alignment Block (Spatial Logic)
        if ('piece', 'ray', 'piece') in batch.edge_index_dict:
                x_dict = self.ray_block(x_dict, 
                                        batch.edge_index_dict[('piece', 'ray', 'piece')],
                                        batch.edge_attr_dict[('piece', 'ray', 'piece')])

        z_dict = x_dict 

        # 3. Pooling (Graph Level)
        # We need to pool nodes back to their respective graphs in the batch.
        # Batch.batch maps nodes to batch index.
        # batch.batch_dict['piece'] -> [N_total_pieces] (indices 0 to T-1)
        
        from torch_geometric.nn import global_mean_pool
        
        piece_embeds = z_dict['piece'] 
        square_embeds = z_dict['square']
        
        # Pool Pieces
        # Some graphs might have NO pieces (rare but possible in empty board? No, kings always exist).
        # We assume batch indices exist.
        batch_index_piece = batch[ 'piece' ].batch
        p_pool = global_mean_pool(piece_embeds, batch_index_piece) # [T, Hidden]
        
        # Pool Squares
        batch_index_square = batch[ 'square' ].batch
        s_pool = global_mean_pool(square_embeds, batch_index_square) # [T, Hidden]
        
        # Concatenate
        graph_embeds = torch.cat([p_pool, s_pool], dim=1) # [T, 2 * Hidden]
        
        # 4. Temporal Process (GRU)
        # Input: [Batch=1, Seq_Len=T, Input_Dim]
        seq_tensor = graph_embeds.unsqueeze(0) 
        
        # GRU
        # output: [1, T, Hidden]
        gru_out, _ = self.global_gru(seq_tensor)
        
        # 5. Predict Win Probability for ALL steps
        # output: [1, T, 1]
        values = self.value_head(gru_out)
        
        return values

    def forward_step(self, graph, h_prev=None):
        """
        Runs one step of inference for minimal latency.
        Args:
            graph (HeteroData): The current board state graph.
            h_prev (Tensor, optional): Previous GRU hidden state [1, 1, Hidden].
        Returns:
            value (Tensor): Win probability logit [1, 1].
            h_new (Tensor): New GRU hidden state [1, 1, Hidden].
        """
        # 1. Spatial Encode (Single Graph)
        x_dict = graph.x_dict.copy()
        x_dict['square'] = self.square_proj(x_dict['square'])
        
        # Edge Weights
        ew_dict = {}
        if ('piece', 'interacts', 'piece') in graph.edge_attr_dict:
                 ew_dict[('piece', 'interacts', 'piece')] = graph['piece', 'interacts', 'piece'].edge_attr[:, 0]

        # Multi-Layer GNN Pass
        for i, conv in enumerate(self.convs):
            # No batch index needed for single graph, but GNN expects edge_index dict
            x_dict = conv(x_dict, graph.edge_index_dict, ew_dict)
        
        z_dict = x_dict 

        # Ray Alignment Block
        if ('piece', 'ray', 'piece') in graph.edge_index_dict:
                x_dict = self.ray_block(x_dict, 
                                        graph.edge_index_dict[('piece', 'ray', 'piece')],
                                        graph.edge_attr_dict[('piece', 'ray', 'piece')])

        z_dict = x_dict 
        
        # 2. Pooling
        from torch_geometric.nn import  global_mean_pool
        
        # For a single graph without batch attribute, we treat all nodes as batch 0
        p_pool = torch.mean(z_dict['piece'], dim=0, keepdim=True) # [1, Hidden]
        s_pool = torch.mean(z_dict['square'], dim=0, keepdim=True) # [1, Hidden]
        
        graph_embed = torch.cat([p_pool, s_pool], dim=1) # [1, 2 * Hidden]
        
        # 3. Temporal Update (GRU Cell equivalent)
        # GRU expects input [Batch, Seq, Feature] if using full GRU with batch_first=True
        # We process a sequence of length 1
        seq_tensor = graph_embed.unsqueeze(0) # [1, 1, 2*Hidden]
        
        if h_prev is None:
            # Init hidden state
             h_prev = torch.zeros(1, 1, self.hidden_channels, device=graph_embed.device)
        
        gru_out, h_new = self.global_gru(seq_tensor, h_prev)
        # gru_out: [1, 1, Hidden]
        # h_new: [1, 1, Hidden]
        
        # 4. Predict
        value = self.value_head(gru_out.squeeze(0).squeeze(0)) # [1]
        
        return value, h_new
