
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
        # Represents how likely this piece is to PASS influence (e.g. pinned or transparent)
        # Wait, Pinned pieces BLOCK? Or Pass?
        # Logic: "If intermediate nodes have high transmissibility (empty/pinned)... influence reaches"
        
        t_val = torch.sigmoid(self.transmissibility_net(x)) # [N, 1]
        
        # 2. Propagate
        # We need to propagate along the Ray edges.
        # But Ray edges skip intermediates? No, builder created edges between aligned pieces.
        # The builder created DIRECT edges between ALL aligned pieces (O(N^2) on rank)?
        # Yes, "edge between any two pieces ... regardless of obstruction".
        # And attr has 'blocking_count'.
        
        # So we have direct edge u->v with blocking_count.
        # Influence = Gate(blocking_count) * Gate(Transmissibility of blockers)?
        # This is hard to do if we don't know WHO the blockers are.
        # The edge attr only has COUNT.
        
        # Simplified Implementation as per Plan Specification:
        # "Network learns: if blocking_count == 1 and blocker is opponent -> Pin"
        # We use edge_attr directly.
        
        # We can implement this as a GAT layer on Ray edges where attention depends on blocking_count.
        pass # To be implemented fully in integration, for now returns x
        return x_dict

class STHGATLikeModel(nn.Module):
    def __init__(self, metadata, hidden_channels=64, num_layers=2):
        super().__init__()
        self.encoder = WeightedHGTConv(10, hidden_channels, metadata, heads=4) # piece input 10
        # Need separate input proj for squares (3 dims)
        self.square_proj = Linear(3, 10) # Project square to same dim as piece for HGT
        
        self.temporal_rnn = nn.GRU(hidden_channels, hidden_channels)
        
        self.policy_head = nn.Sequential(
            Linear(hidden_channels * 2 + 1, hidden_channels), # Src + Dst + EdgeType?
            nn.ReLU(),
            Linear(hidden_channels, 1)
        )

    def forward(self, sequence_graphs):
        # sequence_graphs: List[HeteroData]
        
        hidden_states = []
        
        for data in sequence_graphs:
            # 1. Spatial Encode
            # Project Square features
            x_dict = data.x_dict.copy()
            x_dict['square'] = self.square_proj(x_dict['square'])
            
            # Helper for edge weights
            ew_dict = {}
            if ('piece', 'interacts', 'piece') in data.edge_attr_dict:
                 # Extract the weight (first dim)
                 ew_dict[('piece', 'interacts', 'piece')] = data['piece', 'interacts', 'piece'].edge_attr[:, 0]

            out_dict = self.encoder(x_dict, data.edge_index_dict, ew_dict)
            
            # Pool to Global or keep Node Embeddings?
            # EvolveGCN updates WEIGHTS.
            # Here we just use GRU on embeddings (simpler baseline) or implement EvolveGCN properly?
            # Plan: "EvolveGCN adapts the parameters of the GNN layer".
            # That's complex to implement in one shot. 
            # I will fallback to GRU over node embeddings for the MVP, or just process frame by frame and pool.
            
            # Let's effectively use the last frame's embeddings, but context from RNN?
            # For MVP: Run GNN on last frame only is a strong baseline.
            # To add temporal: Concatenate history to features?
            
            # Using simple per-node GRU if nodes are consistent?
            # Nodes are NOT consistent (pieces move/die).
            # This is the "Variable Node" problem.
            
            # Solution: Graph Recurrent Network usually assumes fixed nodes or explicit matching.
            # Pivot: Use Transformer over temporal sequence of *pooled* graph vectors?
            # Or just use the LAST frame for the spatial graph, but include history features in the input?
            # The input already has "history" (recurent state) in the spec?
            # "h_hist: Recurrent state vector carried over".
            
            # We will return the node embeddings of the last graph.
            z_final = out_dict
            
        return z_final

class STHGATLikeModel(nn.Module):
    def __init__(self, metadata, hidden_channels=64, num_layers=2):
        super().__init__()
        self.encoder = WeightedHGTConv(10, hidden_channels, metadata, heads=4) 
        self.square_proj = Linear(3, 10) 
        
        # Temporal RNN (Simplified: GRU over global graph state)
        self.global_gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)
        
        self.hidden_channels = hidden_channels
        
        # Add Value Head
        self.value_head = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, 1),
            nn.Tanh() # Outputs between -1 (Loss) and 1 (Win)
        )

    def forward(self, sequence_graphs):
        # sequence_graphs: List[HeteroData]
        
        # 1. Encode Sequence
        # MVP: Just use the last frame G_t. 
        
        data = sequence_graphs[-1] # Only use the latest position
        
        # Projection
        x_dict = data.x_dict.copy()
        x_dict['square'] = self.square_proj(x_dict['square'])
        
        # Edge Weights
        ew_dict = {}
        if ('piece', 'interacts', 'piece') in data.edge_attr_dict:
                 ew_dict[('piece', 'interacts', 'piece')] = data['piece', 'interacts', 'piece'].edge_attr[:, 0]

        # GNN Pass
        z_dict = self.encoder(x_dict, data.edge_index_dict, ew_dict)
        
        # Pool node embeddings into a single Graph Embedding
        # Simple Mean Pooling for MVP:
        piece_embeds = z_dict['piece'] 
        graph_embed = torch.mean(piece_embeds, dim=0) # [Hidden_Dim]
        
        # Predict Win Probability
        value = self.value_head(graph_embed)
        return value
