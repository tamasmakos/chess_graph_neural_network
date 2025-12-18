import torch
import torch.nn as nn
from chessgnn.model import RayAlignmentBlock

def test_ray_alignment_transmissibility():
    """
    Verifies that the transmissibility (t_val) correctly modulates the ray element messages.
    """
    in_channels = 16
    block = RayAlignmentBlock(in_channels)
    
    # Mock data
    # 2 pieces: Source (0) -> Target (1)
    # x_dict['piece'] has 2 nodes
    x = torch.randn(2, in_channels)
    x_dict = {'piece': x}
    
    # Edge index: 0 -> 1
    edge_index_ray = torch.tensor([[0], [1]], dtype=torch.long)
    
    # Edge attr: [dist=1, blocking=0]
    edge_attr_ray = torch.tensor([[1.0, 0.0]], dtype=torch.float)
    
    # Case 1: Force transmissibility to 0
    # specific weight/bias manipulation to ensure sigmoid output is near 0
    block.transmissibility_net.weight.data.fill_(0.0)
    block.transmissibility_net.bias.data.fill_(-20.0) # sigmoid(-20) ~ 0
    
    out_dict_0 = block(x_dict.copy(), edge_index_ray, edge_attr_ray)
    
    # Expectation: No change in x (or very small) because message was multiplied by ~0
    # RayAlignmentBlock adds residual: x_new = x + msg
    # msg should be 0
    diff_0 = out_dict_0['piece'] - x
    print(f"Diff with t_val~0 (Should be 0): {diff_0.abs().sum().item()}")
    
    # Case 2: Force transmissibility to 1
    block.transmissibility_net.bias.data.fill_(10.0) # sigmoid(10) ~ 1
    
    out_dict_1 = block(x_dict.copy(), edge_index_ray, edge_attr_ray)
    
    diff_1 = out_dict_1['piece'] - x
    print(f"Diff with t_val~1 (Should be > 0): {diff_1.abs().sum().item()}")
    
    assert diff_0.abs().sum().item() < 1e-4, "Transmissibility 0 failed to block message"
    assert diff_1.abs().sum().item() > 1e-4, "Transmissibility 1 failed to send message"
    
    print("Test Passed!")

if __name__ == "__main__":
    test_ray_alignment_transmissibility()
