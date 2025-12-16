FROM nvcr.io/nvidia/pytorch:25.09-py3

# Install PyTorch for CUDA 11.8 (Compatible with host driver 12.0)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and dependencies
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
