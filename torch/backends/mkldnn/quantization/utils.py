from torch.quantization import qconfig
import torch
from torch import nn
from enum import Enum

# Customized config for MKLDNN backend.
# Note: This is not used now since the PyTorch API is not ready
mkldnn_backend_config_dict = {
    "name": "MKLDNN",
    "operator": {
    }
}

# define a function to return the backend config dict
def get_mkldnn_backend_config_dict():
    return mkldnn_backend_config_dict
