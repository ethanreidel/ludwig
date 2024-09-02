import torch.nn as nn
from ludwig.schema.model_types.lmm import LMMModelConfig

class LinearConnector(nn.Module):
    def __init__(self, config: LMMModelConfig):
        super().__init__()
        self.connector = nn.Linear(config.vision_hidden_size, config.hidden_size)