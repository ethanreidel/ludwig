from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import ast

import torch
import torch.utils.checkpoint
from torch import nn

from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ludwig.models.base import BaseModel
from ludwig.schema.model_types.lmm import LMMModelConfig
from ludwig.utils.lmm_utils import load_pretrained_from_config
from ludwig.constants import MODEL_LMM

class LMM(BaseModel):
    @staticmethod
    def type() -> str:
        return MODEL_LMM
    
    def __init__(
            self,
            config_obj: LMMModelConfig,
            random_seed=None,
            _device=None,
            **_kwargs,
    ):
        super().__init__(random_seed=random_seed)

        self.config_obj = config_obj
        self._random_seed = random_seed

        self.model_name = self.config_obj.base_model
        self.vision_tower_config = AutoConfig.from_pretrained(self.config_obj.vision_tower)
        self.llm_config = AutoConfig.from_pretrained(self.config_obj.language_model)
        self.projector_config = AutoConfig.from_pretrained(self.config_obj.projector)

        #call load model from here?
        