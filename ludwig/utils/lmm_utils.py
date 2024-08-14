import logging
import copy
import tempfile
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union
from packaging import version

import transformers
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig, BitsAndBytesConfig

if TYPE_CHECKING:
    from ludwig.schema.model_types.lmm import LMMModelConfig


logger = logging.getLogger(__name__)

transformers_436 = version.parse(transformers.__version__) >= version.parse("4.36.0")

def load_pretrained_from_config(
        config_obj: LMMModelConfig,
        model_config: Optional[AutoConfig] = None,
        weights_save_path: Optional[str] = None,
) -> PreTrainedModel:
    load_kwargs = {}

    #deal with quantization later here:

    
    





    return

def to_device(
        model: PreTrainedModel,
        device: Union[str, torch.DeviceObjType],
        config_obj: "LMMModelConfig", # noqa F821,
        curr_device: torch.DeviceObjType
) -> Tuple[PreTrainedModel, torch.DeviceObjType]:
    #add 

    return