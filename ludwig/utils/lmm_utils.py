import logging
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union
from packaging import version

import transformers
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoProcessor, AutoConfig, BitsAndBytesConfig, LlavaForConditionalGeneration

if TYPE_CHECKING:
    from ludwig.schema.model_types.lmm import LMMModelConfig


logger = logging.getLogger(__name__)

transformers_436 = version.parse(transformers.__version__) >= version.parse("4.36.0")


#parameters listed here for llava training
#https://github.com/TinyLLaVA/TinyLLaVA_Factory/blob/main/tinyllava/model/configuration_tinyllava.py

#for now, we just want to have 1 default for each component


def load_pretrained_from_config(
        config_obj: LMMModelConfig,
        model_config: Optional[AutoConfig] = None,
        weights_save_path: Optional[str] = None,
) -> PreTrainedModel:
    load_kwargs = {}

    #placeholder until lmms/language_model,vision_tower,projector built out
    #deal with defaults?
    if config_obj.vision_tower:
        load_kwargs["vision_tower"] = config_obj.vision_tower
    if config_obj.projector:
        load_kwargs["projector"] = config_obj.projector
    if config_obj.language_model:
        load_kwargs["language_model"] = config_obj.language_model

    #LLM(language_model)
    #VisionTower(vision_tower)
    #connector(projector)




    #deal with quantization later here:

    pretrained_model_name_or_path = weights_save_path or config_obj.base_model
    #TODO -> add load kwargs for llava (quantization etc)
    #figure out how to load model with modular components (llm, vision tower projector)
    model: PreTrainedModel = LlavaForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
    return model

def to_device(
        model: PreTrainedModel,
        device: Union[str, torch.DeviceObjType],
        config_obj: "LMMModelConfig", # noqa F821,
        curr_device: torch.DeviceObjType
) -> Tuple[PreTrainedModel, torch.DeviceObjType]:
    #add 

    return


#load utils here

def load_vision_tower():
    pass

def load_llm():
    pass

def load_projector():
    pass