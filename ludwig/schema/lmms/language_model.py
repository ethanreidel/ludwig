from abc import ABC, abstractmethod
from typing import List, Optional, Type, TYPE_CHECKING
import os

from ludwig.api_annotations import DeveloperAPI
from ludwig.error import ConfigValidationError
from ludwig.schema import utils as schema_utils
from ludwig.schema.metadata import LMM_METADATA
from ludwig.schema.metadata.parameter_metadata import convert_metadata_to_json
from ludwig.schema.utils import ludwig_dataclass
from ludwig.utils.registry import Registry
from transformers import AutoConfig

MODEL_PRESETS = {
    ""
}

#build this out later

@DeveloperAPI
def BaseLMMLanguageModelDataclassField():
    description = (
        "placeholder: base language model to use for LMM"
    )
    def validate(model_name: str):
        """Validates and upgrades the given model name to its full path, if applicable.

        If the name exists in `MODEL_PRESETS`, returns the corresponding value from the dict; otherwise checks if the
        given name (which should be a full path) exists locally or in the transformers library.
        """
        if isinstance(model_name, str):
            if model_name in MODEL_PRESETS:
                return MODEL_PRESETS[model_name]
            if os.path.isdir(model_name):
                return model_name
            try:
                AutoConfig.from_pretrained(model_name)
                return model_name
            except OSError:
                raise ConfigValidationError(
                    f"Specified base model `{model_name}` could not be loaded. If this is a private repository, make "
                    f"sure to set HUGGING_FACE_HUB_TOKEN in your environment. Check that {model_name} is a valid "
                    "pretrained CausalLM listed on huggingface or a valid local directory containing the weights for a "
                    "pretrained CausalLM from huggingface. See: "
                    "https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for a full list."
                )
        raise ValidationError(
            f"`base_model` should be a string, instead given: {model_name}. This can be a preset or any pretrained "
            "CausalLM on huggingface. See: https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads"
        )