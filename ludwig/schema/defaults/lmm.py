from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import IMAGE, TEXT
from ludwig.schema import utils as schema_utils
from ludwig.schema.defaults.base import BaseDefaultsConfig
from ludwig.schema.defaults.utils import DefaultsDataclassField
from ludwig.schema.features.base import BaseFeatureConfig
from ludwig.schema.features.utils import lmm_defaults_config_registry
from ludwig.schema.utils import ludwig_dataclass


@DeveloperAPI
@ludwig_dataclass
class LMMDefaultsConfig(BaseDefaultsConfig):
    text: BaseFeatureConfig = DefaultsDataclassField(feature_type=TEXT, defaults_registry=lmm_defaults_config_registry)
    image: BaseFeatureConfig = DefaultsDataclassField(
        feature_type=IMAGE, defaults_registry=lmm_defaults_config_registry
    )


@DeveloperAPI
class LMMDefaultsField(schema_utils.DictMarshmallowField):
    def __init__(self):
        super().__init__(LMMDefaultsConfig)
