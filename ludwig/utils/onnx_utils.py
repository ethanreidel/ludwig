# import importlib.util
import logging

from ludwig.api import LudwigModel
from ludwig.api_annotations import DeveloperAPI
from ludwig.constants import NAME
from ludwig.model_export.onnx_exporter import OnnxExporter

logger = logging.getLogger(__name__)


def get_input_names(model: LudwigModel):
    input_names = []
    for feature in model.config["input_features"]:
        name = feature[NAME]
        input_names.append(str(name))
    return input_names


def get_output_names(model: LudwigModel):
    output_names = []
    for feature in model.config["output_features"]:
        name = feature[NAME]
        output_names.append(str(name))
    return output_names


@DeveloperAPI
def export_onnx(model_path: str, export_path: str, model_name):
    onnx_exporter = OnnxExporter()
    model = LudwigModel.load(model_path)
    onnx_exporter.export(
        model_path,
        export_path,
        model_name,
        config_input_names=get_input_names(model),
        config_output_names=get_output_names(model),
        quantize=False,
    )
