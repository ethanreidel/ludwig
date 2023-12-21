# import importlib.util
import logging

# from ludwig.api import LudwigModel
from ludwig.api_annotations import DeveloperAPI

# from ludwig.constants import NAME
from ludwig.model_export.onnx_exporter import OnnxExporter

# import os
# import tempfile
# from typing import Any, Dict, List


# from ludwig.types import ModelConfigDict
# from ludwig.utils.fs_utils import open_file

# import torch


logger = logging.getLogger(__name__)


@DeveloperAPI
def export_onnx(model_path: str, export_path: str, model_name="ludwig_model"):
    onnx_exporter = OnnxExporter()
    # model = onnx_exporter.load_model(model_path)

    # input_names = _get_input_spec(model)
    # output_names = _get_output_spec(model)

    onnx_exporter.export(model_path, export_path, model_name)
