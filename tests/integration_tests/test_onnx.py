# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

import numpy as np
import onnx
import onnxruntime as ort

# import pandas as pd
import pytest

from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, TRAINER

# from ludwig.models.inference import to_inference_module_input_from_dataframe
from ludwig.utils.onnx_utils import export_onnx

# from tests.integration_tests import utils
from tests.integration_tests.utils import binary_feature, generate_data, image_feature, LocalTestBackend


def initialize_onnx(
    tmpdir, config, backend, training_data_csv_path
):  # this should train a ludwig model, and convert it to onnx and return both
    # this function should return both trained ludwig model and onnx model
    # create and train ludwig model based off config
    ludwig_model = LudwigModel(config, backend=backend)
    ludwig_model.train(
        dataset=training_data_csv_path,
        skip_save_training_description=True,
        skip_save_training_statistics=True,
        skip_save_model=True,
        skip_save_progress=True,
        skip_save_log=True,
        skip_save_processed_input=True,
    )

    ludwig_model_path = os.path.join(tmpdir, "ludwig_model")
    onnx_model_path = os.path.join(tmpdir, "export_results")
    model_name = "model.onnx"

    os.makedirs(str(onnx_model_path), exist_ok=True)
    ludwig_model.save(ludwig_model_path)
    export_onnx(ludwig_model_path, onnx_model_path, model_name=model_name)
    onnx_model = os.path.join(onnx_model_path, model_name)
    trained_ludwig_model = LudwigModel.load(ludwig_model_path)

    return trained_ludwig_model, onnx_model


def validate_onnx(tmpdir, config, backend, training_data_csv_path, tolerance=1e-8):
    ludwig_model, onnx_model = initialize_onnx(tmpdir, config, backend, training_data_csv_path)

    onnx.checker.check_model(onnx_model, full_check=True)
    preds_dict, _ = ludwig_model.predict(dataset=training_data_csv_path, return_type=dict)
    onnx_session = ort.InferenceSession(onnx_model)

    input_name = onnx_session.get_inputs()[0].name
    input_shape = onnx_session.get_inputs()[0].shape  # e.g. [128, 3, 12, 12]
    x = np.random.random(input_shape).astype(np.float32)
    output_name = onnx_session.get_outputs()[0].name  # prints binary_#####
    # df = pd.read_csv(training_data_csv_path)

    output = onnx_session.run([output_name], {input_name: x})

    assert preds_dict is not None
    assert output is not None
    # TODO Compare ONNX model with Ludwig Model


@pytest.mark.parametrize(
    "kwargs",
    [
        {"encoder": {"type": "stacked_cnn"}},
    ],
)
def test_onnx_e2e_image(csv_filename, tmpdir, kwargs):
    data_csv_path = os.path.join(tmpdir, csv_filename)
    image_dest_folder = os.path.join(tmpdir, "generated_images")
    input_features = [image_feature(image_dest_folder, **kwargs)]
    output_features = [binary_feature()]
    backend = LocalTestBackend()
    config = {
        "input_features": input_features,
        "output_features": output_features,
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    training_data_csv_path = generate_data(input_features, output_features, data_csv_path)

    validate_onnx(tmpdir, config, backend, training_data_csv_path)
