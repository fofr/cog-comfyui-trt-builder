import json
import os
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

api_json_file = "workflow_api.json"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        checkpoint_loader = workflow["4"]["inputs"]
        checkpoint_loader["ckpt_name"] = kwargs["checkpoint"]

        # Get checkpoint filename without extension
        checkpoint_filename = os.path.splitext(kwargs["checkpoint"])[0]

        builder = workflow["3"]["inputs"]
        builder["filename_prefix"] = f"{checkpoint_filename}_DYN"
        builder["batch_size_min"] = kwargs["batch_size_min"]
        builder["batch_size_opt"] = kwargs["batch_size_opt"]
        builder["batch_size_max"] = kwargs["batch_size_max"]
        builder["height_min"] = kwargs["height_min"]
        builder["height_opt"] = kwargs["height_opt"]
        builder["height_max"] = kwargs["height_max"]
        builder["width_min"] = kwargs["width_min"]
        builder["width_opt"] = kwargs["width_opt"]
        builder["width_max"] = kwargs["width_max"]
        builder["context_min"] = kwargs["context_min"]
        builder["context_opt"] = kwargs["context_opt"]
        builder["context_max"] = kwargs["context_max"]
        builder["filename_prefix"] = kwargs["checkpoint"]

    def predict(
        self,
        checkpoint: str = Input(
            default="",
            description="The checkpoint to use (must be in https://github.com/fofr/cog-comfyui/blob/main/weights.json)",
        ),
        batch_size_min: int = Input(
            default=1,
            ge=1,
            le=100,
            description="The minimum batch size during inference",
        ),
        batch_size_opt: int = Input(
            default=1,
            ge=1,
            le=100,
            description="The optimal batch size during inference",
        ),
        batch_size_max: int = Input(
            default=1,
            ge=1,
            le=100,
            description="The maximum batch size during inference",
        ),
        height_min: int = Input(
            default=512,
            ge=256,
            le=4096,
            description="The minimum height during inference",
        ),
        height_opt: int = Input(
            default=1024,
            ge=256,
            le=4096,
            description="The optimal height during inference",
        ),
        height_max: int = Input(
            default=1536,
            ge=256,
            le=4096,
            description="The maximum height during inference",
        ),
        width_min: int = Input(
            default=512,
            ge=256,
            le=4096,
            description="The minimum width during inference",
        ),
        width_opt: int = Input(
            default=1024,
            ge=256,
            le=4096,
            description="The optimal width during inference",
        ),
        width_max: int = Input(
            default=1536,
            ge=256,
            le=4096,
            description="The maximum width during inference",
        ),
        context_min: int = Input(
            default=1,
            ge=1,
            le=128,
            description="The minimum context during inference",
        ),
        context_opt: int = Input(
            default=1,
            ge=1,
            le=128,
            description="The optimal context during inference",
        ),
        context_max: int = Input(
            default=1,
            ge=1,
            le=128,
            description="The maximum context during inference",
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            checkpoint=checkpoint,
            batch_size_min=batch_size_min,
            batch_size_opt=batch_size_opt,
            batch_size_max=batch_size_max,
            height_min=height_min,
            height_opt=height_opt,
            height_max=height_max,
            width_min=width_min,
            width_opt=width_opt,
            width_max=width_max,
            context_min=context_min,
            context_opt=context_opt,
            context_max=context_max,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR)
