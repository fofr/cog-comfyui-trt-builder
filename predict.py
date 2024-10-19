import json
import os
import tarfile
import math
import subprocess
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

        # print("Server started")
        # gpu_name = (
        #     os.popen("nvidia-smi --query-gpu=name --format=csv,noheader,nounits")
        #     .read()
        #     .strip()
        # )
        # print(f"GPU: {gpu_name}")

    def as_multiple_of_8(self, value):
        return value if value % 8 == 0 else value + 8 - (value % 8)

    def update_workflow(self, workflow, **kwargs):
        checkpoint_loader = workflow["1"]["inputs"]
        checkpoint_loader["ckpt_name"] = kwargs["checkpoint"]

        checkpoint_filename = os.path.splitext(kwargs["checkpoint"])[0]

        builder = workflow["3"]["inputs"]
        builder["filename_prefix"] = f"{checkpoint_filename}_DYN"
        builder["batch_size_min"] = kwargs["batch_size_min"]
        builder["batch_size_opt"] = kwargs["batch_size_opt"]
        builder["batch_size_max"] = kwargs["batch_size_max"]
        builder["height_min"] = self.as_multiple_of_8(kwargs["height_min"])
        builder["height_opt"] = self.as_multiple_of_8(kwargs["height_opt"])
        builder["height_max"] = self.as_multiple_of_8(kwargs["height_max"])
        builder["width_min"] = self.as_multiple_of_8(kwargs["width_min"])
        builder["width_opt"] = self.as_multiple_of_8(kwargs["width_opt"])
        builder["width_max"] = self.as_multiple_of_8(kwargs["width_max"])
        builder["context_min"] = kwargs["context_min"]
        builder["context_opt"] = kwargs["context_opt"]
        builder["context_max"] = kwargs["context_max"]

    def predict(
        self,
        checkpoint: str = Input(
            default="sd3_medium.safetensors",
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

        if checkpoint.startswith("http"):
            local_checkpoint = os.path.join(
                "ComfyUI", "models", "checkpoints", os.path.basename(checkpoint)
            )
            if not os.path.exists(local_checkpoint):
                print(f"Downloading checkpoint from {checkpoint}")
                print(f"Local checkpoint path: {local_checkpoint}")
                subprocess.run(["pget", checkpoint, local_checkpoint], check=True)
                print(f"Downloaded checkpoint to {os.path.basename(local_checkpoint)}")
            else:
                print(f"Checkpoint already exists at {local_checkpoint}")
            checkpoint = os.path.basename(local_checkpoint)

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

        output_tar = os.path.join(OUTPUT_DIR, "output.tar")

        with tarfile.open(output_tar, "w") as tar:
            for file in os.listdir(OUTPUT_DIR):
                file_path = os.path.join(OUTPUT_DIR, file)
                if os.path.isfile(file_path) and file.endswith(".engine"):
                    tar.add(file_path, arcname=file)

        # Split the tar file into 500MB chunks
        chunk_size = 500 * 1024 * 1024  # 500MB in bytes
        file_size = os.path.getsize(output_tar)
        num_chunks = math.ceil(file_size / chunk_size)

        chunk_files = []
        with open(output_tar, "rb") as f:
            for i in range(num_chunks):
                chunk_name = f"{output_tar}.part{i+1}.tar"
                with open(chunk_name, "wb") as chunk:
                    chunk.write(f.read(chunk_size))
                chunk_files.append(Path(chunk_name))

        # Remove the original tar file
        # os.remove(output_tar)

        return chunk_files
