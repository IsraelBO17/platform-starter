# scripts/generate_image.py
"""Generate an image from a prompt with one of three backends.

  python scripts/generate_image.py hf      "..."   # Hugging Face Inference Providers (free monthly credit)
  python scripts/generate_image.py local   "..."   # SD-Turbo on your own machine (free, slower on CPU)
  python scripts/generate_image.py bedrock "..."   # Amazon Nova Canvas (uses the class AWS account)
"""
import base64
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # HF_TOKEN, AWS_PROFILE, AWS_REGION from .env

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def with_hugging_face(prompt: str) -> Path:
    from huggingface_hub import InferenceClient

    client = InferenceClient(api_key=os.environ["HF_TOKEN"])
    image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-schnell")
    path = OUT / "hf.png"
    image.save(path)
    return path


def with_local_model(prompt: str) -> Path:
    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo").to(device)
    # SD-Turbo is distilled to produce an image in a single denoising step.
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    path = OUT / "local.png"
    image.save(path)
    return path


def with_bedrock(prompt: str) -> Path:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt},
        "imageGenerationConfig": {"numberOfImages": 1, "width": 512, "height": 512, "seed": 7},
    }
    response = client.invoke_model(modelId="amazon.nova-canvas-v1:0", body=json.dumps(body))
    payload = json.loads(response["body"].read())
    path = OUT / "bedrock.png"
    path.write_bytes(base64.b64decode(payload["images"][0]))
    return path


if __name__ == "__main__":
    backend, prompt = sys.argv[1], sys.argv[2]
    make = {"hf": with_hugging_face, "local": with_local_model, "bedrock": with_bedrock}[backend]
    print("Saved", make(prompt))
