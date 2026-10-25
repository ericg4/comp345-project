#!/usr/bin/env python3
"""Predict images with saved SmolVLM adapters, including the original Colab run.

    uv run python scripts/predict_smolvlm.py artifacts/ham10000_lora image.jpg

Scores are conditional on the A/B answer choices, not calibrated disease probabilities.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

from common import CLASSES
from train_smolvlm import next_token_logits


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Run folder containing config.json, processor/, best_adapter/")
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    processor = AutoProcessor.from_pretrained(args.checkpoint / "processor")
    base_model = Idefics3ForConditionalGeneration.from_pretrained(
        config["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint / "best_adapter").to(device).eval()
    answer_ids = torch.tensor(config["answer_token_ids"], device=device)
    for path in args.images:
        with Image.open(path) as image:
            inputs = processor(text=config["prompt"], images=[image.convert("RGB")], return_tensors="pt")
        context = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
        with context:
            logits = next_token_logits(model, inputs, device)
        score = logits[:, answer_ids].softmax(-1)[0, 1].item()
        print(json.dumps({"image": str(path), "prediction": CLASSES[int(score >= config.get("threshold", 0.5))],
                          "conditional_melanoma_score": score}))


if __name__ == "__main__":
    main()
