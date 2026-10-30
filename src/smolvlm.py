"""Fine-tune SmolVLM-500M with LoRA on nevus vs. melanoma, or predict with saved adapters.

    uv sync --extra vlm
    uv run python src/smolvlm.py train --epochs 5          # CUDA GPU required
    uv run python src/smolvlm.py predict data/smolvlm_lora image.jpg

The prompt asks for "A" (nevus) or "B" (melanoma), and the score is the softmax over
those two tokens from the frozen vocabulary head; only LoRA adapters on the language
model's q/v projections train. The epoch is chosen by validation average precision;
test is scored once at threshold 0.5. Results go to results/smolvlm_lora/, model
files to data/smolvlm_lora/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import CLASSES, DATA_DIR, RESULTS_DIR, load_split, metrics, pick_device

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
PROMPT_TEXT = "Classify this skin lesion. A: nevus. B: melanoma. Answer only A or B."
IMAGE_SIZE = 512


def load_model(rank):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": IMAGE_SIZE}
    processor.image_processor.max_image_size = {"longest_edge": IMAGE_SIZE}
    prompt = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_TEXT}]},
    ], add_generation_prompt=True)
    answer_ids = [processor.tokenizer.encode(a, add_special_tokens=False)[0] for a in ("A", "B")]

    model = Idefics3ForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                                             attn_implementation="sdpa")
    model.config.use_cache = False
    targets = [name for name, layer in model.named_modules()
               if name.startswith("model.text_model.") and name.endswith(("q_proj", "v_proj"))
               and isinstance(layer, torch.nn.Linear)]
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
                                             target_modules=targets, bias="none"))
    model.get_base_model().model.text_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    return model, processor, prompt, answer_ids


def next_token_logits(model, inputs, device):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # Left padding puts the final prompt token last for every example.
    position_ids = (inputs["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
    return model(**inputs, position_ids=position_ids, use_cache=False, logits_to_keep=1).logits[:, -1].float()


def make_loader(rows, processor, prompt, batch_size, shuffle=False):
    def collate(batch):
        images = [[Image.open(path).convert("RGB")] for path, _ in batch]
        inputs = processor(text=[prompt] * len(batch), images=images, padding=True, return_tensors="pt")
        return inputs, torch.tensor([label for _, label in batch])

    return DataLoader(list(zip(rows["path"], rows["label"])), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate)


@torch.inference_mode()
def predict_probs(model, loader, answer_ids, device, dtype):
    model.eval()
    probs = []
    for inputs, _ in tqdm(loader, desc="Evaluate"):
        with torch.autocast(device.type, dtype=dtype):
            logits = next_token_logits(model, inputs, device)
        probs.append(logits[:, answer_ids].softmax(-1).cpu().numpy())
    return np.concatenate(probs)


def evaluate(labels, probs):
    result = metrics(labels, probs)
    result["accuracy"] = float(accuracy_score(labels, probs[:, 1] >= 0.5))
    result["average_precision"] = float(average_precision_score(labels, probs[:, 1]))
    return result


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    results, model_dir = RESULTS_DIR / "smolvlm_lora", DATA_DIR / "smolvlm_lora"
    results.mkdir(parents=True, exist_ok=True)

    split = load_split()
    model, processor, prompt, token_ids = load_model(args.lora_rank)
    model = model.to(device)
    answer_ids = torch.tensor(token_ids, device=device)
    loaders = {name: make_loader(split[split.split == name], processor, prompt, args.batch_size, name == "train")
               for name in ("train", "val", "test")}
    counts = np.bincount(split[split.split == "train"].label, minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    config = {**vars(args), "model_id": MODEL_ID, "prompt": prompt, "answer_token_ids": token_ids}
    config.pop("func")
    processor.save_pretrained(model_dir / "processor")
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))
    (results / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for step, (inputs, labels) in enumerate(tqdm(loaders["train"], desc=f"Epoch {epoch}")):
            labels = labels.to(device)
            with torch.autocast("cuda", dtype=dtype):
                logits = next_token_logits(model, inputs, device)
                losses = F.cross_entropy(logits, answer_ids[labels], reduction="none") * weights[labels]
            scaler.scale(losses.mean() / args.accumulation).backward()
            total += losses.detach().sum().item()
            if (step + 1) % args.accumulation == 0 or step + 1 == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        val = evaluate(split[split.split == "val"].label.to_numpy(),
                       predict_probs(model, loaders["val"], answer_ids, device, dtype))
        history.append({"epoch": epoch, "train_loss": total / len(loaders["train"].dataset), **val})
        print(json.dumps(history[-1]))
        if best is None or val["average_precision"] > best["average_precision"]:
            best = history[-1]
            model.save_pretrained(model_dir / "best_adapter")
    (results / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(str(model_dir / "best_adapter" / "adapter_model.safetensors")))
    test_rows = split[split.split == "test"]
    test_probs = predict_probs(model, loaders["test"], answer_ids, device, dtype)
    test = evaluate(test_rows.label.to_numpy(), test_probs)
    (results / "test_metrics.json").write_text(json.dumps({"best_epoch": best["epoch"], **test}, indent=2) + "\n")
    test_rows[["image_id", "lesion_id", "label"]].assign(
        prob_mel=test_probs[:, 1], pred=(test_probs[:, 1] >= 0.5).astype(int),
    ).to_csv(results / "test_predictions.csv", index=False)
    print(f"Best epoch {best['epoch']}; test: {json.dumps(test)}")


@torch.inference_mode()
def predict(args):
    from peft import PeftModel
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    config = json.loads((args.model_dir / "config.json").read_text())
    device = torch.device(pick_device())
    processor = AutoProcessor.from_pretrained(args.model_dir / "processor")
    base = Idefics3ForConditionalGeneration.from_pretrained(config["model_id"], torch_dtype=torch.float32,
                                                            attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, args.model_dir / "best_adapter").to(device).eval()
    for path in args.images:
        inputs = processor(text=config["prompt"], images=[Image.open(path).convert("RGB")], return_tensors="pt")
        score = next_token_logits(model, inputs, device)[:, config["answer_token_ids"]].softmax(-1)[0, 1].item()
        print(f"{path}: {CLASSES[int(score >= 0.5)]} (melanoma score {score:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--accumulation", type=int, default=2)
    train_parser.add_argument("--lora-rank", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("model_dir", type=Path, help="Folder with config.json, processor/, best_adapter/")
    predict_parser.add_argument("images", nargs="+", type=Path)
    predict_parser.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
