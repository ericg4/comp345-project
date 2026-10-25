#!/usr/bin/env python3
"""Train SmolVLM LoRA adapters on the project's shared nevus/melanoma split.

    uv sync --extra vlm
    uv run python scripts/train_smolvlm.py --epochs 5

The original vocabulary head stays frozen: A means nevus, B means melanoma.
Select the epoch by validation average precision, then evaluate test once at 0.5.
Results go to results/; model files and their inference config go to artifacts/.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import ARTIFACTS_DIR, CLASSES, HAM_DIR, RESULTS_DIR, SPLIT_CSV, load_split, metrics

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
PROMPT_TEXT = "Classify this skin lesion. A: nevus. B: melanoma. Answer only A or B."


# Images and model inputs
class Images(Dataset):
    def __init__(self, table):
        self.table = table.reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        with Image.open(row.path) as image:
            return image.convert("RGB"), int(row.label)


def make_loaders(split, processor, prompt, batch_size):
    def collate(examples):
        images, labels = zip(*examples)
        inputs = processor(text=[prompt] * len(images), images=[[im] for im in images],
                           padding=True, return_tensors="pt")
        return inputs, torch.tensor(labels)

    return {
        name: DataLoader(Images(split[split.split == name]), batch_size=batch_size,
                         shuffle=(name == "train"), collate_fn=collate)
        for name in ("train", "val", "test")
    }


# The pretrained output head is retained; only LoRA parameters train.
def load_model(rank, image_size):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": image_size}
    processor.image_processor.max_image_size = {"longest_edge": image_size}
    prompt = processor.apply_chat_template([
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_TEXT}]},
    ], add_generation_prompt=True)
    answer_ids = [processor.tokenizer.encode(a, add_special_tokens=False) for a in ("A", "B")]
    assert all(len(ids) == 1 for ids in answer_ids) and answer_ids[0] != answer_ids[1]

    model = Idefics3ForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    targets = [name for name, layer in model.named_modules()
               if name.startswith("model.text_model.") and name.endswith(("q_proj", "v_proj"))
               and isinstance(layer, torch.nn.Linear)]
    assert targets, "No language attention projections found."
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=targets, bias="none",
    ))
    model.get_base_model().model.text_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    model.print_trainable_parameters()
    return model, processor, prompt, [ids[0] for ids in answer_ids]


def next_token_logits(model, inputs, device):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # Left padding puts the final prompt token last for every example.
    position_ids = (inputs["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
    return model(**inputs, position_ids=position_ids, use_cache=False,
                 logits_to_keep=1).logits[:, -1].float()


# Training and evaluation

def train_epoch(model, loader, optimizer, scaler, answer_ids, weights, accumulation, device, dtype):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for step, (inputs, labels) in enumerate(tqdm(loader, desc="Train")):
        labels = labels.to(device)
        window_start = (step // accumulation) * accumulation
        # Count examples, including a final partial batch/accumulation window.
        window_examples = min(accumulation * loader.batch_size,
                              len(loader.dataset) - window_start * loader.batch_size)
        with torch.autocast(device.type, dtype=dtype):
            logits = next_token_logits(model, inputs, device)
            losses = F.cross_entropy(logits, answer_ids[labels], reduction="none") * weights[labels]
        scaler.scale(losses.sum() / window_examples).backward()
        total_loss += losses.detach().sum().item()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def predict(model, loader, answer_ids, device, dtype):
    model.eval()
    batches = []
    for inputs, _ in tqdm(loader, desc="Evaluate"):
        with torch.autocast(device.type, dtype=dtype):
            logits = next_token_logits(model, inputs, device)
        batches.append(logits[:, answer_ids].softmax(-1).cpu().numpy())
    return np.concatenate(batches)


def evaluate(labels, probs):
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score

    result = metrics(labels, probs)  # Same metrics/schema as the other experiment arms.
    result.update({
        "accuracy": float(accuracy_score(labels, probs[:, 1] >= 0.5)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, probs[:, 1] >= 0.5)),
        "pr_auc_average_precision": float(average_precision_score(labels, probs[:, 1])),
    })
    tn, fp = result["confusion_matrix"][0]
    fn, tp = result["confusion_matrix"][1]
    result["melanoma_recall"] = tp / (tp + fn)
    result["nevus_specificity"] = tn / (tn + fp)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=HAM_DIR)
    parser.add_argument("--split-csv", type=Path, default=SPLIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "smolvlm_lora")
    parser.add_argument("--artifacts-dir", type=Path, help="Model files; defaults to artifacts/<output-dir name>")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.artifacts_dir is None:
        args.artifacts_dir = ARTIFACTS_DIR / args.output_dir.name
    if min(args.epochs, args.batch_size, args.accumulation, args.lora_rank) < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, accumulation, rank, and learning rate must be positive.")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Training requires a CUDA GPU. Use a Colab GPU runtime or another CUDA machine.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    split = load_split(args.data_dir, args.split_csv)
    assert split.image_id.is_unique and set(split.split) == {"train", "val", "test"}
    assert (split.groupby("lesion_id").split.nunique() == 1).all(), "Lesion in multiple splits."
    assert (split.label == split.dx.map({name: i for i, name in enumerate(CLASSES)})).all()
    assert all(set(split[split.split == name].label) == {0, 1} for name in ("train", "val", "test"))
    # Protect both saved results and model files from earlier runs.
    for folder in (args.output_dir, args.artifacts_dir):
        if folder.exists():
            raise FileExistsError(f"Run folder already exists: {folder}")
    if args.output_dir.resolve() == args.artifacts_dir.resolve():
        raise ValueError("Results and artifacts must use separate folders.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.artifacts_dir.mkdir(parents=True, exist_ok=False)
    split.drop(columns="path").to_csv(args.output_dir / "split.csv", index=False)

    model, processor, prompt, token_ids = load_model(args.lora_rank, image_size=512)
    model = model.to(device)
    answer_ids = torch.tensor(token_ids, device=device)
    loaders = make_loaders(split, processor, prompt, args.batch_size)
    counts = np.bincount(split[split.split == "train"].label, minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    processor.save_pretrained(args.artifacts_dir / "processor")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(model_id=MODEL_ID, image_size=512, prompt=prompt, answer_token_ids=token_ids,
                  classes=CLASSES, threshold=0.5, selection_metric="pr_auc_average_precision",
                  split_sha256=hashlib.sha256(args.split_csv.read_bytes()).hexdigest())
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.artifacts_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, scaler, answer_ids,
                           weights, args.accumulation, device, dtype)
        val_probs = predict(model, loaders["val"], answer_ids, device, dtype)
        record = {"epoch": epoch, "train_loss": loss,
                  **evaluate(split[split.split == "val"].label.to_numpy(), val_probs)}
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if best is None or record["pr_auc_average_precision"] > best["pr_auc_average_precision"]:
            best = record
            model.save_pretrained(args.artifacts_dir / "best_adapter")
            (args.output_dir / "best_validation.json").write_text(json.dumps(best, indent=2) + "\n")

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(str(args.artifacts_dir / "best_adapter" / "adapter_model.safetensors")))
    test_probs = predict(model, loaders["test"], answer_ids, device, dtype)
    test_rows = split[split.split == "test"]
    test = evaluate(test_rows.label.to_numpy(), test_probs)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test, indent=2) + "\n")
    test_rows[["image_id", "lesion_id", "label"]].assign(
        prob_mel=test_probs[:, 1], pred=(test_probs[:, 1] >= 0.5).astype(int),
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(f"Best epoch: {best['epoch']}; test: {json.dumps(test)}")


if __name__ == "__main__":
    main()
