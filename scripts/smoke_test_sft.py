"""Smoke test: load Qwen2.5-VL + LoRA, tokenise one episode, compute loss."""

import sys
import torch

from vlc.model.qwen_vl import load_model_and_processor, add_lora, LoRAConfig
from vlc.episodes.shapes3d import Shapes3DEpisodeBuilder
from vlc.model.trainer import collate_episodes

print("Loading Qwen2.5-VL-3B-Instruct ...")
model, processor = load_model_and_processor(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    min_pixels=64 * 28 * 28,
    max_pixels=128 * 28 * 28,
)
print("  Model loaded.")

print("Adding LoRA (r=16) ...")
lora_cfg = LoRAConfig(r=16, lora_alpha=32)
model = add_lora(model, lora_cfg)
model.eval()
print("  LoRA added.")

print("\nBuilding 3DShapes episode (8 imgs, 2 steps) ...")
builder = Shapes3DEpisodeBuilder(
    "data/3dshapes.h5", n_images=8, batch_size=4, seed=42
)
ep = builder.build_episode("shape")
print(f"  Episode: K={ep.k}, {ep.total_images} imgs, {len(ep.steps)} steps, "
      f"criterion='{ep.criterion}'")

print("\nCollating episode to model inputs ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batches = collate_episodes([ep], processor, max_length=3000, device=device)
print(f"  Batches: {len(batches)}")

if not batches:
    print("ERROR: collation returned empty batch list. Check logs above.")
    sys.exit(1)

b = batches[0]
iids = b["input_ids"]
labs = b["labels"]
n_answer = (labs != -100).sum().item()
print(f"  input_ids shape:  {iids.shape}")
print(f"  labels non-ignore (answer tokens): {n_answer}")
if "pixel_values" in b and b["pixel_values"] is not None:
    print(f"  pixel_values shape: {b['pixel_values'].shape}")
else:
    print("  pixel_values: None (text-only path)")

print("\nComputing forward pass + loss ...")
with torch.no_grad():
    out = model(
        input_ids=b["input_ids"],
        attention_mask=b.get("attention_mask"),
        pixel_values=b.get("pixel_values"),
        image_grid_thw=b.get("image_grid_thw"),
        labels=b["labels"],
    )
print(f"  Loss: {out.loss.item():.4f}")
print("\nSmoke test PASSED.")
