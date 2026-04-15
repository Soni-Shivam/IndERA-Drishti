"""
model_engine.py
---------------
Offline inference engine for PaliGemma-3b-mix-224.
Loads the model ONCE and keeps it in memory. Call `run_inference(image)` for each frame.
"""

import re
import os
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

MODEL_ID = "google/paligemma-3b-mix-224"
HF_TOKEN = os.getenv("HF_TOKEN")

# INVENTORY_GROUPS = ["electronics", "hardware", "packaging", "tools", "unknown", "human"]

# INVENTORY_GROUPS = [
#     "active electronics", 
#     "mechanical structural", 
#     "electro-mechanical", 
#     "unknown"
# ]

# Merged array for app.py or model_engine.py
INVENTORY_GROUPS = [
    "Integrated Circuits",    # Rectangular, black, flat
    "Cylindrical Capacitors", # Circular top, vertical body
    "Striped Resistors",      # Small, slender, distinct patterns
    "Hex Nuts",               # Small, metallic, hexagonal
    "Long Bolts",             # Slender, metallic, high aspect ratio
    "Batteries",              # Dense, cylindrical, colorful
    "Adhesive Tape",          # Large torus/donut shape
    "USB Connectors",         # Rectangular metallic housing
    "Jumper Wires",           # Irregular, colorful, thin
    "unknown"                 # Fallback for empty space or hands
]
# ---------------------------------------------------------------------------
# Module-level model state (singleton)
# ---------------------------------------------------------------------------
_processor = None
_model = None
_device = None


def load_model() -> None:
    """Load the PaliGemma model into memory. Idempotent — safe to call multiple times."""
    global _processor, _model, _device

    if _model is not None:
        print("[ModelEngine] Model already loaded — skipping.")
        return

    print(f"[ModelEngine] Loading model '{MODEL_ID}'… (this may take a minute)")

    _processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

    _model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    _model.eval()

    _device = next(_model.parameters()).device
    print(f"[ModelEngine] Model loaded successfully on device: {_device}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def parse_classification(vlm_text: str, inventory_groups: list[str]) -> dict:
    """Parse structured fields from the VLM free-text output and match against inventory_groups."""
    parsed = {
        "object": "unknown",
        "group": "unknown",
        "description": "No description provided.",
        "raw_output": vlm_text,
    }

    # 1. Regex extraction
    object_match = re.search(r"Object:\s*([^,\n|]+)", vlm_text, re.IGNORECASE)
    group_match  = re.search(r"Group:\s*([^,\n|]+)",  vlm_text, re.IGNORECASE)
    desc_match   = re.search(r"Description:\s*(.*)", vlm_text, re.IGNORECASE)

    if object_match:
        parsed["object"] = object_match.group(1).strip()
    if group_match:
        parsed["group"] = group_match.group(1).strip()
    if desc_match:
        parsed["description"] = desc_match.group(1).strip()

    # 2. Strict Group Mapping
    # If the group found via regex isn't in our allowed list, try to find a match
    targeted_lower = [g.lower() for g in inventory_groups]
    found_group_lower = parsed["group"].lower()

    final_group = "unknown"
    
    # Case A: Extracted group matches exactly (after trim/lower)
    if found_group_lower in targeted_lower:
        final_group = inventory_groups[targeted_lower.index(found_group_lower)]
    else:
        # Case B: Extracted group is a substring of an allowed group or vice versa
        # e.g. "electronics" matches "active electronics"
        for i, target in enumerate(targeted_lower):
            if found_group_lower in target or target in found_group_lower:
                final_group = inventory_groups[i]
                break

    # 3. Keyword Fallback (if still unknown)
    # Search the entire raw text for any of the targeted group names
    if final_group == "unknown":
        for i, target in enumerate(targeted_lower):
            if target in vlm_text.lower():
                final_group = inventory_groups[i]
                break

    parsed["group"] = final_group

    # 4. Object Fallback
    if parsed["object"] == "unknown" and vlm_text:
        # Take first few words if no Object header found
        first_segment = vlm_text.split(",")[0].split("|")[0].replace("Object:", "").strip()
        parsed["object"] = first_segment[:40] # cap length

    return parsed


def run_inference(image: Image.Image, inventory_groups: list[str] | None = None) -> dict:
    """
    Run a single VLM inference pass on a PIL image.

    Parameters
    ----------
    image : PIL.Image.Image
        RGB image to classify.
    inventory_groups : list[str] | None
        Categories to use. Defaults to INVENTORY_GROUPS.

    Returns
    -------
    dict with keys: object, group, description, raw_output
    """
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    if inventory_groups is None:
        inventory_groups = INVENTORY_GROUPS

    # groups_str = ", ".join(inventory_groups)
    # prompt = (
    #     f"<image> detect the item, classify it into: {groups_str}, "
    #     "and describe its color, shape, and physical characteristics. "
    #     "Answer strictly in this format -> "
    #     "Object: [name], Group: [category], Description: [detailed visual description]"
    # )

    groups_str = ", ".join(inventory_groups)    
    
    # The smart prompt
    # prompt = f"<image> detect the item, classify it strictly into one of these groups: [{groups_str}], and describe its color, shape, and physical characteristics. Answer strictly in this format -> Object: [name], Group: [category], Description: [detailed visual description]"
    # prompt = f"<image> Detect and classify the objects in this warehouse bin. Map each detected item into exactly one of these categories: {groups_str}. Provide the output as a JSON list of objects with their labels and bounding box coordinates"
    
    # prompt = (
    #     f"<image> identify the main item in the image, classify it strictly into one of these groups: "
    #     f"[{groups_str}], and describe its visual appearance. "
    #     f"Answer exactly in this format -> Object: [name], Group: [category], Description: [detailed description]"
    # )
    prompt = (
        f"<image> Identify the main object in this image. "
        f"Answer exactly in this format -> Object: [name of object], Group: [invent a suitable category for it], Description: [brief visual description]"
    )
    print(f"[ModelEngine] Running inference with prompt: {prompt}")
    # print(f"[ModelEngine] Targeted groups: {inventory_groups}")
    image = image.convert("RGB")
    inputs = _processor(text=prompt, images=image, return_tensors="pt").to(_device)

    with torch.no_grad():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[-1]
    raw_text = _processor.decode(
        generated_ids[0][input_length:], skip_special_tokens=True
    ).strip()

    return parse_classification(raw_text, inventory_groups)
