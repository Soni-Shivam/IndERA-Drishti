

import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
from google.colab import userdata

# 2. Authenticate with Hugging Face
try:
    hf_token = userdata.get('HF_TOKEN')
except userdata.SecretNotFoundError:
    print("Error: Please set your HF_TOKEN in the Colab Secrets tab!")
    hf_token = None

# 3. Load the Model and Processor
# We use the 'mix' version which is fine-tuned for general VQA and detection
model_id = "google/paligemma-3b-mix-224"

print("Loading model... (this takes a minute)")
processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # Uses less VRAM, perfect for Colab T4
    device_map="auto",
    token=hf_token
)
print("Model loaded successfully!")


import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import re

def parse_classification(vlm_text, inventory_groups):
    """
    Enhanced parser that handles anything being an Object, Group of objects, humans, and a Detailed Description,
    """
    # Added "description" to the default dictionary
    parsed_data = {
        "object": "unknown",
        "group": "unclassified",
        "description": "No description provided.",
        "raw_output": vlm_text.lower()
    }

    # 1. Regex matching for all three fields
    object_match = re.search(r'Object:\s*([^,]+)', vlm_text, re.IGNORECASE)
    group_match = re.search(r'Group:\s*([^,\n]+)', vlm_text, re.IGNORECASE)
    desc_match = re.search(r'Description:\s*(.*)', vlm_text, re.IGNORECASE)

    if object_match:
        parsed_data["object"] = object_match.group(1).strip()
    if group_match:
        parsed_data["group"] = group_match.group(1).strip()
    if desc_match:
        parsed_data["description"] = desc_match.group(1).strip()

    # 2. Fallback: If group is still unclassified, look for keywords
    if parsed_data["group"] == "unclassified":
        for group in inventory_groups:
            if group.lower() in vlm_text.lower():
                parsed_data["group"] = group
                break

    # 3. If object is unknown, use the first part of the raw text as a guess
    if parsed_data["object"] == "unknown" and len(vlm_text) > 0:
        parsed_data["object"] = vlm_text.split(',')[0].replace('Object:', '').strip()

    return parsed_data


def capture_and_smart_classify(inventory_groups):
    js = Javascript('''
    async function takePhoto() {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture for Smart Prompt';
      capture.style.cssText = 'padding:10px; background:#007bff; color:white; border:none; border-radius:5px; cursor:pointer; margin-bottom:10px; font-weight:bold;';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.maxWidth = '100%';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', 0.8);
    }
    ''')
    display(js)

    try:
        data = eval_js('takePhoto()')
        binary = b64decode(data.split(',')[1])
        filename = 'scanned_item.jpg'
        with open(filename, 'wb') as f:
            f.write(binary)

        image = Image.open(filename).convert("RGB")
        groups_str = ", ".join(inventory_groups)

        # --- UPDATED PROMPT ---
        # Explicitly asking for physical characteristics
        prompt = f"<image> detect the item, classify it into: {groups_str}, and describe its color, shape, and physical characteristics. Answer strictly in this format -> Object: [name], Group: [category], Description: [detailed visual description]"

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            # --- UPDATED TOKENS ---
            # Increased from 30 to 100 to allow space for the descriptive text
            generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)

        input_length = inputs["input_ids"].shape[-1]
        generated_text = processor.decode(generated_ids[0][input_length:], skip_special_tokens=True).strip()

        structured_data = parse_classification(generated_text, inventory_groups)

        print("=" * 60)
        print("INDERA 6-DOF CONTROL PAYLOAD:")
        print(f"Detected Object : {structured_data['object'].upper()}")
        print(f"Routing Bin     : {structured_data['group'].upper()}")
        print(f"Vision Details  : {structured_data['description']}")
        print("-" * 60)
        print(f"(Raw Model Output: {generated_text})")
        print("=" * 60)

        return structured_data

    except Exception as err:
        print(f"Pipeline failed: {err}")




# ==========================================
# RUN THE UPDATED SYSTEM
# ==========================================
my_inventory_groups = ["electronics", "hardware", "packaging", "tools", "unknown", "human"]


# Call the function
final_data = capture_and_smart_classify(my_inventory_groups)