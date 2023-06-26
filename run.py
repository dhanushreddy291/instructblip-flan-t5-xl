import requests
import torch
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

cache_path = "./instructblip-weights"

def generate_caption(**inputs):
    model = InstructBlipForConditionalGeneration.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl", cache_dir=cache_path
    )
    processor = InstructBlipProcessor.from_pretrained(
        "Salesforce/instructblip-flan-t5-xl", cache_dir=cache_path
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    url = inputs["url"]
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    prompt = "Describe the image"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )

    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[
        0
    ].strip()

    return {"response": generated_text}
