import torch
import numpy as np
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

PROMPT_PINS = (
    "thin metallic silver leads or wires, "
    "narrow and reflective, separated from each other, "
    "binary mask of only the silver parts, "
    "exclude gold or yellow pads and large rectangles"
)

def clipseg_zero_shot(image_path: str, out_mask_path: str = "pred_mask.png", thresh: float = 0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Recommended zero-shot checkpoint
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")

    # CLIPSeg supports text prompts at test time
    inputs = processor(
        text=[PROMPT_PINS],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # logits: [batch, height, width] (low-res)
        logits = outputs.logits

    # sigmoid -> probability map
    probs = torch.sigmoid(logits)[0].float().cpu().numpy()  # [h, w] in [0,1]

    # resize back to original image size
    prob_img = Image.fromarray((probs * 255).astype(np.uint8)).resize(image.size, resample=Image.BILINEAR)
    prob = np.array(prob_img).astype(np.float32) / 255.0

    # threshold -> binary mask
    mask = (prob >= thresh).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_mask_path)

    return prob, mask

if __name__ == "__main__":
    prob, mask = clipseg_zero_shot('data/B/images/train/2025-09-16_09-17-15_945.bmp', "pred_mask.png", thresh=0.5)
    print("Saved pred_mask.png")
