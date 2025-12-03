from prompt import OUTPUT_MASK
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


OUTPUT_MASK = "outputs/mask.png"


def run(
    image_path,
    prompt,
    config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
    checkpoint_path="sam2.1_hiera_large.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model = build_sam2(
        config_file=config_path,
        ckpt_path=checkpoint_path,
        device=device,
        mode="eval",
    )

    sam = SAM2AutomaticMaskGenerator(model)

    img = Image.open(image_path).convert("RGB")
    anns = sam.generate(np.array(img))
    if len(anns) == 0:
        return

    crops = []
    masks = []
    for a in anns:
        seg = a["segmentation"]
        masks.append(seg)
        crops.append(Image.fromarray(np.array(img) * seg[..., None]))

    text_inputs = clip_proc(text=[prompt], return_tensors="pt")
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        tfeat = clip_model.get_text_features(**text_inputs)

    img_inputs = clip_proc(images=crops, return_tensors="pt", padding=True)
    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
    with torch.no_grad():
        if hasattr(clip_model, "get_image_features"):
            imfeat = clip_model.get_image_features(
                pixel_values=img_inputs["pixel_values"]
            )
        else:
            out = clip_model(**img_inputs)
            imfeat = out.image_embeds

    sims = F.cosine_similarity(imfeat, tfeat)
    best = int(torch.argmax(sims).item())

    Image.fromarray(masks[best].astype(np.uint8) * 255).save(OUTPUT_MASK)

