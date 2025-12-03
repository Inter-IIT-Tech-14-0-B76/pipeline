import os
import sys
import numpy as np
from PIL import Image
from prompt import Segmentation


def segment_image(image_path, x, y, output_dir="output"):
    """
    Segment an image using a point prompt (x, y) and save the result.
    
    Args:
        image_path (str): Path to input image
        x (int): X coordinate of the point
        y (int): Y coordinate of the point
        output_dir (str): Directory to save output (default: "output")
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Initialize SAM2 segmentation model
    print("Initializing SAM2 model...")
    sam = Segmentation(
        config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint_path="sam2.1_hiera_large.pt"
    )
    
    # Set image and run segmentation
    print("Processing segmentation...")
    sam.set_image(image_np)
    sam.point_prompt((x, y), is_foreground=True)
    
    # Create visualization (dim background, keep foreground bright)
    mask = sam.prev_mask
    dimmed = image_np // 3
    output_vis = image_np.copy()
    output_vis[mask == 0] = dimmed[mask == 0]
    
    # Save outputs
    mask_path = os.path.join(output_dir, "mask.png")
    output_path = os.path.join(output_dir, "segmented_output.png")
    
    Image.fromarray(mask * 255).save(mask_path)
    Image.fromarray(output_vis).save(output_path)
    
    print(f"✓ Mask saved to: {mask_path}")
    print(f"✓ Segmented image saved to: {output_path}")
    
    return mask, output_vis


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python init.py <image_path> <x> <y>")
        print("Example: python init.py abc.jpg 150 200")
        sys.exit(1)
    
    image_path = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    segment_image(image_path, x, y)
