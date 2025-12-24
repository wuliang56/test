
import torch
from sam2_train.build_sam import build_sam2
import cfg

# Create a dummy config class since we can't fully parse args in this environment without proper flags
class Args:
    sam_config = "sam2_hiera_s.yaml"
    sam_ckpt = "/root/autodl-tmp/Medical-SAM2-main/checkpoints/sam2_hiera_small.pt"
    gpu_device = 0

args = Args()
try:
    net = build_sam2(args.sam_config, args.sam_ckpt, device="cpu")
    print("Model built successfully")
    
    print("\n--- Image Encoder Parameters ---")
    for name, param in net.named_parameters():
        if "image_encoder" in name:
            print(name)
            # Just print first few and last few to understand structure
            # We break after a few to avoid spamming, but we want to see depth
            pass

except Exception as e:
    print(f"Error: {e}")
