
import os
import numpy as np
from PIL import Image
import torch
from func_2d.dataset import DRIS
import cfg

# Mock args
class Args:
    data_path = '/root/autodl-tmp/DSM-main/data/DRIS' # Hardcoded based on previous context
    image_size = 1024
    out_size = 1024

args = Args()

def check_mask_values():
    # 1. Check raw file directly
    base_path = '/root/autodl-tmp/DSM-main/data/DRIS/Test-20211018T060000Z-001/Test/Images'
    # Mask path structure: .../Test/Masks/glaucoma/image_name.png (or similar)
    # Let's find a real mask file
    mask_base = '/root/autodl-tmp/DSM-main/data/DRIS/Test-20211018T060000Z-001/Test/Masks'
    
    found_mask = False
    mask_path = ""
    
    for subdir in ['glaucoma', 'normal']: # Corrected names
        p = os.path.join(mask_base, subdir)
        if os.path.exists(p):
            for f in os.listdir(p):
                if f.endswith('.png') or f.endswith('.jpg'):
                    mask_path = os.path.join(p, f)
                    found_mask = True
                    break
        if found_mask: break
    
    if found_mask:
        print(f"Checking raw mask file: {mask_path}")
        img = Image.open(mask_path)
        img_np = np.array(img)
        print(f"Raw Mask Unique Values: {np.unique(img_np)}")
        print(f"Raw Mask Shape: {img_np.shape}")
        print("-" * 30)
    else:
        print("Could not find a raw mask file to check.")

    # 2. Check via Dataset Loader (to see what transforms do)
    # This checks what the model actually receives
    try:
        # Note: We need to point to where DRIS dataset class expects
        # dataset.py DRIS class logic:
        # self.data_path = args.data_path
        # self.image_paths = glob(self.data_path + '/Test-20211018T060000Z-001/Test/Images/**/*.jpg', recursive=True)
        
        # Let's try to instantiate the dataset
        from torchvision import transforms
        transform_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ])
        
        # We need to make sure we use the correct path that the dataset class expects
        # In train_2d.py we used: dris_path = args.data_path.replace('REFUGE', 'DRIS')
        # Here we manually set it
        dris_dataset = DRIS(args, args.data_path, transform=transform_test, mode='Training') # Use Training mode as we use it for Support
        
        if len(dris_dataset) > 0:
            sample = dris_dataset[0]
            mask_tensor = sample['mask']
            print(f"Dataset Output Mask Shape: {mask_tensor.shape}")
            print(f"Dataset Output Mask Unique Values: {torch.unique(mask_tensor)}")
            
            # Check if values are binary 0.0 and 1.0
            unique_vals = torch.unique(mask_tensor)
            is_binary = torch.all(torch.isin(unique_vals, torch.tensor([0.0, 1.0])))
            print(f"Is strictly binary (0.0, 1.0)? {is_binary}")
        else:
            print("Dataset is empty.")
            
    except Exception as e:
        print(f"Error checking dataset: {e}")

if __name__ == "__main__":
    check_mask_values()
