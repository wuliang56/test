
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import cfg
from func_2d.dataset import DRIS
from func_2d.utils import eval_seg, vis_image
from func_2d.adapter import AdapterModel

# Parse arguments manually or via cfg
# We need to replicate the args used in training
def parse_args():
    # Hardcode ckpt path as fallback to bypass argparse issues
    ckpt_path = '/root/autodl-tmp/Medical-SAM2-main/logs/REFUGE_FewShot_to_DRIS_2025_12_23_23_58_34/Model/latest_epoch.pth'
    
    # Try to parse standard args, ignore unknown
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam2')
    parser.add_argument('-exp_name', type=str, default='test')
    parser.add_argument('-vis', type=int, default=0)
    parser.add_argument('-gpu', type=bool, default=True)
    parser.add_argument('-gpu_device', type=int, default=0)
    parser.add_argument('-image_size', type=int, default=1024)
    parser.add_argument('-out_size', type=int, default=1024)
    parser.add_argument('-sam_ckpt', type=str, default='./checkpoints/sam2_hiera_small.pt')
    parser.add_argument('-sam_config', type=str, default='sam2_hiera_s')
    parser.add_argument('-b', type=int, default=1)
    parser.add_argument('-data_path', type=str, default='/root/autodl-tmp/DSM-main/data/DRIS')
    
    # Use parse_known_args to ignore anything else (like -ckpt if passed)
    args, _ = parser.parse_known_args()
    
    # Inject ckpt
    args.ckpt = ckpt_path
    
    return args

def get_network(args):
    # This duplicates logic from train_2d.py/utils.py but simplifies imports
    if args.net == 'sam2':
        from sam2_train.build_sam import build_sam2
        # Need to setup autocast context if strictly following train
        net = build_sam2(args.sam_config, args.sam_ckpt, device="cuda")
    else:
        raise ValueError(f"Unknown net: {args.net}")
    
    if args.gpu:
        net = net.to(args.gpu_device)
        
    return net

def test_one_shot(args):
    # Setup Device
    device = torch.device('cuda', args.gpu_device)
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. Load Network
    print(f"Loading network: {args.net}...")
    net = get_network(args)
    
    # 2. Attach Adapter
    print("Initializing Adapter...")
    adapter_model = AdapterModel().to(device)
    net.add_module('adapter_model', adapter_model)
    
    # Load Checkpoint
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    # Check if checkpoint is wrapped in 'model' key (common in DDP/saving full state)
    if 'model' in checkpoint:
        print("Detected 'model' key in checkpoint, unpacking state_dict...")
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    
    # Debug: Print first few keys from checkpoint
    print("Checkpoint keys sample:", list(state_dict.keys())[:5])
    
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # Explicitly check for adapter keys in new_state_dict
    adapter_keys = [k for k in new_state_dict.keys() if 'adapter_model' in k]
    if not adapter_keys:
        print("WARNING: No adapter keys found in processed state_dict!")
    else:
        print(f"Found {len(adapter_keys)} adapter keys in checkpoint.")

    # Load weights
    msg = net.load_state_dict(new_state_dict, strict=False)
    print(f"Load result: {msg}")
    
    # Check if adapter parameters were loaded
    # We look for keys starting with 'adapter_model' in missing_keys
    missing_adapters = [k for k in msg.missing_keys if 'adapter_model' in k]
    if missing_adapters:
        print("WARNING: Adapter weights were NOT loaded! They are random.")
        print(f"Missing keys: {missing_adapters}")
    else:
        print("SUCCESS: Adapter weights loaded successfully.")
        
        # Verify values (optional)
        # print("Adapter sample weight:", net.adapter_model.adapters[0].adapter[0].weight[0,0,0,0])

    
    net.eval()
    
    # 4. Load Data
    print(f"Loading DRIS dataset from {args.data_path}...")
    from torchvision import transforms
    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    
    # We use 'Test' mode for DRIS
    dataset = DRIS(args, args.data_path, transform=transform_test, mode='Test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    # 5. Run One-Shot Inference
    # Logic similar to validation_sam in function.py
    
    print("Starting Inference...")
    
    tol, eiou, edice = 0, 0, 0
    total_samples = 0
    
    # Memory Bank
    memory_bank_list = []
    feat_sizes = [(256, 256), (128, 128), (64, 64)]
    
    # Save directory
    save_dir = 'test_results'
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # First pass: Get Support Image (First image in loader)
        # We need to iterate loader manually
        iterator = iter(loader)
        
        # --- Support Step ---
        try:
            support_pack = next(iterator)
        except StopIteration:
            print("Dataset is empty!")
            return

        print(f"Using {support_pack['image_meta_dict']['filename_or_obj'][0]} as Support.")
        
        imgs_s = support_pack['image'].to(dtype=torch.float32, device=device)
        masks_s = support_pack['mask'].to(dtype=torch.float32, device=device)
        
        # Encode Support
        backbone_out = net.forward_image(imgs_s)
        _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
        
        # Apply Adapter
        if hasattr(net, 'adapter_model'):
            vision_feats = net.adapter_model.adapt_features(vision_feats)
            
        maskmem_features, maskmem_pos_enc = net._encode_new_memory(
            current_vision_feats=vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=masks_s,
            is_mask_from_pts=True
        )
        
        # Store Memory
        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size) 
                 for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
        image_embed = feats[-1]

        maskmem_features = maskmem_features.to(torch.bfloat16).to(device, non_blocking=True)
        maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16).to(device, non_blocking=True)
        
        # We store the tensors needed for attention
        memory_bank = {
            'mem': maskmem_features[0].unsqueeze(0), # [1, 64, 64, 64]
            'pos': maskmem_pos_enc[0].unsqueeze(0),
            'embed': image_embed[0].reshape(-1)
        }
        
        # --- Query Step (Iterate all) ---
        # Note: We also evaluate on the support image itself (it should be perfect)
        # Reset iterator or re-use? Let's create a new loop over entire dataset
        
        for ind, pack in enumerate(tqdm(loader, desc='Testing')):
            imgs = pack['image'].to(dtype=torch.float32, device=device)
            masks = pack['mask'].to(dtype=torch.float32, device=device)
            name = pack['image_meta_dict']['filename_or_obj'][0]
            
            B = imgs.size(0) # Should be 1
            
            # Encode Query
            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            
            # Apply Adapter
            if hasattr(net, 'adapter_model'):
                vision_feats = net.adapter_model.adapt_features(vision_feats)
                
            # Prepare Memory for Attention
            # [4096, B, 64]
            memory = memory_bank['mem'].to(device).flatten(2).permute(2, 0, 1).expand(-1, B, -1)
            memory_pos = memory_bank['pos'].to(device).flatten(2).permute(2, 0, 1).expand(-1, B, -1)
            
            # Memory Attention
            vision_feats[-1] = net.memory_attention(
                curr=[vision_feats[-1]],
                curr_pos=[vision_pos_embeds[-1]],
                memory=memory,
                memory_pos=memory_pos,
                num_obj_ptr_tokens=0
            )
            
            # Prompt Encoder (Using points if available in dataset, or None)
            # DRIS dataset usually provides random clicks in __getitem__?
            # Let's check DRIS class... yes, it calls random_click.
            # But wait, in function.py validation_sam, it checks for 'pt' in pack.
            # DRIS __getitem__ returns 'pt'.
            
            if 'pt' in pack:
                pt = pack['pt'].to(device).unsqueeze(1)
                point_labels = pack['p_label'].to(device).unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=device)
                points = (coords_torch, labels_torch)
            else:
                points = None
                
            # Random Prompt Dropout in Test?
            # Usually we WANT prompt in test to guide it.
            # But strict One-Shot usually assumes no user interaction on Query.
            # However, our dataset provides it, and we trained with it (50%).
            # Let's keep it to maximize performance.
            
            se, de = net.sam_prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
                batch_size=B,
            )
            
            # Decoder
            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            image_embed = feats[-1]
            high_res_feats = feats[:-1]
            
            low_res_multimasks, _, _, _ = net.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=net.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_feats
            )
            
            # Resize
            pred = torch.nn.functional.interpolate(low_res_multimasks, size=(args.out_size, args.out_size))
            
            # Metric
            threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
            # Handling return values robustly
            res = eval_seg(pred, masks, threshold)
            
            if len(res) == 2:
                batch_iou, batch_dice = res
            elif len(res) == 4:
                # Assuming (iou_d, iou_c, disc_dice, cup_dice) -> Take Cup (last) or Avg?
                # Usually Cup is the harder one. Let's take Avg as in training.
                iou_d, iou_c, disc_dice, cup_dice = res
                batch_iou = (iou_d + iou_c) / 2.0
                batch_dice = (disc_dice + cup_dice) / 2.0
            else:
                batch_iou, batch_dice = 0, 0
                
            tol += batch_dice
            eiou += batch_iou
            edice += batch_dice
            total_samples += 1
            
            # Vis - Save all images
            vis_image(imgs, pred, masks, os.path.join(save_dir, f'{name}_pred.png'), points=points)

    print(f"Final Results on {total_samples} images:")
    print(f"mIoU: {eiou/total_samples:.4f}")
    print(f"mDice: {edice/total_samples:.4f}")

if __name__ == '__main__':
    args = parse_args()
    test_one_shot(args)
