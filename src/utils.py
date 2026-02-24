import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import scipy.ndimage

from tqdm import tqdm
from diffusers.utils import load_image

# Sử dụng sobel kernels để trích edges (do giữ được computational graph)
def get_pytorch_edges(img_tensor):
    gray = img_tensor.mean(dim=1, keepdim=True)
    
    # Sobel kernels
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=gray.device).view(1,1,3,3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=gray.device).view(1,1,3,3)
    
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-4)
    return mag / (mag.max() + 1e-6)

def rescale_image(img_tensor, height, width):
    return F.interpolate(
        img_tensor, 
        size=(height, width), 
        mode='bilinear', 
        align_corners=False, 
        antialias=True
    )

def process_images(pairs, h, w):
    results = [] 
    
    for prompt, path in tqdm(pairs, desc="Preprocessing"):
        raw_img = load_image(path)
        img_tensor = to_tensor(raw_img).unsqueeze(0).to("cuda")
        
        with torch.no_grad():
            img_rescaled = rescale_image(img_tensor, h, w)
            edges = get_pytorch_edges(img_rescaled) # [1, 1, H, W]
            
            edges_np = (edges.squeeze().cpu().numpy() > 0.1).astype(np.uint8) 
            dist_map_np = scipy.ndimage.distance_transform_edt(1 - edges_np)
            dist_map = torch.from_numpy(dist_map_np).float().to("cuda").unsqueeze(0).unsqueeze(0)
            
            edges_rgb = edges.repeat(1, 3, 1, 1)
            canny_pil = to_pil_image(edges_rgb.squeeze(0).cpu())
        
        results.append({
            'prompt': prompt,
            'canny_pil': canny_pil,
            'dist_map': dist_map
        })
        
    return results
