import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import distance_transform_edt

# Dice Coefficient (F1-Score) -> đo mức độ trùng khớp giữa các pixel trắng (cạnh) của hai ảnh.
def dice_coefficient_torch(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def chamfer_distance(pred_canny, target_dist_map):
    return (pred_canny * target_dist_map).sum() / (pred_canny.sum() + 1e-6)

# Tạo ra một heat map, phạt mô hình nếu nó sinh ra nét vẽ ở vị trí xa so với nét vẽ thực
# -> Nhằm ổn định quá trình training, đảm bảo loss không quá to. Sử dụng trong chamfer_distance
def get_target_dist_map(canny_pil):
    canny_np = np.array(canny_pil.convert("L"))
    
    mask = (canny_np == 0)
    
    dist_map = distance_transform_edt(mask)
    return torch.from_numpy(dist_map).float().to("cuda").unsqueeze(0).unsqueeze(0)