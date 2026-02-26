import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from PIL import Image

from utils import get_pytorch_edges

def dice_loss(pred, target, smooth=1e-5):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1.0 - dice_coeff

def get_best_eval_edges(img_tensor, target_tensor, dice_weight, mse_weight):
    best_loss = float('inf')
    best_edges = None
    
    for t in range(500):
        thresh = 0.001 * t
        edges_tmp = get_pytorch_edges(img_tensor, threshold=thresh)
        
        l_dice = dice_loss(edges_tmp, target_tensor)
        l_mse = F.mse_loss(edges_tmp, target_tensor)
        
        loss_val = (l_dice * dice_weight + l_mse * mse_weight).item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_edges = edges_tmp
            
    return best_edges

if __name__ == "__main__":
    WIDTH, HEIGHT = 512, 512
    NUMBER_INFERENCE_STEP = 3
    NUMBER_INFERENCE_TEST_STEP = 20
    NUM_STEP = 400
    LEARNING_RATE = 2e-3

    CONTROLNET_CONDITIONING_SCALE = 4.0
    STRENGTH = 0.8
    
    DICE_WEIGHT = 1.0
    MSE_WEIGHT = 1.0
    CONSISTENCY_WEIGHT = 0.0
    
    NEGATIVE_PROMPT = "low contrast, washed out, illustration, abstract, bad quality"

    image_prompt_pairs = [
        {
            "prompt": "Sunny, full HD, 4k",
            "image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png",
            "segment_image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png"
        },
        {
            "prompt": "Winter, full HD, 4k",
            "image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png",
            "segment_image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png"
        },
        {
            "prompt": "Late night, full HD, 4k",
            "image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png",
            "segment_image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png"
        },
        {
            "prompt": "Dust, full HD, 4k",
            "image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png",
            "segment_image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png"
        },                        
        {
            "prompt": "City, Sunny",
            "image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png",
            "segment_image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000001_000019_gtFine_color.png"
        },
        {
            "prompt": "City, Winter",
            "image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png",
            "segment_image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000001_000019_gtFine_color.png"
        },   
        {
            "prompt": "City, Late night",
            "image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png",
            "segment_image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000001_000019_gtFine_color.png"
        },
        {
            "prompt": "City, Dust",
            "image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/leftImg8bit/train/aachen/aachen_000001_000019_leftImg8bit.png",
            "segment_image_path": "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000001_000019_gtFine_color.png"
        }
    ]

    output_dir = "comparison_results_seg_edge"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)

    print("Loading SD 1.5 & ControlNet Segmentation...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg",
        torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )
    
    pipe.to("cuda")
    pipe.enable_attention_slicing()

    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.controlnet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    pipe.unet.enable_gradient_checkpointing()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    for item in image_prompt_pairs:
        prompt = item['prompt']
        
        init_image = Image.open(item['image_path']).convert("RGB").resize((WIDTH, HEIGHT))
        control_image = Image.open(item['segment_image_path']).convert("RGB").resize((WIDTH, HEIGHT))
        
        init_image_tensor = to_tensor(init_image).unsqueeze(0).to("cuda")
        
        target_edge_tensor = get_pytorch_edges(init_image_tensor, threshold=0.01).detach()
        
        edge_pil = to_pil_image(target_edge_tensor.squeeze(0).cpu())
        
        learnable_vector = torch.nn.Parameter(
            torch.randn(1, 77, 768, device="cuda", dtype=torch.float32) * 0.01
        )
        
        learnable_thresh = torch.nn.Parameter(
            torch.tensor(0.01, device="cuda", dtype=torch.float32)
        )

        optimizer = torch.optim.AdamW([learnable_vector, learnable_thresh], lr=LEARNING_RATE)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_STEP, eta_min=1e-5)

        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=NEGATIVE_PROMPT
            )

        raw_pipe_call = pipe.__call__
        while hasattr(raw_pipe_call, "__wrapped__"):
            raw_pipe_call = raw_pipe_call.__wrapped__

        print("\n=> Đang tính toán Baseline Edge cho Consistency Loss...")
        with torch.no_grad():
            initial_output = raw_pipe_call(
                self=pipe,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=init_image,            
                control_image=control_image,  
                strength=STRENGTH,          
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                num_inference_steps=NUMBER_INFERENCE_STEP,
                height=HEIGHT, width=WIDTH,
                output_type="pt"
            )
            initial_output_image = torch.clamp(initial_output.images.to(torch.float32), 0.0, 1.0)
            initial_output_edges = get_pytorch_edges(initial_output_image).detach()
        
        history_total = []
        
        print(f"\n[{'='*40}]")
        print(f"Bắt đầu xử lý cho prompt: '{prompt[:30]}...'")
        print(f"[{'='*40}]\n")

        for step in range(NUM_STEP):
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            modified_prompt_embeds = prompt_embeds + learnable_vector.to(torch.float16)

            with torch.enable_grad():
                output = raw_pipe_call(
                    self=pipe,
                    prompt_embeds=modified_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    image=init_image,            
                    control_image=control_image,  
                    strength=STRENGTH,          
                    controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                    num_inference_steps=NUMBER_INFERENCE_STEP,
                    height=HEIGHT, width=WIDTH,
                    output_type="pt"
                )

            output_image = torch.clamp(output.images.to(torch.float32), 0.0, 1.0)
            
            current_thresh = torch.clamp(learnable_thresh, 0.0, 1.0)
            
            output_edges = get_pytorch_edges(output_image, threshold=current_thresh)

            loss_dice = dice_loss(output_edges, target_edge_tensor)
            loss_mse = F.mse_loss(output_edges, target_edge_tensor)
            
            loss_consistency = F.mse_loss(output_edges, initial_output_edges)       
            
            total_loss = (loss_dice * DICE_WEIGHT) + (loss_mse * MSE_WEIGHT) + (loss_consistency * CONSISTENCY_WEIGHT)
            
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            history_total.append(total_loss.item())
            
            current_lr = lr_scheduler.get_last_lr()[0]
            
            print(f"Step [{step + 1:03d}] | Total: {total_loss.item():.5f} | "
                  f"Dice: {loss_dice.item():.5f} | MSE: {loss_mse.item():.5f} | Cons: {loss_consistency.item():.5f} | "
                  f"Thresh: {current_thresh.item():.4f} | LR: {current_lr:.6f}")
                  
        final_vector = learnable_vector.data.clone().detach()
        
        # ----------------------------------------------------
        # VẼ BIỂU ĐỒ
        # ----------------------------------------------------
        print("\nĐang vẽ biểu đồ loss...")
        plt.figure(figsize=(10, 6))
        plt.plot(history_total, label="Total Loss", color='blue', linewidth=2)
        plt.title(f"Total Loss Convergence: {prompt[:40]}...")
        plt.xlabel("Tuning Steps")
        plt.ylabel("Loss Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_filename = f"{prompt[:15].replace(' ', '_').lower()}_total_loss_curve.png"
        plot_save_path = os.path.join(plot_dir, plot_filename)
        plt.savefig(plot_save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f" -> Đã lưu biểu đồ loss tại: {plot_save_path}")

        # ----------------------------------------------------
        # GENERATE VÀ EVALUATION
        # ----------------------------------------------------
        print("\nĐang generate hình ảnh so sánh...")
        with torch.no_grad():
            final_modified_embeds = prompt_embeds + final_vector.to(torch.float16)
            
            tuning_output = pipe(
                prompt_embeds=final_modified_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=init_image,
                control_image=control_image,
                strength=STRENGTH,
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                num_inference_steps=NUMBER_INFERENCE_TEST_STEP,
                height=HEIGHT, width=WIDTH,
                output_type="pt"
            )
            tuning_img_pt = torch.clamp(tuning_output.images.to(torch.float32), 0.0, 1.0)
            tuning_img_pil = to_pil_image(tuning_img_pt[0].cpu())
            
            baseline_output = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=init_image,
                control_image=control_image,
                strength=STRENGTH,
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                num_inference_steps=NUMBER_INFERENCE_TEST_STEP,
                height=HEIGHT, width=WIDTH,
                output_type="pt"
            )
            baseline_img_pt = torch.clamp(baseline_output.images.to(torch.float32), 0.0, 1.0)
            baseline_img_pil = to_pil_image(baseline_img_pt[0].cpu())
            
            tuning_edge_pt = get_best_eval_edges(tuning_img_pt, target_edge_tensor, DICE_WEIGHT, MSE_WEIGHT)
            tuning_edge_pil = to_pil_image(tuning_edge_pt.squeeze(0).cpu())

            baseline_edge_pt = get_best_eval_edges(baseline_img_pt, target_edge_tensor, DICE_WEIGHT, MSE_WEIGHT)
            baseline_edge_pil = to_pil_image(baseline_edge_pt.squeeze(0).cpu())

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f"Tuning vs Origin: {prompt[:40]}...", fontsize=16, fontweight='bold')
        
        axes[0, 0].imshow(init_image)
        axes[0, 0].set_title("Input image", fontsize=12, fontweight='bold')
        axes[0, 1].imshow(tuning_img_pil)
        axes[0, 1].set_title("Generated Color (tunning)", color='green', fontsize=12, fontweight='bold')
        axes[0, 2].imshow(baseline_img_pil)
        axes[0, 2].set_title("Generated Color (Original)", color='red', fontsize=12, fontweight='bold')

        axes[1, 0].imshow(edge_pil, cmap='gray')
        axes[1, 0].set_title("Target Edges (Ground Truth)", fontsize=12, fontweight='bold')
        axes[1, 1].imshow(tuning_edge_pil, cmap='gray')
        axes[1, 1].set_title("Generated Edge (tunning)", color='green', fontsize=12, fontweight='bold')
        axes[1, 2].imshow(baseline_edge_pil, cmap='gray')
        axes[1, 2].set_title("Generated Edge (original)", color='red', fontsize=12, fontweight='bold')

        for ax in axes.flatten():
            ax.axis('off')

        plt.tight_layout(w_pad=3.0)
        fig.canvas.draw()

        box0 = axes[0, 0].get_position()
        box1 = axes[0, 1].get_position()
        box2 = axes[0, 2].get_position()
        
        x_line_1 = (box0.x1 + box1.x0) / 2
        x_line_2 = (box1.x1 + box2.x0) / 2

        line_1 = lines.Line2D([x_line_1, x_line_1], [0.05, 0.90], transform=fig.transFigure, color="black", linewidth=3)
        line_2 = lines.Line2D([x_line_2, x_line_2], [0.05, 0.90], transform=fig.transFigure, color="black", linewidth=3)
        fig.add_artist(line_1)
        fig.add_artist(line_2)

        filename = f"{prompt[:15].replace(' ', '_').lower()}_seg_edge_comp.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f" -> Đã lưu bảng so sánh thành công: {save_path}")

    print("\n[HOÀN THÀNH] Toàn bộ quá trình chạy đã kết thúc!")