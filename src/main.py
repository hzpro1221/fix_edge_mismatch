import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from PIL import Image

from utils import (
    process_images_edges,
    get_pytorch_edges
)

def dice_loss(pred, target, smooth=1e-5):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1.0 - dice_coeff

if __name__ == "__main__":
    WIDTH, HEIGHT = 512, 512
    NUMBER_INFERENCE_STEP = 8
    NUMBER_INFERENCE_TEST_STEP = 8
    NUM_STEP = 20
    LEARNING_RATE = 1e-4
    
    CONTROLNET_CONDITIONING_SCALE = 0.7
    
    DICE_WEIGHT = 1.0
    MSE_WEIGHT = 0.0
    CONSISTENCY_WEIGHT = 0.1
    
    NEGATIVE_PROMPT = "low contrast, washed out, illustration, abstract, bad quality"

    image_prompt_pairs = [
        (
            "Sunny", 
            "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000171_000019_gtFine_labelIds.png",
            "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000171_000019_gtFine_color.png"
        ),
        (
            "Winter", 
            "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000171_000019_gtFine_labelIds.png",
            "/root/KhaiDD/prompt_tunning_controlnet/data/cityscape/data/gtFine/train/aachen/aachen_000171_000019_gtFine_color.png"
        ),                                                                          
    ]

    print("Processing Semantic Edges...")
    edge_pairs = [(p[0], p[1]) for p in image_prompt_pairs]
    edge_data = process_images_edges(pairs=edge_pairs, h=HEIGHT, w=WIDTH)
    
    for i, item in enumerate(edge_data):
        item['color_path'] = image_prompt_pairs[i][2]
    
    output_dir = "comparison_results_seg_edge"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)

    print("Loading SD 1.5 & ControlNet Segmentation...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg",
        torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
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
    
    for item in edge_data:
        prompt = item['prompt']
        edge_pil = item['edge_pil'] 
        target_edge_tensor = item['target_edge_tensor']
        
        color_path = item['color_path']
        seg_pil = Image.open(color_path).convert("RGB").resize((WIDTH, HEIGHT))

        learnable_vector = torch.nn.Parameter(
            torch.randn(1, 77, 768, device="cuda", dtype=torch.float32) * 0.01
        )
        optimizer = torch.optim.AdamW([learnable_vector], lr=LEARNING_RATE)

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

        prev_output_edges = None
        best_loss = float('inf')
        best_vector = None
        
        # Chỉ cần khởi tạo mảng cho Total Loss
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
                    image=seg_pil, 
                    controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                    num_inference_steps=NUMBER_INFERENCE_STEP,
                    height=HEIGHT, width=WIDTH,
                    output_type="pt"
                )

            output_image = torch.clamp(output.images.to(torch.float32), 0.0, 1.0)
            
            output_edges = get_pytorch_edges(output_image)

            loss_dice = dice_loss(output_edges, target_edge_tensor)
            loss_mse = F.mse_loss(output_edges, target_edge_tensor)
            
            if prev_output_edges is not None:
                loss_consistency = F.mse_loss(output_edges, prev_output_edges)
            else:
                loss_consistency = torch.tensor(0.0, device="cuda")            
            
            total_loss = (loss_dice * DICE_WEIGHT) + (loss_mse * MSE_WEIGHT) + (loss_consistency * CONSISTENCY_WEIGHT)
            
            total_loss.backward()
            optimizer.step()

            current_loss_val = total_loss.item()
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_vector = learnable_vector.data.clone().detach()

            prev_output_edges = output_edges.detach()
            
            # Chỉ lưu Total Loss
            history_total.append(current_loss_val)
            
            print(f"Step [{step + 1:03d}] | Total: {total_loss.item():.5f} | "
                  f"Dice: {loss_dice.item():.5f} | MSE: {loss_mse.item():.5f} | Consistency: {loss_consistency.item():.5f}")
        
        # ----------------------------------------------------
        # VẼ BIỂU ĐỒ CHỈ VỚI TOTAL LOSS
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

        print("\nĐang generate hình ảnh so sánh...")
        with torch.no_grad():
            final_modified_embeds = prompt_embeds + best_vector.to(torch.float16)
            
            tuning_output = pipe(
                prompt_embeds=final_modified_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=seg_pil, 
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                num_inference_steps=NUMBER_INFERENCE_TEST_STEP,
                height=HEIGHT, width=WIDTH,
                output_type="pt"
            )
            tuning_img_pt = torch.clamp(tuning_output.images.to(torch.float32), 0.0, 1.0)
            tuning_edge_pt = get_pytorch_edges(tuning_img_pt)
            
            tuning_img_pil = to_pil_image(tuning_img_pt[0].cpu())
            tuning_edge_pil = to_pil_image(tuning_edge_pt.squeeze(0).cpu())

            baseline_output = pipe(
                prompt_embeds=prompt_embeds, 
                negative_prompt_embeds=negative_prompt_embeds,
                image=seg_pil,
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                num_inference_steps=NUMBER_INFERENCE_TEST_STEP,
                height=HEIGHT, width=WIDTH,
                output_type="pt"
            )
            baseline_img_pt = torch.clamp(baseline_output.images.to(torch.float32), 0.0, 1.0)
            baseline_edge_pt = get_pytorch_edges(baseline_img_pt)
            
            baseline_img_pil = to_pil_image(baseline_img_pt[0].cpu())
            baseline_edge_pil = to_pil_image(baseline_edge_pt.squeeze(0).cpu())

        input_color_pil = seg_pil

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f"Tuning vs Baseline (Seg -> Edge Loss): {prompt[:40]}...", fontsize=16, fontweight='bold')
        
        axes[0, 0].imshow(input_color_pil)
        axes[0, 0].set_title("Input Seg Map", fontsize=12, fontweight='bold')
        
        axes[0, 1].imshow(tuning_img_pil)
        axes[0, 1].set_title("Generated Color (TUNING)", color='green', fontsize=12, fontweight='bold')
        
        axes[0, 2].imshow(baseline_img_pil)
        axes[0, 2].set_title("Generated Color (BASELINE)", color='red', fontsize=12, fontweight='bold')

        axes[1, 0].imshow(edge_pil, cmap='gray')
        axes[1, 0].set_title("Target Edges (Ground Truth)", fontsize=12, fontweight='bold')
        
        axes[1, 1].imshow(tuning_edge_pil, cmap='gray')
        axes[1, 1].set_title("Generated Edge (TUNING)", color='green', fontsize=12, fontweight='bold')
        
        axes[1, 2].imshow(baseline_edge_pil, cmap='gray')
        axes[1, 2].set_title("Generated Edge (BASELINE)", color='red', fontsize=12, fontweight='bold')

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