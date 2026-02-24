import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.transforms.functional import to_tensor, to_pil_image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from IPython.display import display

import os
import matplotlib.pyplot as plt

from utils import (
    process_images, 
    get_pytorch_edges
)
from losses import ( 
    dice_coefficient_torch, 
    chamfer_distance, 
    get_target_dist_map
)

if __name__ == "__main__":
    WIDTH, HEIGHT = 512, 512
    NUMBER_INFERENCE_STEP = 30
    NUMBER_INFERENCE_TEST_STEP = 50
    NUM_STEP = 20
    LEARNING_RATE = 1e-3
    
    DICE_WEIGHT = 0.3
    SSIM_WEIGHT = 0.3
    CHAMFER_WEIGHT = 0.4
    CONTROLNET_CONDITIONING_SCALE = 0.2

    image_prompt_pairs = [
        # ("Cyberpunk", "/root/KhaiDD/prompt_tunning_controlnet/data/439921836_3842411459416323_5840693379872517580_n.jpg"),
        ("Cinematic wildlife photography, natural atmosphere, 8k", "/root/KhaiDD/prompt_tunning_controlnet/data/457858398_17897638527052725_2651477979417305260_n.jpg"),
        # ("Portrait", "/root/KhaiDD/prompt_tunning_controlnet/data/467902762_8786841014697354_4401281500707445075_n.jpg"),
        # ("Wildlife", "/root/KhaiDD/prompt_tunning_controlnet/data/473723145_1057803019697778_5812087022264039605_n.jpg"),
        # ("Cinematic", "/root/KhaiDD/prompt_tunning_controlnet/data/474047330_1029392195893231_1448405286663373416_n.jpg"),
        # ("Cosmic", "/root/KhaiDD/prompt_tunning_controlnet/data/FB_IMG_1725192745457.jpg")
    ]

    edges_data = process_images(pairs=image_prompt_pairs, h=HEIGHT, w=WIDTH)

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0-small",
        torch_dtype=torch.float16, variant="fp16"
    )
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet, vae=vae,
        torch_dtype=torch.float16, variant="fp16"
    )
    
    pipe.to("cuda")
    pipe.enable_attention_slicing()

    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.controlnet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    pipe.unet.enable_gradient_checkpointing()

    results_list = []
    
    for item in edges_data:
        prompt = item['prompt']
        canny_image = item['canny_pil']
        target_dist_map = item['dist_map'] 
        
        target_canny_tensor = to_tensor(canny_image).unsqueeze(0).to("cuda", dtype=torch.float32)
        target_canny_tensor = target_canny_tensor[:, 0:1, :, :] 

        learnable_vector = torch.nn.Parameter(
            torch.randn(1, 77, 2048, device="cuda", dtype=torch.float32) * 0.01
        )
        optimizer = torch.optim.AdamW([learnable_vector], lr=LEARNING_RATE)

        with torch.no_grad():
            (prompt_embeds, negative_prompt_embeds,
             pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(prompt=prompt)

        raw_pipe_call = pipe.__call__
        while hasattr(raw_pipe_call, "__wrapped__"):
            raw_pipe_call = raw_pipe_call.__wrapped__

        pbar = tqdm(range(NUM_STEP), desc=f"Tuning: {prompt[:15]}")
        for step in pbar:
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            modified_prompt_embeds = prompt_embeds + learnable_vector.to(torch.float16)

            with torch.enable_grad():
                output = raw_pipe_call(
                    self=pipe,
                    prompt_embeds=modified_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    image=canny_image,
                    controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                    num_inference_steps=NUMBER_INFERENCE_STEP, 
                    height=HEIGHT, width=WIDTH,
                    output_type="pt"
                )

            output_image = torch.clamp(output.images.to(torch.float32), 0.0, 1.0)
            output_canny = get_pytorch_edges(output_image)

            loss_dice = 1 - dice_coefficient_torch(output_canny, target_canny_tensor)
            loss_ssim = F.mse_loss(output_canny, target_canny_tensor)
            loss_chamfer = chamfer_distance(output_canny, target_dist_map)

            total_loss = (loss_dice * DICE_WEIGHT + 
                          loss_ssim * SSIM_WEIGHT + 
                          loss_chamfer * CHAMFER_WEIGHT)

            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_([learnable_vector], 1.0)
            optimizer.step()

            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        print(f"\nGenerating tuned result for '{prompt}' with {NUMBER_INFERENCE_TEST_STEP} steps...")
        with torch.no_grad():
            final_modified_embeds = prompt_embeds + learnable_vector.to(torch.float16)
            test_output = pipe(
                prompt_embeds=final_modified_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                image=canny_image,
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                num_inference_steps=NUMBER_INFERENCE_TEST_STEP, 
                height=HEIGHT, width=WIDTH,
                output_type="pt"
            )
            
            test_output_img = torch.clamp(test_output.images.to(torch.float32), 0.0, 1.0)
            test_output_canny = get_pytorch_edges(test_output_img)

        results_list.append((
            prompt,
            canny_image,
            to_pil_image(test_output_img[0].cpu()),
            to_pil_image(test_output_canny[0].cpu())
        ))

    output_dir = "modified"
    os.makedirs(output_dir, exist_ok=True)
    print("Saving modified result...")
    for prompt, target_canny, gen_img, gen_canny in results_list:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Tuning: {prompt}", fontsize=14, fontweight='bold')
        
        axes[0].imshow(target_canny, cmap='gray')
        axes[0].set_title("Target (Ground Truth)")
        axes[0].axis('off')
        
        axes[1].imshow(gen_img)
        axes[1].axis('off')
        
        axes[2].imshow(gen_canny, cmap='gray')
        axes[2].set_title("Generated Edge")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        filename = f"{prompt.replace(' ', '_').lower()}_tuning.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) 
        
        print(f" -> Saved: {filename}")

    baseline_results = [] 
    baseline_dir = "baseline"
    os.makedirs(baseline_dir, exist_ok=True)

    for item in tqdm(edges_data, desc="Baseline Generation"):
        prompt = item['prompt']
        canny_image = item['canny_pil']
        
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                image=canny_image,
                controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
                num_inference_steps=NUMBER_INFERENCE_TEST_STEP, 
                height=HEIGHT,
                width=WIDTH,
                output_type="pt"
            )

            output_image = torch.clamp(output.images.to(torch.float32), 0.0, 1.0)
            output_canny = get_pytorch_edges(output_image)

            gen_img_pil = to_pil_image(output_image[0].cpu())
            gen_canny_pil = to_pil_image(output_canny[0].cpu())

            baseline_results.append((
                prompt, 
                canny_image,      
                gen_img_pil,      
                gen_canny_pil     
            ))

    for prompt, target_canny, gen_img, gen_canny in baseline_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Baseline (No Tuning): {prompt}", fontsize=14, fontweight='bold')
        
        axes[0].imshow(target_canny, cmap='gray')
        axes[0].set_title("Target Canny")
        axes[0].axis('off')
        
        axes[1].imshow(gen_img)
        axes[1].axis('off')
        
        axes[2].imshow(gen_canny, cmap='gray')
        axes[2].set_title("Generated Edge")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        filename = f"{prompt.replace(' ', '_').lower()}_baseline.png"
        plt.savefig(os.path.join(baseline_dir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)