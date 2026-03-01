import torch
import os
import random
import json
import warnings
from typing import Generator
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from PIL import Image

# Suppress standard Hugging Face initialization warnings for a clean console
warnings.filterwarnings("ignore")

class MangoEngine:
    """
    A custom latent diffusion inference engine optimized for CPU execution.
    Utilizes decoupled UNet, VAE, and CLIP text encoders for manual tensor manipulation.
    """
    
    def __init__(self) -> None:
        print("Initializing MangoEngine components...")
        self.device = "cpu"
        model_id = "nota-ai/bk-sdm-tiny"
        
        # Explicitly load individual components rather than using a high-level pipeline wrapper
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        print("MangoEngine initialization complete.")

    def generate_stream(self, prompt: str, output_path: str, url_path: str) -> Generator[str, None, None]:
        """
        Executes the latent diffusion denoising loop.
        Yields Server-Sent Events (SSE) formatted JSON strings to track inference progress.
        """
        yield f"data: {json.dumps({'status': 'Processing text embeddings...'})}\n\n"

        # 1. Text Embedding Generation
        text_input = self.tokenizer(
            prompt, padding="max_length", max_length=self.tokenizer.model_max_length, 
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Generate unconditional embeddings for Classifier-Free Guidance (CFG)
        uncond_input = self.tokenizer(
            [""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # 2. Latent Space Initialization
        latents = torch.randn((1, 4, 64, 64)).to(self.device)
        self.scheduler.set_timesteps(20)
        latents = latents * self.scheduler.init_noise_sigma

        # 3. Dynamic CFG Routing
        # Randomly adjust the guidance scale to vary output aesthetics
        routing_mode = random.choice(["dreamy", "standard", "crunchy"])
        if routing_mode == "dreamy":
            cfg_scale = random.uniform(4.0, 5.0)
        elif routing_mode == "standard":
            cfg_scale = 7.5
        elif routing_mode == "crunchy":
            cfg_scale = random.uniform(8.0, 9.0)

        total_steps = len(self.scheduler.timesteps)

        # 4. Inference Loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Stream current step to the client
            yield f"data: {json.dumps({'step': i + 1, 'total': total_steps, 'vibe': routing_mode.upper()})}\n\n"
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Apply Classifier-Free Guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        yield f"data: {json.dumps({'status': 'Decoding latents to RGB pixels...'})}\n\n"

        # 5. Image Decoding & Post-Processing
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).round().astype("uint8")
        
        # Save payload to OS directory
        pil_image = Image.fromarray(image)
        pil_image.save(output_path)
        
        # 6. Transmit final media path to client
        yield f"data: {json.dumps({'image_url': f'/{url_path}'})}\n\n"