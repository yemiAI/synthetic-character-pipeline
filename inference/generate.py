import os
import json
import argparse
import torch
from PIL import Image
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DDIMScheduler

IDENTITY_TOKEN = (
    "A photorealistic portrait of a muscular male gym instructor, ultra-defined jawline, "
    "short styled hair, expressive hazel eyes, clear skin with visible pores, slight sweat "
    "glisten on forehead, wearing a form-fitting athletic tank top and gym shorts, clearly clothed, "
    "upper body in focus, cinematic lighting, high detail, 8k, shallow depth of field, "
    "highly detailed facial features, sharp cheekbones, professional fitness studio backdrop"
)



class CharacterGenerator:
    def __init__(self,
                 model_path="stabilityai/stable-diffusion-xl-base-1.0",
                 refiner_path="stabilityai/stable-diffusion-xl-refiner-1.0",
                 lora_dir="./models/loras/identity",
                 lora_filename="lora.safetensors",
                 use_refiner=False,
                 allow_nsfw=False):

        self.use_refiner = use_refiner
        self.allow_nsfw = allow_nsfw

        # Load the SDXL pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

        # No safety checker in SDXL
        if not self.allow_nsfw:
            print("⚠️  SDXL does not include a safety checker. Use external tools if NSFW moderation is needed.")

        # Load LoRA weights manually
        lora_path = os.path.join(lora_dir, lora_filename)
        if not os.path.isfile(lora_path):
            raise FileNotFoundError(f"LoRA weights not found at {lora_path}")

        lora_state_dict = load_file(lora_path)
        attn_processors = self.pipe.unet.attn_processors

        for name, processor in attn_processors.items():
            if hasattr(processor, "load_state_dict"):
                processor.load_state_dict({
                    k.replace(f"{name}.", ""): v
                    for k, v in lora_state_dict.items()
                    if k.startswith(name)
                }, strict=False)

        self.pipe.unet.set_attn_processor(attn_processors)

        # Optional refiner
        self.refiner = None
        if self.use_refiner:
            self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                refiner_path,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True
            ).to("cuda")
            self.refiner.scheduler = DDIMScheduler.from_config(self.refiner.scheduler.config)
            self.refiner.enable_xformers_memory_efficient_attention()

            if not self.allow_nsfw:
                print("⚠️  SDXL refiner also has no safety checker.")

    def generate(self, context="in a forest", seed=42, steps=30, denoising=0.4):
        prompt = f"A high-quality photo of {IDENTITY_TOKEN} {context}"
        generator = torch.manual_seed(seed)

        # Base generation
        #commented out to solve OOM| image = self.pipe(prompt=prompt, num_inference_steps=steps, generator=generator).images[0]
        image = self.pipe(prompt=prompt, num_inference_steps=steps, generator=generator, height=768, width=768).images[0]



        # Optional refinement
        if self.refiner:
            image = self.refiner(prompt=prompt, image=image, strength=denoising, generator=generator).images[0]

        return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_refiner", action="store_true", help="Use the SDXL refiner after generation")
    parser.add_argument("--allow_nsfw", action="store_true", help="Allow NSFW content (no moderation)")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    with open("inference/prompt_templates.json") as f:
        prompts = json.load(f)

    gen = CharacterGenerator(use_refiner=args.use_refiner, allow_nsfw=args.allow_nsfw)

    for i, p in enumerate(prompts):
        print(f"🔹 Generating image {i+1}/{len(prompts)}: '{p}'")
        img = gen.generate(context=p, seed=100 + i)
        img.save(f"outputs/gen_{i}.png")


if __name__ == "__main__":
    main()
