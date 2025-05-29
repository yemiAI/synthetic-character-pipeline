import os
import json
import torch
from PIL import Image
from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

IDENTITY_TOKEN = "Male gym instructor"

class CharacterGenerator:
    def __init__(self,
                 model_path="stabilityai/stable-diffusion-xl-base-1.0",
                 lora_dir="./models/loras/identity",
                 lora_filename="lora.safetensors"):

        lora_path = os.path.join(lora_dir, lora_filename)

        if not os.path.isfile(lora_path):
            raise FileNotFoundError(f"LoRA weights not found at {lora_path}")

        # Load the SDXL pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

        # Load LoRA weights using safetensors
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

    def generate(self, context="in a forest", seed=42, steps=30):
        prompt = f"A photo of {IDENTITY_TOKEN} {context}"
        generator = torch.manual_seed(seed)
        output = self.pipe(prompt=prompt, num_inference_steps=steps, generator=generator)
        return output.images[0]


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Load prompts
    with open("inference/prompt_templates.json") as f:
        prompts = json.load(f)

    gen = CharacterGenerator()

    for i, p in enumerate(prompts):
        img = gen.generate(context=p, seed=100 + i)
        img.save(f"outputs/gen_{i}.png")
