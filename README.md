# Synthetic Character Generation Pipeline

A containerised pipeline for generating consistent Flux-style synthetic characters using Stable Diffusion XL and LoRA adapters. Built for realism, prompt adherence, and NSFW compatibility.

## Features
- ⚡ Consistent character generation via LoRA
- 🧠 Powered by Stable Diffusion XL
- 🐳 Dockerised for RunPod and local use
- 🧪 Includes basic unit and e2e tests

## Usage

Build the Docker image:
```bash
docker build -t flux-character-gen .

**Run the generator:**
docker run --rm --gpus all -v $(pwd)/outputs:/app/outputs flux-character-gen

Generated outputs will appear in the `outputs/` directory.

## Requirements

- NVIDIA GPU with compatible drivers
- Docker with GPU support
- Internet connection for initial model download

## Project Structure

- `inference/` - generation scripts and pipeline logic
- `models/loras/` - LoRA adapter configs
- `tests/` - unit and integration tests
- `outputs/` - generated images (created at runtime)

## License

MIT License (or update to your preferred license)
