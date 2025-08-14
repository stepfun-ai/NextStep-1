import torch
import os
import argparse
from gen_pipeline import NextStepPipeline
from loguru import logger
from nextstep.model_zoos import MODEL_ZOOS

def main():
    parser = argparse.ArgumentParser(description="NextStep inference script")
    parser.add_argument(
        "--model_name_or_path",
        default="stepfun-ai/NextStep-1.1",
        type=str,
        help="Path to the model checkpoint or HuggingFace hub path"
    )
    parser.add_argument(
        "--vae_name_or_path",
        default="stepfun-ai/NextStep-1-f8ch16-Tokenizer",
        type=str,
        help="Path to the VAE checkpoint or HuggingFace hub path"
    )
    args = parser.parse_args()

    # Load model
    model_name_or_path = MODEL_ZOOS["stepfun-ai/NextStep-1.1"]
    vae_name_or_path = MODEL_ZOOS["stepfun-ai/NextStep-1-f8ch16-Tokenizer"]
    logger.info("Starting NextStepPipeline initialization...")
    pipeline = NextStepPipeline(model_name_or_path=model_name_or_path, vae_name_or_path=vae_name_or_path, device="cuda", dtype=torch.bfloat16)
    logger.info("NextStepPipeline initialized, setting model to eval mode...")
    pipeline.model.eval()
    logger.info("Model set to eval mode")

    # Set prompts
    prompt = "A photo of a dog"
    negative_prompt = "lowres, bad anatomy, bad hands, bad texts, bad layout, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

    # Generate image
    with torch.no_grad():
        images = pipeline.generate_image(
            prompt,
            hw=(512, 512),
            num_images_per_caption=1,
            positive_prompt="",
            negative_prompt=negative_prompt,
            cfg=7.5,
            cfg_img=1.0,
            cfg_schedule="constant",
            use_norm=False,
            num_sampling_steps=30,
            timesteps_shift=1.0,
            sde_type="sde",
            seed=None,
        )

    # Save image
    images[0].save("output.jpg")
    print(f"Saved image to output.jpg")

if __name__ == "__main__":
    main()