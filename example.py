import torch
from PIL import Image

from nextstep.models.pipeline_nextstep import NextStepPipeline

device = "cuda"
dtype = torch.bfloat16

# Replace with absolute dir
vae_name_or_path = "/path/to/stepfun-ai/NextStep-1-f8ch16-Tokenizer"

## Image Edit
edit_model_name_or_path = "stepfun-ai/NextStep-1-Large-Edit"

edit_pipeline = NextStepPipeline(model_name_or_path=edit_model_name_or_path, vae_name_or_path=vae_name_or_path).to(
    device, dtype
)

cfgs = [7.5, 2.0]
hw = (512, 512)

editing_caption = (
    "<image>"
    + "Add a pirate hat to the dog's head. Change the background to a stormy sea with dark clouds. Include the text 'Captain Paws' in bold white letters at the top portion of the image."
)
positive_prompt = None
negative_prompt = "Copy original image."

input_image = Image.open("./assets/dog.jpg")
input_image = input_image.resize(hw)

output_imgs = edit_pipeline.generate_image(
    captions=editing_caption,
    images=input_image,
    num_images_per_caption=1,
    positive_prompt=positive_prompt,
    negative_prompt=negative_prompt,
    hw=hw,
    use_norm=True,
    cfg=cfgs[0],
    cfg_img=cfgs[1],
    cfg_schedule="constant",
    timesteps_shift=3.2,
    num_sampling_steps=50,
    seed=42,
)
output_imgs = output_imgs[0]
output_imgs.save(f"./test_edit.png")

## Text to image
t2i_model_name_or_path = "stepfun-ai/NextStep-1-Large"
t2_pipeline = NextStepPipeline(model_name_or_path=t2i_model_name_or_path, vae_name_or_path=vae_name_or_path).to(device, dtype)

cfg = 7.5
hw = (512, 512)

caption = "A frosty beer mug with 'CHEERS TO US' engraved on the side."
positive_prompt = "masterpiece, film grained, best quality."
negative_prompt = "lowres, bad anatomy, bad hands, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

output_imgs = t2_pipeline.generate_image(
    captions=caption,
    images=None,
    num_images_per_caption=4,
    positive_prompt=positive_prompt,
    negative_prompt=negative_prompt,
    hw=hw,
    use_norm=True,
    cfg=cfg,
    cfg_schedule="constant",
    timesteps_shift=1.0,
    num_sampling_steps=50,
    seed=42,
)
for i, img in enumerate(output_imgs):
    img.save(f"./test_t2i_{i}.png")
