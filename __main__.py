from Loader import load
from Pipeline import generate
from PIL import Image
from transformers import CLIPTokenizer
import torch


def demo() -> None:
    DEVICE: str = "cpu"
    ALLOW_CUDA: bool = False
    ALLOW_MPS: bool = False

    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
        DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
    model_file = "../data/v1-5-pruned-emaonly.ckpt"
    models = load(model_file, DEVICE)

    # TEXT TO IMAGE prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed,
    # ultra sharp, cinematic, 100mm lens, 8k resolution."
    prompt = "A cat stretching on the floor, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
    uncond_prompt = ""  # Also known as negative prompt
    do_cfg = True
    cfg_scale = 8  # min: 1, max: 14

    # IMAGE TO IMAGE
    input_image = None
    # Comment to disable image to image
    # image_path = "../images/dog.jpg"
    # input_image = Image.open(image_path)
    # Higher values means more noise will be added to the input image, so the result will further from the input image.
    # Lower values means less noise is added to the input image, so output will be closer to the input image.
    strength = 0.9

    output_image = generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name="ddpm",
        n_inference_steps=50,
        seed=42,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    # Combine the input image and the output image into a single image.
    Image.fromarray(output_image)
    return


def main() -> None:
    demo()
    return


if __name__ == "__main__":
    main()
