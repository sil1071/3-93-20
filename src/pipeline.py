import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pipelines.models import TextToImageRequest
from diffusers import AutoencoderTiny
from diffusers import UNet2DConditionModel, LCMScheduler
from torch import Generator


def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "silencer107/beta3924-7",
        torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")
    pipeline.unet = torch.compile(pipeline.unet, mode='reduce-overhead', fullgraph=True)
    pipeline.vae = AutoencoderTiny.from_pretrained("silencer107/beta3924-7", torch_dtype=torch.float16)
    pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
    pipeline(prompt="")

    return pipeline


def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images[0]
