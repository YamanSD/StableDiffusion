from CLIP import CLIP
from VAE import Encoder, Decoder
from UNET import Diffusion

from .converter import load_from_standard_weights


def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = load_from_standard_weights(ckpt_path, device)

    encoder = Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }
