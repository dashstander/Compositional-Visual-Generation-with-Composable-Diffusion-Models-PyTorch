# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from PIL import Image
import torch
from typing import List
from composable_diffusion.download import load_checkpoint
from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)



def show_images(batch: torch.Tensor, file_name):
    """ Display a batch of images inline. """
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    Image.fromarray(reshaped.numpy()).save(file_name)


class Predictor(BasePredictor):
    device = 'cuda:0'

    def setup(self):
        assert torch.cuda.is_available()
        # Initial diffusion setup
        options = model_and_diffusion_defaults()
        options['use_fp16'] = True
        options['timestep_respacing'] = '100'
        self.diff_model, self.diff_sampler = create_model_and_diffusion(**options)
        self.options = options
        # Initial upsampler setup
        options_up = model_and_diffusion_defaults_upsampler()
        options_up['use_fp16'] = True
        options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
        self.up_model, self.up_sampler = create_model_and_diffusion(**options_up)
        self.options_up = options_up
        # Finish model setups
        self.diff_model.eval(), self.up_model().eval()
        self.diff_model.convert_to_fp16(), self.up_model.convert_to_fp16()
        self.diff_model.to(self.device), self.up_model.to(self.device)
        self.diff_model.load_state_dict(load_checkpoint('base', self.device))
        self.up_model.load_state_dict(load_checkpoint('upsample', self.device))


    def tokenize(self, prompts):
        tokens_list = [self.diff_model.tokenizer.encode(prompt) for prompt in prompts]
        outputs = [self.diff_model.tokenizer.padded_tokens_and_mask(
            tokens, self.options['text_ctx']
        ) for tokens in tokens_list]

        cond_tokens, cond_masks = zip(*outputs)
        cond_tokens, cond_masks = list(cond_tokens), list(cond_masks)
        uncond_tokens, uncond_mask = self.diff_model.tokenizer.padded_tokens_and_mask(
            [], self.options['text_ctx']
        )
        return cond_masks, cond_masks, uncond_tokens, uncond_mask
    

    def diffusion_sample(self, prompts: List[str], batch_size: int, full_batch_size: int, guidance_scale: float):
        cond_tokens, cond_masks, uncond_tokens, uncond_masks = self.tokenize(prompts)
        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=torch.tensor(
                cond_tokens + [uncond_tokens], device=self.device
            ),
            mask=torch.tensor(
                cond_masks + [uncond_masks],
                dtype=torch.bool,
                device=self.device,
            ),
        )

        masks = [True] * len(prompts) + [False]
        # coefficients = th.tensor([0.5, 0.5], device=device).reshape(-1, 1, 1, 1)
        masks = torch.tensor(masks, dtype=torch.bool, device=self.device)

        # Create a classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[:1]
            combined = torch.cat([half] * x_t.size(0), dim=0)
            model_out = self.diff_model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps = eps[masks].mean(dim=0, keepdim=True)
            # cond_eps = (coefficients * eps[masks]).sum(dim=0)[None]
            uncond_eps = eps[~masks].mean(dim=0, keepdim=True)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps] * x_t.size(0), dim=0)
            return torch.cat([eps, rest], dim=1)

        # Sample from the base model.
        self.diff_model.del_cache()
        samples = self.diff_sampler.p_sample_loop(
            model_fn,
            (full_batch_size, 3, self.options["image_size"], self.options["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.diff_model.del_cache()
        return samples

    def upsample(self, samples, prompts: List[str], upsample_temp: float, batch_size: int):
        tokens = self.up_model.tokenizer.encode(" ".join(prompts))
        tokens, mask = self.up_model.tokenizer.padded_tokens_and_mask(
            tokens, self.options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,

            # Text tokens
            tokens=torch.tensor(
                [tokens] * batch_size, device=self.device
            ),
            mask=torch.tensor(
                [mask] * batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
        )

        # Sample from the base model.
        self.up_model.del_cache()
        up_shape = (batch_size, 3, self.options_up["image_size"], self.options_up["image_size"])
        up_samples = self.up_sampler.ddim_sample_loop(
            self.up_model,
            up_shape,
            noise=torch.randn(up_shape, device=self.device) * upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.up_model.del_cache()
        return up_samples
    

    def predict(
        self,
        prompt: str = Input(description='Prompts that describe the image you want to create. Different prompts delimited by double bar (`||`), e.g. "A dog || A cold and rainy day"'),
        guidance_scale: float = Input(default=10),
        upsample_temp: float = Input(default=0.98),
    ) -> Path:
        prompts = [x.strip() for x in prompt.split('||')]
        batch_size = 1
        # Create the text tokens to feed to the model.
        full_batch_size = batch_size * (len(prompts) + 1)
        initial_samples = self.diffusion_sample(prompts, batch_size, full_batch_size, guidance_scale)
        final_samples = self.upsample(initial_samples, prompts, upsample_temp, batch_size)
