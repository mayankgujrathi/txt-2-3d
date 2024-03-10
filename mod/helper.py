import os, torch
from mod.diff.sample import sample_latents
from mod.diff.gaussian_diffusion import diffusion_from_config
from mod.prebuild.download import load_config, load_model
from mod.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget, decode_latent_mesh

class Client:

  def __init__(self, show_device=True):
    self.__xm = None
    self.__ldm = None
    self.__diff = None
    self.__latent = None
    self.__guidance_scale = 15.0
    self.__cur_prompt: str = None
    self.device: str = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if show_device:
      print("Using", self.device)

  def __get_folder_path(self):
    if self.__cur_prompt is None:
      return '.'
    return self.__cur_prompt.replace(" ", "_")
  
  def load_weights(self):
    self.__xm = load_model('transmitter', device=self.device)
    self.__ldm = load_model('text300M', device=self.device)
    self.__diff = diffusion_from_config(load_config('diffusion'))

  def prompt(self, text: str, num_sample: int = 1, steps: int = 128):
    self.__cur_prompt = text
    self.__latent = sample_latents(
        batch_size=num_sample,
        model=self.__ldm,
        diffusion=self.__diff,
        guidance_scale=self.__guidance_scale,
        model_kwargs=dict(texts=[self.__cur_prompt] * num_sample),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
      )
  
  def plot_nerf(self, dim: int = 64):
    import matplotlib.pyplot as plt
    cameras = create_pan_cameras(dim, self.device)
    for i, latent in enumerate(self.__latent):
        images = decode_latent_images(self.__xm, latent, cameras, rendering_mode='nerf')
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i])
            ax.axis('off')
        plt.tight_layout()
        plt.show()

  def render_nerf(self, dim: int = 64):
    cameras = create_pan_cameras(dim, self.device)
    cur_folder = self.__get_folder_path()
    for i, latent in enumerate(self.__latent):
        images = decode_latent_images(self.__xm, latent, cameras, rendering_mode='nerf')
        os.makedirs(f"{cur_folder}_{i}/", exist_ok=True)
        for idx, image in enumerate(images):
          image.save(f"{cur_folder}_{i}/image_{idx}.png", bit_format='png')
        display(gif_widget(images))

  def save_ply(self):
    cur_folder = self.__get_folder_path()
    for i, latent in enumerate(self.__latent):
        t = decode_latent_mesh(self.__xm, latent).tri_mesh()
        with open(f'{cur_folder}_{i}/example_mesh_{i}.ply', 'wb') as f:
            t.write_ply(f)

  def save_obj(self):
    cur_folder = self.__get_folder_path()
    for i, latent in enumerate(self.__latent):
        t = decode_latent_mesh(self.__xm, latent).tri_mesh()
        with open(f'{cur_folder}_{i}/example_mesh_{i}.obj', 'w') as f:
            t.write_obj(f)
