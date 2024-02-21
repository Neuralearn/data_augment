import cv2
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
import random
import hashlib
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler



def _create_pipeline(model_id):
  scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
  pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id,
                                                        scheduler=scheduler,
                                                        revision="fp16",
                                                        torch_dtype=torch.float16)

  pipe = pipe.to("cuda")
  pipe.enable_xformers_memory_efficient_attention()
  return pipe

def _generate_inputs(im_path,mask_path, mask_id):
  print("the mask id is ===", mask_id)

  source_image = Image.open(im_path)
  source_image = source_image.convert("RGB")
  
  sd_mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

  out = (sd_mask+(-mask_id*np.ones_like(sd_mask)))
  mask=-(np.clip(1e10*np.multiply(out,out),a_min=0,a_max=255)-255)

  pil_image = source_image.resize((512,512))
  pil_mask = Image.fromarray(mask).resize((512,512))
  return pil_image, pil_mask

def _augpaint(pipe, prompt, pil_image, pil_mask, num_images_per_prompt, guidance_scale, num_inference_steps,random_seed):

  generator = torch.Generator(device="cuda").manual_seed(random_seed)

  encoded_images = []
  for i in range(num_images_per_prompt):
      image = pipe(prompt=prompt, guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps, generator=generator,
                    image=pil_image, mask_image=pil_mask).images[0]
      encoded_images.append(image.resize((550,825)))
  return encoded_images

def _create_hash():
  randint = random.randint(0, 100000000)
  hash = hashlib.sha256(str(randint).encode("utf-8")).hexdigest()[:10]
  return hash

