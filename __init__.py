"""Stable Diffusion Inpainting augmentation plugin.
"""

import os
import shutil
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import numpy as np
from diffusers.utils import load_image, make_image_grid
import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
import random
import hashlib
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone.core.utils import add_sys_path


model_id = "stabilityai/stable-diffusion-2-inpainting"
id2label = {0: 'nan',1: 'accessories',2: 'bag',3: 'belt',4: 'blazer',5: 'blouse',6: 'bodysuit',7: 'boots',8: 'bra',
    9: 'bracelet',10: 'cape',11: 'cardigan',12: 'clogs',13: 'coat',14: 'dress',15: 'earrings',16: 'flats',
    17: 'glasses',18: 'gloves',19: 'hair',20: 'hat',21: 'heels',22: 'hoodie',23: 'intimate',24: 'jacket',25: 'jeans',
    26: 'jumper',27: 'leggings',28: 'loafers',29: 'necklace',30: 'panties',31: 'pants',32: 'pumps',33: 'purse',
    34: 'ring',35: 'romper',36: 'sandals',37: 'scarf',38: 'shirt',39: 'shoes',40: 'shorts',41: 'skin',42: 'skirt',
    43: 'sneakers',44: 'socks',45: 'stockings',46: 'suit',47: 'sunglasses',48: 'sweater',49: 'sweatshirt',50: 'swimwear',
    51: 't-shirt',52: 'tie',53: 'tights',54: 'top',55: 'vest',56: 'wallet',57: 'watch',58: 'wedges'}

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    # pylint: disable=no-name-in-module,import-error
    
    from utils import (
        _create_pipeline,
        _create_hash,
        _augpaint,
        _generate_inputs,
    )
    
def transform_sample(sample, select_class, prompt, num_images_per_prompt, guidance_scale, num_inference_steps, random_seed):
    
    label2id = {label: id for id, label in id2label.items()}

    hash = _create_hash()
    filename = sample.filepath.split("/")[-1][:-4]+"_"+str(hash)+".png"
    pipe = _create_pipeline(model_id)
    im,mask = _generate_inputs(
        sample.filepath, sample.ground_truth.mask_path,
        label2id[select_class])

    output_images = _augpaint(pipe, prompt, im, mask,num_images_per_prompt, guidance_scale, num_inference_steps, random_seed)
    new_samples = []
    for i,out in enumerate(output_images):
        im_saved = out.save(sample.filepath[:-4]+"_"+str(hash)+"_"+str(i)+".png")
        
        shutil.copy(sample.ground_truth.mask_path,
                sample.ground_truth.mask_path[:-4]+"_"+str(hash)+"_"+str(i)+".png",
                )

        new_samples.append(fo.Sample(
            filepath=sample.filepath[:-4]+"_"+str(hash)+"_"+str(i)+".png",
            ground_truth=fo.Segmentation(
                mask_path=sample.ground_truth.mask_path[:-4]+"_"+str(hash)+"_"+str(i)+".png"),
        ))

    return new_samples

class SDAugment(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="augment_with_sd_inpainting",
            label="Augment with Stable Diffusion Inpainting",
            description="Apply Augmentation with Stable Diffusion Inpainting Model to an image based on a mask found in image.",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Augment with Stable Diffusion Inpainting",
            description="Apply an Stable Diffusion Inpainting to the (image,mask) pair the sample.",
        )

        inputs.int(
            "num_augs",
            label="Number of augmentations per sample",
            description="The number of random augmentations to apply to each sample",
            default=1,
            view=types.FieldView(),
            required=True,
        )

        target_view = ctx.view.select(ctx.selected)
        
        for sample in target_view:
            mask = cv2.imread(sample.ground_truth.mask_path,cv2.IMREAD_GRAYSCALE)
            unique_choices = tuple([id2label[i] for i in np.unique(mask)[1:]])
            

        class_choices = types.Dropdown(label="Class")
        for clas in unique_choices:
            class_choices.add_choice(clas, label=clas)


        inputs.enum(
            "class_choices",
            class_choices.values(),
            default="skin",
            view=class_choices,
        )

        inputs.str(
            "prompt",
            label="Prompt",
            description="The prompt to generate new data from",
            required=True,
        )


        inference_steps_slider = types.SliderView(
                label="Num Inference Steps",
                componentsProps={"slider": {"min": 50, "max": 200, "step": 10}},
            )
        inputs.int(
            "inference_steps",
             default=50,
              view=inference_steps_slider
        )


        guidance_scale_slider = types.SliderView(
                label="Guidance_scale",
                componentsProps={"slider": {"min": 1, "max": 30, "step": 2}},
            )
        inputs.int(
            "guidance_scale",
             default=7,
              view=guidance_scale_slider
        )

        random_seed_slider = types.SliderView(
                label="Random_Seed",
                componentsProps={"slider": {"min": 1, "max": 1000, "step": 1}},
            )
        inputs.int(
            "random_seed",
             default=100,
              view=random_seed_slider
        )

        return types.Property(inputs, view=form_view)

    def execute(self, ctx):

        num_images_per_prompt = ctx.params.get("num_augs", 1)

        select_class = ctx.params.get("class_choices", "skin")
        
        prompt = ctx.params.get("prompt", "None provided")
        
        guidance_scale = int(ctx.params.get("guidance_scale", 7))
        
        num_inference_steps = int(ctx.params.get("num_inference_steps", 50))
        
        random_seed = int(ctx.params.get("random_seed", 100))
        

        target_view = ctx.view.select(ctx.selected)

        for sample in target_view:
            new_samples = transform_sample(sample, select_class, prompt,
                 num_images_per_prompt, guidance_scale, num_inference_steps, random_seed)
            for s in new_samples:
                sample._dataset.add_sample(s)
            break
            

        ctx.trigger("reload_dataset")


def register(plugin):

    plugin.register(SDAugment)
