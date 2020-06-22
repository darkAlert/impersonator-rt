import os
import numpy as np
import torch
from lwganrt.utils import cv_utils


def load_textures(path, device=None):
    src_dir = os.path.dirname(path)
    extentions = ['.png', '.PNG']
    texture_dict = {}

    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if len(line) <= 0:
                continue

            # Get file paths:
            tex_dir = os.path.join(src_dir, line)
            tex_paths = []
            for file in os.listdir(tex_dir):
                if any(file.endswith(ext) for ext in extentions):
                    tex_paths.append(os.path.join(tex_dir, file))
            tex_paths.sort()

            # Open files:
            images = []
            for path in tex_paths:
                images.append(cv_utils.read_cv2_img(path))

            if device is not None:
                texture_dict[line] = transform(images).to(device)
            else:
                texture_dict[line] = transform(images)

    return texture_dict


def transform(images):
    transformed_images = []
    for image in images:
        image = image.astype(np.float32)
        image /= 255.0
        image = image * 2 - 1
        image = np.transpose(image, (2, 0, 1))
        transformed_images.append(image)

    images = np.stack(transformed_images, axis=0)
    images = torch.from_numpy(images)

    return images