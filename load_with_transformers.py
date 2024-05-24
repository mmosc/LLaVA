from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import process_images
from llava.eval.run_llava import load_images

from tqdm import tqdm

import glob
import torch
import os
import numpy as np

def main():
    model_path = "liuhaotian/llava-v1.5-7b"
    images_path = "/share/hel/datasets/mmimdb/dataset/*.jpeg"
    # images_path = "/home/marta/jku/LLaVA/data/mmimdb/dataset/*.jpeg"
    # TODO add general paths like in Hassaku
    encoded_data_path = "/share/hel/datasets/mmimdb/dataset/llava_encoded_images/"
    # encoded_data_path = "/home/marta/jku/LLaVA/data/mmimdb/dataset/llava_encoded_images/"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    del model.model.layers
    del model.lm_head

    image_files = glob.glob(images_path)[:3]
    image_names = [os.path.basename(x).split('.')[0] for x in image_files]
    # print(f'Encoding {one_image_path}...')
    # Not sure if I should follow run_llava line 100 on
    # or model_vqa to encode the image
    image_tensors = []
    for image_file in tqdm(image_files):
        image = load_images([image_file])
        image_tensor = process_images(
            image,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        image_tensor = model.encode_images(image_tensor)
        print(image_tensor.reshape(image_tensor.shape[-2],image_tensor.shape[-1]).shape)
        image_tensor = image_tensor.reshape(image_tensor.shape[-2],image_tensor.shape[-1]).mean(dim=0).cpu().detach().numpy()
        print(image_tensor.shape)
        image_tensors.append(image_tensor)

    # images = load_images(image_files)
    # print(images)
    # feature_file_path = os.path.join(encoded_data_path, 'llava_images.npz')
    # np.savez(feature_file_path, indices=image_names, values=images.cpu())
    #
    # image_tensors = process_images(
    #     images,
    #     image_processor,
    #     model.config
    # ).to(model.device, dtype=torch.float16)
    #
    # image_tensors = model.encode_images(image_tensors)

    feature_file_path = os.path.join(encoded_data_path, 'llava_image_tokens_means.npz')
    np.savez(feature_file_path, indices=image_names, values=image_tensors)

if __name__ == '__main__':
    main()
