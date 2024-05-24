from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import process_images
from llava.eval.run_llava import load_images
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

    image_files = glob.glob(images_path)
    image_names = [os.path.basename(x).split('.')[0] for x in image_files]
    # print(f'Encoding {one_image_path}...')
    # Not sure if I should follow run_llava line 100 on
    # or model_vqa to encode the image
    images = load_images(image_files)

    image_tensors = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    image_tensors = model.encode_images(image_tensors)

    feature_file_path = os.path.join(encoded_data_path, 'llava_images.npz')
    np.savez(feature_file_path, indices=image_names, values=image_tensors.cpu().detach().numpy())

if __name__ == '__main__':
    main()
