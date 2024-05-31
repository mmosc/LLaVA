from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.eval.run_llava import load_images
from llava.constants import IMAGE_TOKEN_INDEX

from tqdm import tqdm

import glob
import torch
import os
import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=False,
                        default="This is the poster of a movie. What is the genre of the movie? Describe the picture. What feelings does it convey? ")
    args = parser.parse_args()

    # model_path = "liuhaotian/llava-v1.6-vicuna-34b"
    model_path = "liuhaotian/llava-v1.6-34b"

    # model_path = "liuhaotian/llava-v1.5-7b"
    images_path = "/share/hel/datasets/mmimdb/dataset/*.jpeg"
    # images_path = "/share/hel/datasets/buckethead/*.jpg"

    # images_path = "/home/marta/jku/LLaVA/data/mmimdb/dataset/*.jpeg"
    # TODO add general paths like in Hassaku
    encoded_data_path = "/share/hel/datasets/mmimdb/dataset/llava_encoded_images/"
    # encoded_data_path = "/home/marta/jku/LLaVA/data/mmimdb/dataset/llava_encoded_images/"

    tokenizer, model, image_processor, context_len = load_pretrained_model( # device_map="auto", device="cuda", use_flash_attn=False, **kwargs
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        # device='cpu'
        load_4bit=True,
    )
    image_files = glob.glob(images_path)[:5]
    image_names = [os.path.basename(x).split('.')[0] for x in image_files]

    prompt = args.prompt
    print(prompt)
    # print(image_files)
    # print(f'Encoding {one_image_path}...')
    # Not sure if I should follow run_llava line 100 on
    # or model_vqa to encode the image
    # image_tensors = []
    # image_sizes = []
    # Tokenize the prompt
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    for image_file in tqdm(image_files):
        print(image_file)
        image = load_images([image_file])
        image_size = image[0].size
        image_tensor = process_images(
            image,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)
        # Up to here it's fine since the following tensors are different
        # print(image_size, image_tensor.shape, image_tensor.mean())

        # Generate from both prompt and image
        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor,
                image_sizes=image_size,
                # do_sample=False, # True if args.temperature > 0 else False,
                # temperature=0, # args.temperature,
                # top_p=args.top_p,
                # num_beams=args.num_beams,
                max_new_tokens=512,
                use_cache=False,
                tokenizer=tokenizer,
            )
            # print(output_ids)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(outputs)

if __name__ == '__main__':
    main()
