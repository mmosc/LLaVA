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
        model_name=get_model_name_from_path(model_path),
        # device='cpu'
    )

    image_files = glob.glob(images_path)[:3]
    image_names = [os.path.basename(x).split('.')[0] for x in image_files]

    prompt = ("You are given either the description of the poster of a movie or the description of the movie. "
              "Given this information, identify the genre of the movie.")
    print(image_files)
    # print(f'Encoding {one_image_path}...')
    # Not sure if I should follow run_llava line 100 on
    # or model_vqa to encode the image
    image_tensors = []
    image_sizes = []
    for image_file in tqdm(image_files):
        image = load_images([image_file])
        image_tensor = process_images(
            image,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        # Convert images to tokens
        # image_tensor = model.encode_images(image_tensor)
        # image_tensor = image_tensor.cpu().detach().numpy()
        # image_tensors.append(image_tensor)
        # image_sizes.append(image[0].size)

        # Tokenize the prompt
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        # Generate from both prompt and image
        with torch.inference_mode():


            print(image_tensor)
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                # do_sample=False, # True if args.temperature > 0 else False,
                # temperature=0, # args.temperature,
                # top_p=args.top_p,
                # num_beams=args.num_beams,
                # max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(outputs)

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

    # feature_file_path = os.path.join(encoded_data_path, 'llava_image_tokens.npz')
    # np.savez(feature_file_path, indices=image_names, values=image_tensors)

if __name__ == '__main__':
    main()