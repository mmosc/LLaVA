from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import process_images
from llava.eval.run_llava import load_images
import glob
import torch

def main():
    model_path = "liuhaotian/llava-v1.5-7b"
    images_path = "/share/hel/datasets/mmimdb/dataset/*.jpeg"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    del model.model.layers
    del model.lm_head

    print(model)
    image_files = glob.glob(images_path)[:3]
    # print(f'Encoding {one_image_path}...')
    # Not sure if I should follow run_llava line 100 on
    # or model_vqa to encode the image
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    print(images_tensor[0])
    print(images_tensor.shape)


if __name__ == '__main__':
    main()
