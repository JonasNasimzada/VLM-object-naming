import argparse
import ast
import os
import pickle
import sys
from os import path
from os.path import isfile, join

import cv2
from PIL import Image
from transformers import BitsAndBytesConfig, pipeline, AutoModelForCausalLM, GenerationConfig

sys.path.append(path.abspath('../Show_and_Tell'))

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 512  # Maximum number of tokens for caption generation


def generate_flamingo(image):
    from open_flamingo import create_model_and_transforms
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=1,

    )
    model.to(device)
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    vision_x = [image_processor(image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    lang_x = tokenizer(
        ["<image>\nWhat is shown in this image?<|endofchunk|>"],  # TODO might require a different prompt
        return_tensors="pt",
    )
    generated_text = model.generate(
        vision_x=vision_x.to(device),
        lang_x=lang_x["input_ids"].to(device),
        attention_mask=lang_x["attention_mask"].to(device),
        max_new_tokens=MAX_TOKENS,
        num_beams=3,
    )
    prompt = tokenizer.decode(generated_text[0])

    return prompt


def generate_blip2(image):
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    model.to(device)
    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def generate_llava(image):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model_id = "llava-hf/llava-1.5-7b-hf"
    pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})
    prompt = "USER: <image>\nWhat is shown in this image?\nASSISTANT:"
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": MAX_TOKENS})
    return outputs[0]["generated_text"]


def generate_git(image):
    processor = AutoProcessor.from_pretrained("microsoft/git-base-textcaps")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textcaps")
    model.to(device)
    pixel_values = pixel_values.to(device)

    generated_ids = model.generate(pixel_values=pixel_values, max_length=MAX_TOKENS)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def generate_molmo(image):
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-O-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-O-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )
    inputs = processor.process(
        images=[image],
        text="Describe this image."
    )
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=MAX_TOKENS, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text


def generate_caption(image, vlm_model="BLIP2"):
    if vlm_model == "BLIP2":
        generated_text = generate_blip2(image)
    elif vlm_model == "flamingo":
        generated_text = generate_flamingo(image)
    elif vlm_model == "llava":
        generated_text = generate_llava(image)
    elif vlm_model == "git":
        generated_text = generate_git(image)
    elif vlm_model == "molmo":
        generated_text = generate_molmo(image)
    else:
        raise ValueError(f"Unsupported VLM: {vlm_model}")
    return generated_text


def crop_images(dataframe):
    for row in dataframe.itertuples():
        image = os.path.join(args.vg_path, f"{row.vg_image_id}.jpg")
        if not os.path.exists(image):
            print(f"Image {image} does not exist, skipping...")
            continue
        full_image = cv2.imread(image, cv2.IMREAD_COLOR)

        bbox_xywh = row.bbox_xywh
        bbox_xywh = ast.literal_eval(bbox_xywh)
        cropped_image = full_image[bbox_xywh[1]:bbox_xywh[1] + bbox_xywh[3],
        bbox_xywh[0]:bbox_xywh[0] + bbox_xywh[2]]
        cv2.imwrite(f"{args.images}/{row.vg_image_id}.jpg", full_image)
        cv2.imwrite(f"{args.images}/{row.vg_image_id}_cropped.jpg", cropped_image)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm', nargs='+', required=False, choices=["BLIP2", "flamingo", "llava", "git", "molmo"],
                        help="Specify the VLM model to use for caption generation.",
                        default="BLIP2")
    parser.add_argument('--vg_path', type=str, required=False, default="../VG_100K")
    parser.add_argument('--csv', type=str, required=False, default="manynames-en.tsv")
    parser.add_argument('--images', type=str, required=False, default="images")
    parser.add_argument('--crop_images', type=bool, required=False, default=False)
    args = parser.parse_args()

    if args.crop_images:
        df = pd.read_csv(args.csv, encoding="utf-8", sep="\t")
        crop_images(df)

    images = [f for f in os.listdir(args.images) if isfile(join(args.images, f))]
    for vlm in args.vlm:
        caption_dict = {}
        i = 0
        for image_path in images:
            original_image = Image.open(os.path.join(args.images, image_path))

            caption_image = generate_caption(original_image, vlm_model=vlm)

            caption_dict[image_path] = caption_image
            if i % 1000 == 1:
                with open(f'captions/{vlm}.pkl', 'wb') as f:
                    pickle.dump(caption_dict, f)
                print("image number:", i, "image:", image_path)

            i += 1

        with open(f'captions/{vlm}.pkl', 'wb') as b:
            pickle.dump(caption_dict, b)
