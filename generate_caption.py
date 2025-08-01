import argparse
import ast
import os
from os import path
import pickle
import sys
from os.path import isfile, join

import cv2
from PIL import Image

sys.path.append(path.abspath('../Show_and_Tell'))

import pandas as pd
from huggingface_hub import hf_hub_download
# from open_flamingo import create_model_and_transforms
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 512  # Maximum number of tokens for caption generation


def generate_flamingo(image):
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b",
        tokenizer_path="anas-awadalla/mpt-7b",
        cross_attn_every_n_layers=1,
    )
    checkpoint_path = hf_hub_download("anas-awadalla/mpt-7b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    vision_x = [image_processor(image).unsqueeze(0)]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)

    tokenizer.padding_side = "left"  # For generation padding tokens should be on the left
    lang_x = tokenizer(
        ["<image><|endofchunk|>"],  # TODO might require a different prompt
        return_tensors="pt",
    )
    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
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


def generate_caption(image, vlm="BLIP2"):
    if vlm == "BLIP2":
        generated_text = generate_blip2(image)
    elif vlm == "flamingo":
        generated_text = generate_flamingo(image)
    else:
        raise ValueError(f"Unsupported VLM: {vlm}")
    return generated_text



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm', nargs='+', required=False, choices=["BLIP2", "flamingo"], default="BLIP2")
    parser.add_argument('--vg_path', type=str, required=False, default="../VG_100K")
    parser.add_argument('--csv', type=str, required=False, default="manynames-en.tsv")
    parser.add_argument('--images', type=str, required=False, default="images")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8", sep="\t")

    for row in df.itertuples():
        image_path = os.path.join(args.vg_path, f"{row.vg_image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist, skipping...")
            continue
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        bbox_xywh = row.bbox_xywh
        bbox_xywh = ast.literal_eval(bbox_xywh)
        cropped_image = original_image[bbox_xywh[1]:bbox_xywh[1] + bbox_xywh[3],
                        bbox_xywh[0]:bbox_xywh[0] + bbox_xywh[2]]
        cv2.imwrite(f"{args.images}/{row.vg_image_id}.jpg", cropped_image)
        cv2.imwrite(f"{args.images}/{row.vg_image_id}_cropped.jpg", cropped_image)

    images = [f for f in os.listdir(args.images) if isfile(join(args.images, f))]
    for vlm in args.vlm:
        caption_dict = {}
        i = 0
        for image in images:
            original_image = cv2.imread(os.path.join(args.images, image), cv2.IMREAD_COLOR)

            caption_image = generate_caption(original_image, vlm=vlm)

            caption_dict[row.vg_object_id] = caption_image
            if i % 1000 == 1:
                with open(f'captions/{vlm}.pkl', 'wb') as f:
                    pickle.dump(caption_dict, f)

            i += 1

        with open(f'captions/{vlm}.pkl', 'wb') as b:
            pickle.dump(caption_dict, b)