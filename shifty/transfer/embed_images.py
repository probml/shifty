from functools import partial
import numpy as np
from time import time
import torch
import datasets
from sentence_transformers import SentenceTransformer
from transformers import AutoFeatureExtractor, ResNetModel
from datasets import load_dataset
import typer
from typing import Optional

def process_with_clip(examples, model):
    return {'embedding': model.encode(examples['image'])}

def process_with_resnet(examples, model, feature_extractor):
    images = examples["image"]
    inputs = feature_extractor(images, return_tensors="pt") # crop and convert to pt
    with torch.no_grad():
        outputs = model(**inputs)
    embed = outputs.pooler_output
    batch_size, embed_size = embed.shape[0], embed.shape[1]
    embed = np.reshape(embed, (batch_size, embed_size))
    #examples["embed"] = 
    return {"embedding": embed }

def get_data(data_name, max_images):        
    if data_name == "cats-dogs":
        ds = load_dataset("Bingsu/Cat_and_Dog", split="train")
    else:
        raise ValueError(f"unknown data_name {data_name}")
    if max_images is not None:
        ds = ds.select(range(0, max_images))
    return ds

def get_model(model_name):
    if model_name == 'clip': 
         # https://huggingface.co/sentence-transformers/clip-ViT-B-32  
        model = SentenceTransformer('clip-ViT-B-32')
        transform = partial(process_with_clip, model=model)
    elif model_name == 'resnet':
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
        model = ResNetModel.from_pretrained("microsoft/resnet-18")
        transform = partial(process_with_resnet, model=model, feature_extractor=feature_extractor)
    else:
        raise ValueError(f"unknown model_name {model_name}")
    return model, transform

def main(model_name: str = "resnet",
        data_name: str = "cats-dogs",
        max_images: int = None,
        file_name: str = "embeddings.hf",
        hub_name: str = None):

    # before we spend time processing, let's make sure we can load and save the daya!
    ds = get_data(data_name, 2)
    print('saving initial version to ', file_name)
    ds.save_to_disk(file_name)

    # now get full dataset 
    ds = get_data(data_name, max_images)

    # get model to process data
    model, transform = get_model(model_name)
    
    # do the processing
    init_time = time()
    ds = ds.map(transform, batched=True, batch_size=32)
    ds.set_format("np", columns=["embedding"], output_all_columns=True)
    #ds.remove_columns('image')
    end_time = time()
    print('time in seconds to process the dataset: ', end_time - init_time)
    print('embedding size', ds[0]['embedding'].shape)

    print('saving final version to ', file_name)
    ds.save_to_disk(file_name)

    # Atfer saving, you can load back like this
    #ds = datasets.load_from_disk(file_name)
    # Note that formatting is lost due to a bug so you need
    #ds.set_format("np", columns=["embedding"], output_all_columns=True)

    if hub_name is not None:
        # must run huggingface-cli login
        #hub_name = 'murphyk/dogs-cats-small-clip-embedding'
        ds.push_to_hub(hub_name)

    return ds

if __name__ == "__main__":
    typer.run(main)