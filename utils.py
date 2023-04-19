import os
import json
import pickle
import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
from pathlib import Path
import streamlit as st

class Embedding:
    def __init__(self, device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base')
        self.model = AutoModel.from_pretrained('intfloat/e5-base')
        self.device = device
        self.model.to(device)
    
    @staticmethod
    def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def encode(self, text, is_question=False):
        one_record = False
        if isinstance(text, str):
            text = [text]
            one_record=True
        prefix = 'query: ' if is_question else 'passage: '
        text = [prefix + i for i in text]
        batch_dict = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            outputs = self.model(**batch_dict)
            embeddings = Embedding.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
        return embeddings[0] if one_record else embeddings


def arxiv_data_generator(path="../data/arxiv-metadata-oai-snapshot.json"):
    with open(path, "r") as datafile:
        for line in datafile:
            json_record = json.loads(line)
            try:
                text = ""
                text += f"title: {json_record['title']}\n"
                text += f"abstract: {json_record['abstract']}"
                yield text
            except:
                continue

@st.cache_resource
def get_data_embed():
    if Path("../data/paper_list.pk").is_file():
        data_list = pickle.load(open("../data/paper_list.pk", "rb"))
        data_embedding = np.load("../data/paper_embed.npy")
        return data_list, data_embedding
    else:
        data_gen = arxiv_data_generator()
        data_list = []
        for e, text in enumerate(data_gen):
            data_list.append(text)
            if e % 1000000 == 0:
                print(f"processed {e+1} papers...")
        print(f"total papers processed: {len(data_list)}")
        embedding = Embedding()
        data_embedding = np.zeros((len(data_list), 768))
        for b in tqdm(range(0, len(data_list), 256)):
            data_embedding[b:(b+256)] = embedding.encode(data_list[b:(b+256)])
        
        pickle.dump(data_list, open("../data/paper_list.pk", "wb"))
        np.save("../data/paper_embed.npy", data_embedding)
        return data_list, data_embedding

def get_query_matches(query, paper_list, paper_embed, topK=5):
    embedding = Embedding(device='cpu')
    query_embed = embedding.encode(query, is_question=True)
    similarities = query_embed.reshape(1, -1) @ paper_embed.T
    matches = np.argsort(similarities[0])[-topK:][::-1]
    result = [paper_list[i] for i in matches]
    return result