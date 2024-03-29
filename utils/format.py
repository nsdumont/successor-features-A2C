import os
import json
import numpy as np
import re
import torch
import torch_ac
import gymnasium as gym

import utils

def default_preprocess_obss(obss, device=None):
    return torch.tensor(obss, device=device)

def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape, "text": 0}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device), 
                "text": torch.zeros(len(obss))
                })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and ("image" in list(obs_space.spaces.keys())):
        if ("mission" in list(obs_space.spaces.keys()) and (obs_space.spaces["mission"].shape is not None)):
            obs_space = {"image": obs_space.spaces["image"].shape, "text": obs_space.spaces["mission"].shape}
        
            def preprocess_obss(obss, device=None):
                return torch_ac.DictList({
                    "image": preprocess_images([obs[0]["image"] for obs in obss], device=device),
                    "text": preprocess_images([obs[0]["mission"] for obs in obss], device=device)
                })
        else:
            obs_space = {"image": obs_space.spaces["image"].shape, "text": (100,)}

            vocab = Vocabulary(obs_space["text"][0])
            vocab.load_vocab( dict(zip( [str(i) for i in range(10)], list(np.arange(10)))) )
            def preprocess_obss(obss, device=None):
                return torch_ac.DictList({
                    "image": preprocess_images([obs["image"] for obs in obss], device=device),
                    "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
                })
            preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)



def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z1-9]+)", text.lower())
        var_indexed_text = np.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = np.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.float)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
