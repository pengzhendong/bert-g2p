# Copyright (c) 2024 Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial
from typing import List, Union

import torch
from g2p_mix import G2pMix
from modelscope import snapshot_download
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BertG2p:
    def __init__(self, model: str, device: str = "cpu"):
        self.device = device
        self.g2p = partial(G2pMix(tn=True).g2p, sandhi=True)

        repo_dir = snapshot_download(model)
        self.tokenizer = AutoTokenizer.from_pretrained(repo_dir)
        self.model = AutoModel.from_pretrained(repo_dir).to(self.device)
        self.model.eval()

    def normalize(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = [token for token in tokens if token != "[UNK]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def encode(self, texts: List[str], layer: int = -1):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        # hidden_states: num_hidden_layers * (batch_size, sequence_length, hidden_size)
        # use the hidden state of the last layer as default
        hidden_states = self.model(**inputs, output_hidden_states=True)["hidden_states"][layer]

        # remove [CLS], [SEP] and [PAD]
        tokens = [self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True) for ids in inputs.input_ids]
        token_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))
        hidden_states = [hidden_states[idx][1 : len(ids) + 1] for idx, ids in enumerate(token_ids)]
        return tokens, token_ids, hidden_states

    @staticmethod
    def find_bpes(words, bpes, bpe_ids):
        begin = 0
        for word in words:
            cur = ""
            for end in range(begin, len(bpes)):
                cur += bpes[end].replace("##", "")
                if cur == word["word"].lower():
                    word["bpes"] = bpes[begin : end + 1]
                    word["bpe_ids"] = bpe_ids[begin : end + 1]
                    begin = end + 1
                    break
        return words

    @staticmethod
    def match(words, bpes, bpe_ids, bpe_embeddings):
        # sum the token(BPE) embeddings to word embedding
        words = BertG2p.find_bpes(words, bpes, bpe_ids)
        bpe_embeddings = torch.split(bpe_embeddings, [len(word["bpes"]) for word in words], dim=0)
        word_embeddings = [torch.sum(embeddings, dim=0, keepdim=True) for embeddings in bpe_embeddings]
        word_embeddings = torch.cat(word_embeddings, dim=0)
        # mean the word embedding to phone embeddings
        word2phones = torch.tensor([len(word["phones"]) for word in words]).to(word_embeddings.device)
        return (word_embeddings / word2phones.unsqueeze(1)).repeat_interleave(word2phones, dim=0)

    def __call__(self, texts: Union[str, List[str]], layer: int = -1):
        is_list = not isinstance(texts, str)
        if not is_list:
            texts = [texts]
        with torch.inference_mode():
            texts = [self.normalize(text) for text in texts]
            bpes, bpe_ids, bpe_embeddings = self.encode(texts, layer)
            words = [[word.to_dict() for word in self.g2p(text)] for text in texts]
            phone_embeddings = list(map(BertG2p.match, words, bpes, bpe_ids, bpe_embeddings))
        if not is_list:
            return words[0], bpe_embeddings[0], phone_embeddings[0]
        return words, bpe_embeddings, phone_embeddings
