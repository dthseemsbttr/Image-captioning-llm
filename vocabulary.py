import razdel
import re
import torch
from collections import defaultdict


class Vectorizer:
    pad = "<PAD>"
    unk = "<UNK>"
    sos = "<SOS>"
    eos = "<EOS>"

    def __init__(self, annotations):

        words_with_dot_list = annotations.apply(
            lambda x: self.tokenize(x))
        words_with_dot = words_with_dot_list.explode()
        words = words_with_dot.apply(lambda x: re.sub(r'[^\w\s]', '', x))
        self.counts = words.value_counts()
        words = sorted(list(self.counts[self.counts >= 1].index))
        self.vocabulary = [Vectorizer.pad, Vectorizer.unk,
                           Vectorizer.sos, Vectorizer.eos, *words]

        text2seq = {word: i for i, word in enumerate(self.vocabulary)}
        self.padding_idx = text2seq[Vectorizer.pad]
        self.unknown_idx = text2seq[Vectorizer.unk]
        self.start_of_sentance_idx = text2seq[Vectorizer.sos]
        self.end_of_sentance_idx = text2seq[Vectorizer.eos]
        self.text2seq = defaultdict(lambda: self.unknown_idx,  text2seq)
        self.seq2text = {i: word for i, word in enumerate(self.vocabulary)}
        max_len = max(words_with_dot_list.apply(lambda x: len(x)))
        self.max_len = max_len + 2

    def __len__(self):
        return len(self.vocabulary)

    def tokenize(self, text):
        text_sub = re.sub(r'[^\w\s]', '', text)
        text_list = [_.text for _ in razdel.tokenize(text_sub.lower())]
        return text_list

    def encode(self, text):
        no_pad = [self.start_of_sentance_idx] + list(map(lambda x: self.text2seq.get(
            x, self.unknown_idx), self.tokenize(text))) + [self.end_of_sentance_idx]
        len_pad = self.max_len - len(no_pad)
        return torch.tensor(no_pad + [self.text2seq['<PAD>']]*len_pad)

    def decode(self, encode_text):
        with_pad = list(map(self.seq2text.get, encode_text.tolist(
        ) if not isinstance(encode_text, list) else encode_text))
        return ' '.join(list(filter(lambda x: x != '<PAD>', with_pad)))
