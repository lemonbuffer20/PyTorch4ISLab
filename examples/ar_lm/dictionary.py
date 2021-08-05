from typing import List
from collections import OrderedDict, Counter


class LMDictionary(object):

    def __init__(self, eos="</s>", unk="<unk>"):
        self.token_to_idx = OrderedDict()  # {"the": 0, "some": 1, ... }
        self.counter = Counter()  # {"the": 1238, "some": 978, ...}
        self.idx_to_token = list()
        self._is_finalized = False

        self.add_token(eos, n=0)
        self.add_token(unk, n=0)

        self.eos_token = eos
        self.unk_token = unk

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def add_token(self, token, n: int = 1) -> int:
        if self._is_finalized:
            raise ValueError("Dictionary is finalized, can't add token.")

        if token in self.token_to_idx:  # already exist
            idx = self.token_to_idx[token]
            self.counter[token] += n
        else:  # create new token entry
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.counter[token] = n
        return idx

    def get_token_idx(self, token: str, use_unknown: bool = True) -> int:
        if token not in self.token_to_idx:
            if use_unknown:
                token = self.unk_token  # <unk>
            else:
                raise KeyError(f"Token {token} is not in Dictionary.")
        return self.token_to_idx[token]

    def get_idx_token(self, idx: int) -> str:
        try:
            return self.idx_to_token[idx]
        except IndexError:
            raise IndexError(f"Index {idx} invalid for length {len(self.idx_to_token)}.")

    def finalize(self,
                 min_count_threshold: int = 0,
                 max_num_words_threshold: int = -1,
                 pad_to_multiple: int = 1):
        """
        If some special tokens should be in certain order, set special_token_index to be first appear.
        ex) ["pad", "unk", "eos"]  the index of each will be 0, 1, 2, respectively.
        """
        # sort by frequency, apply thresholding
        if max_num_words_threshold < 0:
            max_num_words_threshold = len(self)

        new_token_to_idx = OrderedDict()
        new_counter = Counter()
        new_idx_to_token = list()

        for token, count in self.counter.most_common(n=max_num_words_threshold):
            if len(new_token_to_idx) >= max_num_words_threshold:
                break

            if count >= min_count_threshold:
                new_token_to_idx[token] = len(new_token_to_idx)
                new_counter[token] = count
                new_idx_to_token.append(token)

        self.token_to_idx = new_token_to_idx
        self.counter = new_counter
        self.idx_to_token = new_idx_to_token
        assert len(self.token_to_idx) == len(self.counter) == len(self.idx_to_token)

        if pad_to_multiple > 1:
            i = 0
            while len(self) % pad_to_multiple != 0:
                dummy_token = f"dummy{i:03d}"
                assert dummy_token not in self.token_to_idx
                self.add_token(dummy_token, n=0)
                i += 1

        self._is_finalized = True

    def encode(self, text: str, use_unknown: bool = True) -> List[int]:
        data = []
        words = text.strip().replace("\n", "").split()
        for w in words:
            data.append(self.get_token_idx(w, use_unknown=use_unknown))
        return data

    def decode(self, sequence: List[int]) -> str:
        words = []
        for s in sequence:
            words.append(self.get_idx_token(s))
        text = " ".join(words)
        return text
