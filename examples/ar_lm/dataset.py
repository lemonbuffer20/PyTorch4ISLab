from typing import Optional, List
import os

from dictionary import LMDictionary


class PTBDataset(object):

    def __init__(self,
                 data_root: str,
                 mode: str = "train",
                 dictionary: Optional[LMDictionary] = None):
        self.data_root = data_root

        mode = mode.lower()
        if mode not in ("train", "valid", "test"):
            raise ValueError(f"PTB dataset mode {mode} invalid.")

        self.data_file = os.path.join(data_root, f"ptb.{mode}.txt")
        if not os.path.isfile(self.data_file):
            raise ValueError(f"PTB dataset file {self.data_file} not exist.")

        # setup dictionary
        if dictionary is not None:
            # assume already created dictionary is given
            self.dictionary = dictionary
        else:
            # we need to create dictionary
            dictionary = LMDictionary()
            with open(self.data_file, "r", encoding="utf-8") as f:
                eos_token = dictionary.eos_token
                for line in f:
                    words = line.strip().replace("\n", "").lower().split() + [eos_token]
                    for word in words:
                        dictionary.add_token(word)
            dictionary.finalize()
            self.dictionary = dictionary

        # tokenize
        data = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            eos_token = self.dictionary.eos_token
            for line in f:
                words = line.strip().replace("\n", "").lower().split() + [eos_token]
                for w in words:
                    data.append(self.dictionary.get_token_idx(w, use_unknown=True))
        self.data: List[int] = data

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)

    def __len__(self) -> int:
        return len(self.data)
