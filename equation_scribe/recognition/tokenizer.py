"""
Simple character-level tokenizer for LaTeX/text targets.

This minimal tokenizer provides:
- vocabulary construction from an explicit character set + special tokens
- encode/decode helpers returning token ids and text
- helpers to save/load vocab as JSON (useful for training/inference)

This tokenizer is intended as a fast prototype for Vision->LaTeX experiments.
For production we recommend migrating to a SentencePiece / BPE tokenizer as discussed in the roadmap.
"""
from __future__ import annotations

import json
from typing import List, Dict, Optional


class CharTokenizer:
    def __init__(
        self,
        extra_chars: Optional[List[str]] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        # define base printable characters useful for LaTeX and body text
        base_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        base_punct = list("`~!@#$%^&*()-_=+[]{}\\|;:'\".,<>/?")
        base_whitespace = [" ", "\n", "\t"]
        chars = base_chars + base_punct + base_whitespace
        if extra_chars:
            for ch in extra_chars:
                if ch not in chars:
                    chars.append(ch)

        # default special tokens
        defaults = {
            "pad": "<pad>",
            "bos": "<s>",
            "eos": "</s>",
            "unk": "<unk>",
        }
        if special_tokens:
            defaults.update(special_tokens)

        self.special_tokens = defaults

        # build vocab: special tokens first (fixed order), then chars
        vocab = [defaults["pad"], defaults["bos"], defaults["eos"], defaults["unk"]] + chars
        self.id2tok = {i: t for i, t in enumerate(vocab)}
        self.tok2id = {t: i for i, t in self.id2tok.items()}

        # convenience properties
        self.pad_token = defaults["pad"]
        self.bos_token = defaults["bos"]
        self.eos_token = defaults["eos"]
        self.unk_token = defaults["unk"]

        self.pad_token_id = self.tok2id[self.pad_token]
        self.bos_token_id = self.tok2id[self.bos_token]
        self.eos_token_id = self.tok2id[self.eos_token]
        self.unk_token_id = self.tok2id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.id2tok)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_token_id)
        for ch in text:
            ids.append(self.tok2id.get(ch, self.unk_token_id))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        toks = [self.id2tok.get(i, self.unk_token) for i in ids]
        if strip_special:
            # remove leading BOS/EOS/PAD/UNK if present
            if toks and toks[0] == self.bos_token:
                toks = toks[1:]
            if toks and toks[-1] == self.eos_token:
                toks = toks[:-1]
            toks = [t for t in toks if t not in (self.pad_token, self.unk_token)]
        return "".join(toks)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id2tok": self.id2tok}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        id2tok = {int(k): v for k, v in data["id2tok"].items()}
        tok2id = {v: k for k, v in id2tok.items()}
        tok = cls()
        tok.id2tok = id2tok
        tok.tok2id = tok2id
        tok.pad_token = tok.id2tok[0]
        tok.bos_token = tok.id2tok[1]
        tok.eos_token = tok.id2tok[2]
        tok.unk_token = tok.id2tok[3]
        tok.pad_token_id = 0
        tok.bos_token_id = 1
        tok.eos_token_id = 2
        tok.unk_token_id = 3
        return tok
