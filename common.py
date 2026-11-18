"""Shared code for the CSCI167 Rotten Tomatoes project.

FROZEN once experiments start. Everyone imports this file so that data splits,
preprocessing, training loop, stopping rule and result logging are identical
across models. Record `COMMON_VERSION` (a hash of this file) with your results.

Usage pattern in a notebook:
    from common import *
    set_seed(seed)                      # BEFORE building model/loaders
    data = load_data()
    train_dl, val_dl, test_dl, vocab = make_vocab_loaders(data, batch_size=64, seed=seed)
    model = MyModel(len(vocab))         # forward(ids, mask) -> logits [B, 2]
    train_and_eval(model, (train_dl, val_dl, test_dl), ...)
"""
import collections
import copy
import csv
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATASET_ID = "cornell-movie-review-data/rotten_tomatoes"
PAD, UNK = 0, 1
MAX_LEN = 64            # sentences are short; truncate longer ones
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

with open(__file__, "rb") as _f:
    COMMON_VERSION = hashlib.sha256(_f.read()).hexdigest()[:8]


# ----------------------------------------------------------------- reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_info():
    """Package versions + hardware, for the report's reproducibility section."""
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "common_version": COMMON_VERSION}
    for pkg in ("datasets", "transformers", "pandas", "scikit-learn"):
        try:
            from importlib.metadata import version
            info[pkg] = version(pkg)
        except Exception:
            info[pkg] = None
    try:
        from huggingface_hub import HfApi
        info["dataset_revision"] = HfApi().dataset_info(DATASET_ID).sha
    except Exception:
        info["dataset_revision"] = None
    info["dataset_id"] = DATASET_ID
    return info


def save_env_info(owner: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    info = env_info()
    with open(os.path.join(RESULTS_DIR, f"env_{owner}.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


# ----------------------------------------------------------------------- data
def load_data():
    """Official splits: train 8530 / validation 1066 / test 1066."""
    from datasets import load_dataset
    ds = load_dataset(DATASET_ID)
    return {s: (list(ds[s]["text"]), list(ds[s]["label"]))
            for s in ("train", "validation", "test")}


def subset_train(texts, labels, frac, seed):
    """Stratified random subset of the TRAIN split only (val/test untouched)."""
    if frac >= 1.0:
        return texts, labels
    rng = random.Random(seed)
    keep = []
    for c in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == c]
        rng.shuffle(idx)
        keep += idx[:max(1, int(round(frac * len(idx))))]
    keep.sort()
    return [texts[i] for i in keep], [labels[i] for i in keep]


def build_vocab(texts, max_size=20000, min_freq=2):
    """Vocabulary from TRAIN text only (no leakage). Text is already tokenized."""
    cnt = collections.Counter(w for t in texts for w in t.split())
    itos = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common(max_size - 2) if c >= min_freq]
    return {w: i for i, w in enumerate(itos)}


def _encode(text, vocab):
    return [vocab.get(w, UNK) for w in text.split()][:MAX_LEN] or [UNK]


class _VocabDS(Dataset):
    def __init__(self, texts, labels, vocab):
        self.x = [_encode(t, vocab) for t in texts]
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _collate_vocab(batch):
    n = max(len(x) for x, _ in batch)
    ids = torch.zeros(len(batch), n, dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return {"ids": ids, "mask": ids != PAD}, y


def make_vocab_loaders(data, batch_size=64, seed=0, train_frac=1.0):
    """Loaders for from-scratch models. Batches are ({'ids','mask'}, labels)."""
    tr_x, tr_y = subset_train(*data["train"], train_frac, seed)
    vocab = build_vocab(tr_x)
    g = torch.Generator().manual_seed(seed)
    mk = lambda x, y, shuffle: DataLoader(
        _VocabDS(x, y, vocab), batch_size=batch_size, shuffle=shuffle,
        collate_fn=_collate_vocab, generator=g if shuffle else None)
    train = mk(tr_x, tr_y, True)
    val = mk(*data["validation"], False)
    test = mk(*data["test"], False)
    return train, val, test, vocab


class _TextDS(Dataset):
    def __init__(self, texts, labels):
        self.t, self.y = texts, labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.t[i], self.y[i]


def make_hf_loaders(data, tokenizer_name="distilbert-base-uncased", batch_size=32,
                    seed=0, train_frac=1.0):
    """Loaders for pretrained HF models. Batches are ({'input_ids','attention_mask'}, labels)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def collate(batch):
        enc = tok([t for t, _ in batch], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt")
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}, y

    tr_x, tr_y = subset_train(*data["train"], train_frac, seed)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(_TextDS(tr_x, tr_y), batch_size=batch_size, shuffle=True,
                       collate_fn=collate, generator=g)
    val = DataLoader(_TextDS(*data["validation"]), batch_size=batch_size, collate_fn=collate)
    test = DataLoader(_TextDS(*data["test"]), batch_size=batch_size, collate_fn=collate)
    return train, val, test, tok


class HFClassifier(nn.Module):
    """Pretrained transformer + classification head (Hugging Face).
    freeze_encoder=True trains only the classifier head; False fine-tunes everything."""

    def __init__(self, name="distilbert-base-uncased", freeze_encoder=False):
        super().__init__()
        from transformers import AutoModelForSequenceClassification
        self.m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
        if freeze_encoder:
            for n, p in self.m.named_parameters():
                p.requires_grad = n.startswith(("classifier", "pre_classifier", "score"))

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


# -------------------------------------------------------------------- training
def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    loss, preds, ys = 0.0, [], []
    for inputs, y in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs)
        loss += crit(logits, y.to(device)).item()
        preds += logits.argmax(1).cpu().tolist()
        ys += y.tolist()
    preds, ys = np.array(preds), np.array(ys)
    acc = float((preds == ys).mean())
    f1s = []  # macro F1
    for c in (0, 1):
        tp = ((preds == c) & (ys == c)).sum()
        p = tp / max(1, (preds == c).sum())
        r = tp / max(1, (ys == c).sum())
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return {"loss": loss / len(ys), "acc": acc, "f1": float(np.mean(f1s)),
            "preds": preds.tolist(), "labels": ys.tolist()}


def train_and_eval(model, loaders, run_name, model_name, seed, optimizer="adam",
                   lr=1e-3, weight_decay=0.0, momentum=0.9, schedule="constant",
                   epochs=20, patience=5, batch_size=None, train_frac=1.0,
                   extra=None, results_file="results.csv", save_checkpoint=True):
    """Train with a fixed epoch budget + early stopping on VALIDATION accuracy,
    restore the best-val weights, then evaluate once on TEST. Logs one CSV row.

    optimizer: 'adam' | 'sgd' (SGD with momentum); schedule: 'constant' | 'cosine'.
    """
    train_dl, val_dl, test_dl = loaders
    device = get_device()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(optimizer)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if schedule == "cosine" else None)
    crit = nn.CrossEntropyLoss()

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_epoch, best_state, bad = -1.0, 0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl, correct, n = 0.0, 0, 0
        for inputs, y in train_dl:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = y.to(device)
            opt.zero_grad()
            logits = model(**inputs)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        if sched:
            sched.step()
        v = evaluate(model, val_dl, device)
        hist["train_loss"].append(tl / n)
        hist["train_acc"].append(correct / n)
        hist["val_loss"].append(v["loss"])
        hist["val_acc"].append(v["acc"])
        print(f"[{run_name}] ep {ep:02d} train_loss {tl/n:.4f} train_acc {correct/n:.4f} "
              f"val_loss {v['loss']:.4f} val_acc {v['acc']:.4f}")
        if v["acc"] > best_acc:
            best_acc, best_epoch, bad = v["acc"], ep, 0
            best_state = copy.deepcopy({k: t.cpu() for k, t in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                print(f"[{run_name}] early stop at epoch {ep}")
                break
    train_seconds = time.time() - t0

    model.load_state_dict(best_state)
    test = evaluate(model, test_dl, device)          # test touched ONCE, after selection
    trainable, total = count_params(model)

    os.makedirs(os.path.join(RESULTS_DIR, "history"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "history", f"{run_name}.json"), "w") as f:
        json.dump({**hist, "best_epoch": best_epoch, "test_preds": test["preds"],
                   "test_labels": test["labels"]}, f)
    if save_checkpoint:
        os.makedirs(os.path.join(RESULTS_DIR, "checkpoints"), exist_ok=True)
        torch.save(best_state, os.path.join(RESULTS_DIR, "checkpoints", f"{run_name}.pt"))

    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "run_name": run_name, "model": model_name, "seed": seed,
           "train_frac": train_frac, "n_train": len(train_dl.dataset),
           "n_val": len(val_dl.dataset), "n_test": len(test_dl.dataset),
           "optimizer": optimizer, "lr": lr, "weight_decay": weight_decay,
           "schedule": schedule, "batch_size": batch_size or train_dl.batch_size,
           "epochs_max": epochs, "patience": patience, "epochs_run": len(hist["val_acc"]),
           "best_epoch": best_epoch, "best_val_acc": round(best_acc, 4),
           "test_acc": round(test["acc"], 4), "test_f1": round(test["f1"], 4),
           "trainable_params": trainable, "total_params": total,
           "train_seconds": round(train_seconds, 1), "device": str(device),
           "common_version": COMMON_VERSION, "extra": json.dumps(extra or {})}
    path = os.path.join(RESULTS_DIR, results_file)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{run_name}] best_val {best_acc:.4f} (ep {best_epoch}) | TEST acc {test['acc']:.4f} "
          f"f1 {test['f1']:.4f} | {train_seconds:.0f}s | trainable params {trainable:,}")
    return row


# --------------------------------------------------------------------- analysis
def plot_history(run_name, title=None):
    """Train vs validation learning curves for a finished run."""
    import matplotlib.pyplot as plt
    with open(os.path.join(RESULTS_DIR, "history", f"{run_name}.json")) as f:
        h = json.load(f)
    ep = range(1, len(h["val_acc"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, h["train_loss"], label="train"); ax[0].plot(ep, h["val_loss"], label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[1].plot(ep, h["train_acc"], label="train"); ax[1].plot(ep, h["val_acc"], label="validation")
    ax[1].axvline(h["best_epoch"], ls=":", c="gray"); ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy"); ax[1].legend()
    fig.suptitle(title or run_name); fig.tight_layout()
    return fig


def summarize(pattern="results*.csv"):
    """Merge every person's results CSV and report mean +/- std over seeds."""
    import glob
    import pandas as pd
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    keys = ["model", "train_frac", "optimizer", "lr", "weight_decay", "schedule", "batch_size", "extra"]
    g = df.groupby(keys, dropna=False)
    out = g.agg(n_seeds=("seed", "nunique"),
                val_acc=("best_val_acc", "mean"), val_std=("best_val_acc", "std"),
                test_acc=("test_acc", "mean"), test_std=("test_acc", "std"),
                test_f1=("test_f1", "mean"), params=("trainable_params", "first"),
                seconds=("train_seconds", "mean")).reset_index()
    return out.round(4)
