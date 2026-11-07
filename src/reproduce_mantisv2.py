"""Reproduce the MantisV2 column of Tables 2 and 3 (arXiv:2602.17868) for datasets 7-12:
UCI-HAR, Blink, CAP-ID, CAP-OOD, Epilepsy-EEG, FingerMovements.

Protocol (paper Sections 3-4 + official README, github.com/vfeofanov/mantis):
  * every series is linearly interpolated to length 512
  * frozen MantisV2 encoder (paris-noah/MantisV2), output of transformer layer 2 (0-indexed),
    output_token='combined' (cls token concatenated with the mean of the other tokens) -> 512 dims per channel
  * channels are encoded independently and the embeddings concatenated
  * Random Forest, 200 trees, unlimited depth, 3 seeds (random_state 0, 1, 2), accuracy on the test split

Usage:
    python reproduce_mantisv2.py                      # all six datasets
    python reproduce_mantisv2.py --datasets Blink FingerMovements
"""
import argparse
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"  # datasets are downloaded here (not in git)
CACHE = DATA / "embeddings"
RESULTS = HERE.parent / "results"

# MantisV2 column of Tables 2 and 3 (mean, std over 3 seeds)
PAPER = {
    "UCI-HAR": (0.9013, 0.002),
    "Blink": (0.9956, 0.002),
    "CAP-ID": (0.8152, 0.001),
    "CAP-OOD": (0.7859, 0.000),
    "Epilepsy-EEG": (0.9558, 0.001),
    "FingerMovements": (0.5500, 0.010),
}
SEEDS = (0, 1, 2)

# TF-C preprocessed data (github.com/mims-harvard/TFC-pretraining), figshare article ids.
# The ndownloader zip links return a bot-check page to scripts, so the files come from the figshare API.
TFC_ARTICLES = {"HAR": 19930244, "Epilepsy": 19930199}
# WOODS preprocessed CAP (github.com/jc-audet/WOODS, woods/download.py)
CAP_GDRIVE_ID = "1NFwX2CqLrenWF4az0c6J-OglAoD48PAT"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------- model

def load_model():
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer

    layer, token = 2, "combined"
    network = MantisV2(device="cpu", return_transf_layer=layer, output_token=token)
    network = network.from_pretrained("paris-noah/MantisV2")
    network.return_transf_layer, network.output_token = layer, token  # as in getting_started/intermediate_layers.ipynb
    model = MantisTrainer(device="cpu", network=network)
    dim = model.transform(np.random.randn(2, 1, 512).astype(np.float32)).shape[1]
    log(f"MantisV2 loaded: {sum(p.numel() for p in network.parameters()) / 1e6:.2f}M params, {dim} dims per channel")
    return model


def resize(X, length=512):
    return F.interpolate(torch.tensor(X, dtype=torch.float), size=length, mode="linear", align_corners=False).numpy()


def embed(model, X, batch_size=512):
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    return model.transform(resize(X), batch_size=batch_size)


# ----------------------------------------------------------------------------- data loaders

def load_ts_archive(name):
    """<name>.zip from timeseriesclassification.com, equal-length .ts files parsed directly
    (aeon's loader needs numba, which does not support NumPy 2.4)."""
    folder = DATA / name
    if not folder.exists():
        zip_path = DATA / f"{name}.zip"
        r = requests.get(f"https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip", timeout=300)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(folder)
        zip_path.unlink()
    out = {}
    for split in ["TRAIN", "TEST"]:
        path = next(folder.rglob(f"{name}_{split}.ts"))
        X, y, in_data = [], [], False
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith("@data"):
                    in_data = True
                    continue
                if in_data:
                    *channels, label = line.split(":")
                    X.append([[float(v) if v != "?" else np.nan for v in ch.split(",")] for ch in channels])
                    y.append(label)
        out[split.lower()] = (np.asarray(X, dtype=np.float32), np.asarray(y))
    return out


def load_tfc(name):
    folder = DATA / name
    folder.mkdir(parents=True, exist_ok=True)
    files = None
    out = {}
    for split in ["train", "test"]:
        path = folder / f"{split}.pt"
        if not path.exists():
            if files is None:
                files = requests.get(f"https://api.figshare.com/v2/articles/{TFC_ARTICLES[name]}", timeout=30).json()["files"]
            url = next(f["download_url"] for f in files if f["name"] == f"{split}.pt")
            path.write_bytes(requests.get(url, timeout=600).content)
        d = torch.load(path, weights_only=False)
        X, y = np.asarray(d["samples"], dtype=np.float32), np.asarray(d["labels"])
        if X.ndim == 2:
            X = X[:, None, :]
        out[split] = (X, y)
    return out


def cap_h5_path():
    folder = DATA / "CAP"
    h5 = list(folder.rglob("*.h5")) if folder.exists() else []
    if h5:
        return h5[0]
    import gdown

    folder.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / "cap.zip"
    if not zip_path.exists():
        log("downloading WOODS CAP (8.7 GB)...")
        gdown.download(id=CAP_GDRIVE_ID, output=str(zip_path), quiet=False)
    log("extracting CAP...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(folder)
    zip_path.unlink()
    return next(folder.rglob("*.h5"))


def cap_embeddings(model, chunk=1000):
    """Embed each recording machine once (cached to disk); reused by both CAP splits."""
    import h5py

    CACHE.mkdir(parents=True, exist_ok=True)
    Z, y = {}, {}
    with h5py.File(cap_h5_path(), "r") as h:
        for m in sorted(h.keys()):
            cache = CACHE / f"CAP_{m}.npz"
            if cache.exists():
                d = np.load(cache)
                Z[m], y[m] = d["Z"], d["y"]
            else:
                t0 = time.time()
                data, parts = h[f"{m}/data"], []
                for start in range(0, data.shape[0], chunk):
                    X = np.transpose(data[start:start + chunk], (0, 2, 1))  # (n, 3000, 19) -> (n, 19, 3000)
                    parts.append(embed(model, X))
                    log(f"  {m}: {min(start + chunk, data.shape[0])}/{data.shape[0]}")
                Z[m], y[m] = np.concatenate(parts), h[f"{m}/labels"][:].ravel()
                np.savez(cache, Z=Z[m], y=y[m])
                log(f"{m}: embedded {Z[m].shape} in {time.time() - t0:.0f}s")
    return Z, y


# ----------------------------------------------------------------------------- evaluation

def random_forest(name, Z_train, y_train, Z_test, y_test, embed_s):
    accs, t0 = [], time.time()
    for seed in SEEDS:
        rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=seed)
        rf.fit(Z_train, y_train)
        accs.append(float(np.mean(rf.predict(Z_test) == np.asarray(y_test))))
        log(f"  {name} seed {seed}: {accs[-1]:.4f}")
    paper_mean, paper_std = PAPER[name]
    row = dict(dataset=name, ours_mean=np.mean(accs), ours_std=np.std(accs), paper_mean=paper_mean,
               paper_std=paper_std, diff=np.mean(accs) - paper_mean, per_seed=json.dumps(accs),
               n_train=len(y_train), n_test=len(y_test), feat_dim=Z_train.shape[1],
               embed_s=round(embed_s, 1), rf_s=round(time.time() - t0, 1))
    log(f"{name}: ours {row['ours_mean']:.4f} +/- {row['ours_std']:.4f} | paper {paper_mean:.4f} +/- {paper_std:.3f} "
        f"| diff {row['diff']:+.4f}")
    return row


def evaluate_raw(model, name, X_train, y_train, X_test, y_test):
    log(f"{name}: train {X_train.shape}, test {X_test.shape}, classes {len(np.unique(y_train))}")
    t0 = time.time()
    Z_train, Z_test = embed(model, X_train), embed(model, X_test)
    return random_forest(name, Z_train, y_train, Z_test, y_test, time.time() - t0)


def run(names):
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = load_model()
    rows = []

    if "FingerMovements" in names:
        d = load_ts_archive("FingerMovements")
        rows.append(evaluate_raw(model, "FingerMovements", *d["train"], *d["test"]))
    if "Blink" in names:
        d = load_ts_archive("Blink")
        rows.append(evaluate_raw(model, "Blink", *d["train"], *d["test"]))
    if "UCI-HAR" in names:
        d = load_tfc("HAR")
        rows.append(evaluate_raw(model, "UCI-HAR", *d["train"], *d["test"]))
    if "Epilepsy-EEG" in names:
        d = load_tfc("Epilepsy")
        rows.append(evaluate_raw(model, "Epilepsy-EEG", *d["train"], *d["test"]))

    if {"CAP-ID", "CAP-OOD"} & set(names):
        t0 = time.time()
        Z, y = cap_embeddings(model)
        embed_s = time.time() - t0
        machines = sorted(Z)
        if "CAP-ID" in names:
            # Paper: merge all domains, 63.75% / 11.25% / 25% train/val/test. Seed and stratification are not
            # reported; we use a plain random permutation with seed 0 (sizes match the paper: 25748 / 10098).
            Z_all, y_all = np.concatenate([Z[m] for m in machines]), np.concatenate([y[m] for m in machines])
            n = len(y_all)
            perm = np.random.RandomState(0).permutation(n)
            n_train, n_test = int(0.6375 * n), int(round(0.25 * n))
            tr, te = perm[:n_train], perm[n - n_test:]
            log(f"CAP-ID: total {n}, train {len(tr)}, val (unused) {n - n_train - n_test}, test {len(te)}")
            rows.append(random_forest("CAP-ID", Z_all[tr], y_all[tr], Z_all[te], y_all[te], embed_s))
            del Z_all
        if "CAP-OOD" in names:
            # Paper: Machines 0, 1, 3 train, Machine 2 validation (unused), Machine 4 test
            train_m, test_m = ["Machine0", "Machine1", "Machine3"], ["Machine4"]
            Z_tr, y_tr = np.concatenate([Z[m] for m in train_m]), np.concatenate([y[m] for m in train_m])
            Z_te, y_te = np.concatenate([Z[m] for m in test_m]), np.concatenate([y[m] for m in test_m])
            log(f"CAP-OOD: train {len(y_tr)}, test {len(y_te)}")
            rows.append(random_forest("CAP-OOD", Z_tr, y_tr, Z_te, y_te, embed_s))

    # Merge with results from earlier runs so partial runs build up one table
    out = RESULTS / "results_datasets_7_12.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old.dataset.isin(df.dataset)], df])
    df["order"] = df.dataset.map(list(PAPER).index)
    df = df.sort_values("order").drop(columns="order")
    df.to_csv(out, index=False)
    log("results saved to " + str(out))
    print(df[["dataset", "ours_mean", "ours_std", "paper_mean", "paper_std", "diff"]].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(PAPER), choices=list(PAPER))
    run(parser.parse_args().datasets)
