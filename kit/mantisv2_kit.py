"""MantisV2 kit: run the paper baseline and our three improvements on any classification dataset.

Same methods and settings as our runs on datasets 7-12:
  rf          frozen MantisV2 (layer 2, cls + mean token) + Random Forest, 200 trees   (paper protocol)
  logreg      same features + logistic regression                                     (paper Section 5.3)
  head        small trained head on the same features, channels concatenated          (no mixing, for the ablation)
  mixer       idea 1: mixer v2 = gated cross-channel attention + channel dropout
  pool        idea 2: learned mix of all 6 layers + attention pooling over tokens
  pool_mixer  idea 2 + idea 1 together
  lora        idea 3: LoRA (rank 8) on the attention weights of the encoder + mixer v2 head

Command line:
    python kit/mantisv2_kit.py --dataset Blink --source uea
    python kit/mantisv2_kit.py --dataset SleepEEG --source tfc --methods rf mixer lora
    python kit/mantisv2_kit.py --dataset MyData --source numpy --path data/MyData

From a notebook:
    from mantisv2_kit import run
    df = run('Blink', source='uea')
"""
import argparse
import json
import os
import subprocess
import time
import traceback
import zipfile

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ALL_METHODS = ['rf', 'logreg', 'head', 'mixer', 'pool', 'pool_mixer', 'lora']
SEEDS = (0, 1, 2)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# TF-C preprocessed datasets (Zhang et al. 2022) on figshare: name -> article id
TFC_ARTICLES = {'SleepEEG': 19930178, 'Epilepsy': 19930199, 'FD-A': 19930205, 'FD-B': 19930226,
                'HAR': 19930244, 'Gesture': 19930247, 'EMG': 19930250, 'ECG': 19930253}


def log(*msg):
    print(time.strftime('%H:%M:%S'), *msg, flush=True)


# ============================================================================ data

def _download(url, path):
    r = requests.get(url, timeout=900)
    r.raise_for_status()
    with open(path, 'wb') as fh:
        fh.write(r.content)


def _parse_ts(path):
    """Parse a .ts file (UEA/UCR format). Series of different lengths are resampled to the longest one."""
    X, y, in_data = [], [], False
    with open(path, encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower().startswith('@data'):
                in_data = True
                continue
            if in_data:
                *channels, label = line.split(':')
                X.append([np.array([float(v) if v not in ('?', 'NaN') else np.nan for v in ch.split(',')],
                                   dtype=np.float32) for ch in channels])
                y.append(label)
    L = max(len(ch) for row in X for ch in row)
    out = np.zeros((len(X), len(X[0]), L), dtype=np.float32)
    for i, row in enumerate(X):
        for c, ch in enumerate(row):
            if len(ch) == L:
                out[i, c] = ch
            else:  # unequal length: stretch to L
                ch = np.nan_to_num(ch)
                out[i, c] = np.interp(np.linspace(0, len(ch) - 1, L), np.arange(len(ch)), ch)
    return out, np.asarray(y)


def load_uea(name, data_dir):
    """Any dataset from timeseriesclassification.com (UEA multivariate or UCR univariate), e.g. 'Blink'."""
    folder = os.path.join(data_dir, name)
    if not os.path.exists(folder):
        zip_path = os.path.join(data_dir, f'{name}.zip')
        log(f'downloading {name} from timeseriesclassification.com')
        _download(f'https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip', zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(folder)
        os.remove(zip_path)
    return load_ts_folder(name, folder)


def load_ts_folder(name, folder):
    """A folder (searched recursively) with <name>_TRAIN.ts and <name>_TEST.ts, e.g. a zip you downloaded yourself."""
    out = {}
    for split in ['TRAIN', 'TEST']:
        found = [os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs if f == f'{name}_{split}.ts']
        if not found:
            raise FileNotFoundError(f'{name}_{split}.ts not found in {folder} (the name must match exactly)')
        out[split.lower()] = _parse_ts(found[0])
    return out


def load_tfc(name, data_dir):
    """TF-C preprocessed datasets: SleepEEG, Epilepsy, FD-A, FD-B, HAR, Gesture, EMG, ECG."""
    folder = os.path.join(data_dir, f'TFC-{name}')
    os.makedirs(folder, exist_ok=True)
    files = None
    out = {}
    for split in ['train', 'test']:
        path = os.path.join(folder, f'{split}.pt')
        if not os.path.exists(path):
            if files is None:
                files = requests.get(f'https://api.figshare.com/v2/articles/{TFC_ARTICLES[name]}', timeout=30).json()['files']
            log(f'downloading TF-C {name} {split}.pt from figshare')
            _download([f['download_url'] for f in files if f['name'] == f'{split}.pt'][0], path)
        d = torch.load(path, weights_only=False)
        X, y = np.asarray(d['samples'], dtype=np.float32), np.asarray(d['labels'])
        out[split] = (X, y)
    return out


def load_numpy(path):
    """Your own data. A folder with X_train.npy, y_train.npy, X_test.npy, y_test.npy
    (X shape: samples x channels x time, or samples x time for one channel).
    If there are only X.npy and y.npy, a stratified 75/25 train/test split with seed 0 is made."""
    f = lambda n: os.path.join(path, n)
    if os.path.exists(f('X_train.npy')):
        return {'train': (np.load(f('X_train.npy')), np.load(f('y_train.npy'), allow_pickle=True)),
                'test': (np.load(f('X_test.npy')), np.load(f('y_test.npy'), allow_pickle=True))}
    X, y = np.load(f('X.npy')), np.load(f('y.npy'), allow_pickle=True)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=0)
    return {'train': (X[tr], y[tr]), 'test': (X[te], y[te])}


def load_dataset(name, source, path=None, data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    if source == 'uea':
        d = load_uea(name, data_dir)
    elif source == 'tfc':
        d = load_tfc(name, data_dir)
    elif source == 'ts':
        d = load_ts_folder(name, path or os.path.join(data_dir, name))
    elif source == 'numpy':
        d = load_numpy(path or os.path.join(data_dir, name))
    else:
        raise ValueError("source must be 'uea', 'tfc', 'ts' or 'numpy'")
    for k, (X, y) in d.items():
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[:, None, :]
        d[k] = (X, np.asarray(y).ravel())
    return d


# ============================================================================ encoder

_NET = None


def encoder():
    """Pretrained MantisV2, layer 2, 'combined' token (the paper's best zero-shot setting). Loaded once."""
    global _NET
    if _NET is None:
        from mantis.architecture import MantisV2
        net = MantisV2(device=DEVICE, return_transf_layer=2, output_token='combined').from_pretrained('paris-noah/MantisV2')
        net.return_transf_layer, net.output_token = 2, 'combined'
        _NET = net.to(DEVICE).eval()
    return _NET


def resize(X, length=512):
    """Every series is linearly interpolated to 512 steps (paper protocol). Returns float16 to save memory."""
    out = []
    for i in range(0, len(X), 2000):
        x = torch.tensor(np.nan_to_num(np.asarray(X[i:i + 2000], dtype=np.float32)))
        out.append(F.interpolate(x, size=length, mode='linear', align_corners=False).numpy().astype(np.float16))
    return np.concatenate(out)


def features(Xr):
    """Paper features: one 512-d embedding per channel -> (N, C, 512)."""
    from mantis.trainer import MantisTrainer
    model = MantisTrainer(device=DEVICE, network=encoder())
    N, C, _ = Xr.shape
    out = [model.transform(Xr[i:i + 2000].astype(np.float32)) for i in range(0, N, 2000)]
    return np.concatenate(out).reshape(N, C, 512).astype(np.float16)


@torch.no_grad()
def tokens(Xr, bs=1024):
    """For idea 2: per channel and per layer, the cls token + the 32 patch tokens averaged in 4 groups.
    Returns (N, C, 6 layers, 5 tokens, 256) float16."""
    net = encoder()
    N, C, _ = Xr.shape
    flat = Xr.reshape(N * C, 1, 512)
    out = []
    for i in range(0, N * C, bs):
        h = net.tokgen_unit(torch.from_numpy(flat[i:i + bs]).to(DEVICE).float())
        h = torch.cat([h, net.transf_unit.cls_token.expand(h.shape[0], 1, -1)], 1)[:, None]
        layers = []
        for layer in net.transf_unit.transformer.layers:
            h = layer(h)
            t = h[:, 0]
            layers.append(torch.cat([t[:, 32:], t[:, :32].reshape(-1, 4, 8, 256).mean(2)], 1))
        out.append(torch.stack(layers, 1).half().cpu().numpy())
    return np.concatenate(out).reshape(N, C, 6, 5, 256)


# ============================================================================ models

class MixerV2(nn.Module):
    """Per-channel projection, channel dropout and (if mix) gated cross-channel attention. The gate starts at 0,
    so the model starts as the plain concatenation head and only mixes channels if that helps."""

    def __init__(self, C, K, mix, d=64, drop=0.3, chan_drop=0.1, d_in=512):
        super().__init__()
        self.mix, self.chan_drop = mix, chan_drop
        self.proj = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        if mix:
            self.chan_emb = nn.Parameter(torch.zeros(1, C, d))
            self.norm = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, 2, dropout=drop, batch_first=True)
            self.gate = nn.Parameter(torch.zeros(1))
        self.out = nn.Sequential(nn.Dropout(drop), nn.Linear(C * d, K))

    def forward(self, Z):
        z = self.proj(Z.float())
        if self.training and self.chan_drop > 0:
            z = z * (torch.rand(z.shape[0], z.shape[1], 1, device=z.device) > self.chan_drop).float()
        if self.mix:
            h = self.norm(z + self.chan_emb)
            z = z + torch.tanh(self.gate) * self.attn(h, h, h, need_weights=False)[0]
        return self.out(z.flatten(1))


class PoolHead(nn.Module):
    """Idea 2: softmax-weighted mix of the 6 layers + attention pooling (4 queries) over the tokens, then MixerV2."""

    def __init__(self, C, K, mix, L=6, heads=4):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(L))
        self.norm = nn.LayerNorm(256)
        self.query = nn.Parameter(torch.randn(heads, 256) * 0.02)
        self.head = MixerV2(C, K, mix, d_in=heads * 256)

    def forward(self, T):
        w = torch.softmax(self.layer_w, 0)
        t = (self.norm(T.float()) * w[:, None, None]).sum(2)
        a = torch.softmax(t @ self.query.T / 16, 2)
        return self.head((a.transpose(2, 3) @ t).flatten(2))


class LoRA(nn.Module):
    """Frozen linear layer + trainable low-rank update B @ A (rank r)."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


class LoraModel(nn.Module):
    """Idea 3: MantisV2 cut after layer 2, LoRA on QKV and output projections of layers 0-2, MixerV2 head."""

    def __init__(self, C, K, r=8):
        super().__init__()
        from mantis.architecture import MantisV2
        self.enc = MantisV2(device=DEVICE, return_transf_layer=2, output_token='combined')
        self.enc.load_state_dict(encoder().state_dict())
        self.enc.remove_transf_layers()
        for p in self.enc.parameters():
            p.requires_grad = False
        for layer in self.enc.transf_unit.transformer.layers:
            layer.attention.wQKV = LoRA(layer.attention.wQKV, r)
            layer.attention.wO = LoRA(layer.attention.wO, r)
        self.head = MixerV2(C, K, True)

    def forward(self, X):
        B, C, L = X.shape
        return self.head(self.enc(X.reshape(B * C, 1, L).float()).reshape(B, C, 512))


# ============================================================================ training

def fit(model, A, y, tr_idx, te_idx, seed, epochs, patience, bs, lr=1e-3, amp=False):
    """One training loop for every model: AdamW, cosine schedule, label smoothing 0.1,
    20% of the training split held out for early stopping. Returns (test accuracy, epochs run)."""
    amp = amp and DEVICE == 'cuda'
    fit_idx, va_idx = train_test_split(tr_idx, test_size=0.2, stratify=y[tr_idx], random_state=seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    batch = lambda idx: torch.from_numpy(A[idx]).to(DEVICE)

    def predict(idx):
        model.eval()
        preds = []
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.float16, enabled=amp):
            for i in range(0, len(idx), 256):
                preds.append(model(batch(idx[i:i + 256])).argmax(1).cpu().numpy())
        return np.concatenate(preds)

    best, best_state, wait = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(fit_idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(batch(b)), torch.from_numpy(y[b]).long().to(DEVICE), label_smoothing=0.1)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        va = np.mean(predict(va_idx) == y[va_idx])
        if va >= best:
            best, wait = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return float(np.mean(predict(te_idx) == y[te_idx])), ep + 1


def settings(n_train, quick=False):
    """Training settings we used: small datasets (< 10,000 training samples) vs large ones like CAP."""
    if quick:
        return dict(head=dict(epochs=3, patience=2, bs=64), pool=dict(epochs=2, patience=1, bs=64),
                    lora=dict(epochs=1, patience=1, bs=32))
    if n_train < 10000:
        return dict(head=dict(epochs=100, patience=20, bs=64), pool=dict(epochs=100, patience=20, bs=64),
                    lora=dict(epochs=30, patience=8, bs=32))
    return dict(head=dict(epochs=40, patience=8, bs=256), pool=dict(epochs=20, patience=5, bs=256),
                lora=dict(epochs=4, patience=2, bs=64))


# ============================================================================ main entry point

def run(dataset, source='uea', path=None, methods=None, seeds=SEEDS, out_dir='results', data_dir='data',
        quick=False, paper=None):
    """Run the chosen methods on one dataset and save results/<dataset>_results.csv.
    paper: optional accuracy reported in the MantisV2 paper, added to the table for comparison."""
    methods = methods or ALL_METHODS
    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, f'{dataset}_results.csv')
    t_all = time.time()
    log(f'{dataset}: device {DEVICE}' + (f' ({torch.cuda.get_device_name(0)})' if DEVICE == 'cuda' else
                                          ' - no GPU, the trained methods will be slow'))

    d = load_dataset(dataset, source, path, data_dir)
    (Xa, ya), (Xb, yb) = d['train'], d['test']
    le = LabelEncoder().fit(np.concatenate([ya, yb]))
    y = le.transform(np.concatenate([ya, yb]))
    tr, te = np.arange(len(ya)), np.arange(len(ya), len(ya) + len(yb))
    L0 = Xa.shape[2]
    Xr = resize(np.concatenate([Xa, Xb]))
    del Xa, Xb, d
    N, C, _ = Xr.shape
    K = len(le.classes_)
    S = settings(len(tr), quick)
    log(f'{dataset}: train {len(tr)}, test {len(te)}, channels {C}, original length {L0}, classes {K}')

    rows = []

    def record(method, accs, t0, **extra):
        row = dict(dataset=dataset, method=method, mean=float(np.mean(accs)), std=float(np.std(accs)),
                   per_seed=json.dumps([round(a, 4) for a in accs]), paper=paper, n_train=len(tr), n_test=len(te),
                   channels=C, classes=K, time_s=round(time.time() - t0), **extra)
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv, index=False)
        log(f'{dataset:16s} {method:12s} {row["mean"]:.4f} +/- {row["std"]:.4f}  {extra or ""}  {row["time_s"]}s')

    def safe(fn):
        try:
            fn()
        except Exception as e:
            traceback.print_exc()
            log('FAILED:', type(e).__name__, e)

    Z = None
    if {'rf', 'logreg', 'head', 'mixer'} & set(methods):
        t0 = time.time()
        Z = features(Xr)
        log(f'features {Z.shape} in {time.time() - t0:.0f}s')
        Zflat = Z.reshape(N, -1).astype(np.float32)

    if 'rf' in methods:
        def m():
            t0, accs = time.time(), []
            for s in seeds:
                rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s).fit(Zflat[tr], y[tr])
                accs.append(float(np.mean(rf.predict(Zflat[te]) == y[te])))
            record('rf', accs, t0)
        safe(m)

    if 'logreg' in methods:
        def m():
            t0 = time.time()
            sc = StandardScaler().fit(Zflat[tr])
            clf = LogisticRegression(max_iter=500).fit(sc.transform(Zflat[tr]), y[tr])
            record('logreg', [float(np.mean(clf.predict(sc.transform(Zflat[te])) == y[te]))], t0)
        safe(m)

    def heads(name, A, make, cfg, extra_fn, amp=False):
        def m():
            t0, accs, eps, extras = time.time(), [], [], []
            for s in seeds:
                torch.manual_seed(s)
                np.random.seed(s)
                model = make().to(DEVICE)
                acc, ep = fit(model, A, y, tr, te, s, amp=amp, **cfg)
                accs.append(acc)
                eps.append(ep)
                extras.append(extra_fn(model))
            record(name, accs, t0, epochs=json.dumps(eps), **{k: json.dumps([e[k] for e in extras]) for k in extras[0]})
        safe(m)

    gate = lambda h: round(torch.tanh(h.gate).item(), 3) if h.mix else 0.0
    if 'head' in methods:
        heads('head', Z, lambda: MixerV2(C, K, False), S['head'], lambda m: {})
    if 'mixer' in methods:
        heads('mixer', Z, lambda: MixerV2(C, K, True), S['head'], lambda m: {'gate': gate(m)})
    del Z

    if {'pool', 'pool_mixer'} & set(methods):
        gb = N * C * 6 * 5 * 256 * 2 / 1e9
        if gb > 20:
            log(f'skipping pool methods: tokens would need {gb:.0f} GB of RAM')
        else:
            t0 = time.time()
            T = tokens(Xr)
            log(f'tokens {T.shape} ({gb:.1f} GB) in {time.time() - t0:.0f}s')
            lw = lambda m: {'gate': gate(m.head),
                            'layer_weights': [round(v, 3) for v in torch.softmax(m.layer_w, 0).tolist()]}
            if 'pool' in methods:
                heads('pool', T, lambda: PoolHead(C, K, False), S['pool'], lw)
            if 'pool_mixer' in methods:
                heads('pool_mixer', T, lambda: PoolHead(C, K, True), S['pool'], lw)
            del T

    if 'lora' in methods:
        heads('lora', Xr, lambda: LoraModel(C, K), S['lora'], lambda m: {'gate': gate(m.head)}, amp=True)

    df = pd.DataFrame(rows)
    log(f'{dataset} done in {(time.time() - t_all) / 60:.1f} min, saved {csv}')
    print(df[['dataset', 'method', 'mean', 'std', 'per_seed']].round(4).to_string(index=False))
    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run MantisV2 baseline + our improvements on one dataset.')
    p.add_argument('--dataset', required=True, help="dataset name, e.g. Blink (uea), SleepEEG (tfc), or your folder name")
    p.add_argument('--source', default='uea', choices=['uea', 'tfc', 'ts', 'numpy'])
    p.add_argument('--path', help='folder with the .npy files (source numpy) or the .ts files (source ts)')
    p.add_argument('--methods', nargs='+', default=ALL_METHODS, choices=ALL_METHODS)
    p.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    p.add_argument('--paper', type=float, help='accuracy reported in the paper, stored in the CSV')
    p.add_argument('--out', default='results')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--quick', action='store_true', help='1-3 epochs only, to check that everything runs')
    a = p.parse_args()
    run(a.dataset, a.source, a.path, a.methods, a.seeds, a.out, a.data_dir, a.quick, a.paper)
