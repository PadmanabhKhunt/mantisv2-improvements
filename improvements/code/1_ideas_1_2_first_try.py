# MantisV2 improvement: learned token pooling (idea 2) + cross-channel mixer (idea 1) on the frozen encoder.
# Same code as the Kaggle notebook "MantisV2 Improvement - Channel Mixer and Token Pooling", one block per cell.

# %% cell 1: setup
# !pip install -q mantis-tsfm
import os, time, json, zipfile, subprocess, requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from mantis.architecture import MantisV2
from mantis.trainer import MantisTrainer

device = 'cuda'
DATA, OUT = '/tmp/data', '/kaggle/working'
os.makedirs(DATA, exist_ok=True)
net = MantisV2(device=device, return_transf_layer=2, output_token='combined').from_pretrained('paris-noah/MantisV2')
net.return_transf_layer, net.output_token = 2, 'combined'
net.eval()
print(torch.cuda.get_device_name(0))

# %% cell 2: data loaders (same as the reproduction notebook)
def load_ts_archive(name):
    folder = f'{DATA}/{name}'
    if not os.path.exists(folder):
        subprocess.run(['wget', '-q', '-O', f'{DATA}/{name}.zip',
                        f'https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip'], check=True)
        with zipfile.ZipFile(f'{DATA}/{name}.zip') as z:
            z.extractall(folder)
    out = {}
    for split in ['TRAIN', 'TEST']:
        path = [os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs if f == f'{name}_{split}.ts'][0]
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
                    X.append([[float(v) if v != '?' else np.nan for v in ch.split(',')] for ch in channels])
                    y.append(label)
        out[split.lower()] = (np.asarray(X, dtype=np.float32), np.asarray(y))
    return out

TFC = {'HAR': 19930244, 'Epilepsy': 19930199}

def load_tfc(name):
    folder = f'{DATA}/{name}'
    os.makedirs(folder, exist_ok=True)
    files = requests.get(f'https://api.figshare.com/v2/articles/{TFC[name]}', timeout=30).json()['files']
    out = {}
    for split in ['train', 'test']:
        path = f'{folder}/{split}.pt'
        if not os.path.exists(path):
            url = [f['download_url'] for f in files if f['name'] == f'{split}.pt'][0]
            with open(path, 'wb') as fh:
                fh.write(requests.get(url, timeout=600).content)
        d = torch.load(path, weights_only=False)
        X, y = np.asarray(d['samples'], dtype=np.float32), np.asarray(d['labels'])
        if X.ndim == 2:
            X = X[:, None, :]
        out[split] = (X, y)
    return out

# %% cell 3: frozen encoder -> all tokens of all 6 layers
@torch.no_grad()
def tokens(X, bs=512):
    """Returns (N, C, 6 layers, 33 tokens, 256) in float16. Tokens 0-31 are patches, token 32 is the cls token."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    N, C, _ = X.shape
    X = F.interpolate(torch.tensor(X), size=512, mode='linear', align_corners=False).reshape(N * C, 1, 512)
    out = []
    for i in range(0, N * C, bs):
        h = net.tokgen_unit(X[i:i + bs].to(device))
        cls = net.transf_unit.cls_token.expand(h.shape[0], 1, -1)
        h = torch.cat([h, cls], 1)[:, None]
        layers = []
        for layer in net.transf_unit.transformer.layers:
            h = layer(h)
            layers.append(h[:, 0])
        out.append(torch.stack(layers, 1).half().cpu())
    return torch.cat(out).reshape(N, C, 6, 33, 256)

def fixed_features(T):
    """The paper's features: layer 2, cls token + mean of patch tokens, channels concatenated."""
    t = T[:, :, 2].float()
    return torch.cat([t[:, :, 32], t[:, :, :32].mean(2)], -1).reshape(len(T), -1).numpy()

# sanity check: our token extraction must give exactly the official features
Xc = np.random.randn(8, 2, 300).astype(np.float32)
official = MantisTrainer(device=device, network=net).transform(
    F.interpolate(torch.tensor(Xc), size=512, mode='linear', align_corners=False).numpy())
print('max difference to official features:', np.abs(fixed_features(tokens(Xc)) - official).max())

# %% cell 4: the new head
class Head(nn.Module):
    """pool='fixed': paper aggregation (layer 2, cls + mean token). pool='learned': softmax-weighted mix of the
    6 layers + attention pooling over the 33 tokens (idea 2).
    chan='concat': channels concatenated as in the paper. chan='mixer': transformer across channels (idea 1)."""
    def __init__(self, C, K, pool, chan, d=128, heads=4, drop=0.2):
        super().__init__()
        self.pool, self.chan = pool, chan
        if pool == 'fixed':
            self.proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
        else:
            self.layer_w = nn.Parameter(torch.zeros(6))
            self.norm = nn.LayerNorm(256)
            self.query = nn.Parameter(torch.randn(heads, 256) * 0.02)
            self.proj = nn.Sequential(nn.LayerNorm(heads * 256), nn.Linear(heads * 256, d))
        if chan == 'concat':
            self.out = nn.Sequential(nn.Dropout(drop), nn.Linear(C * d, K))
        else:
            self.chan_emb = nn.Parameter(torch.randn(1, C, d) * 0.02)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True)
            self.mixer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, K))

    def forward(self, T):
        T = T.float()
        if self.pool == 'fixed':
            t = T[:, :, 2]
            z = torch.cat([t[:, :, 32], t[:, :, :32].mean(2)], -1)
        else:
            w = torch.softmax(self.layer_w, 0)
            t = (self.norm(T) * w[:, None, None]).sum(2)
            a = torch.softmax(t @ self.query.T / 16, 2)
            z = (a.transpose(2, 3) @ t).flatten(2)
        z = self.proj(z)
        if self.chan == 'concat':
            return self.out(z.flatten(1))
        z = torch.cat([self.cls.expand(len(z), -1, -1), z + self.chan_emb], 1)
        return self.out(self.mixer(z)[:, 0])

# %% cell 5: training (20% of train held out for early stopping) and the logistic regression baseline
def train_head(Ttr, ytr, Tte, yte, C, K, pool, chan, seed, epochs=100, patience=20, bs=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx_tr, idx_va = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=seed)
    model = Head(C, K, pool, chan).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    def predict(T):
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(T), 512):
                preds.append(model(T[i:i + 512].to(device)).argmax(1).cpu())
        return torch.cat(preds).numpy()

    best, best_state, wait = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(idx_tr)
        for i in range(0, len(perm), bs):
            b = torch.tensor(perm[i:i + bs])
            loss = F.cross_entropy(model(Ttr[b].to(device)), ytr_t[b].to(device), label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        va = np.mean(predict(Ttr[torch.tensor(idx_va)]) == ytr[idx_va])
        if va >= best:
            best, wait = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    return float(np.mean(predict(Tte) == yte)), ep + 1, n_params

def logreg(Ztr, ytr, Zte, yte):
    s = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=500).fit(s.transform(Ztr), ytr)
    return float(np.mean(clf.predict(s.transform(Zte)) == yte))

# %% cell 6: run
RF = {'FingerMovements': 0.5500, 'Blink': 0.9956, 'UCI-HAR': 0.9013, 'Epilepsy-EEG': 0.9558}
CONFIGS = [('fixed', 'concat'), ('learned', 'concat'), ('fixed', 'mixer'), ('learned', 'mixer')]
rows = []

def run(name, Xtr, ytr, Xte, yte):
    le = LabelEncoder().fit(ytr)
    ytr, yte = le.transform(ytr), le.transform(yte)
    t0 = time.time()
    Ttr, Tte = tokens(Xtr), tokens(Xte)
    print(f'{name}: tokens {tuple(Ttr.shape)} in {time.time() - t0:.0f}s', flush=True)
    C, K = Ttr.shape[1], len(le.classes_)
    rows.append(dict(dataset=name, method='RF (paper protocol, reproduced)', mean=RF[name]))
    acc = logreg(fixed_features(Ttr), ytr, fixed_features(Tte), yte)
    rows.append(dict(dataset=name, method='LogReg (paper Sec 5.3)', mean=acc, std=0.0))
    print(f'  LogReg {acc:.4f}', flush=True)
    for pool, chan in CONFIGS:
        t0 = time.time()
        res = [train_head(Ttr, ytr, Tte, yte, C, K, pool, chan, s) for s in (0, 1, 2)]
        accs = [r[0] for r in res]
        rows.append(dict(dataset=name, method=f'head pool={pool} chan={chan}', mean=np.mean(accs), std=np.std(accs),
                         per_seed=json.dumps(accs), epochs=json.dumps([r[1] for r in res]), params=res[0][2],
                         time_s=round(time.time() - t0)))
        print(f'  {pool:7s} {chan:6s} {np.mean(accs):.4f} +/- {np.std(accs):.4f} {accs} '
              f'params {res[0][2]} {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/improvement_results.csv', index=False)

def safe(fn):
    try:
        fn()
    except Exception as e:
        print('FAILED:', type(e).__name__, e, flush=True)

for name in ['FingerMovements', 'Blink']:
    safe(lambda: run(name, *load_ts_archive(name)['train'], *load_ts_archive(name)['test']))
for folder, name in [('HAR', 'UCI-HAR'), ('Epilepsy', 'Epilepsy-EEG')]:
    safe(lambda: run(name, *load_tfc(folder)['train'], *load_tfc(folder)['test']))
print(pd.DataFrame(rows)[['dataset', 'method', 'mean', 'std']].round(4).to_string(index=False))
