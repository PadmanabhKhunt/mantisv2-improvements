# Mixer v2 cells, added to the Kaggle notebook after cells 1-5 of improvement_head_cells.py (the old run cell is removed).

# %% cell A: mixer v2
# Fixes for mixer v1 (too big, overfits on small data):
# 1) residual + gate that starts at 0: the model starts as the plain concat head and only uses mixing if it helps
# 2) smaller: d=64, 2 heads, one attention layer, no feed-forward block
# 3) channel dropout: whole channels are blanked at random while training, so the model cannot rely on one channel
emb = MantisTrainer(device=device, network=net)

def fixed(X):
    """Paper features kept per channel: (N, C, 512) in float16."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    N, C, _ = X.shape
    X = F.interpolate(torch.tensor(X), size=512, mode='linear', align_corners=False).numpy()
    return emb.transform(X).reshape(N, C, 512).astype(np.float16)

class MixerV2(nn.Module):
    def __init__(self, C, K, mix, d=64, drop=0.3, chan_drop=0.1):
        super().__init__()
        self.mix, self.chan_drop = mix, chan_drop
        self.proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, d))
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

def train_v2(Ztr, ytr, Zte, yte, mix, seed, epochs=100, patience=20, bs=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx_tr, idx_va = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=seed)
    Ztr_t, Zte_t = torch.tensor(Ztr, device=device), torch.tensor(Zte, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    model = MixerV2(Ztr.shape[1], int(ytr.max()) + 1, mix).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def predict(Z):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(Z[i:i + 1024]).argmax(1) for i in range(0, len(Z), 1024)]).cpu().numpy()

    va_idx = torch.tensor(idx_va, device=device)
    best, best_state, wait = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.tensor(np.random.permutation(idx_tr), device=device)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(model(Ztr_t[b]), ytr_t[b], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        va = np.mean(predict(Ztr_t[va_idx]) == ytr[idx_va])
        if va >= best:
            best, wait = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    gate = torch.tanh(model.gate).item() if mix else 0.0
    return float(np.mean(predict(Zte_t) == yte)), ep + 1, gate

# %% cell B: run on the 4 small datasets
RF = {'FingerMovements': 0.5500, 'Blink': 0.9956, 'UCI-HAR': 0.9013, 'Epilepsy-EEG': 0.9558, 'CAP-ID': 0.8091, 'CAP-OOD': 0.7858}
rows2 = []

def safe(fn):
    try:
        fn()
    except Exception as e:
        print('FAILED:', type(e).__name__, e, flush=True)

def run_v2(name, Ztr, ytr, Zte, yte, **kw):
    le = LabelEncoder().fit(ytr)
    ytr, yte = le.transform(ytr), le.transform(yte)
    for mix in (False, True):
        t0 = time.time()
        res = [train_v2(Ztr, ytr, Zte, yte, mix, s, **kw) for s in (0, 1, 2)]
        accs, gates = [r[0] for r in res], [round(r[2], 3) for r in res]
        method = 'mixer v2' if mix else 'concat + channel dropout'
        rows2.append(dict(dataset=name, method=method, mean=np.mean(accs), std=np.std(accs), rf=RF[name],
                          per_seed=json.dumps(accs), epochs=json.dumps([r[1] for r in res]), gate=json.dumps(gates),
                          time_s=round(time.time() - t0)))
        print(f'{name:16s} {method:26s} {np.mean(accs):.4f} +/- {np.std(accs):.4f} gate {gates} '
              f'{time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows2).to_csv(f'{OUT}/mixer_v2_results.csv', index=False)

for name in ['FingerMovements', 'Blink']:
    d = load_ts_archive(name)
    safe(lambda: run_v2(name, fixed(d['train'][0]), d['train'][1], fixed(d['test'][0]), d['test'][1]))
for folder, name in [('HAR', 'UCI-HAR'), ('Epilepsy', 'Epilepsy-EEG')]:
    d = load_tfc(folder)
    safe(lambda: run_v2(name, fixed(d['train'][0]), d['train'][1], fixed(d['test'][0]), d['test'][1]))

# %% cell C: CAP (19 channels, about 40,000 samples - enough data for channel mixing)
# !pip install -q gdown h5py
import gdown, h5py
t0 = time.time()
gdown.download(id='1NFwX2CqLrenWF4az0c6J-OglAoD48PAT', output=f'{DATA}/cap.zip', quiet=True)
with zipfile.ZipFile(f'{DATA}/cap.zip') as z:
    z.extractall(f'{DATA}/CAP')
os.remove(f'{DATA}/cap.zip')
h5path = [os.path.join(r, f) for r, _, fs in os.walk(f'{DATA}/CAP') for f in fs if f.endswith('.h5')][0]
print('CAP ready', f'{time.time() - t0:.0f}s', flush=True)
Zc, yc = {}, {}
with h5py.File(h5path, 'r') as h:
    for m in sorted(h.keys()):
        data, parts = h[f'{m}/data'], []
        for s in range(0, data.shape[0], 2000):
            parts.append(fixed(np.transpose(data[s:s + 2000], (0, 2, 1))))
        Zc[m], yc[m] = np.concatenate(parts), h[f'{m}/labels'][:].ravel()
        print(m, Zc[m].shape, f'{time.time() - t0:.0f}s', flush=True)

# %% cell D: CAP-ID and CAP-OOD
ms = sorted(Zc)
Z_all, y_all = np.concatenate([Zc[m] for m in ms]), np.concatenate([yc[m] for m in ms])
n = len(y_all)
perm = np.random.RandomState(0).permutation(n)
tr, te = perm[:int(0.6375 * n)], perm[n - int(round(0.25 * n)):]
safe(lambda: run_v2('CAP-ID', Z_all[tr], y_all[tr], Z_all[te], y_all[te], epochs=40, patience=8, bs=256))
trm = ['Machine0', 'Machine1', 'Machine3']
safe(lambda: run_v2('CAP-OOD', np.concatenate([Zc[m] for m in trm]), np.concatenate([yc[m] for m in trm]),
                    Zc['Machine4'], yc['Machine4'], epochs=40, patience=8, bs=256))
print(pd.DataFrame(rows2)[['dataset', 'method', 'mean', 'std', 'rf', 'gate']].round(4).to_string(index=False))
