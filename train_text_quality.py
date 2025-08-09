# Usage:
#   python train_text_quality.py \
#       --csv labeling_tasks.csv \
#       --clips_root . \
#       --mobileclip_ckpt checkpoints/mobileclip_blt.pt \
#       --out_dir runs/text_quality_blT_pca256

import os
import math
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

import mobileclip  # noqa


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--clips_root", type=str, default=".")
    ap.add_argument("--mobileclip_arch", type=str, default="mobileclip_b")
    ap.add_argument("--mobileclip_ckpt", type=str, required=True)
    ap.add_argument("--pca_dim", type=int, default=256)
    ap.add_argument("--batch_size_text", type=int, default=256)
    ap.add_argument("--batch_size_train", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--num_workers_io", type=int, default=16)  # for video metadata
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, required=True)
    return ap.parse_args()


# ----------------------------- Data ----------------------------- #

GOOD_STATUS = {"completed"}         # label = 1
BAD_STATUS = {"bad_quality"}        # label = 0
DROP_STATUS = {"assigned", "unassigned"}  # ignore


def load_df(csv_path: str) -> pd.DataFrame:
    # pandas handles the quoted JSON in "peaks" fine by default
    df = pd.read_csv(csv_path)
    # filter statuses
    df = df[~df["status"].isin(DROP_STATUS)].copy()
    # map labels
    def lab(s):
        if s in GOOD_STATUS:
            return 1
        if s in BAD_STATUS:
            return 0
        return np.nan  # unknown -> drop
    df["label"] = df["status"].map(lab)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    # pick revised_caption if non-empty, else caption
    def pick_caption(r):
        rc = str(r.get("revised_caption", "")).strip()
        if rc and rc.lower() != "nan":
            return rc
        return str(r.get("caption", "")).strip()
    df["text"] = df.apply(pick_caption, axis=1)
    # absolute video path
    root = Path(os.path.dirname(csv_path)).resolve()
    df["video_path"] = df["video"].apply(lambda p: str((Path(args.clips_root) / p).resolve()))
    # drop rows with empty text
    df = df[df["text"].str.len() > 0].copy()
    df = df.reset_index(drop=True)
    return df


def read_video_size(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return np.nan, np.nan
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    if w is None or h is None or w <= 0 or h <= 0:
        return np.nan, np.nan
    return float(w), float(h)


def compute_resolution_features(paths, num_workers=16):
    widths = np.full(len(paths), np.nan, dtype=float)
    heights = np.full(len(paths), np.nan, dtype=float)
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        fut2idx = {ex.submit(read_video_size, p): i for i, p in enumerate(paths)}
        for fut in tqdm(as_completed(fut2idx), total=len(fut2idx), desc="reading video sizes"):
            i = fut2idx[fut]
            try:
                w, h = fut.result()
            except Exception:
                w, h = np.nan, np.nan
            widths[i] = w
            heights[i] = h
    # make neg_quality = -log(pixel_count)
    pix = widths * heights
    # if missing, impute later using train statistics
    return pix


# ------------------------ MobileCLIP text ------------------------ #

@torch.no_grad()
def encode_texts_mobileclip(texts, arch, ckpt, batch_size=256, device="cuda"):
    model, _, _ = mobileclip.create_model_and_transforms(arch, pretrained=ckpt)
    tokenizer = mobileclip.get_tokenizer(arch)
    model.eval().to(device)
    all_emb = []
    for i in tqdm(range(0, len(texts), batch_size), desc="MobileCLIP text"):
        batch = texts[i:i+batch_size]
        tok = tokenizer(batch).to(device)
        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda") and torch.cuda.is_available())):
            feats = model.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_emb.append(feats.float().cpu())
    return torch.cat(all_emb, dim=0).numpy()  # [N, D]


# ----------------------- Monotone Logistic ----------------------- #

class MonoLogistic(nn.Module):
    def __init__(self, d_feat):
        super().__init__()
        self.W = nn.Linear(d_feat, 1, bias=False)   # weights on PCA features
        self.u_res = nn.Parameter(torch.tensor(0.0))  # w_res = -softplus(u_res) <= 0
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_feat, x_res):
        w_res = -F.softplus(self.u_res)
        logits = self.W(x_feat).squeeze(1) + w_res * x_res.squeeze(1) + self.b
        return logits


class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / torch.exp(self.log_t)


def train_epoch(model, loader, opt, device, pos_weight=None):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total, ys, ps = 0, [], []
    for x_feat, x_res, y in loader:
        x_feat = x_feat.to(device)
        x_res = x_res.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x_feat, x_res)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * y.size(0)
        ys.append(y.detach().cpu().numpy())
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return total / len(loader.dataset), roc_auc_score(y_true, y_pred)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total, ys, ps, lg = 0, [], [], []
    for x_feat, x_res, y in loader:
        x_feat = x_feat.to(device)
        x_res = x_res.to(device)
        y = y.to(device)
        logits = model(x_feat, x_res)
        loss = loss_fn(logits, y)
        total += loss.item() * y.size(0)
        lg.append(logits.cpu())
        ys.append(y.cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return total / len(loader.dataset), roc_auc_score(y_true, y_pred), torch.cat(lg)


def pick_threshold(y_true, probs):
    # maximize F1 on val
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0.05, 0.95, 181):
        preds = (probs >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV...")
    df = load_df(args.csv)
    print(f"Rows after filtering: {len(df)}")
    print(df["status"].value_counts())

    # resolution feature
    print("Reading video resolutions...")
    pix = compute_resolution_features(df["video_path"].tolist(), num_workers=args.num_workers_io)
    df["pixel_count"] = pix

    # build text
    texts = df["text"].tolist()

    # MobileCLIP text embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_emb = encode_texts_mobileclip(
        texts, arch=args.mobileclip_arch, ckpt=args.mobileclip_ckpt,
        batch_size=args.batch_size_text, device=device
    )  # [N, D]

    labels = df["label"].values.astype(np.float32)

    # train/val split (stratified)
    X_txt_tr, X_txt_va, pix_tr, pix_va, y_tr, y_va = train_test_split(
        text_emb, df["pixel_count"].values, labels,
        test_size=0.2, stratify=labels, random_state=args.seed
    )

    # impute pixel_count by median of TRAIN
    med_pix = np.nanmedian(pix_tr[~np.isnan(pix_tr)]) if np.any(~np.isnan(pix_tr)) else 224.0 * 224.0
    pix_tr = np.where(np.isnan(pix_tr) | (pix_tr <= 0), med_pix, pix_tr)
    pix_va = np.where(np.isnan(pix_va) | (pix_va <= 0), med_pix, pix_va)

    # neg_quality = -log(pixel_count)
    nq_tr = -np.log(np.clip(pix_tr, 1.0, None)).astype(np.float32)
    nq_va = -np.log(np.clip(pix_va, 1.0, None)).astype(np.float32)

    # PCA on TRAIN only, keep 256
    print("Fitting PCA...")
    pca = PCA(n_components=args.pca_dim, svd_solver="auto", random_state=args.seed)
    Xp_tr = pca.fit_transform(X_txt_tr).astype(np.float32)
    Xp_va = pca.transform(X_txt_va).astype(np.float32)

    # scale resolution to a sane range
    res_scaler = StandardScaler()
    rq_tr = res_scaler.fit_transform(nq_tr.reshape(-1, 1)).astype(np.float32)
    rq_va = res_scaler.transform(nq_va.reshape(-1, 1)).astype(np.float32)

    # tensors and loaders
    tr_ds = TensorDataset(torch.from_numpy(Xp_tr), torch.from_numpy(rq_tr), torch.from_numpy(y_tr))
    va_ds = TensorDataset(torch.from_numpy(Xp_va), torch.from_numpy(rq_va), torch.from_numpy(y_va))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size_train, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=1024, shuffle=False)

    # model
    model = MonoLogistic(d_feat=args.pca_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train with early stop on val loss
    best_loss = float("inf")
    best_state = None
    bad = 0
    for epoch in range(args.epochs):
        tr_loss, tr_auc = train_epoch(model, tr_loader, opt, device)
        va_loss, va_auc, _ = eval_epoch(model, va_loader, device)
        print(f"epoch {epoch:03d}  train_loss {tr_loss:.4f} AUC {tr_auc:.4f} | val_loss {va_loss:.4f} AUC {va_auc:.4f}")
        if va_loss < best_loss:
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stop")
                break

    model.load_state_dict(best_state)
    model.eval()

    # collect val logits for calibration
    _, va_auc, va_logits = eval_epoch(model, va_loader, device)
    print(f"Val AUC before calibration: {va_auc:.4f}")

    # temperature scaling
    scaler = TempScaler().to(device)
    opt_t = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)
    bce = nn.BCEWithLogitsLoss()

    yv = torch.from_numpy(y_va).to(device)
    def closure():
        opt_t.zero_grad()
        loss = bce(scaler(va_logits.to(device)), yv)
        loss.backward()
        return loss
    opt_t.step(closure)

    with torch.no_grad():
        probs_cal = torch.sigmoid(scaler(va_logits.to(device))).cpu().numpy()
    val_auc_cal = roc_auc_score(y_va, probs_cal)
    thr, f1 = pick_threshold(y_va, probs_cal)
    preds = (probs_cal >= thr).astype(int)
    p, r, f1_, _ = precision_recall_fscore_support(y_va, preds, average="binary", zero_division=0)
    print(f"Val AUC after calibration: {val_auc_cal:.4f}")
    print(f"Chosen threshold {thr:.3f}  F1 {f1:.4f}  P {p:.3f}  R {r:.3f}")

    # save artifacts
    joblib.dump(pca, out_dir / "pca_256.joblib")
    joblib.dump(res_scaler, out_dir / "res_scaler.joblib")
    torch.save(model.state_dict(), out_dir / "mono_logistic.pt")
    torch.save(scaler.state_dict(), out_dir / "temp_scaler.pt")
    meta = {
        "arch": args.mobileclip_arch,
        "ckpt": os.path.abspath(args.mobileclip_ckpt),
        "pca_dim": args.pca_dim,
        "label_map": {"completed": 1, "bad_quality": 0},
        "drop_status": list(DROP_STATUS),
        "val_metrics": {
            "auc_calibrated": float(val_auc_cal),
            "threshold": float(thr),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1_),
        },
        "median_pixel_count": float(med_pix),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {out_dir.resolve()}")

    # quick inference helper on the same process
    def predict_proba(captions, pixel_counts):
        # text → MC text embeddings
        emb = encode_texts_mobileclip(
            captions, arch=args.mobileclip_arch, ckpt=args.mobileclip_ckpt,
            batch_size=args.batch_size_text, device=device
        )
        # transform
        Xp = pca.transform(emb).astype(np.float32)
        nq = -np.log(np.clip(pixel_counts, 1.0, None)).astype(np.float32)
        rq = res_scaler.transform(nq.reshape(-1, 1)).astype(np.float32)
        with torch.no_grad():
            x_feat = torch.from_numpy(Xp).to(device)
            x_res = torch.from_numpy(rq).to(device)
            logits = model(x_feat, x_res)
            logits = scaler(logits)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

if __name__ == "__main__":
    args = parse_args()
    main(args)
