"""
Python 3.10 compatible
requirements:
  pip install torch torchvision torchaudio
  pip install transformers
  pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
  pip install scikit-learn pandas numpy tqdm networkx

Usage:
  python apt_pipeline.py --data path/to/graph_node_link.json --epochs 20 --device cuda

The JSON must be NetworkX node-link format, e.g. produced by:
  from networkx.readwrite import json_graph
  data = json_graph.node_link_data(G)
  json.dump(data, open("graph.json","w"))

This file implements:
  - load_prov_graph(): parse node-link JSON → ProvGraphBatch
  - Embedding backbones: ATTACK-BERT / SecBERT / CySecBERT
  - RGCN-based GNN edge anomaly model
  - Counterfactual edge-mask explanation
"""

from __future__ import annotations
import os, json, time, math, argparse, warnings, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Metrics
try:
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
    SK_OK = True
except Exception:
    SK_OK = False

# HuggingFace
from transformers import AutoTokenizer, AutoModel

# PyG
from torch_geometric.nn import RGCNConv

# ---------------------------
# 0) Utils
# ---------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_device(obj, device):
    if obj is None:
        return None
    return obj.to(device)

# === Plot utils ===
def plot_roc_pr_curves(y_true: np.ndarray,
                       y_score: np.ndarray,
                       title_prefix: str,
                       out_prefix: str):
    """
    Lưu 2 hình:
      - {out_prefix}_roc.png
      - {out_prefix}_pr.png
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} – ROC")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = f"{out_prefix}_roc.png"
    os.makedirs(os.path.dirname(roc_path), exist_ok=True)
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} – Precision–Recall")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    pr_path = f"{out_prefix}_pr.png"
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    print(f"[+] Saved: {roc_path}")
    print(f"[+] Saved: {pr_path}")


# ---------------------------
# 1) DATA STRUCTURE
# ---------------------------

@dataclass
class ProvGraphBatch:
    num_nodes: int
    edge_index: torch.LongTensor            # [2, E]
    edge_type: torch.LongTensor             # [E]
    edge_texts: List[str]                   # len E
    y: torch.FloatTensor                    # [E] (0/1)
    split_idx: Dict[str, torch.LongTensor]  # 'train','val','test' (edge indices)
    node_x: Optional[torch.FloatTensor]     # [N, Dn] or None
    num_edge_types: int

# ---------------------------
# 2) LOAD FROM NETWORKX NODE-LINK JSON
# ---------------------------

def _safe_get(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _as_float_ts(v):
    # try convert timestamp-ish to float (seconds)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        # try int-like str
        try:
            return float(int(v))
        except Exception:
            return None

def _edge_text(src_attr: dict, edge_attr: dict, dst_attr: dict) -> str:
    # Compose a concise sentence for BERT embedding
    s_type = _safe_get(src_attr, ["type", "category", "kind"], "node")
    s_name = _safe_get(src_attr, ["name", "cmd", "path", "label", "id"], str(src_attr.get("id","src")))
    d_type = _safe_get(dst_attr, ["type", "category", "kind"], "node")
    d_name = _safe_get(dst_attr, ["name", "path", "label", "id"], str(dst_attr.get("id","dst")))
    op     = _safe_get(edge_attr, ["type", "op", "action", "event"], "rel")
    ts     = _safe_get(edge_attr, ["ts", "time", "timestamp"], None)
    size   = _safe_get(edge_attr, ["bytes", "size", "length"], None)
    rc     = _safe_get(edge_attr, ["rc", "status", "result"], None)
    ip     = _safe_get(edge_attr, ["ip", "dst_ip", "dip"], None)
    port   = _safe_get(edge_attr, ["port", "dport"], None)

    parts = []
    if ts is not None: parts.append(f"{ts}")
    parts.append(f"{s_type}:{s_name}")
    parts.append(f"{op}")
    parts.append(f"{d_type}:{d_name}")
    if size is not None: parts.append(f"bytes={size}")
    if rc is not None:   parts.append(f"rc={rc}")
    if ip is not None:   parts.append(f"ip={ip}")
    if port is not None: parts.append(f"port={port}")
    return " ".join(str(x) for x in parts)

def load_prov_graph(path: str) -> ProvGraphBatch:
    """
    Parse node-link JSON exported from NetworkX.
    Expected keys:
      {
        "nodes":[{"id": ..., "type": "...", "name": "...", ...}, ...],
        "links":[{"source": id_or_idx, "target": id_or_idx, "type": "...", "ts": ..., "label": 0/1 or "anomaly": true/false, ...}, ...]
      }

    Behavior:
      - node ids can be strings or ints; we remap to 0..N-1
      - edge_type: from 'type'/'op'/'action'/'event'; if missing -> "rel"
      - timestamp: from 'ts'/'time'/'timestamp'; if present → temporal split; else random split
      - label: from 'label' (0/1) or 'anomaly' (bool) else 0
      - node_x: simple features = onehot(node_type) + degree (normalized)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes_raw = data.get("nodes", [])
    links_raw = data.get("links", []) or data.get("edges", [])

    if not nodes_raw or not links_raw:
        raise ValueError("JSON thiếu 'nodes' hoặc 'links' (node-link format).")

    # Map node id -> idx
    node_ids = [n.get("id", i) for i, n in enumerate(nodes_raw)]
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    # Node type mapping (optional)
    node_types = []
    for n in nodes_raw:
        nt = _safe_get(n, ["type", "category", "kind"], "node")
        node_types.append(nt)
    uniq_nt = sorted(set(node_types))
    nt2i = {t:i for i,t in enumerate(uniq_nt)}

    # Build degrees for a simple numeric feature
    indeg = [0]*num_nodes
    outdeg = [0]*num_nodes

    # Edge type mapping
    edge_types_list = []
    timestamps = []
    y_labels = []
    src_idx = []
    dst_idx = []
    edge_texts = []

    # 1st pass: gather edge types
    tmp_edge_types = []
    for e in links_raw:
        et = _safe_get(e, ["type", "op", "action", "event"], "rel")
        tmp_edge_types.append(str(et))
    uniq_et = sorted(set(tmp_edge_types))
    et2i = {t:i for i,t in enumerate(uniq_et)}
    num_edge_types = len(uniq_et) if uniq_et else 1

    # 2nd pass: build arrays
    for e in links_raw:
        s = e.get("source")
        t = e.get("target")
        if s not in id2idx or t not in id2idx:
            # In some node-link dumps, source/target may be indices instead of ids
            try:
                s_idx = int(s)
                t_idx = int(t)
            except Exception:
                raise ValueError("source/target không map được sang node indices.")
        else:
            s_idx = id2idx[s]
            t_idx = id2idx[t]

        indeg[t_idx] += 1
        outdeg[s_idx] += 1
        src_idx.append(s_idx)
        dst_idx.append(t_idx)

        et = _safe_get(e, ["type", "op", "action", "event"], "rel")
        edge_types_list.append(et2i[str(et)])

        ts = _as_float_ts(_safe_get(e, ["ts", "time", "timestamp"], None))
        timestamps.append(ts)

        # label: label or anomaly
        lab = _safe_get(e, ["label"], None)
        if lab is None:
            an = _safe_get(e, ["anomaly", "is_anomaly"], False)
            lab = 1 if (an is True or an == "true" or an == 1) else 0
        try:
            lab = int(lab)
        except Exception:
            lab = 0
        y_labels.append(lab)

        # edge text
        s_attr = nodes_raw[s_idx]
        d_attr = nodes_raw[t_idx]
        edge_texts.append(_edge_text(s_attr, e, d_attr))

    E = len(src_idx)
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    edge_type  = torch.tensor(edge_types_list, dtype=torch.long)
    y         = torch.tensor(y_labels, dtype=torch.float32)

    # Node features: onehot(type) + degree (2 scalars)
    deg = np.stack([np.array(outdeg), np.array(indeg)], axis=1).astype(np.float32)
    if deg.max() > 0:
        deg = deg / (deg.max(axis=0, keepdims=True) + 1e-6)
    nt_idx = np.array([nt2i[t] for t in node_types], dtype=np.int64)
    nt_onehot = np.eye(len(uniq_nt), dtype=np.float32)[nt_idx]
    node_x_np = np.concatenate([nt_onehot, deg], axis=1)  # [N, |NT|+2]
    node_x = torch.tensor(node_x_np, dtype=torch.float32)

    # Split (edge indices)
    idx_all = np.arange(E)
    if all(ts is None for ts in timestamps):
        # random split
        set_seed(1337)
        np.random.shuffle(idx_all)
        n_train = int(0.70 * E)
        n_val   = int(0.15 * E)
        train_idx = idx_all[:n_train]
        val_idx   = idx_all[n_train:n_train+n_val]
        test_idx  = idx_all[n_train+n_val:]
    else:
        # temporal split by timestamp percentiles
        ts_vals = np.array([t if t is not None else -1 for t in timestamps], dtype=np.float64)
        order = np.argsort(ts_vals)
        n_train = int(0.70 * E)
        n_val   = int(0.15 * E)
        train_idx = order[:n_train]
        val_idx   = order[n_train:n_train+n_val]
        test_idx  = order[n_train+n_val:]

    split_idx = {
        "train": torch.tensor(train_idx, dtype=torch.long),
        "val":   torch.tensor(val_idx, dtype=torch.long),
        "test":  torch.tensor(test_idx, dtype=torch.long),
    }

    return ProvGraphBatch(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_texts=edge_texts,
        y=y,
        split_idx=split_idx,
        node_x=node_x,
        num_edge_types=num_edge_types
    )

# ---------------------------
# 3) TEXT EMBEDDERS (HF)
# ---------------------------

HF_MODELS = {
    "ATTACK_BERT": "basel/ATTACK-BERT",
    "SEC_BERT":    "jackaduma/SecBERT",
    "CYSEC_BERT":  "markusbayer/CySecBERT",
}

class TextEmbedder(nn.Module):
    def __init__(self, model_name_or_path: str, out_dim: int = 256, mean_pool: bool = True, fp16: bool = True):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.enc = AutoModel.from_pretrained(model_name_or_path)
        self.h = self.enc.config.hidden_size
        self.norm = nn.LayerNorm(self.h)
        self.proj = nn.Sequential(nn.Linear(self.h, out_dim), nn.ReLU())
        self.mean_pool = mean_pool
        self.fp16 = fp16

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 512, max_len: int = 192, device: str = "cuda") -> torch.FloatTensor:
        self.eval().to(device)
        all_out = []
        dtype_ac = torch.float16 if (self.fp16 and device.startswith("cuda")) else None

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding edges"):
            chunk = texts[i:i+batch_size]
            toks = self.tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            if dtype_ac is not None:
                with torch.autocast(device_type="cuda", dtype=dtype_ac):
                    out = self.enc(**toks).last_hidden_state  # [B, L, H]
            else:
                out = self.enc(**toks).last_hidden_state
            if self.mean_pool:
                mask = toks.attention_mask.unsqueeze(-1)
                emb = (out*mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                emb = out[:,0,:]
            emb = self.norm(emb)
            emb = self.proj(emb)
            all_out.append(emb.detach().cpu())
        return torch.cat(all_out, dim=0)  # [E, out_dim]

# ---------------------------
# 4) MODEL: RGCN + EDGE SCORER
# ---------------------------

class EdgeAnomalyModel(nn.Module):
    def __init__(self, num_nodes: int, num_edge_types: int,
                 node_in_dim: int = 0,
                 node_emb_dim: int = 256,
                 gnn_hidden: int = 256,
                 gnn_out: int = 128,
                 edge_text_dim: int = 256):
        super().__init__()
        self.use_node_x = node_in_dim > 0
        if self.use_node_x:
            self.input_proj = nn.Linear(node_in_dim, node_emb_dim)
        else:
            self.node_emb = nn.Embedding(num_nodes, node_emb_dim)
        self.gnn1 = RGCNConv(node_emb_dim, gnn_hidden, num_relations=num_edge_types)
        self.gnn2 = RGCNConv(gnn_hidden, gnn_out, num_relations=num_edge_types)
        self.dropout = nn.Dropout(0.2)
        self.edge_mlp = nn.Sequential(
            nn.Linear(gnn_out*2 + edge_text_dim, gnn_hidden),
            nn.ReLU(),
            nn.Linear(gnn_hidden, 1),
        )

    def forward(self, edge_index: torch.LongTensor, edge_type: torch.LongTensor,
                e_text_emb: torch.FloatTensor, node_x: Optional[torch.FloatTensor] = None):
        N = int(edge_index.max().item() + 1)
        if self.use_node_x:
            x0 = self.input_proj(node_x)
        else:
            device = edge_index.device
            x0 = self.node_emb(torch.arange(N, device=device))

        h = F.relu(self.gnn1(x0, edge_index, edge_type))
        h = self.dropout(F.relu(self.gnn2(h, edge_index, edge_type)))

        src, dst = edge_index[0], edge_index[1]
        e_repr = torch.cat([h[src], h[dst], e_text_emb], dim=-1)
        logits = self.edge_mlp(e_repr).squeeze(-1)
        return logits, h

# ---------------------------
# 5) LOSS & METRICS
# ---------------------------

class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = p*targets + (1-p)*(1-targets)
        w  = self.alpha * (1-pt).pow(self.gamma)
        return (w * ce).mean()

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    out = {}
    if SK_OK:
        try:
            pr, rc, _ = precision_recall_curve(y_true, y_score)
            out["pr_auc"] = auc(rc, pr)
        except Exception:
            out["pr_auc"] = float("nan")
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            out["roc_auc"] = float("nan")
        f1s = []
        for p, r in zip(pr, rc):
            if p+r > 0:
                f1s.append(2*p*r/(p+r))
        out["f1_best"] = max(f1s) if f1s else float("nan")
    else:
        # crude AP fallback
        order = np.argsort(-y_score)
        cum_pos = 0
        precisions = []
        for i, idx in enumerate(order, 1):
            if y_true[idx] == 1:
                cum_pos += 1
                precisions.append(cum_pos / i)
        out["pr_auc"] = float(np.mean(precisions) if precisions else 0.0)
        out["roc_auc"] = float("nan")
        out["f1_best"] = float("nan")
    return out

# ---------------------------
# 6) TRAIN ONE BACKBONE
# ---------------------------

def train_one_backbone(batch: ProvGraphBatch,
                       backbone_key: str,
                       cache_dir: str = "./cache_emb",
                       epochs: int = 20,
                       lr: float = 2e-3,
                       device: str = "cuda",
                       max_len: int = 192,
                       batch_size_embed: int = 512) -> Tuple[Dict[str,float], Dict[str,any]]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(cache_dir, exist_ok=True)
    model_path = HF_MODELS[backbone_key]
    embedder = TextEmbedder(model_path, out_dim=256, mean_pool=True, fp16=True)

    emb_cache = os.path.join(cache_dir, f"edge_emb_{backbone_key}.pt")
    if os.path.exists(emb_cache):
        e_text_emb = torch.load(emb_cache, map_location="cpu")
    else:
        e_text_emb = embedder.encode_texts(batch.edge_texts, batch_size=batch_size_embed,
                                           max_len=max_len, device=device).cpu()
        torch.save(e_text_emb, emb_cache)
    e_text_emb = e_text_emb.to(device)

    node_in_dim = 0 if batch.node_x is None else batch.node_x.size(-1)
    model = EdgeAnomalyModel(
        num_nodes=batch.num_nodes,
        num_edge_types=batch.num_edge_types,
        node_in_dim=node_in_dim,
        node_emb_dim=256,
        gnn_hidden=256,
        gnn_out=128,
        edge_text_dim=e_text_emb.size(-1),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = FocalBCELoss(gamma=2.0, alpha=0.75)

    edge_index = batch.edge_index.to(device)
    edge_type  = batch.edge_type.to(device)
    y          = batch.y.to(device)
    node_x     = batch.node_x.to(device) if batch.node_x is not None else None

    train_idx = batch.split_idx['train'].to(device)
    val_idx   = batch.split_idx['val'].to(device)
    test_idx  = batch.split_idx['test'].to(device)

    # === new logs ===
    hist_loss, hist_prauc = [], []

    best_val = -1.0
    best_state = None
    t0 = time.time()

    for ep in range(1, epochs+1):
        model.train()
        logits, _ = model(edge_index, edge_type, e_text_emb, node_x)
        loss = loss_fn(logits[train_idx], y[train_idx])
        opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(edge_index, edge_type, e_text_emb, node_x)
            y_true = y[val_idx].detach().cpu().numpy()
            y_score = torch.sigmoid(logits_eval[val_idx]).detach().cpu().numpy()
            m = compute_metrics(y_true, y_score)
            prauc = m.get("pr_auc", float("nan"))
            hist_loss.append(loss.item())
            hist_prauc.append(prauc)
            if not math.isnan(prauc) and prauc > best_val:
                best_val = prauc
                best_state = {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "backbone": backbone_key,
                    "metrics_val": m,
                }

        if ep % 2 == 0:
            print(f"[{backbone_key}] epoch {ep}/{epochs} loss={loss.item():.4f} best_val_prauc={best_val:.4f}")

    # === plot training curve ===
    fig, ax1 = plt.subplots(figsize=(7,4))
    sns.set(style="whitegrid")

    ax1.plot(hist_loss, color='tab:blue', label='Train Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(hist_prauc, color='tab:red', label='Val PR-AUC')
    ax2.set_ylabel("PR-AUC", color='tab:red')

    fig.suptitle(f"Training Curve – {backbone_key}")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/training_{backbone_key}.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved training curve: {plot_path}")

    latency = (time.time() - t0) / max(1, epochs)

    # Evaluate best model on val/test
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    model.eval()
    with torch.no_grad():
        logits_eval, _ = model(edge_index, edge_type, e_text_emb, node_x)
        y_true_val  = batch.y[batch.split_idx['val'] ].numpy()
        y_score_val = torch.sigmoid(logits_eval[val_idx]).cpu().numpy()
        y_true_test  = batch.y[batch.split_idx['test'] ].numpy()
        y_score_test = torch.sigmoid(logits_eval[test_idx]).cpu().numpy()
        m_val  = compute_metrics(y_true_val,  y_score_val)
        m_test = compute_metrics(y_true_test, y_score_test)


    artifacts = {
        "state_dict": {k: v.cpu() for k,v in model.state_dict().items()},
        "backbone": backbone_key,
        "edge_emb": e_text_emb.detach().cpu(),
        "latency_per_epoch": latency,
        "val_metrics": m_val,
        "test_metrics": m_test,
        "curve": {"loss": hist_loss, "prauc": hist_prauc},
    }
    return m_val, artifacts

def select_best(backbone_runs: Dict[str, Tuple[Dict[str,float], Dict[str,any]]]) -> str:
    def score_of(k):
        m, art = backbone_runs[k]
        return (m.get("pr_auc", float("-inf")), -art.get("latency_per_epoch", float("inf")))
    return max(backbone_runs.keys(), key=lambda k: score_of(k))

# ---------------------------
# 7) COUNTERFACTUAL (Edge-mask)
# ---------------------------

class EdgeMaskCounterfactual(nn.Module):
    def __init__(self, trained_model: EdgeAnomalyModel,
                 full_edge_index: torch.LongTensor,
                 full_edge_type : torch.LongTensor,
                 full_node_x    : Optional[torch.FloatTensor],
                 full_edge_text_emb: torch.FloatTensor,
                 sub_edge_idx: torch.LongTensor):
        super().__init__()
        self.model = trained_model
        self.edge_index = full_edge_index
        self.edge_type  = full_edge_type
        self.node_x     = full_node_x
        self.e_text_emb = full_edge_text_emb
        self.sub_idx = sub_edge_idx
        self.m_logits = nn.Parameter(torch.zeros(self.sub_idx.size(0)))
        with torch.no_grad():
            self.model.eval()
            base_logits, _ = self.model(self.edge_index, self.edge_type, self.e_text_emb, self.node_x)
        self.register_buffer("base_logits", base_logits)

    def forward(self):
        device = self.e_text_emb.device
        mask_full = torch.ones(self.e_text_emb.size(0), device=device)
        mask_full[self.sub_idx] = torch.sigmoid(self.m_logits)
        masked_e_text = self.e_text_emb * mask_full.unsqueeze(-1)
        logits, _ = self.model(self.edge_index, self.edge_type, masked_e_text, self.node_x)
        return logits, mask_full

def run_counterfactual(trained_state: Dict[str,any],
                       batch: ProvGraphBatch,
                       sub_edge_idx: torch.LongTensor,
                       target_label: int = 0,
                       lam_sparsity: float = 0.10,
                       lam_smooth: float = 1e-3,
                       steps: int = 300,
                       lr: float = 0.1,
                       device: str = "cuda") -> Dict[str, any]:

    node_in_dim = 0 if batch.node_x is None else batch.node_x.size(-1)
    model = EdgeAnomalyModel(
        num_nodes=batch.num_nodes,
        num_edge_types=batch.num_edge_types,
        node_in_dim=node_in_dim,
        node_emb_dim=256,
        gnn_hidden=256,
        gnn_out=128,
        edge_text_dim=trained_state["edge_emb"].size(-1),
    ).to(device)
    model.load_state_dict(trained_state["state_dict"])
    model.eval()

    e_text_emb = trained_state["edge_emb"].to(device)
    edge_index = batch.edge_index.to(device)
    edge_type  = batch.edge_type.to(device)
    node_x     = to_device(batch.node_x, device)
    sub_edge_idx = sub_edge_idx.to(device)

    expl = EdgeMaskCounterfactual(model, edge_index, edge_type, node_x, e_text_emb, sub_edge_idx).to(device)
    opt = torch.optim.Adam([expl.m_logits], lr=lr)
    tgt = torch.tensor(float(target_label), device=device)

    for _ in tqdm(range(steps), desc="Optimizing counterfactual"):
        logits, mask_full = expl()
        sub_logits = logits[sub_edge_idx]
        pred_loss = F.binary_cross_entropy_with_logits(sub_logits.mean().unsqueeze(0), tgt.unsqueeze(0))
        sparsity = (1 - mask_full[sub_edge_idx]).mean()
        s = torch.sigmoid(expl.m_logits)
        smooth = (s*(1-s)).mean()
        loss = pred_loss + lam_sparsity * sparsity + lam_smooth * smooth
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        logits_final, mask_full = expl()
        sub_mask = mask_full[sub_edge_idx].detach().cpu().numpy()
        base_sub_logits = expl.base_logits[sub_edge_idx].detach().cpu().numpy()
        final_sub_logits = logits_final[sub_edge_idx].detach().cpu().numpy()

    order = np.argsort(sub_mask)  # smallest masks first = most critical to turn off
    critical_idx = sub_edge_idx.detach().cpu().numpy()[order]

    return {
        "critical_edges_ordered": critical_idx.tolist(),
        "mask_values_ordered": sub_mask[order].tolist(),
        "delta_logits_mean": float(final_sub_logits.mean() - base_sub_logits.mean()),
        "final_pred_mean": float(1/(1+np.exp(-final_sub_logits.mean()))),
    }

# ---------------------------
# 8) MAIN
# ---------------------------

def main(args):
    set_seed(1337)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        warnings.warn("CUDA không khả dụng, chuyển sang 'cpu'.")
        args.device = "cpu"

    print("[*] Loading graph from:", args.data)
    batch = load_prov_graph(args.data)

    runs = {}
    backbones = ["ATTACK_BERT", "SEC_BERT", "CYSEC_BERT"]
    for key in backbones:
        print(f"\n=== Train with {key} ===")
        m_val, artifacts = train_one_backbone(batch, key, epochs=args.epochs, device=args.device)
        print(f"VAL ({key}): {m_val}")
        print(f"TEST({key}): {artifacts['test_metrics']}")
        runs[key] = (m_val, artifacts)

    winner = select_best(runs)
    best_art = runs[winner][1]
    print(f"\n>>> WINNER BACKBONE: {winner} | VAL: {best_art['val_metrics']} | TEST: {best_art['test_metrics']}")
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/edge_anom_{winner}.pt"
    torch.save(best_art, ckpt_path)
    print(f"[+] Saved checkpoint: {ckpt_path}")

    # Demo counterfactual on top-K suspicious edges in test
    print("\n[*] Counterfactual demo on top-K suspicious edges in TEST split")
    device = args.device
    node_in_dim = 0 if batch.node_x is None else batch.node_x.size(-1)
    model = EdgeAnomalyModel(
        num_nodes=batch.num_nodes, num_edge_types=batch.num_edge_types,
        node_in_dim=node_in_dim, node_emb_dim=256, gnn_hidden=256, gnn_out=128,
        edge_text_dim=best_art["edge_emb"].size(-1),
    ).to(device)
    model.load_state_dict(best_art["state_dict"]); model.eval()
    e_text_emb = best_art["edge_emb"].to(device)
    edge_index = batch.edge_index.to(device)
    edge_type  = batch.edge_type.to(device)
    node_x     = to_device(batch.node_x, device)

    with torch.no_grad():
        logits, _ = model(edge_index, edge_type, e_text_emb, node_x)
        probs = torch.sigmoid(logits)
    test_idx = batch.split_idx["test"]
    test_scores = probs[test_idx.to(device)].detach().cpu().numpy()
    topk = min(args.topk, len(test_scores))
    sel = test_idx[torch.argsort(torch.tensor(test_scores))[-topk:]]

    cf = run_counterfactual(best_art, batch, sel, target_label=0, steps=args.cf_steps, device=device)
    print("[CF] delta_mean_logit:", cf["delta_logits_mean"])
    print("[CF] final_pred_mean:", cf["final_pred_mean"])
    print("[CF] first 10 critical edges:", cf["critical_edges_ordered"][:10])
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for k, (m, art) in runs.items():
        plt.plot(art["curve"]["prauc"], label=k)
    plt.title("PR-AUC comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Validation PR-AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/compare_prauc.png", bbox_inches="tight")
    print("[+] Saved: plots/compare_prauc.png")
    # === So sánh PR-AUC giữa các backbone (TEST) ===
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score

        plt.figure(figsize=(7, 5))
        for k, (m, art) in runs.items():
            # Rebuild logits on TEST để có y_true/y_score test
            best_art = art
            # Tải lại model & tensors
            node_in_dim = 0 if batch.node_x is None else batch.node_x.size(-1)
            model_cmp = EdgeAnomalyModel(
                num_nodes=batch.num_nodes, num_edge_types=batch.num_edge_types,
                node_in_dim=node_in_dim, node_emb_dim=256, gnn_hidden=256, gnn_out=128,
                edge_text_dim=best_art["edge_emb"].size(-1),
            ).to(args.device)
            model_cmp.load_state_dict(best_art["state_dict"]);
            model_cmp.eval()
            e_text_emb_cmp = best_art["edge_emb"].to(args.device)
            edge_index_cmp = batch.edge_index.to(args.device)
            edge_type_cmp = batch.edge_type.to(args.device)
            node_x_cmp = batch.node_x.to(args.device) if batch.node_x is not None else None

            with torch.no_grad():
                logits_cmp, _ = model_cmp(edge_index_cmp, edge_type_cmp, e_text_emb_cmp, node_x_cmp)
                probs_cmp = torch.sigmoid(logits_cmp)

            test_idx = batch.split_idx["test"]
            y_true_test = batch.y[test_idx].numpy()
            y_score_test = probs_cmp[test_idx.to(args.device)].detach().cpu().numpy()

            prec, rec, _ = precision_recall_curve(y_true_test, y_score_test)
            ap = average_precision_score(y_true_test, y_score_test)
            plt.plot(rec, prec, label=f"{k} (AP={ap:.4f})")

        plt.title("Precision–Recall (TEST) – Backbone Comparison")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/compare_pr_test.png", bbox_inches="tight")
        plt.close()
        print("[+] Saved: plots/compare_pr_test.png")
    except Exception as e:
        print(f"[!] Compare PR plot failed: {e}")
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(7, 5))
    for k, (m, art) in runs.items():
        # (dùng y_true_test, y_score_test đã tính ở đoạn trên)
        fpr, tpr, _ = roc_curve(y_true_test, y_score_test)
        the_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{k} (AUC={the_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC (TEST) – Backbone Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/compare_roc_test.png", bbox_inches="tight")
    plt.close()
    print("[+] Saved: plots/compare_roc_test.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to NetworkX node-link JSON")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    p.add_argument("--topk", type=int, default=50, help="Top-K suspicious edges in TEST for CF")
    p.add_argument("--cf_steps", type=int, default=200, help="CF optimization steps")
    args = p.parse_args()
    main(args)

