"""Dense retriever (GPU): contrastive fine-tune of multilingual-e5-small (MIT) as a bi-encoder, trained ONLY on the
99,948 S1 records the small cross-encoder uses (ce_data.py), which the classifier never trains on, so its kNN candidates
never leak labels. Training pairs are built here: positives = all ground-truth matches of those S1, hard negatives =
their blocking candidates that are not matches (ce_train.parquet rows with y = 0).

Each step: B distinct S1 queries, for each one random true copy (positive) and one random hard negative. InfoNCE over the
batch's 2B candidates (temperature 0.05); batch columns that are true matches of another row's S1 are masked out.

python ret_train.py --work work --ce work/ce/ce_train.parquet --out work/retriever        (then knn_search.py --model work/retriever)
"""
import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/
from run import truth_pairs  # noqa: E402

MODEL, MAX_LEN, EPOCHS, LR, TAU = "intfloat/multilingual-e5-small", 64, 4, 3e-5, 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--ce", type=Path, default=Path("work/ce/ce_train.parquet"), help="ce_data.py output (i, j, y)")
    ap.add_argument("--batch", type=int, default=256, help="distinct S1 per step (256 fits a 16 GB GPU)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    B = a.batch
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ce = pl.read_parquet(a.ce, columns=["i", "j", "y"])
    s1p, cp = str(a.work / "train_s1" / "*.parquet"), str(a.work / "train_cand" / "*.parquet")
    pos = truth_pairs(types.SimpleNamespace(data=a.data), s1p, cp).join(ce.select("i").unique(), on="i").select("i", "j")
    neg = ce.filter(pl.col("y") == 0).select("i", "j")
    txt = lambda p, ids: dict(pl.scan_parquet(p).filter(pl.col("idx").is_in(ids.unique().implode())).select(
        "idx", s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).collect().iter_rows())
    ts, tc = txt(s1p, pos["i"]), txt(cp, pl.concat([pos["j"], neg["j"]]))
    P = {k: v for k, v in pos.group_by("i").agg("j").iter_rows()}
    N = {k: v for k, v in neg.group_by("i").agg("j").iter_rows()}
    Ps = {k: set(v) for k, v in P.items()}
    qs = np.array(sorted(P))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).cuda()
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(qs) // B)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05, anneal_strategy="linear")
    scaler = torch.amp.GradScaler()

    def enc(texts):
        e = tok(["query: " + t for t in texts], max_length=MAX_LEN, truncation=True, padding=True, return_tensors="pt").to("cuda")
        h = net(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).to(h.dtype)
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)

    print(f"{len(qs)} S1 queries, {pos.height} positives, {neg.height} hard negatives, {steps} steps", flush=True)
    t0, n = time.time(), 0
    model.train()
    for ep in range(EPOCHS):
        order = rng.permutation(qs)
        for s in range(0, len(order) - B + 1, B):
            q = order[s:s + B]
            p = [int(rng.choice(P[i])) for i in q]
            hn = [int(rng.choice(N[i])) if i in N else int(rng.choice(P[q[(k + 1) % B]])) for k, i in enumerate(q)]
            with torch.autocast("cuda", dtype=torch.float16):
                zq, zc = enc([ts[i] for i in q]), enc([tc[j] for j in p + hn])
                logits = (zq @ zc.T).float() / TAU
            # a batch column that is a true match of row r (other than r's own positive) is not a negative for r:
            # the fallback negative is another query's positive, and a hard negative can be another S1's match
            cand = p + hn
            bad = torch.tensor([[c != r and cand[c] in Ps[i] for c in range(len(cand))] for r, i in enumerate(q)],
                               device="cuda")
            logits = logits.masked_fill(bad, float("-inf"))
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(q), device="cuda"))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += 1
            if n % 100 == 0:
                print(f"epoch {ep} step {n}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
