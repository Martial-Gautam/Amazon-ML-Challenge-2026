"""Nearest-neighbour candidate source (GPU): for every S1 record, the top-k S2/S3 records of the same country by cosine
of e5 "name | address" embeddings. Used twice: with the off-the-shelf multilingual-e5-small (-> work/n4) and with the
fine-tuned retriever from ret_train.py (-> work/ret). Raw text only, label-free.

python knn_search.py --work work --split train --model intfloat/multilingual-e5-small --out work/n4
python knn_search.py --work work --split test  --model work/retriever                 --out work/ret
-> <out>/{split}_knn.parquet: i, j, knn_rank (0-based), knn_cos
"""
import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 64


def embed(texts, tok, model, batch=2048):
    """L2-normalised mean-pooled e5 vectors (fp16, kept on CPU), length-sorted batches."""
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = torch.empty(len(texts), model.module.config.hidden_size if hasattr(model, "module") else model.config.hidden_size,
                      dtype=torch.float16)
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for n, s in enumerate(range(0, len(order), batch)):
            k = order[s:s + batch]
            enc = tok(["query: " + texts[x] for x in k], max_length=MAX_LEN, truncation=True, padding=True,
                      return_tensors="pt").to("cuda")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            out[torch.from_numpy(k)] = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).half().cpu()
            if n % 500 == 0:
                print(f"embedded {s}/{len(texts)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path("work"), help="normalised records: <work>/{split}_s1, _cand parquet")
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--model", required=True, help="e5 model name or a fine-tuned model folder")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = torch.nn.DataParallel(AutoModel.from_pretrained(a.model).cuda().eval())
    txt = lambda w: (pl.scan_parquet(str(a.work / f"{a.split}_{w}" / "*.parquet"))
                     .select("idx", pl.col("ctry").cast(pl.String),
                             s=pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null(""))
                     .collect())
    s1, cand = txt("s1"), txt("cand")
    res = []
    for ctry in sorted(set(s1["ctry"].unique()) & set(cand["ctry"].unique())):  # open set of countries
        q, c = s1.filter(pl.col("ctry") == ctry), cand.filter(pl.col("ctry") == ctry)
        print(f"{ctry}: {q.height} S1, {c.height} candidates", flush=True)
        C = embed(c["s"].to_list(), tok, model).cuda()
        Q = embed(q["s"].to_list(), tok, model)
        cidx = torch.from_numpy(c["idx"].to_numpy().astype(np.int64)).cuda()
        k = min(a.top, c.height)
        J, S = np.empty((q.height, k), np.uint32), np.empty((q.height, k), np.float16)
        for s in range(0, q.height, 256):
            v, ix = (Q[s:s + 256].cuda() @ C.T).topk(k, dim=1)
            J[s:s + 256], S[s:s + 256] = cidx[ix].cpu().numpy(), v.cpu().numpy()
        del C
        torch.cuda.empty_cache()
        res.append(pl.DataFrame({"i": np.repeat(q["idx"].to_numpy().astype(np.uint32), k), "j": J.ravel(),
                                 "knn_rank": np.tile(np.arange(k, dtype=np.uint8), q.height),
                                 "knn_cos": S.ravel().astype(np.float32)}))
    a.out.mkdir(parents=True, exist_ok=True)
    pl.concat(res).write_parquet(a.out / f"{a.split}_knn.parquet")
    print(f"wrote {a.out / f'{a.split}_knn.parquet'}", flush=True)


if __name__ == "__main__":
    main()
